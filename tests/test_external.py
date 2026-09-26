# The engine with its weights outside Python (T93): Llama(external=...) must describe every tensor exactly where
# the NumPy engine reads it, for every architecture and dtype, so that public/forward.js computes on the same numbers.
# The forward pass itself runs in JavaScript: tests/smoke.mjs checks it against the NumPy one.
import numpy as np
import pytest
from conftest import pack_tokenizer, synthetic_weights, tiny_vocab
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gpt2 import VOCAB as GPT2_VOCAB
from test_gpt2 import gpt2_model
from test_neox import VOCAB as NEOX_VOCAB
from test_neox import neox_model

from llama2_convert import Safetensors, normalize, rotary_dim
from llama2_numpy import OUTLIER_CHANNELS, Llama, outlier_channels


class Outside:
    """What public/forward.js offers the engine, with the checkpoint in a bytes object."""

    def __init__(self, data):
        self.data, self.size, self.plan = data, len(data), None

    def read(self, offset, length):
        return self.data[offset:offset + length]

    def start(self, plan):
        self.plan = plan
        return Engine()


class Engine:
    backend = "outside"

    def bind(self, logits):
        self.logits = logits

    def forward(self, token, pos, need_logits):
        self.logits[:] = token  # something to see that the array is the one bound


def models():
    config, weights = synthetic_weights()
    yield "llama", *hugging_face(config, weights, True), config["vocab_size"], {}
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    yield "qwen2", *qwen2(config, weights, False), config["vocab_size"], {"bias": True}
    tensors, published = gpt2_model()
    yield "gpt2", tensors, published, GPT2_VOCAB, {"arch": "gpt2"}
    tensors, published = neox_model(0.25, True)
    yield "neox", tensors, published, NEOX_VOCAB, {"arch": "neox", "rotary": rotary_dim(normalize(published)),
                                                   "parallel_residual": True}


def widened(checkpoint, tensor):
    """The float32 values a tensor of the plan stands for, read from the file the way the NumPy engine reads it."""
    count = int(np.prod(tensor["shape"]))
    if tensor["kind"] == "int8":
        values = np.frombuffer(checkpoint, dtype=np.int8, count=count, offset=tensor["offset"])
        scales = np.frombuffer(checkpoint, dtype=np.float32, count=count // tensor["group"], offset=tensor["scales"])
        return (values.reshape(-1, tensor["group"]).astype(np.float32) * scales[:, None]).reshape(tensor["shape"])
    dtype = np.float16 if tensor["kind"] == "f16" else np.float32
    return np.frombuffer(checkpoint, dtype=dtype, count=count, offset=tensor["offset"]).astype(np.float32).reshape(tensor["shape"])


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("name, tensors, published, vocab, options", list(models()), ids=[m[0] for m in models()])
def test_the_plan_points_at_what_the_numpy_engine_reads(name, tensors, published, vocab, options, dtype):
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, dtype)
    tokenizer = pack_tokenizer(tiny_vocab(vocab))
    reference = Llama(checkpoint, tokenizer, dtype=dtype, **options)  # NumPy: every matrix widened to float32
    outside = Outside(checkpoint)
    engine = Llama(None, tokenizer, dtype=dtype, external=outside, **options)
    plan = outside.plan
    # int8 stays int8 only where every row is whole groups of 32 (the kernels' groups); else forward.js widens it
    whole = all(n % 32 == 0 for n in (reference.dim, reference.hidden_dim, reference.n_kv_heads * reference.head_size))
    assert plan["arch"] == options.get("arch", "llama") and plan["int8"] is (dtype == "int8" and whole)
    for attribute, tensor in plan["tensors"].items():
        want = getattr(reference, attribute)
        if isinstance(want, tuple):  # an int8 table the NumPy engine keeps as it is (a classifier of its own)
            values, scales = want
            want = (values.astype(np.float32) * scales).reshape(tensor["shape"])
        assert np.array_equal(widened(checkpoint, tensor), np.asarray(want, dtype=np.float32)), attribute
    # the tables Python computes (a checkpoint without them, or GPT-2's zeros) come along as float32 bytes
    for attribute, data in plan["derived"].items():
        assert np.array_equal(np.frombuffer(data, dtype=np.float32).reshape(getattr(reference, attribute).shape),
                              getattr(reference, attribute)), attribute
    assert set(plan["tensors"]) | set(plan["derived"]) >= {"freq_cis_real", "freq_cis_imag"} or name == "gpt2"
    assert plan["shared_classifier"] is (reference.wcls is reference.token_embedding_table)
    if plan["int8"]:
        assert plan["outliers"] == [int(c) for c in outlier_channels(reference.rms_final_weight, min(OUTLIER_CHANNELS, reference.dim))]
    # the logits come back into one array of Python's, which the sampling then reads
    logits = engine.forward(7, 0)
    assert logits is engine.forward(3, 1) and logits[0] == 3


def test_a_file_of_the_wrong_size_is_refused():
    config, weights = synthetic_weights()
    checkpoint = converted(Safetensors(reader(safetensors_file(hugging_face(config, weights, True)[0]))),
                           hugging_face(config, weights, True)[1], "int8")
    with pytest.raises(ValueError, match="asks for"):
        Llama(None, pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype="float32", external=Outside(checkpoint))
    with pytest.raises(ValueError, match="in Python"):
        Llama(None, pack_tokenizer(tiny_vocab(config["vocab_size"])), dtype="int8", external=Outside(checkpoint),
              disable=("kernels",))


def test_a_prompt_goes_through_forward_many_and_writes_the_same():
    """T108: with forward_many, generate() hands the prompt's tokens over in blocks, each at its positions, and the
    text, the counts and what follows are what they are one token at a time. forward_many here is the NumPy forward
    token by token, so any difference is generate()'s bookkeeping."""
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    checkpoint = converted(Safetensors(reader(safetensors_file(tensors))), published, "float32")
    tokenizer = pack_tokenizer(tiny_vocab(config["vocab_size"]))
    plain = Llama(checkpoint, tokenizer)
    prompt = "abcabcabcabc"
    expected = "".join(plain.generate(prompt, steps=40, temperature=0.0))
    expected_stats = dict(plain.stats)
    blocks = []
    batched = Llama(checkpoint, tokenizer)

    def many(tokens, pos):
        blocks.append((list(tokens), pos))
        for i, token in enumerate(tokens):
            batched.forward(token, pos + i, need_logits=False)

    batched.forward_many = many
    # T147: the block is what the engine says now (forward.js's promptBlock: more where the GPU takes the prompt)
    batched.prompt_block = lambda: 5
    assert "".join(batched.generate(prompt, steps=40, temperature=0.0)) == expected
    assert "".join(batched.generate(prompt, steps=40, temperature=0.0, echo=False)) == \
        "".join(plain.generate(prompt, steps=40, temperature=0.0, echo=False))
    prompt_tokens = plain.tokenizer.encode(prompt, plain.specials)
    fed = [plain.bos] + prompt_tokens[:-1]
    assert [pos for _, pos in blocks[:len(blocks) // 2]] == list(range(0, len(fed), 5))
    assert [t for block, _ in blocks[:len(blocks) // 2] for t in block] == fed
    for key in ("tokens", "sampled", "prompt_tokens"):
        assert batched.stats[key] == expected_stats[key], key
