# GGUF as a source (T74): a GGUF written here the way llama.cpp writes one (Q8_0 matrices, F32 vectors, q and k of
# a Llama turned into llama2.c's order, a Qwen2's left alone) must give the very checkpoint that the safetensors of
# the same values gives. The real file is checked by tests/gguf_check.py.
import json
import struct

import numpy as np
import pytest
from conftest import synthetic_weights
from test_bias import qwen2
from test_convert import converted, hugging_face, reader, safetensors_file

import llama2_convert
from llama2_convert import Conversion, Incomplete, Safetensors, gguf_read
from llama2_numpy import rope_frequencies

NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
         "lm_head.weight": "output.weight"}
LAYER = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q",
         "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
         "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}  # a Qwen3's (T203)


def q8_0_blocks(values):
    """llama.cpp's Q8_0 (ggml-quants.c's quantize_row_q8_0_ref): per 32 values d = largest / 127 in float32, the
    values times 1 / d rounded half away from zero, and d kept as float16; a d under float16's smallest half step
    is kept as 0, and its values read back 0. T145: as llama.cpp rounds, to the bit (it rounded by the float16 d
    before), so that tests/gguf_check.py's nearest reference is 0 off for a GGUF of the original's values."""
    groups = values.reshape(-1, 32).astype(np.float32)
    d = np.abs(groups).max(axis=1) / np.float32(127)
    scales = d.astype(np.float16)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d > 0)
        scaled = groups * inverse[:, None]
        ints = np.where(scales[:, None] > 0, np.sign(scaled) * np.floor(np.abs(scaled) + 0.5), 0).astype(np.int8)
    return np.concatenate([scales.view(np.uint8).reshape(-1, 2), ints.view(np.uint8)], axis=1).tobytes(), \
        (ints.astype(np.float32) * scales.astype(np.float32)[:, None]).reshape(values.shape)


def turn(w, heads):
    """What llama.cpp's convert does to q and k of a Llama."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_name(name):
    if name in NAMES:
        return NAMES[name]
    _, _, layer, *rest = name.split(".")
    return f"blk.{layer}.{LAYER['.'.join(rest[:-1])]}.{rest[-1]}"


def gguf_file(tensors, published, vocab_size, arch="llama", pre="gpt-2", theta=10000.0, more=(), extra=None):
    """A GGUF v3 of these Hugging Face tensors, and the tensors as the GGUF holds them (Q8_0 rounds).
    more: further metadata (key, GGUF type, value); extra: {GGUF name: float32 values} written as they are
    (rope_freqs.weight)."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    heads = {"q_proj": published["num_attention_heads"], "k_proj": published["num_key_value_heads"]}
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, published["num_hidden_layers"]),
                (f"{arch}.context_length", 4, published["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, published["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, published["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, published["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, published["num_key_value_heads"]),
                (f"{arch}.rope.freq_base", 6, theta), ("tokenizer.ggml.model", 8, "gpt2"),
                ("tokenizer.ggml.pre", 8, pre), ("tokenizer.ggml.bos_token_id", 4, 1),
                ("tokenizer.ggml.eos_token_id", 4, 2), *more]
    tokens = [f"w{i}" for i in range(vocab_size)]
    out = [b"GGUF", struct.pack("<IQQ", 3, len(tensors) + len(extra or {}), len(metadata) + 3)]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(string(value) if kind == 8 else struct.pack({4: "<I", 6: "<f"}[kind], value))
    out.append(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens)) + b"".join(map(string, tokens)))
    out.append(string("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 5, len(tokens)) + struct.pack(f"<{len(tokens)}i", *[1] * len(tokens)))
    out.append(string("tokenizer.ggml.merges") + struct.pack("<IIQ", 9, 8, 0))
    held, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        stored = tensor
        kind = next((k for k in heads if f".{k}." in name), None)
        if arch == "llama" and kind:
            stored = turn(tensor, heads[kind])
        if tensor.ndim == 2:
            blob, rounded = q8_0_blocks(stored)
            held[name] = rounded  # as the GGUF holds it (turned, for q and k of a Llama)
            type_ = 8
        else:
            blob, type_ = stored.astype(np.float32).tobytes(), 0
            held[name] = stored.astype(np.float32)
        out.append(string(gguf_name(name)) + struct.pack("<I", tensor.ndim) + struct.pack(f"<{tensor.ndim}Q", *reversed(tensor.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    # after every other tensor, as Swallow 8B's GGUF has its rope_freqs
    for name, values in (extra or {}).items():
        blob = np.asarray(values, np.float32).tobytes()
        out.append(string(name) + struct.pack("<I", 1) + struct.pack("<Q", len(values)) + struct.pack("<IQ", 0, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    # the values the GGUF stands for, back in Hugging Face's order: what a safetensors of the same model holds
    same = {}
    for name, values in held.items():
        kind = next((k for k in heads if f".{k}." in name), None)
        same[name] = llama2_convert.unturned(values, heads[kind]) if arch == "llama" and kind else values
    return head + b"".join(blobs), same


def fed(file, dtype, chunk=4096):
    for size in (100, 1000, len(file)):
        try:
            conversion = Conversion.from_gguf(file[:size], dtype=dtype, max_seq_len=1 << 20)
            break
        except Incomplete:
            continue
    for start in range(conversion.base, len(file), chunk):
        conversion.feed(file[start:start + chunk])
    conversion.finish()
    return conversion


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["llama", "llama-own-classifier", "qwen2"])
def test_a_gguf_converts_to_the_checkpoint_of_the_same_values(model, dtype):
    config, weights = synthetic_weights(n_kv_heads=2, shared=model != "llama-own-classifier")
    tensors, published = (qwen2 if model == "qwen2" else hugging_face)(config, weights, model != "llama-own-classifier")
    file, same = gguf_file(tensors, published, config["vocab_size"], "qwen2" if model == "qwen2" else "llama")
    conversion = fed(file, dtype)
    expected = converted(Safetensors(reader(safetensors_file(same))), published, dtype)
    assert bytes(conversion.checkpoint) == expected
    assert conversion.options["arch"] == "llama" and conversion.options["bias"] is (model == "qwen2")
    assert conversion.options["tokenizer_kind"] == "bytebpe" and conversion.options["pretokenizer"] == "gpt2"


@pytest.mark.parametrize("head_size", [0, 16])
def test_a_gguf_says_the_epsilon_and_the_size_of_a_head(head_size):
    """T144 (the review of T124): llama.cpp writes rms_norm_eps as attention.layer_norm_rms_epsilon and head_dim as
    attention.key_length. Unread, the GGUFs of the list (Qwen2.5, TinySwallow: 1e-6) would go back to 1e-5 without a
    word, and heads of another size than dim / heads would be laid out as dim / heads. A key_length of dim / heads
    (every GGUF of the list) is no head_dim in the options: they stay what they were."""
    config, weights = synthetic_weights(n_kv_heads=2, head_size=head_size)
    tensors, published = hugging_face(config, weights, True)
    extra = [("llama.attention.layer_norm_rms_epsilon", 6, 1e-6), ("llama.attention.key_length", 4, config["head_size"])]
    file, same = gguf_file(tensors, published, config["vocab_size"], more=extra)
    conversion = fed(file, "float32")
    assert conversion.options["rms_norm_eps"] == 1e-6
    assert conversion.options.get("head_dim") == (head_size or None)
    assert bytes(conversion.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "float32")


def test_q8_0_comes_back_as_the_same_int8():
    values = (np.random.default_rng(3).standard_normal((4, 64)) * 0.2).astype(np.float32)
    blob, rounded = q8_0_blocks(values)
    back = llama2_convert.q8_0(blob).reshape(values.shape)
    assert np.array_equal(back, rounded)
    ints, scales = llama2_convert.quantize(back)
    assert np.array_equal(ints.astype(np.float32) * scales[:, None], rounded.reshape(-1, 32)), "lossless"


def test_a_short_head_asks_for_more_and_other_files_are_refused():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"])
    with pytest.raises(Incomplete):
        gguf_read(file[:200])
    with pytest.raises(ValueError, match="not a GGUF"):
        gguf_read(b"PK\x03\x04" + file[4:])
    k_quant = bytearray(file)
    metadata, found, base = gguf_read(file)
    # make the first matrix a Q4_K (type 12): its type is the u32 before its u64 offset in the tensor infos
    at = file.index(b"token_embd.weight") + len(b"token_embd.weight") + 4 + 16
    k_quant[at:at + 4] = struct.pack("<I", 12)
    with pytest.raises(ValueError, match="K-quants"):
        Conversion.from_gguf(bytes(k_quant))


def test_a_q8_0_tensor_whose_rows_are_not_groups_of_32_is_refused():
    """Fable's review of T74: the Q8_0 reader counts 34 bytes per 32 values, so a row that is not a multiple of 32
    would be read at the wrong offsets and write nonsense instead of failing."""
    config, weights = synthetic_weights(dim=48, hidden_dim=64, n_heads=2, n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"])
    with pytest.raises(ValueError, match="multiple of 32"):
        Conversion.from_gguf(file)


# ---- T136's second stage: the weights of a GGUF with the vocabulary and config.json of the original repository
def with_original(file, published, vocabulary, name, dtype, chunk=4096):
    """What worker.js does for such an item: the GGUF's header made a safetensors one, then the original's files."""
    header, base = llama2_convert.gguf_weights(file, json.dumps(published))
    conversion = Conversion(header, base, json.dumps(published), vocabulary, name, dtype=dtype, max_seq_len=1 << 20,
                            start=base)
    for start in range(base, len(file), chunk):
        conversion.feed(file[start:start + chunk])
    conversion.finish()
    return conversion


def sentencepiece(vocab_size):
    from make_hf_fixture import field
    NORMAL, UNKNOWN, CONTROL = 1, 2, 3
    pieces = [("<unk>", UNKNOWN), ("<s>", CONTROL), ("</s>", CONTROL)]
    pieces += [(f"▁w{i}", NORMAL) for i in range(vocab_size - len(pieces))]
    model = b"".join(field(1, field(1, text.encode()) + field(2, -float(i)) + field(3, kind)) for i, (text, kind) in enumerate(pieces))
    return model + field(2, field(3, 1)) + field(3, field(1, b"identity"))


EPS = ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)  # llama.cpp always writes it
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def llama3_table(published):
    width = llama2_convert.head_size(published)
    return rope_frequencies(width, 10000.0) / rope_frequencies(width, 10000.0, LLAMA3)


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["sentencepiece", "llama3"])
def test_a_gguf_with_the_originals_vocabulary_is_the_safetensors_conversion(model, dtype):
    """The same checkpoint, tokenizer.bin and options as the safetensors of the same values with the same files:
    the GGUF's own vocabulary and settings play no part, and a rope_freqs table is checked and dropped."""
    config, weights = synthetic_weights(n_kv_heads=2, shared=False)
    tensors, published = hugging_face(config, weights, False)
    published["rms_norm_eps"] = 1e-5
    table = None
    if model == "llama3":
        published["rope_scaling"] = LLAMA3
        table = llama3_table(published)
    file, same = gguf_file(tensors, published, config["vocab_size"], extra=None if table is None else {"rope_freqs.weight": table}, more=[EPS])
    vocabulary, name = (sentencepiece(config["vocab_size"]), "tokenizer.model") if model == "sentencepiece" else \
        (json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                     "vocab": [[f"w{i}", -float(i)] for i in range(config["vocab_size"])]}}).encode(), "tokenizer.json")
    got = with_original(file, published, vocabulary, name, dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, name,
                          dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert (got.options.get("rope_scaling") == LLAMA3) is (model == "llama3")


@pytest.mark.parametrize("change, what", [
    (dict(num_attention_heads=2, num_key_value_heads=2, head_dim=16), "number of heads"),
    (dict(num_key_value_heads=4), "number of key-value heads"),
    (dict(rope_theta=500000.0), "RoPE theta"),
    (dict(rms_norm_eps=1e-6), "RMSNorm epsilon"),
    (dict(model_type="qwen2"), "architecture"),
    (dict(tie_word_embeddings=False), "classifier of its own"),
    (dict(num_hidden_layers=1), "number of layers"),
])
def test_a_gguf_that_is_not_the_originals_is_refused(change, what):
    """What the sizes of the tensors do not show (T136's review, point 5): heads and key-value heads of the same
    product, fewer layers than the GGUF has (it went through, cut to them), a classifier that would silently be the
    embedding, and the numbers that are no tensor."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published["rms_norm_eps"] = 1e-5
    file, _ = gguf_file(tensors, published, config["vocab_size"], more=[EPS])
    llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**published, **change}))


def test_a_rope_freqs_table_that_is_not_the_originals_scaling_is_refused():
    """A GGUF's rope_freqs is dropped for the engine's own tables (rope_frequencies of rope_scaling): only where the
    two agree. Here it comes after every other tensor, as in Swallow 8B's GGUF, and is refused before finish()."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published["rope_scaling"] = LLAMA3
    table = llama3_table(published)
    table[-1] *= 1.01
    file, _ = gguf_file(tensors, published, config["vocab_size"], extra={"rope_freqs.weight": table})
    vocabulary = sentencepiece(config["vocab_size"])
    with pytest.raises(ValueError, match="scales its RoPE otherwise"):
        with_original(file, published, vocabulary, "tokenizer.model", "int8")
    # without the original's config.json (?hf= of a GGUF) the table is refused as before
    with pytest.raises(ValueError, match="rope_freqs table"):
        Conversion.from_gguf(file)


# ---- T203 (T136's fourth stage): a Qwen3, whose GGUF holds q, k and the norms of their heads in Hugging Face's order
def qwen3_gguf(head_size, eps=1e-6):
    from test_qwen3 import qwen3
    config, weights = synthetic_weights(n_kv_heads=2, head_size=head_size)
    tensors, published = qwen3(config, weights, True)
    published["rms_norm_eps"] = eps
    # llama.cpp writes the size of a head as the length of a key, whatever it is, and always the epsilon
    more = [("qwen3.attention.key_length", 4, config["head_size"]), ("qwen3.attention.layer_norm_rms_epsilon", 6, eps)]
    file, same = gguf_file(tensors, published, config["vocab_size"], "qwen3", pre="qwen2", more=more)
    return config, published, file, same


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("head_size", [0, 16])
def test_a_qwen3_gguf_with_the_originals_files_is_the_safetensors_conversion(head_size, dtype):
    """The checkpoint, tokenizer.bin and options of the safetensors of the same values, the norms of q and k with them
    (qk_norm), and heads of another size than dim / heads (head_dim). Unread, the norms would leave a Qwen3 that runs
    as a Llama and writes nonsense without a word."""
    config, published, file, same = qwen3_gguf(head_size)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, published, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary,
                          "tokenizer.json", dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["qk_norm"] is True and got.options.get("head_dim") == (head_size or None)


@pytest.mark.parametrize("head_size", [0, 16])
def test_a_qwen3_gguf_alone_converts_to_the_checkpoint_of_the_same_values(head_size):
    """?hf= of a Qwen3's GGUF alone (its own vocabulary and settings): the same checkpoint, and the options say the
    norms and the size of a head."""
    config, published, file, same = qwen3_gguf(head_size)
    conversion = fed(file, "int8")
    assert bytes(conversion.checkpoint) == converted(Safetensors(reader(safetensors_file(same))), published, "int8")
    assert conversion.options["qk_norm"] is True and conversion.options.get("head_dim") == (head_size or None)
    assert conversion.options["rms_norm_eps"] == pytest.approx(1e-6)


@pytest.mark.parametrize("change, what", [
    (dict(model_type="qwen2"), "architecture"),
    (dict(head_dim=8), "size of a head"),
    (dict(rms_norm_eps=1e-5), "RMSNorm epsilon"),
])
def test_a_qwen3_gguf_that_is_not_the_originals_is_refused(change, what):
    config, published, file, _ = qwen3_gguf(16)
    llama2_convert.gguf_weights(file, json.dumps(published))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**published, **change}))


# ---- T136's third stage: GPT-2 and GPT-NeoX, as llama.cpp's convert_hf_to_gguf.py writes them
GPT2_NAMES = {"wte.weight": "token_embd.weight", "wpe.weight": "position_embd.weight", "ln_f.weight": "output_norm.weight",
              "ln_f.bias": "output_norm.bias", "ln_1": "attn_norm", "attn.c_attn": "attn_qkv", "attn.c_proj": "attn_output",
              "ln_2": "ffn_norm", "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down"}
NEOX_NAMES = {"gpt_neox.embed_in.weight": "token_embd.weight", "embed_out.weight": "output.weight",
              "gpt_neox.final_layer_norm.weight": "output_norm.weight", "gpt_neox.final_layer_norm.bias": "output_norm.bias",
              "input_layernorm": "attn_norm", "attention.query_key_value": "attn_qkv", "attention.dense": "attn_output",
              "post_attention_layernorm": "ffn_norm", "mlp.dense_h_to_4h": "ffn_up", "mlp.dense_4h_to_h": "ffn_down"}
CONV1D = ("attn_qkv", "attn_output", "ffn_up", "ffn_down")  # GPT-2's matrices, which llama.cpp stores as (out, in)


def other_gguf(tensors, config):
    """A GGUF v3 of a GPT-2's or GPT-NeoX's Hugging Face tensors the way llama.cpp writes one (checked on the real
    files by Range, T136's third stage): GPT-2's Conv1D matrices turned to (out, in) and a copy of the embedding as
    output.weight, GPT-NeoX's query_key_value (and its bias) as all of q, then k, then v; Q8_0 matrices (GPT-2's
    positions F32), F32 vectors. Returns the file and, under the Hugging Face names, the values it stands for."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    neox = config["model_type"] == "gpt_neox"
    arch = "gptneox" if neox else "gpt2"
    heads = config["num_attention_heads"] if neox else config["n_head"]
    dim = config["hidden_size"] if neox else config["n_embd"]
    metadata = [("general.architecture", 8, arch), (f"{arch}.embedding_length", 4, dim),
                (f"{arch}.attention.head_count", 4, heads), (f"{arch}.attention.layer_norm_epsilon", 6, 1e-5)]
    if neox:
        metadata += [(f"{arch}.block_count", 4, config["num_hidden_layers"]),
                     (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                     (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                     (f"{arch}.rope.dimension_count", 4, llama2_convert.rotary_dim(config)),
                     (f"{arch}.use_parallel_residual", 7, config["use_parallel_residual"])]
    else:
        metadata += [(f"{arch}.block_count", 4, config["n_layer"]), (f"{arch}.context_length", 4, config["n_positions"]),
                     (f"{arch}.feed_forward_length", 4, config["n_inner"])]
    names = NEOX_NAMES if neox else GPT2_NAMES
    stored = {}  # GGUF name: (Hugging Face name, the values as the GGUF holds them)
    for name, tensor in tensors.items():
        if name.endswith(("masked_bias", "inv_freq")):
            continue  # nothing llama.cpp writes
        short = name.removeprefix("transformer.")
        if short in names:
            stored[names[short]] = (name, tensor)
            continue
        parts = short.split(".")
        at = 3 if neox else 2  # gpt_neox.layers.N. or h.N.
        layer, what, kind = parts[at - 1], ".".join(parts[at:-1]), parts[-1]
        gguf, value = names[what], tensor
        if not neox and kind == "weight" and gguf in CONV1D:
            value = tensor.T
        if neox and gguf == "attn_qkv":
            value = tensor.reshape(heads, 3, dim // heads, -1).swapaxes(0, 1).reshape(tensor.shape)
        stored[f"blk.{layer}.{gguf}.{kind}"] = (name, value)
    if not neox:
        stored["output.weight"] = (None, stored["token_embd.weight"][1])
    out = [b"GGUF", struct.pack("<IQQ", 3, len(stored), len(metadata))]
    for key, kind, value in metadata:
        out.append(string(key) + struct.pack("<I", kind))
        out.append(string(value) if kind == 8 else struct.pack({4: "<I", 6: "<f", 7: "<?"}[kind], value))
    blobs, offset, same = [], 0, {}
    for gguf, (name, value) in stored.items():
        if value.ndim == 2 and gguf != "position_embd.weight":
            blob, held = q8_0_blocks(np.ascontiguousarray(value))
            type_ = 8
        else:
            blob, held, type_ = np.ascontiguousarray(value, np.float32).tobytes(), value.astype(np.float32), 0
        if name is not None:  # back to the Hugging Face tensor these values stand for
            if not neox and gguf.startswith("blk.") and gguf.endswith(".weight") and gguf.split(".")[2] in CONV1D:
                held = held.T
            if neox and ".attn_qkv." in gguf:
                held = held.reshape(3, heads, dim // heads, -1).swapaxes(0, 1).reshape(held.shape)
            same[name] = np.ascontiguousarray(held)
        out.append(string(gguf) + struct.pack("<I", value.ndim) + struct.pack(f"<{value.ndim}Q", *reversed(value.shape))
                   + struct.pack("<IQ", type_, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    head += b"\0" * (-len(head) % 32)
    return head + b"".join(blobs), same


def unigram(vocab_size):
    return json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                       "vocab": [[f"w{i}", -float(i)] for i in range(vocab_size)]}}).encode()


def the_other(model):
    from test_gpt2 import gpt2_model
    from test_neox import neox_model
    if model == "gpt2":
        return gpt2_model()
    return neox_model(*{"neox": (0.25, True), "neox-serial": (1.0, False)}[model])


@pytest.mark.parametrize("dtype", ["int8", "float32"])
@pytest.mark.parametrize("model", ["gpt2", "neox", "neox-serial"])
def test_a_gpt2_or_neox_gguf_with_the_originals_files_is_the_safetensors_conversion(model, dtype):
    """T136's third stage: GPT-2's matrices turned back to Conv1D and GPT-NeoX's q, k and v put back per head give
    the checkpoint, tokenizer.bin and options of the safetensors of the same values. Leaving either out of the reader
    (the transposed entries, llama2_convert.unsplit) makes the checkpoint differ."""
    tensors, config = the_other(model)
    file, same = other_gguf(tensors, config)
    vocabulary = unigram(config["vocab_size"])
    got = with_original(file, config, vocabulary, "tokenizer.json", dtype)
    safetensors = safetensors_file(same)
    size = struct.unpack("<Q", safetensors[:8])[0]
    expected = Conversion(safetensors[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary, "tokenizer.json",
                          dtype=dtype, max_seq_len=1 << 20)
    expected.feed(safetensors)
    expected.finish()
    assert bytes(got.checkpoint) == bytes(expected.checkpoint)
    assert bytes(got.tokenizer) == bytes(expected.tokenizer)
    assert got.options == expected.options
    assert got.options["arch"] == ("gpt2" if model == "gpt2" else "neox")


@pytest.mark.parametrize("model, change, what", [
    ("neox", dict(rotary_pct=0.5), "rotated values"),
    ("neox", dict(use_parallel_residual=False), "parallel residual"),
    ("neox", dict(layer_norm_eps=1e-6), "LayerNorm epsilon"),
    ("neox", dict(num_hidden_layers=1), "number of layers"),
    ("neox", dict(num_attention_heads=2), "number of heads"),
    ("neox", dict(model_type="llama"), "architecture"),
    ("gpt2", dict(n_head=2), "number of heads"),
    ("gpt2", dict(n_layer=1), "number of layers"),
])
def test_a_gpt2_or_neox_gguf_that_is_not_the_originals_is_refused(model, change, what):
    """What the tensors do not say: GPT-NeoX's rotated part and parallel branches (the options carry them, and the
    wrong ones write nonsense without a word, T72), heads of the same product, the layers."""
    tensors, config = the_other(model)
    file, _ = other_gguf(tensors, config)
    llama2_convert.gguf_weights(file, json.dumps(config))  # its own config goes through
    with pytest.raises(ValueError, match=what):
        llama2_convert.gguf_weights(file, json.dumps({**config, **change}))
