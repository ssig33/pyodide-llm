# Llama 3 (T106): the llama3 kind of RoPE scaling, its pre-tokenizer (digits up to three at a time), ignore_merges,
# the BOS its chat template writes, and the special tokens of a chat template read from tokenizer.json.
import json
import struct

import numpy as np
import pytest
from conftest import CORPUS, TEXTS, synthetic_weights
from test_convert import converted, hugging_face, reader, safetensors_file
from test_gguf import gguf_file

import llama2_convert
from llama2_convert import Arrays, Conversion, check_config, gguf_model, gguf_read, rope_table, tokenizer_json_options
from llama2_numpy import Llama, Tokenizer, pretokenize, rope_amplitude, rope_frequencies, rope_tables

LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+")
# config.json of Llama 3.2 1B Instruct (unsloth/Llama-3.2-1B-Instruct@5a8abab4)
SCALING = {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
           "rope_type": "llama3"}
# what llama.cpp's convert wrote as rope_freqs into bartowski/Llama-3.2-1B-Instruct-GGUF@067b946c (Q8_0): for
# head_size 64 and rope_theta 500000, the plain frequency over the scaled one, pair by pair. An implementation of
# the same formula independent of this one (transformers needs PyTorch, which this machine has not).
# config.json of deepseek-ai/deepseek-coder-1.3b-instruct@e063262d (T126)
LINEAR = {"factor": 4.0, "type": "linear"}
LLAMA_CPP_DIVISORS = [1.0] * 15 + [1.6513293, 3.2922628, 9.666731] + [32.0] * 14


def test_the_llama3_frequencies_are_llama_cpps():
    divisors = rope_frequencies(64, 500000.0) / rope_frequencies(64, 500000.0, SCALING)
    assert np.allclose(divisors, LLAMA_CPP_DIVISORS, rtol=1e-6)


def test_without_scaling_the_frequencies_are_what_they_were():
    assert np.array_equal(rope_frequencies(64, 10000.0), 1.0 / 10000.0 ** (np.arange(0, 64, 2) / 64))
    assert rope_amplitude(None) == rope_amplitude(SCALING) == rope_amplitude(LINEAR) == 1.0
    with pytest.raises(ValueError, match="dynamic"):
        rope_frequencies(64, 10000.0, {"rope_type": "dynamic", "factor": 4.0})


def test_the_linear_frequencies_are_the_plain_ones_over_the_factor():
    """T126 (deepseek-coder): positions divided by factor, which is every frequency divided by it"""
    assert np.allclose(rope_frequencies(64, 100000.0) / rope_frequencies(64, 100000.0, LINEAR), 4.0, rtol=1e-12)


# T235: config.json of prism-ml/Ternary-Bonsai-1.7B-unpacked@3aca8400 (head_dim 128, rope_theta 1e6), and the same
# with the optional keys. The frequencies are what transformers 5.16.1's _compute_yarn_parameters made of them (with
# PyTorch 2.14, in float32), its attention_factor 1.138629436111989 for both
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}
YARN_KEYS = {**YARN, "original_max_position_embeddings": 16384, "beta_fast": 16, "beta_slow": 2, "truncate": False}
FAST = [1.0, 0.805842221, 0.649381638, 0.523299158, 0.421696514, 0.339820832, 0.273841977, 0.220673397, 0.177827939,
        0.143301263, 0.115478203, 0.0930572003, 0.0749894157, 0.0604296438, 0.0486967526, 0.0392418988, 0.0316227786,
        0.0254829675]
SLOW = [0.000162345415, 0.000130824774, 0.000105424122, 8.49552089e-05, 6.84604893e-05, 5.51683552e-05, 4.44569851e-05,
        3.58253164e-05, 2.88695483e-05, 2.32643015e-05, 1.87473561e-05, 1.51074091e-05, 1.21741887e-05, 9.81047469e-06,
        7.90569356e-06, 6.37074163e-06, 5.13381246e-06, 4.13704265e-06, 3.33380353e-06, 2.68651957e-06, 2.16491094e-06,
        1.74457659e-06, 1.40585337e-06, 1.13289593e-06, 9.1293532e-07, 7.35681795e-07, 5.92843435e-07, 4.7773824e-07,
        3.84981632e-07, 3.10234441e-07]
TRANSFORMERS_YARN = FAST + [0.0196292847, 0.0150880395, 0.0115702599, 0.00884971209, 0.00674942741, 0.00513110729,
                            0.00388677069, 0.00293220137, 0.0022017851, 0.00164446514, 0.00122056005, 0.000899271923,
                            0.000656733348, 0.000474476255, 0.000338235288, 0.000237012428] + SLOW
TRANSFORMERS_YARN_KEYS = FAST + [0.0205352511, 0.0165481716, 0.0133352149, 0.0107460786, 0.00865964312, 0.00697830599,
                                 0.00544650853, 0.00403620768, 0.00296823028, 0.00216281135, 0.00155825494, 0.00110692508,
                                 0.000772111875, 0.000525583862, 0.000345679902, 0.000215822452] + SLOW


@pytest.mark.parametrize("scaling, expected", [(YARN, TRANSFORMERS_YARN), (YARN_KEYS, TRANSFORMERS_YARN_KEYS)],
                         ids=["bonsai", "keys"])
def test_the_yarn_frequencies_are_transformers(scaling, expected):
    """T235: the fast pairs as they were, the slow ones factor times slower (SLOW is the plain frequency over 4), the
    ones between on a line; and the tables scaled by 0.1 ln(4) + 1."""
    assert np.allclose(rope_frequencies(128, 1e6, scaling), expected, rtol=3e-7)
    assert rope_amplitude(scaling) == pytest.approx(1.138629436111989, rel=1e-15)
    assert rope_amplitude({**scaling, "attention_factor": 1.5}) == 1.5
    cos, sin = rope_tables(8, 128, 1e6, scaling)
    angles = np.arange(8)[:, None] * rope_frequencies(128, 1e6, scaling)
    assert np.allclose(cos, np.cos(angles) * 1.138629436111989) and np.allclose(sin, np.sin(angles) * 1.138629436111989)


def test_only_the_kinds_of_scaling_the_tables_know_are_let_through():
    config, weights = synthetic_weights()
    _, published = hugging_face(config, weights, True)
    check_config({**published, "rope_scaling": SCALING})
    check_config({**published, "rope_scaling": LINEAR})
    check_config({**published, "rope_scaling": YARN})
    with pytest.raises(ValueError, match="RoPE scaling"):
        check_config({**published, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0}})
    with pytest.raises(ValueError, match="YaRN"):
        check_config({**published, "rope_scaling": {"rope_type": "yarn", "factor": 2.0}})


def llama3(max_seq_len=64, scaling=None):
    """A small Llama with Llama 3's scaling (a short original context, so that all three bands are there)."""
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    published = {**published, "rope_theta": 500000.0,
                 "rope_scaling": scaling or {**SCALING, "original_max_position_embeddings": 32}}
    return config, tensors, published


@pytest.mark.parametrize("scaling", [None, LINEAR, {**YARN, "original_max_position_embeddings": 32}],
                         ids=["llama3", "linear", "yarn"])
def test_the_file_and_the_engine_make_the_same_scaled_tables(scaling):
    """float32 files hold the tables (the converter makes them), int8 files do not (the engine does): the two must
    agree, and the engine needs rope_scaling from the options for that (T72's lesson: test the options' path)."""
    config, tensors, published = llama3(scaling=scaling)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(config["vocab_size"])]}}).encode()
    tables = {}
    for dtype in ("float32", "int8"):
        conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                                dtype=dtype, max_seq_len=config["seq_len"])
        conversion.feed(file)
        conversion.finish()
        options = conversion.options
        assert options["rope_scaling"] == published["rope_scaling"]
        llama = Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), dtype=options["dtype"],
                      rope_theta=options["rope_theta"], rope_scaling=options["rope_scaling"], tokenizer_kind="unigram")
        tables[dtype] = llama.freq_cis_real, llama.freq_cis_imag
    header = struct.unpack_from("<7i", converted(Arrays(tensors), published, "float32"), 0)
    plain = rope_table({**published, "rope_scaling": None}, header, 0)
    assert not np.allclose(tables["float32"][0], plain), "the scaling changes the table"
    for which in (0, 1):
        assert np.allclose(tables["int8"][which], tables["float32"][which], atol=1e-6)


def test_llama3_pretokenizer_follows_the_pattern():
    regex = pytest.importorskip("regex", reason="pip install regex to check the pattern itself")
    for text in TEXTS + ["12345678", " 1234 ", "x9999y", "٣٤٥٦"]:
        assert pretokenize(text, "llama3") == regex.findall(LLAMA3_PATTERN, text)


def real_bpe(ignore_merges):
    """A byte-level BPE with Llama 3's pattern, trained here, with ignore_merges on or off."""
    tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers to check against the real one")
    from tokenizers import Tokenizer as Real, decoders, models, pre_tokenizers, trainers
    real = Real(models.BPE(ignore_merges=ignore_merges))
    real.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(tokenizers.Regex(LLAMA3_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    real.decoder = decoders.ByteLevel()
    real.train_from_iterator([CORPUS] * 8, trainers.BpeTrainer(
        vocab_size=900, special_tokens=["<|begin_of_text|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    spec = json.loads(real.to_str())
    # a word the merges cannot make: in the vocabulary, but its merge is gone (588 of Llama 3's words are so)
    word = "".join(llama2_convert_chars("Pyodide"))
    spec["model"]["merges"] = [merge for merge in spec["model"]["merges"]
                               if (merge if isinstance(merge, str) else " ".join(merge)).replace(" ", "") != word]
    spec["model"]["ignore_merges"] = ignore_merges
    return Real.from_str(json.dumps(spec)), spec


def llama2_convert_chars(text):
    from llama2_numpy import BYTE_CHARS
    return [BYTE_CHARS[byte] for byte in text.encode("utf-8")]


@pytest.mark.parametrize("ignore_merges", [False, True])
@pytest.mark.parametrize("text", TEXTS + ["Pyodide", " Pyodide", "12345"])
def test_matches_the_real_tokenizer(ignore_merges, text):
    real, spec = real_bpe(ignore_merges)
    options = tokenizer_json_options(spec)
    assert options["pretokenizer"] == "llama3" and options["ignore_merges"] is ignore_merges
    vocab_size = real.get_vocab_size()
    from llama2_convert import tokenizer_bin, tokenizer_json_pieces
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size, kind="bytebpe",
                     pretokenizer="llama3", ignore_merges=ignore_merges)
    assert mine.encode(text) == real.encode(text, add_special_tokens=False).ids


def test_ignore_merges_makes_a_difference_here():
    """So that the test above tests something: the word the merges cannot make is one token only with the flag."""
    assert len(real_bpe(False)[0].encode("Pyodide", add_special_tokens=False).ids) > 1
    assert len(real_bpe(True)[0].encode("Pyodide", add_special_tokens=False).ids) == 1


CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")


def test_the_special_tokens_of_the_template_go_to_the_engine():
    """T73's leftover: with ?hf= the template's <|im_start|> was spelled out letter by letter."""
    config, weights = synthetic_weights(vocab_size=300)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocab = {"w": 0, "x": 1, "wx": 2, **{f"w{i}": i + 2 for i in range(1, 295)}}
    specials = {"<|im_start|>": 297, "<|im_end|>": 298, "<|unused|>": 299}
    vocabulary = json.dumps({"added_tokens": [{"id": id, "content": text, "special": True} for text, id in specials.items()],
                             "model": {"type": "BPE", "vocab": {**vocab, **specials}, "merges": ["w x"]},
                             "pre_tokenizer": {"type": "ByteLevel"}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype="int8", max_seq_len=config["seq_len"],
                            tokenizer_config=json.dumps({"chat_template": CHATML, "bos_token": "<|im_start|>"}))
    assert conversion.options["specials"] == ["<|im_start|>", "<|im_end|>"]
    tokenizer = Tokenizer(bytes(conversion.tokenizer), config["vocab_size"], kind="bytebpe")
    assert tokenizer.encode("<|im_start|>wx<|im_end|>", conversion.options["specials"]) == [297, 2, 298]


def test_a_llama3_gguf_splits_like_llama3_and_refuses_the_rope_freqs_table():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file, _ = gguf_file(tensors, published, config["vocab_size"], pre="llama-bpe")
    conversion = Conversion.from_gguf(file, dtype="int8", max_seq_len=1 << 20)
    assert conversion.options["pretokenizer"] == "llama3" and conversion.options["ignore_merges"] is True
    metadata, found, base = gguf_read(file)
    with pytest.raises(ValueError, match="rope_freqs"):
        gguf_model(metadata, {**found, "rope_freqs.weight": {"type": 0, "shape": [4], "offset": 0}}, base)


def transformers5(published):
    """The same config.json as transformers 5 saves it: rope_theta, rope_scaling and GPT-NeoX's rotary_pct and
    rotary_emb_base go into one rope_parameters (seen with transformers 5.12.1's LlamaConfig, Qwen2Config and
    GPTNeoXConfig)."""
    old = dict(published)
    theta, base = old.pop("rope_theta", None), old.pop("rotary_emb_base", None)
    rope = {"rope_theta": theta or base or 10000.0, "rope_type": "default", **(old.pop("rope_scaling", None) or {})}
    if "rotary_pct" in old:
        rope["partial_rotary_factor"] = old.pop("rotary_pct")
    return {**old, "rope_parameters": rope}


def options_of(tensors, published, dtype, max_seq_len):
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(published["vocab_size"])]}}).encode()
    conversion = Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), vocabulary, "tokenizer.json",
                            dtype=dtype, max_seq_len=max_seq_len)
    conversion.feed(file)
    conversion.finish()
    return bytes(conversion.checkpoint), conversion.options


@pytest.mark.parametrize("family", ["llama3", "theta", "neox"])
def test_a_config_of_transformers_5_converts_as_the_old_one(family):
    """The float32 file holds the RoPE tables (theta, the scaling, the rotated width): the same bytes and the same
    options from either spelling, for the float32 and the int8 files."""
    from test_neox import POSITIONS, neox_model
    if family == "neox":
        tensors, published = neox_model(0.25, True)
        max_seq_len = POSITIONS
    else:
        config, tensors, published = llama3()
        max_seq_len = config["seq_len"]
        if family == "theta":
            published = {**published, "rope_theta": 1000000.0}
            del published["rope_scaling"]
    for dtype in ("float32", "int8"):
        assert options_of(tensors, transformers5(published), dtype, max_seq_len) == options_of(tensors, published, dtype, max_seq_len)
    if family != "neox":
        # and the tables do differ from the ones of theta 10000 unscaled, which is what an unread one got
        plain = {key: value for key, value in published.items() if key not in ("rope_theta", "rope_scaling")}
        assert options_of(tensors, plain, "float32", max_seq_len)[0] != options_of(tensors, published, "float32", max_seq_len)[0]
