"""llama2_convert.py: a Hugging Face checkpoint, read piece by piece, becomes exactly the checkpoint of the build."""
import json
import struct
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, pack_checkpoint, synthetic_weights
import llama2_convert
from llama2_convert import (Arrays, Safetensors, check_config, checkpoint_form, checkpoint_header, checkpoint_size,
                            convert_weights, normalize,
                            tokenizer_bin, tokenizer_json_options, tokenizer_json_pieces)
from llama2_numpy import FORM, Llama, Tokenizer, check_tokenizer, checkpoint_dtype


def hugging_face(config, weights, shared):
    """The tensors and the config.json Hugging Face would publish for these llama2.c weights."""
    dim, n_heads, n_kv_heads = config["dim"], config["n_heads"], config["n_kv_heads"]
    head_size = config["head_size"]

    def permute(w, heads):  # llama2.c's adjacent pairs -> [first halves, second halves]: what convert undoes
        return w.reshape(heads, head_size // 2, 2, w.shape[1]).transpose(0, 2, 1, 3).reshape(w.shape)

    tensors = {"model.embed_tokens.weight": weights["token_embedding_table"], "model.norm.weight": weights["rms_final_weight"]}
    names = dict(rms_att_weight="input_layernorm", wq="self_attn.q_proj", wk="self_attn.k_proj", wv="self_attn.v_proj",
                 wo="self_attn.o_proj", rms_ffn_weight="post_attention_layernorm", w1="mlp.gate_proj", w2="mlp.down_proj",
                 w3="mlp.up_proj")
    for ours, theirs in names.items():
        for layer in range(config["n_layers"]):
            tensor = weights[ours][layer]
            if ours in ("wq", "wk"):
                tensor = permute(tensor, n_heads if ours == "wq" else n_kv_heads)
            tensors[f"model.layers.{layer}.{theirs}.weight"] = np.ascontiguousarray(tensor)
    if not shared:
        tensors["lm_head.weight"] = weights["wcls"]
    published = dict(model_type="llama", hidden_size=dim, intermediate_size=config["hidden_dim"],
                     num_hidden_layers=config["n_layers"], num_attention_heads=n_heads, num_key_value_heads=n_kv_heads,
                     vocab_size=config["vocab_size"], max_position_embeddings=config["seq_len"], rope_theta=10000.0,
                     tie_word_embeddings=shared)
    if head_size != dim // n_heads:
        published["head_dim"] = head_size
    return tensors, published


def safetensors_file(tensors, stored="F32"):
    header, data = {"__metadata__": {"format": "pt"}}, b""
    for name, tensor in tensors.items():
        if stored == "BF16":  # the upper half of the float32: what bfloat16 is
            raw = (np.ascontiguousarray(tensor, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        else:
            raw = np.ascontiguousarray(tensor, dtype={"F32": np.float32, "F16": np.float16}[stored]).tobytes()
        header[name] = {"dtype": stored, "shape": list(tensor.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


def reader(file, log=None):
    def read(offset, length):
        if log is not None:
            log.append(length)
        return file[offset:offset + length]
    return read


def converted(source, published, dtype, max_seq_len=1 << 20):
    out = bytearray(checkpoint_size(checkpoint_header(published, source, max_seq_len), dtype,
                                    checkpoint_form(published, source)))
    convert_weights(source, published, dtype, max_seq_len, out)
    return bytes(out)


CONFIGS = [dict(n_kv_heads=4), dict(n_kv_heads=2), dict(n_kv_heads=1, shared=False), dict(n_kv_heads=4, hidden_dim=48)]


@pytest.mark.parametrize("config", CONFIGS)
def test_float32_is_the_checkpoint_the_weights_came_from(config, monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 700)  # several pieces per tensor, and not a multiple of a row
    shared = config.get("shared", True)
    config, weights = synthetic_weights(**config)
    tensors, published = hugging_face(config, weights, shared)
    expected = pack_checkpoint(config, weights)
    assert converted(Safetensors(reader(safetensors_file(tensors))), published, "float32") == expected
    assert converted(Arrays(tensors), published, "float32") == expected


@pytest.mark.parametrize("config", CONFIGS)
def test_int8_is_what_quantize_makes_of_float32(tmp_path, config, monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 700)
    shared = config.get("shared", True)
    config, weights = synthetic_weights(**config)
    tensors, published = hugging_face(config, weights, shared)
    source, target = tmp_path / "model.f32", tmp_path / "model.bin"
    source.write_bytes(pack_checkpoint(config, weights))
    subprocess.run([sys.executable, str(ROOT / "quantize.py"), str(source), str(target)], check=True)
    int8 = converted(Safetensors(reader(safetensors_file(tensors))), published, "int8")
    assert int8 == target.read_bytes()
    assert checkpoint_dtype(struct.unpack_from("<7i", int8, 0), len(int8)) == "int8"


def test_half_precision_sources_and_targets():
    config, weights = synthetic_weights(n_kv_heads=2)
    tensors, published = hugging_face(config, weights, True)
    # bfloat16 keeps the upper half of each float32: truncate the weights the same way, and the result is exact
    truncated = {name: (np.ascontiguousarray(tensor).view(np.uint32) >> 16 << 16).view(np.float32) for name, tensor in tensors.items()}
    from_bfloat16 = converted(Safetensors(reader(safetensors_file(tensors, "BF16"))), published, "float32")
    assert from_bfloat16 == converted(Arrays(truncated), published, "float32")
    float16 = converted(Safetensors(reader(safetensors_file(tensors, "F16"))), published, "float16")
    assert checkpoint_dtype(struct.unpack_from("<7i", float16, 0), len(float16)) == "float16"
    reference = np.frombuffer(pack_checkpoint(config, weights), dtype=np.float32, offset=28).astype(np.float16)
    assert np.array_equal(np.frombuffer(float16, dtype=np.float16, offset=28), reference)


def test_it_reads_pieces_and_never_the_whole_file(monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 2048)
    config, weights = synthetic_weights(vocab_size=1000)
    tensors, published = hugging_face(config, weights, True)
    file, log, progress = safetensors_file(tensors), [], []
    source = Safetensors(reader(file, log))
    out = bytearray(checkpoint_size(checkpoint_header(published, source, 1 << 20), "int8"))
    convert_weights(source, published, "int8", 1 << 20, out, progress=lambda done, total: progress.append(done / total))
    header_bytes = log[1]
    assert max(length for length in log if length != header_bytes) <= 2048 * 4
    assert progress == sorted(progress) and progress[-1] == 1.0 and len(progress) > 20


def test_the_context_can_be_cut_and_the_engine_runs_the_result():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    checkpoint = converted(Arrays(tensors), published, "float32", max_seq_len=16)
    assert struct.unpack_from("<7i", checkpoint, 0)[6] == 16
    vocabulary = tokenizer_bin([("<unk>", 0.0, False)] + [(f"▁w{i}", -float(i), True) for i in range(300)], config["vocab_size"])
    check_tokenizer(vocabulary, struct.unpack_from("<7i", checkpoint, 0))
    llama = Llama(checkpoint, vocabulary, tokenizer_kind="unigram")
    assert len(list(llama.generate(" w1 w2", steps=12))) > 0


@pytest.mark.parametrize("change, reason", [
    (dict(model_type="rwkv"), "only Llama, Mistral, Qwen2, Qwen3, Qwen3.5, GPT-2 and GPT-NeoX"), (dict(rope_scaling={"type": "dynamic", "factor": 2.0}), "RoPE scaling"),
    (dict(hidden_act="gelu"), "gelu"), (dict(attention_bias=True), "biases"), (dict(num_attention_heads=5), "heads"),
    (dict(vocab_size=None), "vocab_size"), (dict(head_dim=3), "heads")])
def test_a_model_the_engine_cannot_run_is_refused(change, reason):
    config, weights = synthetic_weights()
    _, published = hugging_face(config, weights, True)
    with pytest.raises(ValueError, match=reason):
        check_config({**published, **change})


def test_a_head_dim_of_null_is_no_head_dim():
    # cyberagent/CAT-Translate-7b writes "head_dim": null, and was refused as heads that do not divide (T124)
    config, weights = synthetic_weights()
    _, published = hugging_face(config, weights, True)
    check_config({**published, "head_dim": None})


def test_missing_and_misshapen_tensors_are_refused():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    missing = {name: tensor for name, tensor in tensors.items() if "layers.1.mlp.up_proj" not in name}
    with pytest.raises(ValueError, match="up_proj.weight is missing"):
        converted(Arrays(missing), published, "float32")
    with pytest.raises(ValueError, match="embed_tokens.weight is"):
        converted(Arrays(tensors), {**published, "vocab_size": 999}, "float32")
    for broken in (b"\x00" * 64, struct.pack("<Q", 20) + b"this is not json....", b"\xff" * 64):
        with pytest.raises(ValueError, match="not a safetensors file"):
            Safetensors(reader(broken))
    wrong = safetensors_file(tensors).replace(b'"F32"', b'"I64"')
    with pytest.raises(ValueError, match="stored as I64"):
        converted(Safetensors(reader(wrong)), published, "float32")


def test_tokenizer_json():
    tokenizer = {"added_tokens": [{"content": "<s>", "special": True}],
                 "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}]},
                 "model": {"type": "Unigram", "unk_id": 0,
                           "vocab": [["<unk>", 0.0], ["<s>", 0.0], ["<0x41>", 0.0], ["▁hello", -1.5], ["猫", -2.0]]
                                    + [[f"w{i}", -3.0] for i in range(45)]}}
    pieces = list(tokenizer_json_pieces(tokenizer))
    assert [matchable for _, _, matchable in pieces[:5]] == [False, False, False, True, True]
    data = tokenizer_bin(pieces, 53)  # three embedding rows more than pieces: padding
    vocabulary = Tokenizer(data, 53, kind="unigram")
    assert vocabulary.vocab[3] == b" hello" and vocabulary.vocab[52] == b"" and vocabulary.scores[1] < Tokenizer.UNMATCHABLE
    with pytest.raises(ValueError, match="do not belong together"):
        tokenizer_bin(pieces, 100)  # the tokenizer of a model with half the vocabulary
    assert tokenizer_json_options(tokenizer) == {"tokenizer_kind": "unigram", "nfkc": True}
    assert tokenizer_json_options({**tokenizer, "normalizer": None})["nfkc"] is False
    with pytest.raises(ValueError, match="vocabulary of 4"):
        tokenizer_bin(pieces, 4)
    with pytest.raises(ValueError, match="only Unigram and byte-level BPE"):
        list(tokenizer_json_pieces({**tokenizer, "model": {"type": "WordPiece"}}))


def test_a_tokenizer_json_precompiled_map_goes_into_tokenizer_bin():
    """T216: a tokenizer.json made from a sentencepiece model says its normalizer as "Precompiled", the model's map in
    base64, alone or in a Sequence. It goes after the pieces as a tokenizer.model's does (the nfkc option was a guess)."""
    import base64
    from conftest import charsmap, vocabulary_conversion
    from llama2_convert import tokenizer_json_charsmap
    map = charsmap({"Ａ": "A", "\n": " "})
    precompiled = {"type": "Precompiled", "precompiled_charsmap": base64.b64encode(map).decode()}
    vocab = [["<unk>", 0.0], ["<s>", 0.0], ["</s>", 0.0], ["▁A", -1.0]] + [[f"▁w{i}", -2.0 - i / 100] for i in range(316)]
    for normalizer in (precompiled, {"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"String": "x"},
                                                                          "content": "x"}, precompiled]}):
        tokenizer = {"added_tokens": [], "normalizer": normalizer, "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab}}
        assert tokenizer_json_charsmap(tokenizer) == map and tokenizer_json_options(tokenizer) == {"tokenizer_kind": "unigram", "nfkc": False}
        conversion = vocabulary_conversion(json.dumps(tokenizer).encode(), "tokenizer.json", 320)
        assert bytes(conversion.tokenizer).endswith(b"charsmap" + struct.pack("<I", len(map)) + map)
        vocabulary = Tokenizer(bytes(conversion.tokenizer), 320, kind="unigram")
        assert vocabulary.encode("Ａ\nw1") == vocabulary.encode("A w1") == [3, 5]
    assert tokenizer_json_charsmap({"normalizer": None}) == b"" and tokenizer_json_charsmap({"normalizer": {"type": "NFKC"}}) == b""


def test_a_transform_nobody_wrote_is_refused():
    """transformed() used to take any unknown name for a slice of rows (T77): a typo must fail loudly."""
    import numpy as np
    from llama2_convert import transformed
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert transformed(values, ("row", 1, 3), 2).tolist() == values[2:4].tolist()
    with pytest.raises(ValueError, match="no transform called 'rows'"):
        transformed(values, ("rows", 1, 3), 2)


def streamed(file, published, dtype, chunk, max_seq_len=1 << 20):
    """The file fed to Stream from its beginning to its end, chunk bytes at a time."""
    (size,) = struct.unpack("<Q", file[:8])
    header = json.loads(file[8:8 + size])
    stream = llama2_convert.Stream(header, 8 + size, published, dtype, max_seq_len)
    out = stream.out
    # the first stretch of the file may be skipped, as the page does once it has read the header
    skip = 8 + size if chunk > 100 else 0
    if skip:
        stream = llama2_convert.Stream(header, 8 + size, published, dtype, max_seq_len, start=skip)
        out = stream.out
    progress = [stream.feed(file[start:start + chunk]) for start in range(skip, len(file), chunk)]
    stream.finish()
    return bytes(out), progress


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype, stored, chunk", [("float32", "F32", 1000), ("int8", "BF16", 4096), ("float16", "F16", 7), ("int8", "F32", 1 << 20)])
def test_the_file_in_its_own_order_gives_the_same_checkpoint(config, dtype, stored, chunk, monkeypatch):
    monkeypatch.setattr(llama2_convert, "PIECE", 700)
    shared = config.get("shared", True)
    config, weights = synthetic_weights(**config)
    tensors, published = hugging_face(config, weights, shared)
    # Hugging Face writes the tensors sorted by name, and a file may hold tensors nobody asks for
    tensors = dict(sorted({**tensors, "model.layers.0.self_attn.rotary_emb.inv_freq": np.arange(8, dtype=np.float32)}.items()))
    file = safetensors_file(tensors, stored)
    expected = converted(Safetensors(reader(file)), published, dtype)
    result, progress = streamed(file, published, dtype, chunk)
    assert result == expected
    assert progress[-1][0] == progress[-1][1] and [done for done, _ in progress] == sorted(done for done, _ in progress)


def test_a_stream_that_ends_early_or_lacks_a_tensor_is_refused():
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    (size,) = struct.unpack("<Q", file[:8])
    header = json.loads(file[8:8 + size])
    out = bytearray(checkpoint_size(checkpoint_header(published, Safetensors(reader(file)), 1 << 20), "int8"))
    stream = llama2_convert.Stream(header, 8 + size, published, "int8", 1 << 20, out)
    stream.feed(file[:len(file) // 2])
    with pytest.raises(ValueError, match="ended before"):
        stream.finish()
    missing = {name: info for name, info in header.items() if "layers.1.mlp.up_proj" not in name}
    with pytest.raises(ValueError, match="up_proj.weight is missing"):
        llama2_convert.Stream(missing, 8 + size, published, "int8", 1 << 20, out)


class Sink:
    """What the worker passes for a checkpoint that goes straight into the WebAssembly memory of forward.js (T93)."""

    def __init__(self):
        self.data = self.opened = None

    def open(self, size, header, dtype, form):
        self.data = bytearray(size)
        self.opened = (list(header), dtype, form)  # what the worker sizes the forward pass's memory from (T115)

    def write(self, offset, array):
        raw = bytes(np.asarray(array, dtype=np.uint8))
        self.data[offset:offset + len(raw)] = raw


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("head_size", [0, 16])
def test_a_sink_gets_the_very_checkpoint(dtype, head_size):
    """And the form (T144: with heads of 16 in a dim of 32 and 4 heads, the worker sizes the keys and values of heads
    twice what the header says; with the size of dim / heads, footprint() counted them 45% short for Qwen3 0.6B)."""
    config, weights = synthetic_weights(head_size=head_size)
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    expected = converted(Safetensors(reader(file)), published, dtype)
    size = struct.unpack("<Q", file[:8])[0]
    sink = Sink()
    stream = llama2_convert.Stream(json.loads(file[8:8 + size]), 8 + size, published, dtype, 1 << 20, sink=sink)
    assert stream.out is None
    for start in range(0, len(file), 777):
        stream.feed(file[start:start + 777])
    stream.finish()
    assert bytes(sink.data) == expected
    assert sink.opened == (list(stream.header), dtype, {**FORM, "head_dim": head_size})


def test_a_dtype_chosen_from_the_header_is_the_one_converted_to():
    """T115: the worker gives the converter a function in place of a dtype (int8 where the forward pass fits a 32-bit
    memory, else int6), which gets the header and the size of either; the checkpoint is then that dtype's."""
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    asked = []

    def choose(header, form, sizes):
        asked.append((list(header), form, dict(sizes)))
        return "int6"

    sink = Sink()
    stream = llama2_convert.Stream(json.loads(file[8:8 + size]), 8 + size, published, choose, 1 << 20, sink=sink)
    stream.feed(file)
    stream.finish()
    header = list(stream.header)
    form = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}
    assert asked == [(header, form, {name: checkpoint_size(header, name) for name in ("int8", "int6")})]
    assert stream.dtype == "int6" and sink.opened[1] == "int6"
    assert bytes(sink.data) == converted(Safetensors(reader(file)), published, "int6")


def test_a_quantizer_of_rows_is_used_for_whole_groups_of_32_only():
    """T89: the converter hands int8 matrices to quantize_rows (the kernels' quantizer in the page) when their rows are
    whole groups of 32, and keeps NumPy's quantize() for the rest; the bytes are the same either way."""
    config, weights = synthetic_weights(dim=48, hidden_dim=64)  # rows of 48: groups of 16, not the kernel's
    tensors, published = hugging_face(config, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    seen = []

    def quantize_rows(values):
        seen.append(values.shape[-1])
        return llama2_convert.quantize(values)

    stream = llama2_convert.Stream(json.loads(file[8:8 + size]), 8 + size, published, "int8", 1 << 20,
                                   quantize_rows=quantize_rows)
    stream.feed(file)
    stream.finish()
    assert bytes(stream.out) == converted(Safetensors(reader(file)), published, "int8")
    assert seen and set(seen) == {64}, "only the rows of 64 (w2) go to it; the rows of 48 stay with NumPy"


def test_a_mistral_is_a_llama_and_its_sliding_window_cuts_the_context():
    """T125: model_type mistral converts as the Llama it is, the same bytes; a sliding window shorter than the
    context cuts the context to it (the window then covers every position), a longer one or none changes nothing"""
    config, weights = synthetic_weights()
    tensors, published = hugging_face(config, weights, True)
    llama = converted(Arrays(tensors), published, "float32")
    mistral = {**published, "model_type": "mistral", "sliding_window": None}
    check_config(normalize(mistral))
    assert converted(Arrays(tensors), mistral, "float32") == llama
    context = published["max_position_embeddings"]
    for window, expected in ((context // 2, context // 2), (4 * context, context)):
        windowed = normalize({**mistral, "sliding_window": window})
        assert windowed["model_type"] == "llama" and windowed["max_position_embeddings"] == expected
        assert struct.unpack_from("<7i", converted(Arrays(tensors), {**mistral, "sliding_window": window}, "float32"), 0)[6] == expected


def test_the_bos_is_the_one_the_tokenizer_names():
    """T143: config.json and the tokenizer may name two BOS tokens (DeepSeek-R1's Distill: 151643, its end of a
    sentence, and <｜begin▁of▁sentence｜> in tokenizer_config.json, which transformers begins a text with). The
    tokenizer's wins, and the answer stops at both."""
    from conftest import vocabulary_conversion
    bos = "<｜begin▁of▁sentence｜>"
    vocab = [["<unk>", 0.0], ["<s>", 0.0], ["</s>", 0.0], [bos, 0.0]] + [[f"▁w{i}", -1.0 - i / 100] for i in range(316)]
    tokenizer = json.dumps({"added_tokens": [{"id": i, "content": text, "special": True} for i, (text, _) in enumerate(vocab[:4])],
                            "model": {"type": "Unigram", "unk_id": 0, "vocab": vocab}}).encode()
    named = lambda config: vocabulary_conversion(tokenizer, "tokenizer.json", 320, tokenizer_config=config,
                                                 bos_token_id=2, eos_token_id=2).options
    for config in (json.dumps({"bos_token": {"content": bos, "special": True}}), json.dumps({"bos_token": bos}), {"bos_token": bos}):
        options = named(config)
        assert (options["bos"], options["stop_tokens"]) == (3, [3, 2, 2]), config
    for config in (None, "", "not json", json.dumps({"bos_token": None}), json.dumps({"bos_token": "<nope>"}), json.dumps({})):
        options = named(config)
        assert (options["bos"], options["stop_tokens"]) == (2, [2, 2]), config
