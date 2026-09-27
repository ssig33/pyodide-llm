# tests/gguf_check.py tensors, held to what T136's stage 2 asks of it (the review of 2026-09-26): the embedding and
# the classifier row by row, Llama 3's rope_freqs.weight against the engine's table, the vocabulary of a
# sentencepiece original, the config after normalize(), and the vocabulary shown but not counted when the page takes
# it from the original. Each test fails when its check is taken out.
import json
import math

import numpy as np
import pytest
from conftest import synthetic_weights
from make_hf_fixture import field
from test_convert import hugging_face, safetensors_file
from test_gguf import gguf_file, other_gguf, the_other

import gguf_check
from llama2_numpy import rope_frequencies

EPS = ("llama.attention.layer_norm_rms_epsilon", 6, 1e-5)
LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
          "original_max_position_embeddings": 64}


def model(tmp_path, vocab_size=320, dim=32, theta=10000.0, extra=None, change=None, original_change=None, shared=True,
          n_kv_heads=4, **config):
    """A GGUF and the directory of its original (config.json and model.safetensors); change(tensors) alters what
    the GGUF is written from, as a GGUF of other weights would be, and original_change(tensors) the original.
    shared=False: a classifier of its own (output.weight, held row by row); n_kv_heads under 4: GQA (T145)."""
    shape, weights = synthetic_weights(dim=dim, hidden_dim=2 * dim, vocab_size=vocab_size, seq_len=128,
                                       n_kv_heads=n_kv_heads, shared=shared)
    tensors, published = hugging_face(shape, weights, shared)
    published = {**published, "rms_norm_eps": 1e-5, "rope_theta": theta, **config}
    written = {name: tensor.copy() for name, tensor in tensors.items()}
    if change:
        change(written)
    file, same = gguf_file(written, published, vocab_size, theta=theta, more=[EPS], extra=extra)
    # the original holds the values the GGUF stands for (Q8_0 rounds), except where change() made them differ
    original = {name: (same[name] if change is None or np.array_equal(written[name], tensors[name]) else tensors[name])
                for name in tensors}
    if original_change:
        original_change(original)
    (tmp_path / "model.gguf").write_bytes(file)
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(original))
    (tmp_path / "config.json").write_text(json.dumps(published))
    return tmp_path / "model.gguf", tmp_path


def summary(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_same_weights_pass(tmp_path, capsys):
    assert gguf_check.check_tensors(*model(tmp_path))
    assert summary(capsys)["mismatches"] == 0


@pytest.mark.parametrize("block", [gguf_check.BLOCK, 4096])  # the whole tensor at once, and a block of rows at a time
def test_eight_swapped_rows_of_the_embedding_are_caught(tmp_path, capsys, monkeypatch, block):
    """Qwen2.5 0.5B's embedding with 8 rows swapped was 0.0122 off as a whole, under the line of 0.02 (the review of
    T136). Here 100000 rows of 32: the whole is about as far off, and every one of the 8 rows is past 0.05."""
    monkeypatch.setattr(gguf_check, "BLOCK", block)

    def swap(tensors):
        embedding = tensors["model.embed_tokens.weight"]
        for a, b in ((10, 20), (300, 4000), (50000, 60000), (99990, 7)):
            embedding[[a, b]] = embedding[[b, a]]

    assert not gguf_check.check_tensors(*model(tmp_path, vocab_size=100000, change=swap))
    result = summary(capsys)
    assert result["worst"] < 0.02, "the error of the whole tensor alone would have let this through"
    # the 8 rows, and the embedding as a whole past TIGHT (T145)
    assert result["rows"]["token_embd.weight"] == 8 and result["mismatches"] == 9
    assert list(result["past_tight"]) == ["token_embd.weight"]


def test_a_row_of_nearly_nothing_is_not_an_error(tmp_path, capsys):
    """Qwen2.5 7B's unused rows are 1.2e-37 at most, and a Q8_0 writes them as 0: no difference that matters."""
    def tiny(tensors):
        tensors["model.embed_tokens.weight"][5] = 1e-37

    assert gguf_check.check_tensors(*model(tmp_path, change=tiny, original_change=tiny))
    assert summary(capsys)["rows"]["token_embd.weight"] == 0


def test_rope_freqs_is_compared_with_the_engines_table(tmp_path, capsys):
    head = 32 // 4
    factors = rope_frequencies(head, 500000.0) / rope_frequencies(head, 500000.0, LLAMA3)
    assert not np.allclose(factors, 1), "the scaling must change some pairs for the test to mean anything"
    assert gguf_check.check_tensors(*model(tmp_path, theta=500000.0, extra={"rope_freqs.weight": factors},
                                           rope_scaling=LLAMA3))
    result = summary(capsys)
    assert result["rope_freqs_diff"] < 1e-6 and result["mismatches"] == 0

    # a table of another scaling (none at all) is a mismatch, not a crash
    other = tmp_path / "other"
    other.mkdir()
    assert not gguf_check.check_tensors(*model(other, theta=500000.0, extra={"rope_freqs.weight": np.ones(head // 2)},
                                               rope_scaling=LLAMA3))
    result = summary(capsys)
    assert result["rope_freqs_diff"] > 1e-3 and result["mismatches"] == 1


def test_the_config_is_read_after_normalize(tmp_path, capsys):
    """transformers 5 writes rope_theta inside rope_parameters (CAT-Translate 1.4b): read as it is, the theta would
    be the default 10000 and differ from the GGUF's."""
    path, directory = model(tmp_path, theta=500000.0)
    config = json.loads((directory / "config.json").read_text())
    del config["rope_theta"]
    config["rope_parameters"] = {"rope_theta": 500000.0, "rope_type": "default"}
    (directory / "config.json").write_text(json.dumps(config))
    assert gguf_check.check_tensors(path, directory)
    assert summary(capsys)["mismatches"] == 0


def sentencepiece(pieces):
    return b"".join(field(1, field(1, text.encode()) + field(2, -1.0) + field(3, 1)) for text in pieces)


@pytest.mark.parametrize("original_vocabulary", [False, True])
def test_a_sentencepiece_vocabulary_is_compared(tmp_path, capsys, original_vocabulary):
    path, directory = model(tmp_path)
    pieces = [f"w{i}" for i in range(320)]
    (directory / "tokenizer.model").write_bytes(sentencepiece(pieces))
    assert gguf_check.check_tensors(path, directory, original_vocabulary)
    assert summary(capsys)["vocab_diffs"] == {"tokenizer.model": 0}

    pieces[3] = "▁w3"  # the id of one piece written differently (llm-jp-4's GGUF: 13 of them)
    (directory / "tokenizer.model").write_bytes(sentencepiece(pieces))
    assert gguf_check.check_tensors(path, directory, original_vocabulary) is original_vocabulary
    result = summary(capsys)
    assert result["vocab_diffs"] == {"tokenizer.model": 1}
    assert result["mismatches"] == (0 if original_vocabulary else 1)


def test_a_unigram_tokenizer_json_is_read_by_id(tmp_path, capsys):
    """llm-jp's tokenizer.json is a Unigram: its vocab is a list of [piece, score], in the order of the ids."""
    path, directory = model(tmp_path)
    vocab = [[f"w{i}", -1.0] for i in range(320)]
    (directory / "tokenizer.json").write_text(json.dumps({"model": {"type": "Unigram", "vocab": vocab}, "added_tokens": []}))
    assert gguf_check.check_tensors(path, directory)
    assert summary(capsys)["vocab_diffs"] == {"tokenizer.json": 0}


def test_names_without_a_hugging_face_counterpart_do_not_raise():
    assert gguf_check.hugging_face_name("rope_freqs.weight") is None
    assert gguf_check.hugging_face_name("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"
    assert math.isfinite(gguf_check.ROW_LINE)


def test_q_and_k_compared_in_blocks_are_read_turned(tmp_path, capsys, monkeypatch):
    """Llama 3.2 3B's q is 3072 x 3072, past BLOCK: compared a block at a time, it was read in Hugging Face's order
    and a GGUF turned the way llama.cpp turns it was 1.4 off (the first run of stage 2)."""
    monkeypatch.setattr(gguf_check, "BLOCK", 256)  # q and k of 32 x 32 go by blocks of 8 rows, one head each
    assert gguf_check.check_tensors(*model(tmp_path))
    result = summary(capsys)
    assert result["orders"] == ["turned (llama2.c order)"] and result["worst"] < 0.02


def llama_cpp_q8_0(values):
    """llama.cpp's quantize_row_q8_0_ref, as ggml-quants.c writes it: d = amax / 127, q = roundf(x / d), d stored
    as float16 (the rounded d is what reads back)."""
    groups = values.reshape(-1, 32).astype(np.float32)
    d = np.abs(groups).max(axis=1) / 127
    q = np.array([[math.floor(abs(x) / di + 0.5) * math.copysign(1, x) if di else 0 for x in g]
                  for g, di in zip(groups, d)])
    return (q * d.astype(np.float16).astype(np.float32)[:, None]).reshape(values.shape).astype(np.float32)


def test_rows_only_the_float16_scale_rounds_pass_and_a_swapped_row_does_not():
    """llm-jp-3 980M's embedding (the first run of stage 2): 8 rows of values near 1e-5, whose Q8_0 scale is under
    float16's smallest steps, came back 7 to 22% off (one as 0), and read as rows of other weights. Against
    llama.cpp's Q8_0 of the original they are the same; a swapped row is far from both."""
    rng = np.random.default_rng(1)
    original = (rng.standard_normal((64, 1536)) * 0.02).astype(np.float32)
    original[5] *= 2.5e-3  # largest value about 2e-4: d about 1.6e-6, in float16's steps of 6e-8
    original[6] *= 5e-4
    gguf = llama_cpp_q8_0(original)
    checked = gguf_check.row_check(gguf_check.row_parts(gguf, original, True))
    assert checked[0] == 0 and checked[4] >= 1
    assert {row for row, *_ in checked[5]} <= {5, 6} and len(checked[5]) == checked[4], "T145: the rows are listed"
    assert gguf_check.row_check(gguf_check.row_parts(gguf, original, False))[0] >= 1, \
        "against the original alone the rounded rows read as other weights"
    gguf[[10, 20]] = gguf[[20, 10]]
    assert gguf_check.row_check(gguf_check.row_parts(gguf, original, True))[0] == 2


def test_a_gguf_made_through_float16_is_held_to_the_q8_0_of_that():
    """mradermacher's RakutenAI 7B chat went through a float16 file: its row 79 (largest values near 1e-5) is 11%
    from llama.cpp's Q8_0 of the bfloat16 original and exactly the Q8_0 of the original made float16."""
    rng = np.random.default_rng(2)
    original = (rng.standard_normal((64, 4096)) * 0.0027).astype(np.float32)
    original[5] = np.sign(rng.standard_normal(4096)) * np.exp(rng.standard_normal(4096) * 2) * 1e-7
    gguf = gguf_check.q8_0_of(original.astype(np.float16).astype(np.float32))
    parts = gguf_check.row_parts(gguf, original, True)
    assert len(parts) == 6 and parts[2][5] > 0 and parts[4][5] == 0
    assert gguf_check.row_check(parts)[0] == 0


def test_a_row_whose_q8_0_reference_overflows_is_still_compared():
    """The review of stage 2 (2026-09-26): a row of the original under about 3.7e-37 has a d under 2.9e-39, whose
    1 / d overflows to inf, and inf * a float16 d of 0 made the reference NaN: np.minimum passed the row whatever the
    GGUF held (llm-jp-4's 490 unused rows went through so; the GGUF's are 0, as llama.cpp writes them). The reference
    is 0 there now: a GGUF of 0 passes, a used row put in its place does not."""
    rng = np.random.default_rng(3)
    original = (rng.standard_normal((64, 1536)) * 0.02).astype(np.float32)
    original[8] = rng.standard_normal(1536).astype(np.float32) * 1.1e-37
    gguf = llama_cpp_q8_0(original)
    assert not gguf[8].any()
    assert not np.isnan(gguf_check.q8_0_of(original)).any()
    assert gguf_check.row_check(gguf_check.row_parts(gguf, original, True))[0] == 0
    gguf[8] = gguf[40]
    assert gguf_check.row_check(gguf_check.row_parts(gguf, original, True))[0] == 1


# ------------------------------------------------------------------------------------------- T145
@pytest.mark.parametrize("tensor", ["model.layers.1.post_attention_layernorm.weight", "model.layers.0.mlp.up_proj.weight"])
def test_a_tensor_a_little_off_is_past_the_tight_line(tmp_path, capsys, tensor):
    """The review of T136's second stage: the whole tensor's 0.02 lets a scale of 1.05 or 4% of noise through, while
    a GGUF made from the original is 0 off its nearest reference (TIGHT). Here 0.4%: an F32 norm, and a Q8_0 matrix."""
    def scale(tensors):
        tensors[tensor] = tensors[tensor] * np.float32(1.004)

    assert not gguf_check.check_tensors(*model(tmp_path, change=scale))
    result = summary(capsys)
    assert result["worst"] < 0.02, "the line of the whole tensor would have let this through"
    assert len(result["past_tight"]) == 1 and result["mismatches"] == 1, result["past_tight"]
    assert 1e-3 < next(iter(result["past_tight"].values())) < 0.02


def test_the_same_weights_are_0_off_their_nearest_reference(tmp_path, capsys):
    assert gguf_check.check_tensors(*model(tmp_path))
    result = summary(capsys)
    assert result["nearest"] < 1e-6 and result["past_tight"] == {}


def test_a_classifier_of_its_own_is_held_row_by_row(tmp_path, capsys):
    assert gguf_check.check_tensors(*model(tmp_path, shared=False))
    result = summary(capsys)
    assert result["rows"] == {"token_embd.weight": 0, "output.weight": 0} and result["mismatches"] == 0

    def swap(tensors):
        classifier = tensors["lm_head.weight"]
        classifier[[3, 200]] = classifier[[200, 3]]

    other = tmp_path / "other"
    other.mkdir()
    assert not gguf_check.check_tensors(*model(other, shared=False, change=swap))
    result = summary(capsys)
    assert result["rows"]["output.weight"] == 2 and result["rows"]["token_embd.weight"] == 0
    assert [row for row, *_ in result["bad_rows"]["output.weight"]] == [3, 200]


@pytest.mark.parametrize("block", [gguf_check.BLOCK, 128])  # the whole tensor, and blocks of one head
def test_gqa_is_read_turned_by_its_own_heads(tmp_path, capsys, monkeypatch, block):
    """k of a GQA model has fewer heads than q (2 heads of 8 rows here, q 4): turned by q's heads it reads as other
    weights, and in blocks a block must be whole heads of k."""
    monkeypatch.setattr(gguf_check, "BLOCK", block)
    assert gguf_check.check_tensors(*model(tmp_path, n_kv_heads=2))
    result = summary(capsys)
    assert result["orders"] == ["turned (llama2.c order)"] and result["mismatches"] == 0 and result["nearest"] < 1e-6


def test_rows_that_pass_only_against_a_q8_0_are_listed_with_their_piece(tmp_path, capsys):
    """T145 (2): the rows the float16 scale of Q8_0 rounds (llm-jp-3 980M's 8, T136) pass, and are listed with their
    id, piece and norm, to be read."""
    def small(tensors):
        # an embedding of 0.01 (with 0.3, no row can be both over 1e-3 of the median and of a d near float16's steps),
        # and row 5 of a d of 1.5 of float16's smallest step: kept as 2 steps, the row reads back 33% off
        rng = np.random.default_rng(5)
        embedding = (rng.standard_normal(tensors["model.embed_tokens.weight"].shape) * 0.01).astype(np.float32)
        row = np.linspace(0.5, 1.0, embedding.shape[1], dtype=np.float32) * np.float32(127 * 1.5 * 2.0 ** -24)
        embedding[5] = row * np.where(np.arange(row.size) % 2, -1, 1).astype(np.float32)
        tensors["model.embed_tokens.weight"] = embedding

    assert gguf_check.check_tensors(*model(tmp_path, change=small, original_change=small))
    result = summary(capsys)
    rows = result["rounded_detail"]["token_embd.weight"]
    assert rows and len(rows) == result["rounded_rows"]["token_embd.weight"]
    assert [(row, piece) for row, piece, *_ in rows] == [(5, "w5")]
    _, _, against, near, norm = rows[0]
    assert against > gguf_check.ROW_LINE and near == 0 and norm < 1e-3
    assert result["past_tight"] == {} and result["mismatches"] == 0


def other_model(tmp_path, model, change=None):
    """T136's third stage: a GPT-2 or GPT-NeoX GGUF as llama.cpp writes one (test_gguf.other_gguf) and its original;
    change(tensors) alters what the GGUF is written from."""
    tensors, config = the_other(model)
    written = {name: tensor.copy() for name, tensor in tensors.items()}
    if change:
        change(written)
    file, _ = other_gguf(written, config)
    (tmp_path / "model.gguf").write_bytes(file)
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(tensors))
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path / "model.gguf", tmp_path


@pytest.mark.parametrize("block", [gguf_check.BLOCK, 256])  # whole, and query_key_value a head's q, k or v at a time
@pytest.mark.parametrize("model, order", [("gpt2", "transposed (out, in)"), ("neox", "split (q, k, v)"),
                                          ("neox-serial", "split (q, k, v)")])
def test_a_gpt2_or_neox_gguf_of_the_same_weights_passes_in_llama_cpps_order(tmp_path, capsys, monkeypatch, block, model,
                                                                             order):
    """GPT-2's Conv1D matrices are held to the original turned to (out, in), GPT-NeoX's query_key_value to it split
    into q, k and v, and the order found is said; GPT-2's output.weight (llama.cpp's copy) to the embedding."""
    monkeypatch.setattr(gguf_check, "BLOCK", block)
    assert gguf_check.check_tensors(*other_model(tmp_path, model))
    result = summary(capsys)
    assert result["mismatches"] == 0 and result["orders"] == [order] and result["nearest"] <= gguf_check.TIGHT


@pytest.mark.parametrize("block", [gguf_check.BLOCK, 256])
@pytest.mark.parametrize("model, name", [("neox", "gpt_neox.layers.1.attention.query_key_value.weight"),
                                         ("neox", "gpt_neox.layers.0.attention.query_key_value.bias"),
                                         ("gpt2", "transformer.h.1.attn.c_attn.weight"),
                                         ("gpt2", "transformer.h.0.mlp.c_proj.weight")])
def test_a_gpt2_or_neox_gguf_of_other_weights_does_not(tmp_path, capsys, monkeypatch, block, model, name):
    """Two heads' rows swapped in a fused q, k, v (or two rows of a matrix): the order is still found, the values
    are not the original's."""
    monkeypatch.setattr(gguf_check, "BLOCK", block)

    def swap(tensors):
        tensor = tensors[name]
        rows = tensor.shape[0] // 4  # a head's q, k and v of a NeoX; a quarter of the rows otherwise
        tensor[:rows], tensor[rows:2 * rows] = tensor[rows:2 * rows].copy(), tensor[:rows].copy()

    assert not gguf_check.check_tensors(*other_model(tmp_path, model, swap))
    assert summary(capsys)["mismatches"] >= 1


@pytest.mark.parametrize("norm", [None, "model.layers.1.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"])
def test_a_qwen3_gguf_is_held_to_its_norms_of_q_and_k(tmp_path, capsys, norm):
    """T203: a Qwen3's GGUF (llama.cpp's attn_q_norm and attn_k_norm, heads of another size than dim / heads) passes
    against its original, and a norm of other values does not (unread, the norms went unchecked: no counterpart)."""
    from test_gguf import qwen3_gguf
    config, published, file, same = qwen3_gguf(16)
    original = {name: tensor.copy() for name, tensor in same.items()}
    if norm:
        original[norm] = original[norm][::-1].copy()  # the GGUF now holds another norm than the original's
    (tmp_path / "model.gguf").write_bytes(file)
    (tmp_path / "model.safetensors").write_bytes(safetensors_file(original))
    (tmp_path / "config.json").write_text(json.dumps(published))
    assert gguf_check.check_tensors(tmp_path / "model.gguf", tmp_path) is (norm is None)
    result = summary(capsys)
    assert (result["mismatches"] == 0) is (norm is None)
