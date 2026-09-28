# Qwen3.5 (T229): attention in every fourth layer with an output gate and RoPE over part of each head, Gated DeltaNet
# in the others. The small model of tests/qwen35_model.py, converted from the safetensors transformers would read,
# must give the logits transformers gave (tests/fixtures/qwen35-logits.json, tests/qwen35_reference.py).
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from qwen35_model import CONFIG, TOKENS, weights
from test_convert import safetensors_file

import llama2_convert
from llama2_numpy import Llama, hadamard

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "qwen35-logits.json").read_text())
VOCABULARY = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                         "vocab": [[f"w{i}", -float(i)] for i in range(CONFIG["vocab_size"])]}}).encode()


def conversion(tensors, config=CONFIG, dtype="float32"):
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    made = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), VOCABULARY,
                                     "tokenizer.json", dtype=dtype, max_seq_len=64, start=8 + size)
    made.feed(file[8 + size:])
    made.finish()
    return made


def engine(made):
    options = {key: value for key, value in made.options.items()
               if key not in ("dtype", "template", "specials", "tokenizer_kind", "stop_tokens", "bos")}
    return Llama(bytes(made.checkpoint), bytes(made.tokenizer), dtype=made.options["dtype"], tokenizer_kind="unigram",
                 **options)


def logits(llama, tokens=TOKENS):
    return np.stack([llama.forward(token, pos).copy() for pos, token in enumerate(tokens)])


def test_the_float32_conversion_computes_what_transformers_does():
    made = conversion(weights())
    assert made.options["arch"] == "qwen35" and made.options["rotary"] == 8
    assert {key: made.options[key] for key in ("interval", "k_heads", "v_heads", "k_head", "v_head", "conv")} == \
        {"interval": 4, "k_heads": 2, "v_heads": 4, "k_head": 16, "v_head": 16, "conv": 4}
    ours, theirs = logits(engine(made)), np.array(REFERENCE["logits"], dtype=np.float32)
    assert ours.shape == theirs.shape
    worst = np.abs(ours - theirs).max() / np.abs(theirs).max()
    assert worst < 1e-5, worst
    assert (ours.argmax(axis=1) == theirs.argmax(axis=1)).all()


def test_a_new_sequence_forgets_the_last():
    """The conv's inputs and the delta rule's states belong to one sequence: position 0 begins again."""
    llama = engine(conversion(weights()))
    first = logits(llama)
    again = logits(llama)
    assert np.array_equal(first, again)


def test_int8_holds_the_same_tensors_where_float32_does():
    """Every tensor of the int8 checkpoint is the float32 one within int8's rounding (a group's largest / 254), the
    vectors the same: the layout is one for both. (Its logits are no test: a small model of random weights moves by
    9% for 0.4% of noise on its matrices.)"""
    wide, narrow = engine(conversion(weights())), engine(conversion(weights(), dtype="int8"))
    from llama2_numpy import TENSOR_NAMES
    compared = 0
    for name in TENSOR_NAMES:
        ours, theirs = getattr(wide, name, None), getattr(narrow, name, None)
        if not isinstance(ours, np.ndarray):
            continue
        if isinstance(theirs, tuple):
            theirs = (theirs[0] * theirs[1]).reshape(ours.shape)
        groups = np.abs(ours).reshape(-1, 32).max(axis=1) if ours.shape[-1] % 32 == 0 else np.abs(ours).max()
        assert np.all(np.abs(ours - theirs).reshape(-1, 32 if ours.shape[-1] % 32 == 0 else ours.size)
                      <= np.reshape(groups, (-1, 1)) / 254 + 1e-7), name
        compared += 1
    assert compared >= 20


@pytest.mark.parametrize("broken", ["plus_one", "rotate", "gated", "neg_exp", "grouped"])
def test_each_step_of_the_conversion_is_needed(broken, monkeypatch):
    """What each transform does shows in the logits: without it they are far from transformers'."""
    tensors = weights()
    if broken == "plus_one":
        monkeypatch.setitem(tensors, "model.norm.weight", tensors["model.norm.weight"] - 1)
    elif broken == "rotate":
        name = "model.layers.3.self_attn.k_proj.weight"
        w = tensors[name].reshape(2, 32, -1)
        tensors[name] = np.concatenate([w[:, 4:8], w[:, :4], w[:, 8:]], axis=1).reshape(64, -1)
    elif broken == "gated":
        name = "model.layers.3.self_attn.q_proj.weight"
        tensors[name] = tensors[name].reshape(4, 2, 32, -1)[:, ::-1].reshape(256, -1).copy()
    elif broken == "neg_exp":
        tensors["model.layers.0.linear_attn.A_log"] = tensors["model.layers.0.linear_attn.A_log"] + 1
    elif broken == "grouped":
        # q and k of the two heads swapped: each head of v reads the other's (heads 0 and 1 of v read head 0)
        name = "model.layers.0.linear_attn.in_proj_qkv.weight"
        w = tensors[name]
        tensors[name] = np.concatenate([w[:32].reshape(2, 16, -1)[::-1].reshape(32, -1),
                                        w[32:64].reshape(2, 16, -1)[::-1].reshape(32, -1), w[64:]]).copy()
    ours, theirs = logits(engine(conversion(tensors))), np.array(REFERENCE["logits"], dtype=np.float32)
    assert np.abs(ours - theirs).max() / np.abs(theirs).max() > 1e-3


# ---- the GGUF llama.cpp's convert_hf_to_gguf.py writes (Qwen3_5TextModel), and Prism's rotation (Bonsai 2)
GGUF_NAMES = {"input_layernorm": "attn_norm", "post_attention_layernorm": "post_attention_norm",
              "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
              "self_attn.o_proj": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
              "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.in_proj_b": "ssm_beta",
              "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.conv1d": "ssm_conv1d", "linear_attn.norm": "ssm_norm",
              "linear_attn.out_proj": "ssm_out", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
              "mlp.down_proj": "ffn_down"}


def tiled(w, axis, start, head, k_heads=2, per_k=2):
    """convert_hf_to_gguf.py's _reorder_v_heads: the heads of v from grouped by the head of k to tiled."""
    moved = np.moveaxis(w, axis, 0)
    rest = moved[start:].reshape(k_heads, per_k, head, *moved.shape[1:]).swapaxes(0, 1).reshape(moved[start:].shape)
    return np.moveaxis(np.concatenate([moved[:start], rest]), 0, axis)


def llama_cpp_tensors(tensors, grouped_out=False):
    """What convert_hf_to_gguf.py makes of transformers' tensors, by llama.cpp's names: the norms 1 + weight (not the
    DeltaNet's gated one), A_log as -exp(A_log), the conv without its middle 1, the heads of v tiled."""
    out = {}
    for name, w in tensors.items():
        w = np.asarray(w, dtype=np.float32)
        if name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            gguf = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                    "lm_head.weight": "output.weight"}[name]
            out[gguf] = w + 1 if name == "model.norm.weight" else w
            continue
        _, _, layer, *rest = name.split(".")
        kind, what = ".".join(rest[:-1]), rest[-1]
        if kind == "linear_attn" and what == "A_log":
            out[f"blk.{layer}.ssm_a"] = tiled(-np.exp(w), 0, 0, 1)
            continue
        if kind == "linear_attn" and what == "dt_bias":
            out[f"blk.{layer}.ssm_dt.bias"] = tiled(w, 0, 0, 1)
            continue
        gguf = GGUF_NAMES[kind]
        if gguf in ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm"):
            w = w + 1
        if gguf == "ssm_conv1d":
            w = tiled(w.reshape(w.shape[0], -1), 0, 64, 16)
        elif gguf == "attn_qkv":
            w = tiled(w, 0, 64, 16)
        elif gguf == "attn_gate":
            w = tiled(w, 0, 0, 16)
        elif gguf in ("ssm_beta", "ssm_alpha"):
            w = tiled(w, 0, 0, 1)
        elif gguf == "ssm_out" and not grouped_out:
            w = tiled(w, 1, 0, 16)
        out[f"blk.{layer}.{gguf}.{what}"] = w
    return out


BLOCK = 16


def rotation(width, seed):
    """Prism's rotation of an input of this width: the signs, and x -> H(signs * x) as a matrix (blocks of BLOCK)."""
    signs = np.where(np.random.default_rng(seed).random(width) < 0.5, -1.0, 1.0).astype(np.float32)
    matrix = hadamard(np.diag(signs), BLOCK)  # row i is H(signs e_i) along the last axis: the columns are H(s e_i)
    return signs, matrix.T  # matrix.T @ x = H(signs * x)


def rotated(tensors, config=CONFIG):
    """The same model stored the way Prism stores Bonsai 2: every matrix but the DeltaNet's two gates for its input
    rotated (W' = W R^T, R x = H(s x), so W' R x = W x), the embedding's rows rotated as R h, and the metadata that says so."""
    dim, hidden = config["hidden_size"], config["intermediate_size"]
    widths = {dim: rotation(dim, 1), 128: rotation(128, 2), hidden: rotation(hidden, 3)}
    folded = {}
    for name, w in tensors.items():
        width = w.shape[-1] if w.ndim == 2 else None
        if name.endswith(("in_proj_b.weight", "in_proj_a.weight")) or width is None:
            folded[name] = w
        elif name == "model.embed_tokens.weight":
            folded[name] = (widths[dim][1] @ w.T).T.astype(np.float32)
        else:
            folded[name] = (w @ widths[width][1].T).astype(np.float32)
    names = ["output.weight"]
    for layer, kind in enumerate(config["layer_types"]):
        kinds = ("attn_q", "attn_k", "attn_v", "attn_output") if kind == "full_attention" else ("attn_qkv", "attn_gate", "ssm_out")
        names += [f"blk.{layer}.{k}.weight" for k in (*kinds, "ffn_gate", "ffn_up", "ffn_down")]
    # q_dim (4 x 32) and the DeltaNet's v (4 x 16) are two widths, 128 and 64; Prism keys its signs by width only,
    # so the DeltaNet's output shares dim's signs here, as Bonsai 2's shares 6144's
    widths[64] = widths[dim]
    metadata = [("prism.hadamard.version", 4, 1), ("prism.hadamard.block_size", 4, BLOCK),
                ("prism.hadamard.transform", 8, "normalized-sylvester-walsh-hadamard"),
                ("prism.hadamard.axis", 8, "input-last-dimension"), ("prism.hadamard.sign_mode", 8, "explicit"),
                ("prism.hadamard.weight_names", (9, 8), names), ("prism.hadamard.inverse_weight_names", (9, 8), ["token_embd.weight"]),
                ("prism.hadamard.sign_widths", (9, 5), [dim, 128, hidden]),
                ("prism.hadamard.sign_values", (9, 5), [int(v) for width in (dim, 128, hidden) for v in widths[width][0]]),
                ("prism.hadamard.gdn_v_grouped", 7, True)]
    return folded, metadata


def gguf(tensors, config=CONFIG, more=(), grouped_out=False):
    """A GGUF v3 of these tensors (float32) and metadata as llama.cpp writes a Qwen3.5's."""
    string = lambda text: struct.pack("<Q", len(text.encode())) + text.encode()
    scalar = {4: "<I", 5: "<i", 6: "<f", 7: "<?"}

    def value(kind, v):
        if kind == 8:
            return string(v)
        if isinstance(kind, tuple):
            item = kind[1]
            return struct.pack("<IQ", item, len(v)) + b"".join(value(item, x) for x in v)
        return struct.pack(scalar[kind], v)

    arch = "qwen35"
    metadata = [("general.architecture", 8, arch), (f"{arch}.block_count", 4, config["num_hidden_layers"]),
                (f"{arch}.context_length", 4, config["max_position_embeddings"]),
                (f"{arch}.embedding_length", 4, config["hidden_size"]),
                (f"{arch}.feed_forward_length", 4, config["intermediate_size"]),
                (f"{arch}.attention.head_count", 4, config["num_attention_heads"]),
                (f"{arch}.attention.head_count_kv", 4, config["num_key_value_heads"]),
                (f"{arch}.attention.key_length", 4, config["head_dim"]), (f"{arch}.attention.value_length", 4, config["head_dim"]),
                (f"{arch}.rope.freq_base", 6, 10000.0), (f"{arch}.rope.dimension_count", 4, 8),
                (f"{arch}.attention.layer_norm_rms_epsilon", 6, 1e-6), (f"{arch}.ssm.conv_kernel", 4, 4),
                (f"{arch}.ssm.state_size", 4, 16), (f"{arch}.ssm.group_count", 4, 2), (f"{arch}.ssm.time_step_rank", 4, 4),
                (f"{arch}.ssm.inner_size", 4, 64), (f"{arch}.full_attention_interval", 4, 4),
                ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "qwen2"),
                ("tokenizer.ggml.tokens", (9, 8), [f"w{i}" for i in range(config["vocab_size"])]),
                ("tokenizer.ggml.merges", (9, 8), []), ("tokenizer.ggml.bos_token_id", 4, 1),
                ("tokenizer.ggml.eos_token_id", 4, 2), *more]
    held = llama_cpp_tensors(tensors, grouped_out)
    out = [b"GGUF", struct.pack("<IQQ", 3, len(held), len(metadata))]
    for key, kind, v in metadata:
        out.append(string(key) + struct.pack("<I", 9 if isinstance(kind, tuple) else kind) + value(kind, v))
    blobs, offset = [], 0
    for name, w in held.items():
        blob = np.ascontiguousarray(w, dtype=np.float32).tobytes()
        out.append(string(name) + struct.pack("<I", w.ndim) + struct.pack(f"<{w.ndim}Q", *reversed(w.shape))
                   + struct.pack("<IQ", 0, offset))
        blobs.append(blob + b"\0" * (-len(blob) % 32))
        offset += len(blobs[-1])
    head = b"".join(out)
    return head + b"\0" * (-len(head) % 32) + b"".join(blobs)


def from_gguf(file, dtype="float32", with_config=False):
    if with_config:
        header, base = llama2_convert.gguf_weights(file, json.dumps(CONFIG))
        made = llama2_convert.Conversion(header, base, json.dumps(CONFIG), VOCABULARY, "tokenizer.json", dtype=dtype,
                                         max_seq_len=64, start=base)
    else:
        made = llama2_convert.Conversion.from_gguf(file, dtype=dtype, max_seq_len=64)
    made.feed(file[made.base if not with_config else base:])
    made.finish()
    return made


@pytest.mark.parametrize("with_config", [False, True], ids=["alone", "with its config.json"])
def test_a_gguf_is_the_safetensors_conversion(with_config):
    """llama.cpp's GGUF of the same model: its norms, decay, conv and tiled heads of v back to transformers' ways, the
    same checkpoint to the byte."""
    made = from_gguf(gguf(weights()), with_config=with_config)
    assert bytes(made.checkpoint) == bytes(conversion(weights()).checkpoint)


def test_prisms_rotation_computes_what_the_model_unrotated_does():
    """Bonsai 2's way: the matrices stored for rotated inputs, the embedding's rows rotated, the signs in the metadata.
    The engine rotates the inputs back and computes transformers' logits."""
    folded, metadata = rotated(weights())
    file = gguf(folded, more=metadata, grouped_out=True)
    assert from_gguf(file).options["hadamard"] == BLOCK
    # with the original's config.json (which says nothing of the rotation), the GGUF's header carries it
    made = from_gguf(file, with_config=True)
    assert made.options["hadamard"] == BLOCK
    ours, theirs = logits(engine(made)), np.array(REFERENCE["logits"], dtype=np.float32)
    worst = np.abs(ours - theirs).max() / np.abs(theirs).max()
    assert worst < 1e-4, worst
    # without the engine's rotation the same file computes something else
    unrotated = dict(made.options, hadamard=0)
    bare = Llama(bytes(made.checkpoint)[:-4 * (64 + 128 + 64 + 96)], bytes(made.tokenizer), dtype="float32",
                 tokenizer_kind="unigram", **{k: v for k, v in unrotated.items()
                                             if k not in ("dtype", "template", "specials", "tokenizer_kind", "stop_tokens", "bos")})
    assert np.abs(logits(bare) - theirs).max() / np.abs(theirs).max() > 0.1


@pytest.mark.parametrize("change, what", [
    ("prism.hadamard.transform", "normalized-sylvester"), ("prism.hadamard.weight_names", "folded into other"),
    ("prism.hadamard.version", "version"), ("prism.hadamard.sign_values", "signs")])
def test_a_rotation_the_engine_does_not_do_is_refused(change, what):
    folded, metadata = rotated(weights())
    changed = []
    for key, kind, v in metadata:
        if key == change:
            v = {"prism.hadamard.transform": lambda v: "other", "prism.hadamard.weight_names": lambda v: v[:-1],
                 "prism.hadamard.version": lambda v: 2, "prism.hadamard.sign_values": lambda v: [2] + v[1:]}[key](v)
        changed.append((key, kind, v))
    with pytest.raises(ValueError, match="rotation"):
        from_gguf(gguf(folded, more=changed, grouped_out=True))


def test_hadamard_is_its_own_inverse_and_sylvesters():
    block = 16
    x = np.random.default_rng(1).standard_normal(3 * block).astype(np.float32)
    rows = np.arange(block)
    matrix = np.array([[(-1) ** bin(i & j).count("1") for j in rows] for i in rows], dtype=np.float64) / np.sqrt(block)
    assert np.allclose(hadamard(x, block), (x.reshape(3, block) @ matrix.T).reshape(-1), atol=1e-6)
    assert np.allclose(hadamard(hadamard(x, block), block), x, atol=1e-6)
