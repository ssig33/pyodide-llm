# gguf_check.py
# The check T74 needs before a GGUF reader is written: a mistake there does not raise, it only makes the text
# worse (the way T72 broke), so what "right" means is fixed first, on a real model, and the reader is held to it.
#
# The GGUF reading here is a reference of its own, written from the format's description and kept apart from
# public/llama2_convert.py on purpose: the reader that goes into the page must not share code with what checks it.
#
#   python3 tests/gguf_check.py tensors <model.gguf> <directory of the same model: config.json, model.safetensors
#                                       or its shards and model.safetensors.index.json> [--original-vocabulary]
#       Every tensor of the GGUF against the Hugging Face one: its name, shape and relative error, and for the
#       q and k matrices which order the GGUF holds (Hugging Face's, or turned the way llama.cpp and llama2.c turn
#       them). For Q8_0, also how many int8 values equal what llama2_convert.quantize() makes of the original.
#       The embedding and the classifier also row by row (T136 stage 2), and Llama 3's rope_freqs.weight against
#       the engine's rope_frequencies() of the original's rope_scaling. The metadata against config.json (after
#       normalize()), and the vocabulary against tokenizer.json and tokenizer.model. T145: every tensor also against
#       the nearest of what a GGUF made the usual ways holds (TIGHT), and the rows that pass only against a Q8_0 of
#       the original are listed with their id, piece and norm. --original-vocabulary (stage 2:
#       the page takes the vocabulary from the original) shows a difference in the GGUF's vocabulary without
#       counting it. The last line is the summary as JSON. T136's third stage: GPT-2 (its Conv1D matrices, which
#       llama.cpp turns to (out, in), and its output.weight, a copy of the embedding) and GPT-NeoX (its
#       query_key_value, which llama.cpp splits into all of q, k, then v); the order found is said as for q and k.
#   python3 tests/gguf_check.py logits <out A> <out B> <text file> [tokens = 300]
#       Two converted checkpoints (the <out> of tests/perplexity_prepare.py) on the same text: the largest logit
#       difference, how often the most likely token agrees, and the perplexity of each. The acceptance of T74 is
#       this between the int8 made from the GGUF and the int8 made from safetensors: top-1 agreement of 99% and
#       perplexity within 0.2% (exit 1 otherwise). That line holds for models whose int8 costs about 0.6% or less
#       (SmolLM2 135M, Pythia 410M): where it costs 1 to 3% (Pythia 70M and 160M, GPT-2), an int8 of the original
#       rounded again as well (its scales made float16) is outside it too (T136's third stage, the review).
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))

# ------------------------------------------------------------------------------------------- the reference reader
# GGUF v2/v3: "GGUF", version u32, tensor count u64, metadata count u64, the metadata (key, type u32, value), the
# tensor infos (name, n_dims u32, dims u64 innermost first, type u32, offset u64), then the data, aligned to
# general.alignment (32 when absent). Offsets count from the start of the data.
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
F32, F16, Q8_0, BF16 = 0, 1, 8, 30
# T98: the 4- and 5-bit types of the GGUF files that are about int4, read only to measure them (widened to float32)
Q4_0, Q4_1, Q5_0, Q4_K, Q6_K = 2, 3, 6, 12, 14
TYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0",
              Q4_K: "Q4_K", Q6_K: "Q6_K"}
# bytes per value: a block of 32 values (or a super-block of 256) and its scales
BYTES = {F32: 4, F16: 2, BF16: 2, Q8_0: 34 / 32, Q4_0: 18 / 32, Q4_1: 20 / 32, Q5_0: 22 / 32, Q4_K: 144 / 256, Q6_K: 210 / 256}


def half(raw):
    return raw.copy().view(np.float16).astype(np.float32)


def low_high(qs):
    """16 bytes of 4-bit values per block: the low halves are values 0..15, the high halves 16..31 (ggml's order)."""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


def widen_q4_0(raw):  # a float16 scale, 16 bytes: (q - 8) * d
    blocks = raw.reshape(-1, 18)
    return (low_high(blocks[:, 2:]).astype(np.float32) - 8) * half(blocks[:, :2])


def widen_q4_1(raw):  # a float16 scale and a float16 minimum, 16 bytes: q * d + m
    blocks = raw.reshape(-1, 20)
    return low_high(blocks[:, 4:]).astype(np.float32) * half(blocks[:, :2]) + half(blocks[:, 2:4])


def widen_q5_0(raw):  # a float16 scale, 32 fifth bits (u32), 16 bytes: (q | fifth << 4) - 16, times d
    blocks = raw.reshape(-1, 22)
    fifth = blocks[:, 2:6].copy().view("<u4")
    bits = (fifth >> np.arange(32, dtype=np.uint32)) & 1
    return ((low_high(blocks[:, 6:]) | (bits << 4)).astype(np.float32) - 16) * half(blocks[:, :2])


def widen_q4_k(raw):
    """Super-blocks of 256: float16 d and dmin, 12 bytes of eight 6-bit scales and eight 6-bit minimums, 128 bytes
    of 4-bit values. Sub-block j of 32: value * d * scale[j] - dmin * min[j]; the values go 64 at a time, the low
    halves of 32 bytes first, then their high halves."""
    blocks = raw.reshape(-1, 144)
    d, dmin, packed, qs = half(blocks[:, 0:2]), half(blocks[:, 2:4]), blocks[:, 4:16].astype(np.int32), blocks[:, 16:]
    scales, minimums = np.empty((len(blocks), 8), np.int32), np.empty((len(blocks), 8), np.int32)
    scales[:, :4], minimums[:, :4] = packed[:, 0:4] & 63, packed[:, 4:8] & 63
    scales[:, 4:] = (packed[:, 8:12] & 0x0F) | ((packed[:, 0:4] >> 6) << 4)
    minimums[:, 4:] = (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 6) << 4)
    values = qs.reshape(-1, 4, 32)
    values = np.stack([values & 0x0F, values >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    return (values * (d * scales)[:, :, None] - (dmin * minimums)[:, :, None]).reshape(-1, 256)


def widen_q6_k(raw):
    """Super-blocks of 256: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 int8 scales (one per 16 values),
    a float16 d. Two halves of 128; in each, value l (0..31) of the four quarters is made of ql[l], ql[l + 32]
    (low and high nibbles) and the four 2-bit fields of qh[l], minus 32."""
    blocks = raw.reshape(-1, 210)
    ql, qh = blocks[:, :128].reshape(-1, 2, 64), blocks[:, 128:192].reshape(-1, 2, 32)
    scales, d = blocks[:, 192:208].copy().view(np.int8).astype(np.float32), half(blocks[:, 208:210])
    quarters = [(ql[:, :, :32] & 0x0F) | (((qh >> 0) & 3) << 4), (ql[:, :, 32:] & 0x0F) | (((qh >> 2) & 3) << 4),
                (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4), (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)]
    values = np.stack(quarters, axis=2).astype(np.float32) - 32  # (blocks, half, quarter, 32)
    scale = scales.reshape(-1, 2, 4, 2).repeat(16, axis=3)  # one per 16 values
    return (values * scale * d[:, :, None, None]).reshape(-1, 256)


WIDEN = {Q4_0: (32, 18, widen_q4_0), Q4_1: (32, 20, widen_q4_1), Q5_0: (32, 22, widen_q5_0),
         Q4_K: (256, 144, widen_q4_k), Q6_K: (256, 210, widen_q6_k)}


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, fmt):
        (value,) = struct.unpack_from(fmt, self.data, self.at)
        self.at += struct.calcsize(fmt)
        return value

    def string(self):
        size = self.take("<Q")
        self.at += size
        return bytes(self.data[self.at - size:self.at]).decode("utf-8", errors="replace")

    def value(self, kind):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item, count = self.take("<I"), self.take("<Q")
            return [self.value(item) for _ in range(count)]
        return self.take(SCALARS[kind])


def read_gguf(path):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    r = Reader(data)
    assert bytes(data[:4]) == b"GGUF", "not a GGUF file"
    r.at = 4
    version, tensors, entries = r.take("<I"), r.take("<Q"), r.take("<Q")
    assert version in (2, 3), f"GGUF version {version}: only 2 and 3 count tensors in 64 bits (1 used 32)"
    metadata = {}
    for _ in range(entries):
        key = r.string()
        metadata[key] = r.value(r.take("<I"))
    infos = {}
    for _ in range(tensors):
        name = r.string()
        dims = [r.take("<Q") for _ in range(r.take("<I"))]
        infos[name] = {"shape": tuple(reversed(dims)), "type": r.take("<I"), "offset": r.take("<Q")}
    alignment = metadata.get("general.alignment", 32)
    base = (r.at + alignment - 1) // alignment * alignment
    return version, metadata, infos, data, base


def tensor(info, data, base, first=0, last=None):
    """float32 values of rows first..last, and for Q8_0 also the int8 values and the float16 scales as stored."""
    shape = info["shape"]
    last = shape[0] if last is None else last
    row = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    if len(shape) > 1:
        shape = (last - first, *shape[1:])
    count = math.prod(shape)
    size = BYTES.get(info["type"], 0)
    start = base + info["offset"] + int(first * row * size)
    if info["type"] == F32:
        return np.frombuffer(data, np.float32, count, start).reshape(shape), None
    if info["type"] == F16:
        return np.frombuffer(data, np.float16, count, start).astype(np.float32).reshape(shape), None
    if info["type"] == BF16:
        wide = np.frombuffer(data, np.uint16, count, start).astype(np.uint32) << 16
        return wide.view(np.float32).reshape(shape), None
    if info["type"] == Q8_0:
        # blocks of 32: a float16 scale, then 32 int8
        blocks = np.frombuffer(data, np.uint8, count // 32 * 34, start).reshape(-1, 34)
        scales = blocks[:, :2].copy().view(np.float16).reshape(-1)
        values = blocks[:, 2:].copy().view(np.int8)
        return (values * scales.astype(np.float32)[:, None]).reshape(shape), (values, scales)
    if info["type"] in WIDEN:
        values, block_bytes, widen = WIDEN[info["type"]]
        raw = np.frombuffer(data, np.uint8, count // values * block_bytes, start)
        return widen(raw).reshape(shape), None
    raise ValueError(f"type {info['type']} is not one this reads ({', '.join(TYPE_NAMES.values())})")


# ------------------------------------------------------------------------------------------- names and orders
# llama.cpp's names and Hugging Face's, for each architecture: the tensors outside the layers, where a layer's go,
# and the layer's. GPT-2's are openai-community/gpt2's (no "transformer." in front; check_tensors() tries both), and
# its output.weight is the copy of the embedding llama.cpp writes for a GPT-2 (T136's third stage)
NAMES = {
    "llama": ({"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
               "output.weight": "lm_head.weight"}, "model.layers.{}.",
              {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
               "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
               "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}),
    "gpt2": ({"token_embd.weight": "wte.weight", "position_embd.weight": "wpe.weight", "output_norm.weight": "ln_f.weight",
              "output_norm.bias": "ln_f.bias", "output.weight": "wte.weight"}, "h.{}.",
             {"attn_norm": "ln_1", "attn_qkv": "attn.c_attn", "attn_output": "attn.c_proj", "ffn_norm": "ln_2",
              "ffn_up": "mlp.c_fc", "ffn_down": "mlp.c_proj"}),
    "gptneox": ({"token_embd.weight": "gpt_neox.embed_in.weight", "output_norm.weight": "gpt_neox.final_layer_norm.weight",
                 "output_norm.bias": "gpt_neox.final_layer_norm.bias", "output.weight": "embed_out.weight"},
                "gpt_neox.layers.{}.",
                {"attn_norm": "input_layernorm", "attn_qkv": "attention.query_key_value", "attn_output": "attention.dense",
                 "ffn_norm": "post_attention_layernorm", "ffn_up": "mlp.dense_h_to_4h", "ffn_down": "mlp.dense_4h_to_h"}),
}
NAMES["qwen2"] = NAMES["llama"]
# T203: a Qwen3's are a Llama's and the norms of each head of q and k
NAMES["qwen3"] = (NAMES["llama"][0], NAMES["llama"][1],
                  {**NAMES["llama"][2], "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"})


def hugging_face_name(name, arch="llama"):
    """blk.3.attn_q.weight -> model.layers.3.self_attn.q_proj.weight (Llama; T74 starts with it). None for a
    tensor Hugging Face has no counterpart of (rope_freqs.weight: llama.cpp's table of Llama 3's RoPE scaling,
    made from config.json's rope_scaling, checked on its own)."""
    fixed, layer, parts = NAMES.get(arch, NAMES["llama"])
    if name in fixed:
        return fixed[name]
    pieces = name.split(".", 2)
    if len(pieces) != 3 or pieces[0] != "blk":
        return None
    _, number, rest = pieces
    tensor, kind = rest.rsplit(".", 1)  # .weight, or .bias (Qwen2's q, k and v)
    return f"{layer.format(number)}{parts[tensor]}.{kind}" if tensor in parts else None


def turned(w, heads):
    """What llama.cpp's convert_hf_to_gguf.py does to q and k (the same turn as llama2.c's export): the two
    halves of each head, as Hugging Face keeps them for rotate_half, back to adjacent pairs. Written out here
    rather than taken from llama2_convert.permute_heads, which is what is being checked."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def split(w, heads):
    """What llama.cpp's convert_hf_to_gguf.py does to GPT-NeoX's query_key_value (and its bias): q, k and v of each
    head in turn, as Hugging Face keeps them, to all of q, then all of k, then all of v. Written out here rather than
    taken from llama2_convert.unsplit, which is what is being checked."""
    return w.reshape(heads, 3, w.shape[0] // heads // 3, *w.shape[1:]).swapaxes(0, 1).reshape(w.shape)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ------------------------------------------------------------------------------------------- the vocabularies
def protobuf(data):
    """(field, value) of one protobuf message, nested messages as bytes: enough of the format for a sentencepiece
    model (again apart from llama2_convert's own reader)."""
    i = 0

    def varint():
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < len(data):
        key = varint()
        kind = key & 7
        if kind == 0:
            yield key >> 3, varint()
        elif kind in (1, 5):
            i += 8 if kind == 1 else 4
            yield key >> 3, data[i - (8 if kind == 1 else 4):i]
        elif kind == 2:
            size = varint()
            i += size
            yield key >> 3, data[i - size:i]
        else:
            raise ValueError(f"protobuf wire type {kind}")


def original_vocabularies(directory):
    """{file name: the pieces by id} of the original's tokenizer.json and tokenizer.model, where they are."""
    found = {}
    path = directory / "tokenizer.json"
    if path.exists():
        parsed = json.loads(path.read_text())
        vocab = parsed["model"]["vocab"]
        # BPE writes {piece: id}, Unigram [[piece, score], ...] in the order of the ids (llm-jp's)
        by_id = dict(enumerate(p for p, _ in vocab)) if isinstance(vocab, list) else {i: p for p, i in vocab.items()}
        by_id.update({t["id"]: t["content"] for t in parsed.get("added_tokens", [])})
        found["tokenizer.json"] = [by_id.get(i) for i in range(max(by_id) + 1)]
    for name in ("tokenizer.model", "spiece.model"):
        if (directory / name).exists():
            pieces = [dict(protobuf(value)).get(1, b"").decode("utf-8")
                      for field, value in protobuf((directory / name).read_bytes()) if field == 1]
            found[name] = pieces
    return found


def compare_vocabulary(metadata, pieces, name, rows):
    """How many ids hold another piece in the GGUF than in the original, and whether the only difference in
    length is llama.cpp's padding: it fills the vocabulary up to the rows of the embedding with pieces of its
    own ([PAD151665] ..., token type 5, unused), where tokenizer.json ends earlier (T136: Qwen2.5's 151665 of
    151936)."""
    tokens = metadata.get("tokenizer.ggml.tokens", [])
    kinds = metadata.get("tokenizer.ggml.token_type", [])
    differ = sum(a != b for a, b in zip(tokens, pieces))
    padding = len(tokens) > len(pieces) and len(tokens) == rows and len(kinds) == len(tokens) \
        and all(kind == 5 for kind in kinds[len(pieces):])
    print(f"\nvocabulary: GGUF {len(tokens)} pieces ({metadata.get('tokenizer.ggml.model')}), {name} {len(pieces)}; "
          f"{differ} differ at the same id"
          + (f"; the GGUF's last {len(tokens) - len(pieces)} are its padding (unused) up to vocab_size" if padding else ""))
    return differ + (0 if len(tokens) == len(pieces) or padding else abs(len(tokens) - len(pieces)))


# ------------------------------------------------------------------------------------------- the two checks
BLOCK = 8 << 20  # values; larger tensors are compared in blocks of rows
# T136 stage 2: the embedding and the classifier are held row by row. The error of the whole tensor under 0.02
# let 8 rows of Qwen2.5 0.5B's embedding be swapped (0.0122; 26 rows before it failed), while one wrong row is
# a token that means another. A row of Q8_0 is off by about 0.005 (row_check() says what else passes).
ROW_LINE = 0.05
BY_ROW = ("token_embd.weight", "output.weight")


def q8_0_of(original):
    """What llama.cpp's Q8_0 makes of these values (quantize_row_q8_0_ref): per 32, d = largest / 127, the values
    rounded (half away from zero) times 1/d, and d kept as float16. A row whose values are all below about 7.6e-6
    has a d under float16's smallest step, and comes back as 0 or coarse (llm-jp-3 980M's unused rows, T136)."""
    groups = original.reshape(-1, 32).astype(np.float32)
    d = np.abs(groups).max(axis=1) / 127
    inverse = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    scaled = groups * inverse[:, None]
    scale = d.astype(np.float16).astype(np.float32)[:, None]
    # a d under float16's half step is stored as 0 and reads back 0 (1/d may have overflowed to inf: inf * 0 is NaN)
    return np.where(scale > 0, np.sign(scaled) * np.floor(np.abs(scaled) + 0.5) * scale, 0).reshape(original.shape)


# T145: a GGUF made from the original holds its values as they are where it keeps them (F32: 0 off) and llama.cpp's
# Q8_0 of them where it quantizes (sarashina2.2's two and llm-jp-3 980M: 0 off, int8 99.999% the same; the review of
# T136's second stage). So each tensor is also held to TIGHT against the nearest of those, where the whole tensor's
# 0.02 lets 4% of noise or a scale of 1.05 through. The base and the instruct model of one family are 8.3e-3 apart at
# their nearest tensor. Should one of the list's GGUFs be past it, the line goes back to what it was (the review).
TIGHT = 1e-3


def references(original, q8_0):
    """What a GGUF made from original the usual ways holds: the values (an F32 or F16 tensor, near enough for F16),
    and for a Q8_0 one llama.cpp's Q8_0 of them, or of them made float16 first (mradermacher's RakutenAI 7B chat)."""
    if not q8_0:
        return [original]
    return [original, q8_0_of(original), q8_0_of(original.astype(np.float16).astype(np.float32))]


def squares(values, original, q8_0):
    """[(the difference squared, the reference squared)] against each of references(), summed in float64, to be
    summed further over the blocks of a large tensor."""
    wide = values.astype(np.float64)
    return [(float(((wide - reference) ** 2).sum()), float((reference.astype(np.float64) ** 2).sum()))
            for reference in references(original, q8_0)]


def nearest(sums):
    """The relative error against the nearest reference, of squares()."""
    return min(math.sqrt(difference / max(norm, 1e-300)) for difference, norm in sums)


def row_parts(values, original, q8_0):
    """Per row (in float64: a row of 1e-37 squares to nothing in float32): the norm of the difference and of the
    original, and for a Q8_0 tensor the same against q8_0_of(original) and q8_0_of(the original made float16):
    some GGUFs went through a float16 file first (mradermacher's RakutenAI 7B chat, T136), and float16 rounds the
    values under 6.1e-5 more coarsely than bfloat16 holds them."""
    norm = lambda a: np.linalg.norm(a.reshape(len(a), -1).astype(np.float64), axis=1)
    parts = [norm(values - original), norm(original)]
    if q8_0:
        for rounded in (q8_0_of(original), q8_0_of(original.astype(np.float16).astype(np.float32))):
            parts += [norm(values - rounded), norm(rounded)]
    return parts


def row_check(parts):
    """(how many rows are past ROW_LINE, the worst error, its row, up to 16 of the rows past the line as [row, error,
    the row's norm over the median], how many rows are past it against the original but not against its Q8_0, and up
    to 64 of those as [row, error against the original, against the nearest Q8_0, the row's norm over the median]).
    A row's error is against the original, or against llama.cpp's Q8_0 of it where that is nearer (row_parts()): a
    row of values so small that the float16 scale rounds them is the format's rounding, while a row of another
    token is far from all of them. A norm under 1e-3 of the median counts as that (Qwen2.5 7B's unused rows,
    largest value 1.2e-37)."""
    norms = parts[1]
    median = max(float(np.median(norms)), 1e-30)
    floor = 1e-3 * median
    against = np.nan_to_num(parts[0] / np.maximum(norms, floor), nan=np.inf)  # a NaN is past the line
    each = against
    for difference, norm in zip(parts[2::2], parts[3::2]):
        each = np.fmin(each, difference / np.maximum(norm, floor))  # a reference of NaN (float16 overflow) is no reference
    bad = np.flatnonzero(each > ROW_LINE)
    rounded = np.flatnonzero((against > ROW_LINE) & (each <= ROW_LINE))
    return (len(bad), float(each.max()), int(each.argmax()),
            [[int(i), round(float(each[i]), 4), float(f"{norms[i] / median:.3g}")] for i in bad[:16]],
            len(rounded),
            [[int(i), round(float(against[i]), 4), round(float(each[i]), 4), float(f"{norms[i] / median:.3g}")]
             for i in rounded[:64]])


def large(info, data, base, hf, target, shape, quantize, by_row, head_rows=0, split_heads=0):
    """relative error, the error against the nearest reference (squares()) and int8 equality of a large matrix, a
    block of rows at a time; by_row: also the error of each row (row_check()). head_rows: q or k, whose heads of head_rows rows the GGUF may hold turned: the
    blocks are whole heads, and the order is the one of the first block (the order found is returned too).
    split_heads: GPT-NeoX's query_key_value of that many heads, which the GGUF may hold split (split()): the blocks
    are one head's q, k or v, and the original's rows of it are fetched from where Hugging Face keeps them."""
    rows = max(1, BLOCK // math.prod(shape[1:]))
    if head_rows:
        rows = max(head_rows, rows // head_rows * head_rows)
    size = shape[0] // 3 // split_heads if split_heads else 0
    if split_heads:
        rows = size
    difference = total = same = count = 0.0
    parts, sums = [], None
    order = ""
    for first in range(0, shape[0], rows):
        last = min(first + rows, shape[0])
        values, raw = tensor(info, data, base, first, last)
        original = hf.rows(target, first, last).reshape(last - first, *shape[1:]).astype(np.float32)
        if split_heads:
            part, head = divmod(first // size, split_heads)
            source = (head * 3 + part) * size
            moved = hf.rows(target, source, source + size).reshape(size, *shape[1:]).astype(np.float32)
            if not order and source != first:  # the first head's q is where it is in either order: not that block
                order = "split (q, k, v)" if relative(values, moved) < relative(values, original) else "as Hugging Face"
            if order.startswith("split"):
                original = moved
        if head_rows:
            turn = turned(original, (last - first) // head_rows)
            if not order:
                order = "turned (llama2.c order)" if relative(values, turn) < relative(values, original) else "as Hugging Face"
            if order.startswith("turned"):
                original = turn
        difference += float(((values - original) ** 2).sum())
        total += float((original ** 2).sum())
        block = squares(values, original, raw is not None)
        sums = block if sums is None else [(a + c, b + d) for (a, b), (c, d) in zip(sums, block)]
        if by_row:
            parts.append(row_parts(values, original, raw is not None))
        if raw is not None:
            ours, _ = quantize(original.reshape(-1, original.shape[-1]))
            same += float((ours.reshape(-1) == raw[0].reshape(-1)).sum())
            count += ours.size
    rowwise = None
    if by_row:
        rowwise = row_check([np.concatenate(column) for column in zip(*parts)])
    return (math.sqrt(difference / max(total, 1e-30)), nearest(sums), f"{same / count * 100:.2f}%" if count else "",
            rowwise, order)


def config_pairs(config, arch="llama"):
    """(GGUF key, config.json name, the value config.json says, counted) of the original's config.json after
    normalize(): what the page reads (CAT-Translate 1.4b writes transformers 5's rope_parameters). The context is
    shown and not counted: a Mistral's sliding window cuts it in normalize(). GPT-2's and GPT-NeoX's (T136's third
    stage): LayerNorm's epsilon, and GPT-NeoX's rotated part of a head and parallel branches, which no tensor says."""
    from llama2_convert import head_size, normalize, rotary_dim

    config = normalize(config)
    heads = config["num_attention_heads"]
    if arch in ("gpt2", "gptneox"):
        pairs = [("block_count", "num_hidden_layers", config.get("num_hidden_layers"), True),
                 ("embedding_length", "hidden_size", config.get("hidden_size"), True),
                 ("feed_forward_length", "intermediate_size", config.get("intermediate_size"), True),
                 ("attention.head_count", "num_attention_heads", heads, True),
                 # transformers' default where config.json says none
                 ("attention.layer_norm_epsilon", "layer_norm_eps",
                  config.get("layer_norm_eps", config.get("layer_norm_epsilon", 1e-5)), True),
                 ("context_length", "max_position_embeddings", config.get("max_position_embeddings"), False)]
        if arch == "gptneox":
            pairs += [("rope.dimension_count", "rotary_pct (as values)", rotary_dim(config), True),
                      ("use_parallel_residual", "use_parallel_residual", config.get("use_parallel_residual", True), True)]
        return pairs, config
    return [("block_count", "num_hidden_layers", config.get("num_hidden_layers"), True),
            ("embedding_length", "hidden_size", config.get("hidden_size"), True),
            ("feed_forward_length", "intermediate_size", config.get("intermediate_size"), True),
            ("attention.head_count", "num_attention_heads", heads, True),
            ("attention.head_count_kv", "num_key_value_heads", config.get("num_key_value_heads", heads), True),
            ("attention.key_length", "head_dim", head_size(config), True),
            ("rope.freq_base", "rope_theta", config.get("rope_theta", 10000.0), True),
            ("attention.layer_norm_rms_epsilon", "rms_norm_eps", config.get("rms_norm_eps"), True),
            ("context_length", "max_position_embeddings", config.get("max_position_embeddings"), False)], config


def rope_factors(config, width):
    """What llama.cpp writes as rope_freqs.weight: how many times slower each pair of a head turns under
    config.json's rope_scaling, from the engine's own rope_frequencies() (so the check is that the table the page
    makes is the GGUF's)."""
    from llama2_numpy import rope_frequencies

    theta = float(config.get("rope_theta", 10000.0))
    return rope_frequencies(width, theta) / rope_frequencies(width, theta, config.get("rope_scaling"))


def check_tensors(gguf_path, directory, original_vocabulary=False):
    """True when the GGUF holds the original's weights. original_vocabulary (T136 stage 2): the page takes the
    vocabulary from the original, so a difference in the GGUF's is shown and not counted. The summary (worst
    error, mismatches, the rope_freqs difference, the vocabulary differences) is printed last as one JSON line."""
    from llama2_convert import Safetensors, Shards, quantize

    version, metadata, infos, data, base = read_gguf(gguf_path)
    raw_config = json.loads((directory / "config.json").read_text())
    index = directory / "model.safetensors.index.json"
    if not (directory / "model.safetensors").exists() and index.exists():
        # a model split over several files (T136: Qwen2.5 3B and 7B): every shard as one source, as convert_hf.py reads it
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        maps = [np.memmap(directory / name, dtype=np.uint8, mode="r") for name in files]
        hf = Shards([Safetensors(lambda offset, length, data=data: data[offset:offset + length]) for data in maps])
    else:
        hf_data = np.memmap(directory / "model.safetensors", dtype=np.uint8, mode="r")
        hf = Safetensors(lambda offset, length: hf_data[offset:offset + length])
    arch = metadata["general.architecture"]
    print(f"GGUF v{version}, {len(infos)} tensors, architecture {arch}, "
          f"types {sorted({TYPE_NAMES.get(i['type'], i['type']) for i in infos.values()})}")

    print("\n| metadata | GGUF | config.json (normalized) |\n|---|---|---|")
    pairs, config = config_pairs(raw_config, arch)
    mismatched = 0
    for key, name, theirs, counted in pairs:
        ours = metadata.get(f"{arch}.{key}")
        if ours is None and key == "attention.key_length":
            ours = metadata.get(f"{arch}.embedding_length", 0) // metadata.get(f"{arch}.attention.head_count", 1)
        # the GGUF writes floats as float32: 1e-6 of relative difference is its rounding
        same = ours is not None and theirs is not None and math.isclose(float(ours), float(theirs), rel_tol=1e-6)
        mismatched += counted and not same
        note = "" if same else " **differs**" if counted else " (differs, not counted)"
        print(f"| {arch}.{key} | {ours} | {name} = {theirs}{note} |")
    rows = infos["token_embd.weight"]["shape"][0] if "token_embd.weight" in infos else None
    # a GPT-2 always shares its classifier: llama.cpp writes a copy of the embedding as output.weight, which is held
    # to the original's embedding below (T136's third stage)
    tied = "output.weight" not in infos or arch == "gpt2"
    for what, ours, theirs in (("embedding rows", rows, config.get("vocab_size")),
                               ("classifier shared with the embedding", tied, bool(config.get("tie_word_embeddings", False)))):
        mismatched += ours != theirs
        print(f"| {what} | {ours} | {theirs}{'' if ours == theirs else ' **differs**'} |")

    vocab_diffs = {}
    originals = original_vocabularies(directory)
    for name, pieces in originals.items():
        vocab_diffs[name] = compare_vocabulary(metadata, pieces, name, config.get("vocab_size"))
        if original_vocabulary:
            print("(not counted: the page takes the vocabulary from the original)")
        else:
            mismatched += vocab_diffs[name] > 0

    heads, kv_heads = config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"])
    print(f"\n| tensor | type | shape | relative error | nearest reference (line {TIGHT}) | order | int8 equal to quantize() "
          f"| rows past {ROW_LINE} (worst) |\n|---|---|---|---:|---:|---|---:|---|")
    worst, orders, rope_difference, bad_rows, row_detail, rounded_rows = 0.0, set(), None, {}, {}, {}
    near_worst, past_tight, rounded_detail = 0.0, {}, {}
    for name, info in infos.items():
        if name == "rope_freqs.weight":
            values, _ = tensor(info, data, base)
            factors = rope_factors(config, 2 * len(values))
            rope_difference = relative(values.astype(np.float64), factors) if values.shape == factors.shape else math.inf
            counted = rope_difference > 1e-5
            mismatched += counted
            print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | {rope_difference:.2e} against "
                  f"rope_frequencies() of rope_scaling = {config.get('rope_scaling')}{' **differs**' if counted else ''} | | | | |")
            continue
        target = hugging_face_name(name, arch)
        if target is not None and target not in hf and f"transformer.{target}" in hf:
            target = f"transformer.{target}"  # a GPT-2 of the other spelling (rinna's)
        if target is None or target not in hf:
            print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | | | no {target} in safetensors | | |")
            mismatched += 1
            continue
        shape = tuple(hf.shape(target))
        by_row = name in BY_ROW
        rowwise = None
        # GPT-2's Conv1D matrices go whole, to be read turned (a square one has the same shape either way)
        conv1d = arch == "gpt2" and name.startswith("blk.") and name.endswith(".weight") and len(shape) == 2
        if math.prod(shape) > BLOCK and len(shape) > 1 and tuple(info["shape"]) == shape and not conv1d:
            # a large matrix (Qwen's embedding is 545 MB as float32) is compared a block of rows at a time: two
            # whole copies side by side are what took this machine down
            # (q and k of an 8B are 16.8M values: before T136's stage 2 they were compared here as Hugging Face holds
            # them, and a turned GGUF read as 1.4 off)
            head_rows = 0
            if name.endswith(("attn_q.weight", "attn_k.weight")):
                head_rows = shape[0] // (heads if "attn_q" in name else kv_heads)
            split_heads = heads if arch == "gptneox" and name.endswith("attn_qkv.weight") else 0
            error, near, equal, rowwise, order = large(info, data, base, hf, target, shape, quantize, by_row, head_rows,
                                                       split_heads)
            if order:
                orders.add(order)
        else:
            values, raw = tensor(info, data, base)
            original = hf.rows(target, 0, shape[0]).reshape(shape).astype(np.float32)
            order = ""
            if name.endswith(("attn_q.weight", "attn_k.weight", "attn_q.bias", "attn_k.bias")):
                n = heads if "attn_q" in name else kv_heads
                as_is, turn = relative(values, original), relative(values, turned(original, n))
                order = "turned (llama2.c order)" if turn < as_is else "as Hugging Face"
                orders.add(order)
                if turn < as_is:
                    original = turned(original, n)
            if arch == "gptneox" and name.endswith(("attn_qkv.weight", "attn_qkv.bias")):
                # T136's third stage: llama.cpp splits query_key_value into all of q, k, then v
                as_is, moved = relative(values, original), relative(values, split(original, heads))
                order = "split (q, k, v)" if moved < as_is else "as Hugging Face"
                orders.add(order)
                if moved < as_is:
                    original = split(original, heads)
            if conv1d:
                # and GPT-2's Conv1D matrices, (in, out), it stores as every other model's, (out, in)
                transposed = np.ascontiguousarray(original.T)
                as_is = relative(values, original) if values.shape == original.shape else math.inf
                turn = relative(values, transposed) if values.shape == transposed.shape else math.inf
                order = "transposed (out, in)" if turn < as_is else "as Hugging Face (in, out)"
                orders.add(order)
                if turn < as_is:
                    original = transposed
            error = relative(values, original) if values.shape == original.shape else float("nan")
            near = nearest(squares(values, original, raw is not None)) if values.shape == original.shape else float("nan")
            equal = ""
            if raw is not None and values.shape == original.shape:
                ours, _ = quantize(original.reshape(-1, original.shape[-1]))
                equal = f"{(ours.reshape(-1) == raw[0].reshape(-1)).mean() * 100:.2f}%"
            if by_row and values.shape == original.shape:
                rowwise = row_check(row_parts(values, original, raw is not None))
        worst = max(worst, error) if not math.isnan(error) else math.inf
        near_worst = max(near_worst, near) if not math.isnan(near) else math.inf
        near_note = f"{near:.2e}"
        if not near <= TIGHT:
            past_tight[name] = near
            mismatched += 1
            near_note += " **past**"
        rows_note = ""
        if rowwise is not None:
            bad_rows[name] = rowwise[0]
            row_detail[name] = rowwise[3]
            rounded_rows[name] = rowwise[4]
            rounded_detail[name] = rowwise[5]
            mismatched += rowwise[0]
            rows_note = (f"{rowwise[0]} ({rowwise[1]:.4f} at row {rowwise[2]}){' **differs**' if rowwise[0] else ''}"
                         + (f"; {rowwise[4]} more only as Q8_0's float16 scale rounds them" if rowwise[4] else ""))
        print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | {error:.5f} | {near_note} | {order} | {equal} | {rows_note} |")
    if any(rounded_detail.values()):
        # T145: what passes only against a Q8_0 of the original, to be read: an unused piece of a small norm, or not.
        # The pieces are the original's (what the page reads, T136's second stage), or else the GGUF's
        pieces = next(iter(originals.values()), None) or metadata.get("tokenizer.ggml.tokens", [])
        print("\n| tensor | row (id) | piece | error against the original | against the nearest Q8_0 | norm / median |"
              "\n|---|---:|---|---:|---:|---:|")
        for name, rows in rounded_detail.items():
            for detail in rows:
                detail.insert(1, pieces[detail[0]] if detail[0] < len(pieces) else None)  # [row, piece, ...]
                row, piece, against, near, norm = detail
                print(f"| {name} | {row} | {json.dumps(piece, ensure_ascii=False)} | {against:.4f} | {near:.4f} | {norm} |")
            if rounded_rows[name] > len(rows):
                print(f"| {name} | and {rounded_rows[name] - len(rows)} more | | | | |")
    ok = mismatched == 0 and worst < 0.02
    print(f"\nworst relative error {worst:.5f}; against the nearest reference {near_worst:.2e} ({len(past_tight)} "
          f"past {TIGHT}); q and k are stored {' and '.join(sorted(orders)) or '(none)'}; {mismatched} mismatches")
    print(json.dumps({"worst": worst, "nearest": near_worst, "past_tight": past_tight, "mismatches": mismatched,
                      "rows": bad_rows, "bad_rows": row_detail, "rounded_rows": rounded_rows,
                      "rounded_detail": rounded_detail, "orders": sorted(orders), "rope_freqs_diff": rope_difference,
                      "vocab_diffs": vocab_diffs, "ok": ok}, ensure_ascii=False))
    return ok


def check_logits(a, b, text_file, count):
    from llama2_numpy import Llama

    def load(out):
        return Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), Path(f"{out}.tokenizer.bin").read_bytes(),
                     kernels=None, **json.loads(Path(f"{out}.json").read_text()))

    # One model at a time: without the kernels an int8 model is widened to float32 (SmolLM2 135M: 540 MB each),
    # and two of them side by side took this 6.6 GB machine down twice. The logits of the first are kept
    # (count x vocab_size float32, 59 MB for SmolLM2) and the model is let go before the second is loaded.
    import gc

    def logits_of(out, tokens=None):
        llama = load(out)
        encoded = llama.tokenizer.encode(Path(text_file).read_text())[:count]
        if tokens is not None:
            assert encoded == tokens[1:], "the two tokenize differently"
        tokens = [llama.bos] + encoded
        rows = np.stack([np.asarray(llama.forward(tokens[pos], pos), dtype=np.float32) for pos in range(len(tokens) - 1)])
        del llama
        gc.collect()
        return tokens, rows

    tokens, first = logits_of(a)
    _, second = logits_of(b, tokens)
    largest, agree, nll = 0.0, 0, [0.0, 0.0]
    for pos in range(len(tokens) - 1):
        one, two = first[pos].astype(np.float64), second[pos].astype(np.float64)
        largest = max(largest, float(np.abs(one - two).max()))
        agree += int(one.argmax() == two.argmax())
        for i, logits in enumerate((one, two)):
            shifted = logits - logits.max()
            nll[i] -= shifted[tokens[pos + 1]] - math.log(np.exp(shifted).sum())
    n = len(tokens) - 1
    first_ppl, second_ppl = math.exp(nll[0] / n), math.exp(nll[1] / n)
    # The line T74 is held to (Fable, 2026-09-24): the two int8 differ only by the rounding of the scales
    # (float16 in Q8_0, float32 here), so they must agree on the most likely token 99 times in 100 and be within
    # 0.2% of perplexity. The float32 against int8 of the same model measures 96.3% and 0.6%: a mistake in the
    # reader (a wrong order, a wrong scale) lands far outside this. Only for a model as little bothered by int8: on
    # Pythia 70M and 160M and GPT-2 (int8 costs 1 to 3% there) two int8 of the same quality agree 94.5 to 98.6% of
    # the time and differ by up to 0.2% (T136's third stage, the review): read those against such a rounding again.
    ok = agree / n >= 0.99 and abs(second_ppl / first_ppl - 1) <= 0.002
    print(json.dumps({"a": Path(a).name, "b": Path(b).name, "tokens": n, "largest logit difference": largest,
                      "top-1 agreement": agree / n, "perplexity a": first_ppl, "perplexity b": second_ppl,
                      "within the line": ok}))
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "tensors":
        sys.exit(0 if check_tensors(sys.argv[2], Path(sys.argv[3]), "--original-vocabulary" in sys.argv[4:]) else 1)
    elif sys.argv[1] == "logits":
        sys.exit(0 if check_logits(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300) else 1)
