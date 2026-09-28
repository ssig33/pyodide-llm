# Shared helpers for the engine tests: they run on native Python + NumPy, no Pyodide and no torch.
# Synthetic checkpoints and tokenizers are built here, so that no binary has to live in the repository.
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "public"))
sys.path.insert(0, str(ROOT))  # quantize.py

# tmp_path lives under the system's temporary directory, which on the development machine is a tmpfs, that is memory
# (AGENTS.md), and was written there without a word (the owner, 2026-09-26): here it is the repository's .tmp, which
# .gitignore has. pytest reads this where it first makes a tmp_path, after the conftests; a caller may say otherwise.
(ROOT / ".tmp").mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(ROOT / ".tmp"))

import llama2_numpy  # noqa: E402


def model_file(name):
    """A file that `make models` produces, or a skip when it is not there."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is missing: run `make models` first")
    return path


def checkpoint_vocab_size(name):
    """The vocabulary size in a checkpoint header, read without loading the weights."""
    with open(model_file(name), "rb") as f:
        return abs(struct.unpack("<7i", f.read(28))[5])


# ------------------------------------------------------------------------------------------- texts to split
# The texts the pre-tokenizers are checked on (test_bytebpe.py against the real tokenizers, test_llama3.py against
# the patterns). Here and not in test_bytebpe.py, which skips as a whole without tokenizers and took test_llama3.py's
# own tests along when it imported them from there (T144).
CORPUS = (
    "The quick brown fox jumps over the lazy dog. 日本語の文章も混ぜる。"
    "Don't stop; it's theirs, they'll go. I'D LIKE 'IT'.\n"
    "Pyodide は WebAssembly 版の Python で、ブラウザの中で NumPy が動く。"
    "def forward(x, w):\n\treturn w @ x  # matmul\n\n\n"
    "価格は1,234,567円（税込）です。2026-09-21T00:00:00Z\r\n"
    "絵文字 \U0001f600\U0001f389 と外字 \U00029E3D、全角ＡＢＣ１２３、半角ｶﾅ。"
    "https://example.com/a/b?c=1&d=2#frag  'single' \"double\" `tick`\n"
    "   spaces\tand\ttabs\n\n\nnewlines   \nTHE END. the end. The End?!  "
)
# every kind of boundary the patterns care about
TEXTS = [CORPUS, " ", "  ", "\n", "\r\n", " \n ", "0123", " 42 ", "a", " a", "  a", "\ta", "(abc", "、あ",
         "a 1b", " 1,234", "1a2", "v1.2.3", "第1章 2節", "it's a dog's life", "IT'S", "end.  ", "x\n\n\ny",
         # a line of spaces between line breaks, as pasted code has (the review of T106: Qwen's \s*[\r\n]+ takes the
         # whole run up to its last line break, this took it up to the first)
         "a\n  \nb", "\n \n \n", "def f(x):\n    a = 1\n    \n    return a\n", " \t\n \r\n x"]


# ------------------------------------------------------------------------------------------- sentencepiece
def charsmap(mapping):
    """A sentencepiece precompiled_charsmap for {text: its normalized text} (T216): the length of a Darts-clone double
    array, the array, and the normalized texts, each ended by a NUL. Built here the simple way (each node takes the
    first base no other node has and whose children's places are free; a unit: bit 31 a value, bits 0-7 its label,
    bit 8 a leaf below, bits 10 up the offset to its base), which llama2_numpy.Charsmap walks as Darts-clone does."""
    trie, texts = {}, bytearray()
    for key, normal in mapping.items():
        node = trie
        for byte in key.encode("utf-8"):
            node = node.setdefault(byte, {})
        node[None] = len(texts)
        texts += normal.encode("utf-8") + b"\0"
    units, used, bases = {}, {0}, set()

    def place(node, at):
        labels = sorted(label for label in node if label is not None)
        places = labels + ([0] if None in node else [])
        base = 1
        while base in bases or any(base ^ label in used for label in places):
            base += 1
        bases.add(base)
        used.update(base ^ label for label in places)
        units[at] = units.get(at, 0) | (at ^ base) << 10 | (0x100 if None in node else 0)
        if None in node:
            units[base] = node[None] | 1 << 31
        for label in labels:
            units[base ^ label] = label
            place(node[label], base ^ label)

    place(trie, 0)
    array = struct.pack(f"<{max(units) + 1}I", *(units.get(i, 0) for i in range(max(units) + 1)))
    return struct.pack("<I", len(array)) + array + bytes(texts)


# ------------------------------------------------------------------------------------- synthetic checkpoints

class NoWeights:
    """A converter's sink that keeps nothing: for the options and the tokenizer.bin, which it has before the weights."""

    def open(self, *args):
        pass

    def write(self, *args):
        pass


def vocabulary_conversion(tokenizer, name, vocab_size, tokenizer_config=None, **config):
    """llama2_convert.Conversion of a small model with this tokenizer (bytes, as the file name says) and vocabulary,
    and these keys of config.json besides (bos_token_id ...): its options and tokenizer, with no weights fed (T143)."""
    import json
    import llama2_convert
    from test_convert import hugging_face, safetensors_file
    settings, weights = synthetic_weights(vocab_size=vocab_size)
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    return llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps({**published, **config}), tokenizer,
                                     name, dtype="float32", max_seq_len=settings["seq_len"], start=8 + size,
                                     tokenizer_config=tokenizer_config, sink=NoWeights())


def rope_tables(seq_len, head_size, rope_theta=10000.0):
    angles = np.arange(seq_len)[:, None] / rope_theta ** (np.arange(0, head_size, 2) / head_size)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def synthetic_weights(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=4,
                      vocab_size=320, seq_len=24, shared=True, seed=0, head_size=0):
    """Random weights for a tiny model, plus its configuration. Small enough for a naive reference.
    head_size: where a head is not dim / n_heads (T124), q and the attention's output are n_heads * head_size wide."""
    rng = np.random.default_rng(seed)
    head_size = head_size or dim // n_heads
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    cos, sin = rope_tables(seq_len, head_size)
    weights = {
        "token_embedding_table": normal(vocab_size, dim),
        "rms_att_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "wq": normal(n_layers, q_dim, dim), "wk": normal(n_layers, kv_dim, dim),
        "wv": normal(n_layers, kv_dim, dim), "wo": normal(n_layers, dim, q_dim),
        "rms_ffn_weight": (1.0 + normal(n_layers, dim) * 0.1).astype(np.float32),
        "w1": normal(n_layers, hidden_dim, dim), "w2": normal(n_layers, dim, hidden_dim),
        "w3": normal(n_layers, hidden_dim, dim),
        "rms_final_weight": (1.0 + normal(dim) * 0.1).astype(np.float32),
        "freq_cis_real": cos, "freq_cis_imag": sin,
    }
    weights["wcls"] = weights["token_embedding_table"] if shared else normal(vocab_size, dim)
    config = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                  n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len,
                  head_size=head_size, q_dim=q_dim, kv_dim=kv_dim, shared=shared)
    return config, weights


# the tensor order of the llama2.c "legacy" format, as quantize.py and llama2_numpy.py read it
TENSOR_ORDER = ["token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo",
                "rms_ffn_weight", "w1", "w2", "w3", "rms_final_weight", "freq_cis_real", "freq_cis_imag"]


def pack_checkpoint(config, weights):
    """The float32 checkpoint: a 7 int header (negative vocab size means an unshared classifier) then tensors."""
    vocab_size = config["vocab_size"] if config["shared"] else -config["vocab_size"]
    out = [struct.pack("<7i", config["dim"], config["hidden_dim"], config["n_layers"], config["n_heads"],
                       config["n_kv_heads"], vocab_size, config["seq_len"])]
    names = TENSOR_ORDER + ([] if config["shared"] else ["wcls"])
    out += [np.ascontiguousarray(weights[name], dtype=np.float32).tobytes() for name in names]
    return b"".join(out)


# ------------------------------------------------------------------------------------- synthetic tokenizer

def pack_tokenizer(pieces):
    """llama2.c's tokenizer.bin: max piece length, then (score, length, bytes) per piece."""
    out = [struct.pack("<i", max(len(text) for _, text in pieces))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in pieces]
    return b"".join(out)


def tiny_vocab(vocab_size=320):
    """A vocabulary of that many pieces: <unk>, byte fallbacks, then words and characters.

    Byte and control pieces get the unmatchable score convert_hf.py writes, so they never match text directly.
    """
    pieces = [(-1e9, b"<unk>"), (-1e9, b"<s>"), (-1e9, b"</s>")]
    pieces += [(-1e9, b"<0x%02X>" % byte) for byte in range(256)]
    words = [" ", " the", " cat", " s", "a", "t", "o", "n", "e", "h", "c", " a", " o",
             "流", "行", "、", " 流行り", "Ａ", "!", "?", " \n", "\n", " hello", " world",
             " he", "l", "w", "r", "d", " t", "i", "g", "猫", " 猫", "り"]
    for i, word in enumerate(words):
        pieces.append((-float(i) * 0.5, word.encode("utf-8")))
    assert len(pieces) <= vocab_size, "the vocabulary does not fit"
    while len(pieces) < vocab_size:  # padding rows, never matchable
        pieces.append((-1e9, b"<pad%d>" % len(pieces)))
    return pieces[:vocab_size]


def tiny_tokenizer(kind="bpe", nfkc=False):
    pieces = tiny_vocab(320)
    return llama2_numpy.Tokenizer(pack_tokenizer(pieces), len(pieces), kind=kind, nfkc=nfkc)


def detokenize(tokenizer, tokens, bos=llama2_numpy.BOS):
    """Join the pieces the way generate() does: the first one loses the dummy prefix."""
    out, previous = [], bos
    for token in tokens:
        out.append(tokenizer.decode(previous, token, bos))
        previous = token
    return b"".join(out).decode("utf-8", "replace")


# ------------------------------------------------------------------------------------- naive reference

def naive_logits(config, weights, tokens):
    """llama2.c written out with Python loops: the reference forward() is compared against."""
    dim, n_layers = config["dim"], config["n_layers"]
    n_heads, n_kv_heads, head_size = config["n_heads"], config["n_kv_heads"], config["head_size"]
    kv_mul = n_heads // n_kv_heads
    cos, sin = weights["freq_cis_real"], weights["freq_cis_imag"]

    eps = config.get("eps", 1e-5)  # config.json's rms_norm_eps (T124)

    def rmsnorm(vector, weight):
        return weight * vector / math.sqrt(sum(float(v) * float(v) for v in vector) / len(vector) + eps)

    def head_norm(vector, name, l):
        # Qwen3 (T124): every head of q and k normalized on its own, with one weight of a head's size
        if name not in weights:
            return vector
        return np.concatenate([rmsnorm(vector[h:h + head_size], weights[name][l]) for h in range(0, len(vector), head_size)])

    def rope(vector, pos, heads):
        out = vector.copy()
        for h in range(heads):
            for i in range(head_size // 2):
                a, b = h * head_size + 2 * i, h * head_size + 2 * i + 1
                out[a] = vector[a] * cos[pos, i] - vector[b] * sin[pos, i]
                out[b] = vector[a] * sin[pos, i] + vector[b] * cos[pos, i]
        return out

    x = [weights["token_embedding_table"][token].astype(np.float64) for token in tokens]
    for l in range(n_layers):
        queries, keys, values = [], [], []
        for pos in range(len(tokens)):
            xb = rmsnorm(x[pos], weights["rms_att_weight"][l])
            # Qwen2 adds a bias to q, k and v before the rotation
            bias = lambda name: weights[name][l] if name in weights else 0.0
            queries.append(rope(head_norm(weights["wq"][l] @ xb + bias("bq"), "q_norm", l), pos, n_heads))
            keys.append(rope(head_norm(weights["wk"][l] @ xb + bias("bk"), "k_norm", l), pos, n_kv_heads))
            values.append(weights["wv"][l] @ xb + bias("bv"))
        for pos in range(len(tokens)):
            attended = np.zeros(n_heads * head_size, dtype=np.float64)
            for h in range(n_heads):
                kv = h // kv_mul
                q = queries[pos][h * head_size:(h + 1) * head_size]
                scores = np.array([float(q @ keys[t][kv * head_size:(kv + 1) * head_size]) / math.sqrt(head_size)
                                   for t in range(pos + 1)])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                for t in range(pos + 1):
                    attended[h * head_size:(h + 1) * head_size] += \
                        scores[t] * values[t][kv * head_size:(kv + 1) * head_size]
            x[pos] = x[pos] + weights["wo"][l] @ attended
            xb = rmsnorm(x[pos], weights["rms_ffn_weight"][l])
            h1 = weights["w1"][l] @ xb
            h1 = h1 / (1.0 + np.exp(-h1)) * (weights["w3"][l] @ xb)
            x[pos] = x[pos] + weights["w2"][l] @ h1
    return [weights["wcls"] @ rmsnorm(vector, weights["rms_final_weight"]) for vector in x]
