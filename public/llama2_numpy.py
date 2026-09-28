# llama2_numpy.py
# Llama 2 inference with NumPy. A port of tairov/llama2.py (itself a port of karpathy/llama2.c) in which
# every loop over vector elements became a NumPy call, so the interpreter only sequences the layers.
#
# This file is under the Mozilla Public License 2.0 (the LICENSE file at the top of the repository), and it is
# derived from two works under the MIT License, whose notice follows: tairov/llama2.py
# (https://github.com/tairov/llama2.py; its LICENSE names no copyright holder) and karpathy/llama2.c
# (https://github.com/karpathy/llama2.c), Copyright (c) 2023 Andrej.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import codecs
import heapq
import math
import re
import struct
import time
import unicodedata

import numpy as np

BOS = 1  # beginning-of-sequence token, also what the model emits when a story is over


def byte_chars():
    """GPT-2's byte <-> character table: every byte becomes one printable character, so that a byte-level BPE
    vocabulary is plain text. 0x20 is "\u0120", 0x0A is "\u010a"."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(printable, chars)}


BYTE_CHARS = byte_chars()
CHAR_BYTES = {char: byte for byte, char in BYTE_CHARS.items()}


def letter(char):
    return unicodedata.category(char)[0] == "L"


def number(char):
    return unicodedata.category(char)[0] == "N"


class CharClasses(dict):
    r"""What the pre-tokenizer's patterns tell apart, one character for each character (T200): the ASCII letters,
    the apostrophe, the space, \r and \n as they are (the contractions and the line breaks name them) and ſ (U+017F,
    which (?i:'s) takes for s), any other whitespace "\t", any other letter (\p{L}) "a", a number (\p{N}) "0",
    anything else "!". A text goes through str.translate() with it, and the standard re module runs the patterns on
    what comes out: \p{L} and \p{N}, which re has not, become [A-Za-zſ] and 0. A character is classed the first time
    it is seen.

    Whitespace is what the real tokenizers' regex (Oniguruma) calls \s (T206): str.isspace less \x1c to \x1f, which
    Oniguruma takes for neither whitespace nor a letter nor a number."""

    def __missing__(self, code):
        char = chr(code)
        if char.isascii() and char.isalpha() or char in "' \r\nſ":
            kind = char
        elif char.isspace() and not "\x1c" <= char <= "\x1f":
            kind = "\t"
        elif letter(char):
            kind = "a"
        elif number(char):
            kind = "0"
        else:
            kind = "!"
        self[code] = kind
        return kind


CHAR_CLASSES = CharClasses()
# pretokenize()'s patterns on the classes: \p{L} is [A-Za-zſ], \p{N} is 0, \s is [ \t\r\n], anything else ['!].
# Under (?i) re takes ſ for s, as Oniguruma does; GPT-2's case-sensitive 's does not.
CONTRACTED = "'s|'t|'re|'ve|'m|'ll|'d"
SPACES = r"[ \t\r\n]+(?![^ \t\r\n])|[ \t\r\n]+"
PATTERNS = {
    "gpt2": re.compile(CONTRACTED + r"| ?[A-Za-zſ]+| ?0+| ?['!]+|" + SPACES),
    "qwen": re.compile(f"(?i:{CONTRACTED})" + r"|[^\r\nA-Za-zſ0]?[A-Za-zſ]+|0| ?['!]+[\r\n]*|[ \t\r\n]*[\r\n]+|" + SPACES),
    "llama3": re.compile(f"(?i:{CONTRACTED})" + r"|[^\r\nA-Za-zſ0]?[A-Za-zſ]+|0{1,3}| ?['!]+[\r\n]*|[ \t\r\n]*[\r\n]+|" + SPACES),
}
DIGITS = re.compile("0|[^0]+")  # Digits(individual_digits) on the classes: every number a piece of its own


def pretokenize(text, pattern):
    r"""Split text the way the tokenizer.json's pre_tokenizer does. Those patterns need \p{L} and \p{N}, which the
    standard re module has not, so they run on the text's character classes (CharClasses) and the pieces are cut
    from the text at the same places (T200: a loop of Python over the characters did it before, a tenth as fast).

    "gpt2" is what ByteLevel(use_regex) applies:
        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    "gpt2-digits" is the same after Digits(individual_digits), which SmolLM2 puts in front of it: every number is a
    piece of its own, \p{N} as Rust's char::is_numeric says (², Ⅱ and ① too, not only \d: T206), and GPT-2's
    pattern runs on each piece as if the piece were the whole text.
    "qwen" is the pattern Qwen2 spells out (the contractions match whatever the case, digits come one by one,
    and a piece of anything-but-a-line-break may lead a word):
        (?i:'s|…)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    "llama3" is Llama 3's, which is Qwen's with the digits taken up to three at a time (\p{N}{1,3}).
    All four are checked against the real patterns in tests/test_bytebpe.py and tests/test_llama3.py.
    """
    classes = text.translate(CHAR_CLASSES)
    if pattern == "gpt2-digits":
        gpt2 = PATTERNS["gpt2"]
        return [text[match.start():match.end()] for piece in DIGITS.finditer(classes)
                for match in gpt2.finditer(classes, piece.start(), piece.end())]
    return [text[match.start():match.end()] for match in PATTERNS.get(pattern, PATTERNS["gpt2"]).finditer(classes)]


# T216: what may follow the pieces of a tokenizer.bin: this, the length of a sentencepiece model's precompiled_charsmap
# (uint32) and the charsmap itself
CHARSMAP = b"charsmap"


class Charsmap:
    """sentencepiece's normalizer, from the model's own precompiled_charsmap (T216): the length of a Darts-clone double
    array (uint32), the array, and the normalized texts, each ended by a NUL. The array's keys are UTF-8 texts and its
    values where their normalized text begins. At each place the longest key that begins there is replaced, and where
    none does, one character is kept: sentencepiece's Normalizer::NormalizePrefix, which walks the array as Darts-clone's
    commonPrefixSearch does (a unit: bit 31 a value, bits 0-7 its label, bit 8 a leaf below, and its offset from bit 10,
    shifted by 8 more where bit 9 says). The model's own map: not Python's NFKC, whose Unicode is another version (the
    four letters T126 found in tiny-lm), and not a table of the nmt_ kinds, which differ between models (rinna's)."""

    def __init__(self, blob):
        (size,) = struct.unpack_from("<I", blob)
        self.units = struct.unpack_from(f"<{size // 4}I", blob, 4)
        self.texts = bytes(blob[4 + size:])
        self.found = {}  # the normalized text at each value, read once
        # T222: the bytes a key may begin with, the root's children (the first step of the walk below). The text
        # between two of them is kept as it is, without a walk at each byte. No continuation byte of UTF-8 begins a
        # character, so the text is cut between characters, as the walk cuts it
        units, root = self.units, self.offset(self.units[0])
        begins = bytes(byte for byte in [*range(0x80), *range(0xC0, 0x100)]
                       if root ^ byte < len(units) and units[root ^ byte] & 0x800000FF == byte)
        self.starts = frozenset(begins)  # a walk begins at once where one is, without a search
        self.begins = re.compile(b"[" + re.escape(begins) + b"]") if begins else None

    @staticmethod
    def offset(unit):
        return (unit >> 10) << ((unit & 0x200) >> 6)

    def replaced(self, text):
        data, units, out, i = text.encode("utf-8"), self.units, [], 0
        offset, begins, starts = self.offset, self.begins, self.starts
        root, size = offset(units[0]), len(units)
        while i < len(data):
            if data[i] not in starts:
                found = begins.search(data, i) if begins else None
                if not found:
                    out.append(data[i:])
                    break
                out.append(data[i:found.start()])
                i = found.start()
            node, longest, value = root, 0, 0
            for j in range(i, len(data)):
                node ^= data[j]
                if node >= size or units[node] & 0x800000FF != data[j]:
                    break
                unit = units[node]
                node ^= offset(unit)
                if unit & 0x100:
                    longest, value = j + 1 - i, units[node] & 0x7FFFFFFF
            if longest:
                if value not in self.found:
                    self.found[value] = self.texts[value:self.texts.index(b"\0", value)]
                out.append(self.found[value])
                i += longest
            else:  # one character as it is (Python's texts are always well-formed UTF-8)
                length = 1 if data[i] < 0xC0 else 2 if data[i] < 0xE0 else 3 if data[i] < 0xF0 else 4
                out.append(data[i:i + length])
                i += length
        return b"".join(out).decode("utf-8")


class Tokenizer:
    """llama2.c's tokenizer.bin: sentencepiece pieces with their scores.

    kind="bpe" merges the best-scoring adjacent pair, like llama2.c (Llama 2 vocabulary);
    kind="unigram" picks the segmentation with the best total score (sentencepiece unigram models);
    kind="bytebpe" is Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). It merges by the rank of the merge
    that makes the piece, which is the same loop as "bpe" because the converter writes minus the rank as the
    score. Its vocabulary is written in the byte <-> character table above, so the pieces are plain text.
    """

    UNMATCHABLE = -1e8  # convert_hf.py gives control and byte pieces a score below this

    def __init__(self, data, vocab_size, kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", ignore_merges=False,
                 collapse=False, unknown=None):
        self.kind, self.nfkc, self.nfc, self.pretokenizer = kind, nfkc, nfc, pretokenizer
        self.collapse = collapse  # a sentencepiece model's remove_extra_whitespaces: see normalized()
        # a sentencepiece model without byte pieces (rinna's) writes a character it lacks as its unknown piece, a run
        # of them as one; the others spell it in bytes
        self.unknown = unknown
        self.ignore_merges = ignore_merges
        self.vocab, self.scores = [], []
        offset = 4  # skip max_token_length
        for _ in range(vocab_size):
            score, length = struct.unpack_from("<fi", data, offset)
            offset += 8
            self.vocab.append(bytes(data[offset:offset + length]))
            self.scores.append(score)
            offset += length
        # T216: the sentencepiece model's own normalizer, where the converter put it after the pieces
        rest = bytes(data[offset:offset + len(CHARSMAP) + 4])
        self.charsmap = None
        if rest[:len(CHARSMAP)] == CHARSMAP:
            (size,) = struct.unpack_from("<I", rest, len(CHARSMAP))
            start = offset + len(CHARSMAP) + 4
            self.charsmap = Charsmap(bytes(data[start:start + size]))
        # first occurrence wins, like list.index() in llama2.py
        self.index = {}
        for i, piece in enumerate(self.vocab):
            self.index.setdefault(piece, i)
        # raw byte tokens look like b"<0x0A>"; they spell out whatever the vocabulary lacks
        self.byte_tokens = [self.index.get(b"<0x%02X>" % byte, byte + 3) for byte in range(256)]
        # byte-level pieces are text, and the merging works on that text rather than on bytes
        self.text_index = {}
        if self.kind == "bytebpe":
            for i, piece in enumerate(self.vocab):
                self.text_index.setdefault(piece.decode("utf-8", "replace"), i)
        # the Viterbi search's pieces as text, each the token index finds for it (the first of equal pieces), those
        # it can match only, and how long such a piece that begins with two given characters can be (T200)
        self.pieces, self.reach = {}, {}
        if self.kind == "unigram":
            for piece, i in self.index.items():
                try:
                    text = piece.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # no text encodes to it
                if self.scores[i] > self.UNMATCHABLE:
                    self.pieces[text] = i
                    if len(text) > 1 and self.reach.get(text[:2], 0) < len(text):
                        self.reach[text[:2]] = len(text)
        self.unknown_score = min(score for score in self.scores if score > self.UNMATCHABLE) - 10.0

    def encode(self, text, specials=()):
        """specials: pieces such as "</s>" that stand for their token wherever they are written (a chat template
        puts them between the turns). As plain text they would be spelled out letter by letter."""
        tokens, first = [], True
        for part in re.split("(" + "|".join(re.escape(special) for special in specials) + ")", text) if specials else [text]:
            if part in specials:
                tokens.append(self.index[part.encode("utf-8")])
            elif part and (part := self.normalized(part)):
                if self.kind == "bytebpe":
                    # no dummy prefix: a byte-level vocabulary spells the space out as a character of its own
                    tokens += self.encode_bytebpe(part)
                else:
                    # sentencepiece's dummy prefix: the model saw every text start with a space (but not the
                    # text after a special token)
                    part = " " + part if first else part
                    tokens += self.encode_unigram(part) if self.kind == "unigram" else self.encode_bpe(part)
            first = False
        return tokens

    def normalized(self, text):
        """The text as the model's normalizer makes it: a sentencepiece model's own map (charsmap, T216: nmt_nfkc makes
        tabs, newlines and a few more characters a space, and drops or keeps the other control characters, as each
        model's map says), NFKC or NFC (a tokenizer.json's), and collapse: sentencepiece's remove_extra_whitespaces,
        runs of spaces one and none at either end. Without them a newline was spelled with byte + 3 in a vocabulary
        with no byte pieces: rinna's った (the review of T126)."""
        if self.charsmap:
            text = self.charsmap.replaced(text)
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.collapse:
            text = re.sub(" {2,}", " ", text).strip(" ")
        if self.kind != "bytebpe":
            # sentencepiece writes a space as U+2581, so one written in the text is a space too (after the collapse,
            # which sees spaces only; an nmt_ map made it one before). The pieces here spell it " " (T216)
            text = text.replace("\u2581", " ")
        return text

    def encode_bpe(self, text):
        # First encode every individual character; a character the vocabulary lacks becomes its UTF-8 bytes
        tokens = []
        for char in text:
            piece = char.encode("utf-8")
            if piece in self.index:
                tokens.append(self.index[piece])
            elif self.unknown is not None:
                tokens += [] if tokens[-1:] == [self.unknown] else [self.unknown]  # a run of them is one
            else:
                tokens.extend(self.byte_tokens[byte] for byte in piece)

        # Merge the best consecutive pair each iteration, according to the scores in vocab_scores (the first of equal
        # scores). The pairs wait in a heap by (minus the score, where the pair's left piece began), so the best and
        # the first of equal ones comes out first; a merge pushes the two pairs next to it, and the pairs it ended
        # are left in the heap and passed over when they come out (stamp: a piece's stamp changes when it merges).
        # T200 had the scores in a list and took the best with max() and index(): 0.17 s for 1000 tokens (T207).
        # The pieces are a list linked by after and before, the first one never goes (a merge keeps the left piece).
        pieces, count = [self.vocab[token] for token in tokens], len(tokens)
        after, before, stamp, waiting = list(range(1, count + 1)), list(range(-1, count - 1)), [0] * count, []

        def pair(k, m):
            """the pieces k and m joined, if they merge: (minus the score, k, m, their stamps, the token)"""
            id = self.index.get(pieces[k] + pieces[m])
            if id is not None and self.scores[id] > -1e10:
                return -self.scores[id], k, m, stamp[k], stamp[m], id

        for k in range(count - 1):
            if joined := pair(k, k + 1):
                waiting.append(joined)
        heapq.heapify(waiting)
        while waiting:
            _, k, m, stamp_k, stamp_m, id = heapq.heappop(waiting)
            if stamp[k] != stamp_k or stamp[m] != stamp_m:
                continue  # a merge has changed one of the two since
            tokens[k], pieces[k] = id, pieces[k] + pieces[m]
            stamp[k] += 1
            stamp[m] += 1
            after[k] = after[m]
            if after[k] < count:
                before[after[k]] = k
            for left, right in ((before[k], k), (k, after[k])):  # the pairs with the merged piece
                if left >= 0 and right < count and (joined := pair(left, right)):
                    heapq.heappush(waiting, joined)
        merged, k = [], 0
        while k < count:
            merged.append(tokens[k])
            k = after[k]
        return merged

    def encode_bytebpe(self, text):
        # The pre-tokenizer keeps merges inside a word: the pieces never cross from a word into the next.
        tokens = []
        for part in pretokenize(text, self.pretokenizer):
            word = part.encode("utf-8").decode("latin-1").translate(BYTE_CHARS)  # each byte as its character
            if self.ignore_merges and word in self.text_index:
                # Llama 3: a piece the vocabulary has is that one token, whatever the merges would make of it
                tokens.append(self.text_index[word])
                continue
            # a byte the vocabulary lacks (SmolLM2 has no piece for 21 of them: 0x04, 0xF1, ...) is left out, and its
            # neighbours may then merge, as Hugging Face's BPE does without byte_fallback or an unk_token (T215)
            symbols = [char for char in word if char in self.text_index]
            while len(symbols) > 1:
                best_score, best_id, best_idx = self.UNMATCHABLE, -1, -1
                for i in range(len(symbols) - 1):
                    id = self.text_index.get(symbols[i] + symbols[i + 1])
                    if id is not None and self.scores[id] > best_score:
                        best_score, best_id, best_idx = self.scores[id], id, i
                if best_idx == -1:
                    break
                symbols[best_idx:best_idx + 2] = [self.vocab[best_id].decode("utf-8", "replace")]
            tokens += [self.text_index[symbol] for symbol in symbols]
        return tokens

    def encode_unigram(self, text):
        # Viterbi: best[j] is the best total score of any segmentation of text[:j]. The pieces that begin at i are
        # no longer than the reach of text[i:i + 2] (one character where no longer piece begins with those two) (T200)
        n, pieces, scores, reach = len(text), self.pieces, self.scores, self.reach
        best = [0.0] + [-math.inf] * n
        back = [None] * (n + 1)
        for i in range(n):
            here = best[i]
            if here == -math.inf:
                continue
            for j in range(i + 1, min(n, i + reach.get(text[i:i + 2], 1)) + 1):
                id = pieces.get(text[i:j])
                if id is not None and here + scores[id] > best[j]:
                    best[j], back[j] = here + scores[id], (i, [id])
            # a character the vocabulary lacks becomes its UTF-8 bytes, at a penalty
            if back[i + 1] is None or back[i + 1][0] != i:
                score = best[i] + self.unknown_score
                if score > best[i + 1]:
                    spelled = [self.unknown] if self.unknown is not None else [self.byte_tokens[byte] for byte in text[i].encode("utf-8")]
                    best[i + 1], back[i + 1] = score, (i, spelled)
        tokens, j = [], len(text)
        while j > 0:
            i, ids = back[j]
            if self.unknown is not None and ids == [self.unknown] and tokens[:1] == ids:
                ids = []  # sentencepiece makes a run of unknown characters one unknown piece
            tokens[:0] = ids
            j = i
        return tokens

    def decode(self, prev_token, token, bos=1):
        piece = self.vocab[token]
        if self.kind == "bytebpe":
            # back through the byte <-> character table; a piece that is not written in it (an added token such
            # as <|im_end|>) is its own text
            text = piece.decode("utf-8", "replace")
            return bytes(CHAR_BYTES[char] for char in text) if all(char in CHAR_BYTES for char in text) else piece
        # Following the first token, sentencepiece decoder strips the leading whitespace (the dummy prefix)
        if prev_token == bos and piece.startswith(b" "):
            piece = piece[1:]
        # Some tokens designate raw bytes, and look like b"<0x0A>"
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece


def partial_rope(heads, cos, sin, rotary):
    """GPT-NeoX rotates the first rotary values of every head and leaves the rest alone."""
    turned = rope(heads[:, :rotary], cos[:rotary // 2], sin[:rotary // 2])
    return np.concatenate([turned, heads[:, rotary:]], axis=1) if rotary < heads.shape[1] else turned


# the epsilon of RMSNorm where config.json does not say another (T124: Qwen3's 1e-6 moved perplexity by 0.12%)
RMS_EPS = 1e-5


def rmsnorm(x, weight, eps=RMS_EPS):
    return weight * (x / np.sqrt(x.dot(x) / x.size + np.float32(eps)))


def head_norm(x, weight, eps=RMS_EPS):
    """rmsnorm() of every head of x (heads one after another), all with the same weight of one head's size."""
    heads = x.reshape(-1, weight.size)
    return (weight * heads / np.sqrt((heads * heads).mean(axis=1, keepdims=True) + np.float32(eps))).reshape(-1)


def l2norm(x, eps=1e-6):
    """Every row of x over its length (transformers' l2norm of Gated DeltaNet's q and k, T229)."""
    return x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + np.float32(eps))


def softplus(x):
    """ln(1 + e^x), as PyTorch's (x past 20 is x)."""
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def hadamard(x, block):
    """Prism's rotation (T229): every block of x times the normalized Walsh-Hadamard matrix of Sylvester's order
    (H[i, j] = (-1)^popcount(i & j) / sqrt(block)), which is its own inverse."""
    y = np.array(x, dtype=np.float32).reshape(-1, block)
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        y = np.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], axis=2)
        h *= 2
    return (y.reshape(np.shape(x)) / np.float32(math.sqrt(block))).astype(np.float32)


def layernorm(x, weight, bias):
    """GPT-2 normalizes by the mean and the variance, and adds a bias."""
    centred = x - x.mean()
    return weight * (centred / np.sqrt(centred.dot(centred) / x.size + 1e-5)) + bias


def gelu(x):
    """GPT-2's gelu_new: the tanh approximation, written with exp so that the kernel can do the same."""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * (1.0 / (1.0 + np.exp(-2.0 * inner)))


def rope(x, cos, sin):
    # Rotate each pair (x[2i], x[2i+1]) of every head by the angle for this position
    pairs = x.reshape(-1, cos.size, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = x0 * cos - x1 * sin
    out[..., 1] = x0 * sin + x1 * cos
    return out.reshape(-1, 2 * cos.size)


def rope_frequencies(width, theta, scaling=None):
    """The angle per position of each pair of a head's first width values (float64), for the RoPE tables.

    scaling: config.json's rope_scaling, of three kinds. "linear" (T126: deepseek-coder): every pair turns factor times
    slower, as if the positions were divided by factor. "llama3": the pairs that turn slowly (a wavelength past
    original_max_position_embeddings / low_freq_factor) turn factor times slower, the fast ones (shorter than
    original / high_freq_factor) as before, and the ones between are a blend of the two (transformers'
    _compute_llama3_parameters). "yarn" (T235: the ternary Bonsai): the pairs that turn fewer than beta_slow times
    within original_max_position_embeddings turn factor times slower, the ones that turn more than beta_fast times as
    before, and the ones between are a blend along a line over their index (transformers' _compute_yarn_parameters);
    the tables are then scaled by rope_amplitude().
    """
    frequencies = 1.0 / theta ** (np.arange(0, width, 2, dtype=np.float64) / width)
    if not scaling:
        return frequencies
    kind = scaling.get("rope_type", scaling.get("type"))
    if kind == "linear":
        return frequencies / float(scaling["factor"])
    if kind == "yarn":
        factor, original = float(scaling["factor"]), float(scaling["original_max_position_embeddings"])
        # the index of the pair that turns so many times within the original context, and the line between the two
        where = lambda turns: width * math.log(original / (turns * 2 * math.pi)) / (2 * math.log(theta))
        low, high = where(scaling.get("beta_fast") or 32), where(scaling.get("beta_slow") or 1)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, width - 1)
        # both ends the same would divide by zero: transformers moves the far one a little
        ramp = np.clip((np.arange(width // 2, dtype=np.float64) - low) / ((high - low) or 0.001), 0, 1)
        return frequencies / factor * ramp + frequencies * (1 - ramp)
    if kind != "llama3":
        raise ValueError(f"RoPE scaling of the {kind} kind is not supported.")
    factor, low, high = float(scaling["factor"]), float(scaling["low_freq_factor"]), float(scaling["high_freq_factor"])
    original = float(scaling["original_max_position_embeddings"])
    wavelength = 2 * math.pi / frequencies
    smooth = (original / wavelength - low) / (high - low)
    blended = (1 - smooth) * frequencies / factor + smooth * frequencies
    return np.where(wavelength < original / high, frequencies,
                    np.where(wavelength > original / low, frequencies / factor, blended))


def rope_amplitude(scaling=None):
    """What the cos and sin tables are multiplied by: 1 but for YaRN (T235), whose attention_factor (0.1 ln(factor) + 1
    unless config.json says it) scales q and k, and so every score by its square."""
    if not scaling or scaling.get("rope_type", scaling.get("type")) != "yarn":
        return 1.0
    if scaling.get("attention_factor") is not None:
        return float(scaling["attention_factor"])
    factor = float(scaling["factor"])
    grown = lambda m=1.0: 0.1 * m * math.log(factor) + 1.0 if factor > 1 else 1.0
    if scaling.get("mscale") and scaling.get("mscale_all_dim"):
        return grown(float(scaling["mscale"])) / grown(float(scaling["mscale_all_dim"]))
    return grown()


def rope_tables(seq_len, width, theta, scaling=None):
    """The cos and sin tables (float64, seq_len rows of width // 2) of the RoPE of a head's first width values."""
    angles = np.arange(seq_len, dtype=np.float64)[:, None] * rope_frequencies(width, theta, scaling)
    amplitude = rope_amplitude(scaling)
    return np.cos(angles) * amplitude, np.sin(angles) * amplitude


REPETITION_WINDOW = 64  # the repetition penalty looks at this many of the latest tokens
# T195: what sample() says, on the kernels and in NumPy, when the largest logit is NaN or an infinity (a NaN anywhere,
# +inf anywhere, or all -inf): a broken model or an overflow. No token is drawn from logits like these
NOT_FINITE = "The model computed logits that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed."
# The KV cache starts with room for this many positions and doubles when a run gets there: a context of 4096
# tokens is 200 MB of cache for llm-jp-3-150m, which a short text should not have to pay for (and WebAssembly
# never gives memory back)
KV_START = 256
# The classifier's input has a few channels that the final norm's weight blows up (openai-community/gpt2: 12 to 17
# times, 316 against a median of 0.3). With the int8 kernels the activations are quantized in groups of 32, so one
# such channel sets the scale of its group and the other 31 round to nothing: GPT-2 lost 17% of perplexity to
# that (T92). These many channels, the largest of the norm's weight, are taken out of the vector before it is
# quantized and multiplied in float32 by their own columns of the classifier (vocab_size * 8 multiply-adds next to
# vocab_size * dim). With them GPT-2 is back to +0.35%. Only a norm whose largest weight is OUTLIER_RATIO times
# its median gets this (GPT-2: 13.9; tiny-lm 1.1, llm-jp-3 150M 1.3, Pythia 160M 1.3, rinna GPT-2 1.4, SmolLM2 135M
# 1.9): the others gain nothing from it (tiny-lm stays at +0.31%) and would pay about 3% of speed.
OUTLIER_CHANNELS = 8
OUTLIER_RATIO = 4.0


# what T52 can leave out, each of them something that already has a fallback
SWITCHES = ("kernels", "int8", "relaxed", "sampler", "kv16")


def load_kernels(path, without_relaxed=False):
    """The WASM SIMD kernels of kernels/*.ts, as ctypes functions, or None when they cannot be used.

    They are Emscripten side modules: ctypes.CDLL links them into Pyodide's own memory, so they work in place
    on NumPy arrays. Anything may go wrong here (no file, not Pyodide, a future Emscripten that loads side
    modules differently), and then NumPy does the work as before.
    """
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        i32, p = ctypes.c_int32, ctypes.c_void_p
        signatures = dict(matmul_f32=[p, p, p, i32, i32, i32], quantize_x=[p, p, p, i32, i32], quantize6_x=[p, p, p, i32],
                          matmul_q8=[p, p, p, p, p, i32, i32, i32], rmsnorm=[p, p, p, i32, ctypes.c_float], rope=[p, p, p, i32, i32, i32],
                          attention=[p, p, p, p, p, i32, i32, i32, i32, i32, i32],
                          attention_f16=[p, p, p, p, p, i32, i32, i32, i32, i32, i32], to_f16=[p, p, i32], from_f16=[p, p, i32],
                          swiglu=[p, p, p, i32], add_inplace=[p, p, i32],
                          add_columns=[p, p, p, i32, i32],
                          layernorm=[p, p, p, p, i32], gelu=[p, p, p, i32],
                          penalize=[p, p, i32, ctypes.c_float], widen_bf16=[p, p, i32], widen_q8_0=[p, p, i32],
                          sample=[p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_double, p, p])
        kernels = {}
        for name, argtypes in signatures.items():
            kernels[name] = getattr(lib, name)
            kernels[name].argtypes, kernels[name].restype = argtypes, i32 if name == "sample" else None
    except Exception:
        return None
    if without_relaxed:
        return kernels
    try:
        # a browser without relaxed SIMD (shipping Safari) refuses to compile this one: then int8 uses matmul_q8
        relaxed = ctypes.CDLL(path.replace(".so", "_relaxed.wasmlib")).matmul_q8r
        relaxed.argtypes, relaxed.restype = [p, p, p, p, p, p, i32, i32, i32], None
        kernels["matmul_q8r"] = relaxed
    except Exception:
        pass
    return kernels


def kernel_quantizer(path):
    """llama2_convert.quantize() on the SIMD kernels (T89): int8 values in groups of 32 and one float32 scale per
    group, the same bytes as NumPy's, six times faster (quantize_x with no bias: the activations' quantizer is the
    same computation). For the converter's quantize_rows; None where the kernels cannot be loaded.
    six=True: quantize6() and pack6() in one pass on the kernel quantize6_x (T98), the same bytes: the packed groups
    (24 bytes each) and their scales."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    quantize_x, quantize6_x = kernels["quantize_x"], kernels["quantize6_x"]

    def quantize_rows(values, six=False):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if six:
            packed = np.empty(values.size // 32 * 24, dtype=np.uint8)
            scales = np.empty(values.size // 32, dtype=np.float32)
            quantize6_x(packed.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size)
            return packed.reshape(-1, 24), scales
        quantized = np.empty(values.size, dtype=np.int8)
        scales = np.empty(values.size // 32, dtype=np.float32)
        quantize_x(quantized.ctypes.data, scales.ctypes.data, values.ctypes.data, values.size, 0)
        return quantized.reshape(-1, 32), scales

    return quantize_rows


def kernel_widener(path):
    """llama2_convert.bfloat16() on the SIMD kernels (T123): the same float32, a shift of every 16 bits, several
    times faster than NumPy's two passes. For the converter's bfloat16; None where the kernels cannot be loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    widen = kernels["widen_bf16"]

    def bfloat16(raw):
        halves = np.frombuffer(raw, dtype=np.uint16)
        out = np.empty(halves.size, dtype=np.float32)
        widen(out.ctypes.data, halves.ctypes.data, halves.size)
        return out

    return bfloat16


def kernel_q8_0(path):
    """llama2_convert.q8_0() on the SIMD kernels (T136): GGUF's Q8_0 blocks widened to the same float32, each int8
    times its block's float16 scale. For the converter's q8_0; None where the kernels cannot be loaded."""
    kernels = load_kernels(path) if path else None
    if not kernels:
        return None
    widen = kernels["widen_q8_0"]

    def q8_0(raw):
        blocks = np.frombuffer(raw, dtype=np.uint8)
        if blocks.size % 34:
            raise ValueError("Q8_0 data is not whole blocks of 34 bytes.")
        out = np.empty(blocks.size // 34 * 32, dtype=np.float32)
        widen(out.ctypes.data, blocks.ctypes.data, blocks.size // 34)
        return out

    return q8_0


# T98: int6, six bits a weight. An int6 group is an int8 group whose values are multiples of 4 (-128..124: six
# significant bits) and whose scale is a quarter: the same products as six bits and a whole scale, to the bit (a
# power of two), and everything past the packing is int8. So the kernels widen a group straight into the int8 their
# dot products take, with no offset to take out. A group of 32 takes 24 bytes: the four low bits of the six of
# value j and of value j + 16 share byte j (0..15), and the top two bits of values k, k + 8, k + 16 and k + 24 share
# byte 16 + k (0..7), at bits 0, 2, 4 and 6. Masks and shifts by constants only (kernels/six.ts).


def pack6(values):
    """int8 values that are multiples of 4, groups of 32 (rows of them) -> 24 bytes a group (uint8)."""
    b = (np.asarray(values, dtype=np.int8).reshape(-1, 32).view(np.uint8) >> 2) & 63  # the six bits
    low = (b[:, :16] & 15) | ((b[:, 16:] & 15) << 4)
    top = b >> 4
    high = top[:, 0:8] | (top[:, 8:16] << 2) | (top[:, 16:24] << 4) | (top[:, 24:32] << 6)
    return np.concatenate([low, high], axis=1)


def unpack6(packed):
    """24 bytes a group -> the 32 int8 values of it (rows of them), multiples of 4."""
    b = np.asarray(packed, dtype=np.uint8).reshape(-1, 24)
    low = np.concatenate([b[:, :16] & 15, b[:, :16] >> 4], axis=1)
    top = np.concatenate([(b[:, 16:] >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1)
    return ((low | (top << 4)) << 2).astype(np.uint8).view(np.int8)


def quantize6(values):
    """float32 values, whole rows of groups of 32 -> (int8 values, float32 scales) in six bits: v = round(x / s) in
    -32..31 with s = the largest |x| of the group over 31, given as 4 v and s / 4 (see pack6)."""
    groups = np.asarray(values, dtype=np.float32).reshape(-1, 32)
    scales = (np.abs(groups).max(axis=1) / 31.0).astype(np.float32)
    inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
    six = np.clip(np.rint(groups * inverse[:, None]), -32, 31).astype(np.int8)
    return (six * 4).astype(np.int8), scales / np.float32(4)


class Tensor:
    """Where a tensor of the checkpoint is, when the weights live outside Python (T93: the forward pass runs in
    public/forward.js on its own WebAssembly memory). kind: "int8" (values, then one float32 scale per group of
    the last dimension at scales), "int6" (the same with the values packed, see pack6), "f32" or "f16". Offsets count
    from the start of the checkpoint file."""

    __slots__ = ("kind", "offset", "shape", "group", "scales")

    def __init__(self, kind, offset, shape, group=0, scales=0):
        self.kind, self.offset, self.shape, self.group, self.scales = kind, offset, tuple(shape), group, scales

    def plan(self):
        return {"kind": self.kind, "offset": self.offset, "shape": list(self.shape), "group": self.group,
                "scales": self.scales}


class Places:
    """Where the tensors of a checkpoint are, taken in file order (Llama's take() with external=, T93): a Tensor for
    each, from the header's 28 bytes on. dtype: the checkpoint's as numpy has it (int6 is int8 with six=True)."""

    def __init__(self, dtype, six=False):
        self.dtype, self.six, self.offset = np.dtype(dtype), six, 28

    def take(self, *shape, matrix=True, widen=True):
        count = math.prod(shape)
        if self.dtype == np.int8 and matrix:
            group = 32
            while shape[-1] % group:
                group //= 2
            stored = count * 3 // 4 if self.six else count
            tensor = Tensor("int6" if self.six else "int8", self.offset, shape, group, self.offset + stored)
            self.offset += stored + 4 * (count // group)
            return tensor
        tensor = Tensor("f16" if self.dtype == np.float16 else "f32", self.offset, shape)
        self.offset += count * (2 if self.dtype == np.float16 else 4)
        return tensor


def external_tensors(header, dtype, form=None):
    """Where every tensor of a Llama checkpoint with this header, dtype and form (FORM) is, {name: Tensor.plan()}, as
    Llama(external=) hands them to public/forward.js, before any of its bytes are there (T156: the worker sends the
    layers' matrices to the GPU as they come, and keeps the rest; llama_tensors() is the one order)."""
    form = form_of(form)
    if form["arch"] != "llama":
        raise ValueError("Only a Llama's tensors are placed before the model is built.")
    probe = Llama.__new__(Llama)
    (probe.dim, probe.hidden_dim, probe.n_layers, probe.n_heads, probe.n_kv_heads, vocab_size,
     probe.seq_len) = (int(value) for value in header)
    probe.vocab_size = abs(vocab_size)
    probe.head_size = int(form["head_dim"]) or probe.dim // probe.n_heads
    probe.q_dim, kv_dim = probe.n_heads * probe.head_size, probe.n_kv_heads * probe.head_size
    six = str(dtype) == "int6"
    places = Places(np.int8 if six else dtype, six)
    probe.llama_tensors(places.take, vocab_size > 0, True, kv_dim, form["bias"], places.dtype,
                        lambda width: (np.zeros(width // 2), np.zeros(width // 2)), form["qk_norm"])
    return {name: getattr(probe, name).plan() for name in TENSOR_NAMES if isinstance(getattr(probe, name, None), Tensor)}


# the attributes of Llama that are tensors of the file, in no particular order
TENSOR_NAMES = ("token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo", "rms_ffn_weight", "w1", "w2", "w3",
                "rms_final_weight", "freq_cis_real", "freq_cis_imag", "wcls", "bq", "bk", "bv", "positions",
                "ln_att_bias", "ln_ffn_bias", "ln_final_bias", "bo", "b1", "b2", "q_norm", "k_norm",
                # T229: a Qwen3.5's (qwen35_layout())
                "wg", "wqkv", "wz", "wb", "wa", "conv", "ssm_a", "ssm_dt", "ssm_norm", "wout",
                "signs_dim", "signs_q", "signs_v", "signs_hidden")


def outlier_channels(weight, count=OUTLIER_CHANNELS, ratio=OUTLIER_RATIO):
    """The count channels with the largest final-norm weight, in order, or none when the weight has no outliers
    (its largest is less than ratio times its median)."""
    size = np.abs(np.asarray(weight, dtype=np.float32))
    if size.max() < ratio * np.median(size):
        return np.zeros(0, dtype=np.intp)
    return np.sort(np.argsort(-size)[:count])


def outlier_columns(classifier, channels):
    """The columns of the int8 classifier for these channels, as float32, one column per row (len(channels),
    vocab_size): what the add_columns kernel multiplies (see OUTLIER_CHANNELS)."""
    values, scales = classifier  # (vocab_size, dim / group, group) int8 and (vocab_size, dim / group, 1) float32
    group = values.shape[-1]
    columns = np.stack([values[:, c // group, c % group].astype(np.float32) * scales[:, c // group, 0] for c in channels])
    return np.ascontiguousarray(columns)


# The form of a checkpoint: what sets its tensors and sizes its forward pass besides the 7 ints of the header, which
# the legacy file cannot say (see Llama.__init__), with the value of a file that says nothing. The converter writes
# them into the options, and one dict of these names goes to everything that lays the file out or sizes it
# (llama2_convert.layout(), checkpoint_size() and Writer, checkpoint_dtype() below, forward.js's footprint()), so
# that another one is added where it is used, not along the way (T144).
FORM = {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0,
        # T229: a Qwen3.5 (arch "qwen35") has every interval-th layer of attention and the others of Gated DeltaNet,
        # whose q and k have k_heads heads of k_head values, v v_heads of v_head, and a causal conv of conv taps.
        # hadamard: the block of the rotation its matrices were stored in (Prism's, 0 for none; T229)
        "interval": 0, "k_heads": 0, "v_heads": 0, "k_head": 0, "v_head": 0, "conv": 0, "hadamard": 0}


def qwen35_layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, form):
    """(name, shape, is a matrix) of every tensor of a Qwen3.5's checkpoint, in file order (T229): llama2_convert's
    layout() and Llama.qwen35_tensors() both follow this list. The layers of attention (every interval-th, the last of
    each run) and of Gated DeltaNet are stacked apart, each kind with its own tensors; the norms and the FFN are the
    same in every layer. q and its output gate are two matrices (Hugging Face interleaves them by head). Is a matrix:
    True for what int8 quantizes, False for float32 vectors, None for the RoPE tables. With a rotation (hadamard),
    the signs it multiplies each kind of input by come last: of dim, of q_dim (the attention's output), of the
    DeltaNet's v (its output) and of hidden_dim."""
    head = int(form["head_dim"]) or dim // n_heads
    full = n_layers // int(form["interval"])
    linear = n_layers - full
    q_dim, kv_dim = n_heads * head, n_kv_heads * head
    k_dim, v_dim = int(form["k_heads"]) * int(form["k_head"]), int(form["v_heads"]) * int(form["v_head"])
    v_heads = int(form["v_heads"])
    tensors = [("token_embedding_table", (abs(vocab_size), dim), True), ("rms_att_weight", (n_layers, dim), False),
               ("wq", (full, q_dim, dim), True), ("wg", (full, q_dim, dim), True), ("wk", (full, kv_dim, dim), True),
               ("wv", (full, kv_dim, dim), True), ("wo", (full, dim, q_dim), True),
               ("q_norm", (full, head), False), ("k_norm", (full, head), False),
               ("wqkv", (linear, 2 * k_dim + v_dim, dim), True), ("wz", (linear, v_dim, dim), True),
               ("wb", (linear, v_heads, dim), True), ("wa", (linear, v_heads, dim), True),
               ("conv", (linear, 2 * k_dim + v_dim, int(form["conv"])), False), ("ssm_a", (linear, v_heads), False),
               ("ssm_dt", (linear, v_heads), False), ("ssm_norm", (linear, int(form["v_head"])), False),
               ("wout", (linear, dim, v_dim), True),
               ("rms_ffn_weight", (n_layers, dim), False), ("w1", (n_layers, hidden_dim, dim), True),
               ("w2", (n_layers, dim, hidden_dim), True), ("w3", (n_layers, hidden_dim, dim), True),
               ("rms_final_weight", (dim,), False),
               ("freq_cis_real", (seq_len, head // 2), None), ("freq_cis_imag", (seq_len, head // 2), None)]
    if vocab_size < 0:
        tensors.append(("wcls", (abs(vocab_size), dim), True))
    if int(form["hadamard"]):
        tensors += [("signs_dim", (dim,), False), ("signs_q", (q_dim,), False), ("signs_v", (v_dim,), False),
                    ("signs_hidden", (hidden_dim,), False)]
    return tensors


def form_of(options=None):
    """The form (FORM's keys, each with its default where options has none) out of options, a dict with those and
    any others (the options of a model, a manifest's). Dicts from JavaScript are read too (a JsProxy). A key given as
    None (a JSON null, or JavaScript's undefined) says nothing, as head_size() reads "head_dim": null: it had
    checkpoint_dtype() fail on int(None) while footprint() counted dim / heads (the review of T144)."""
    options = options.to_py() if hasattr(options, "to_py") else (options or {})
    return {key: default if options.get(key) is None else options[key] for key, default in FORM.items()}


def checkpoint_dtype(header, size, form=None):
    """"float32", "float16", "int8" or "int6": what a checkpoint file of size bytes with this header (7 ints) holds.

    The legacy format does not say, but the header fixes the size of each variant. Anything else is no checkpoint
    this engine can read, and the ValueError says so before hundreds of megabytes are read for nothing.
    form: what the file cannot say either (FORM, taken out of a model's options): the tensors differ with it.
    """
    form = form_of(form)
    arch = form["arch"]
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (int(value) for value in header)
    head_size = int(form["head_dim"]) or (dim // n_heads if n_heads and dim % n_heads == 0 else 0)
    limit = 1 << 24
    if not (0 < dim < limit and 0 < hidden_dim < limit and 0 < n_layers < 4096 and 0 < n_kv_heads <= n_heads <= dim
            and 0 < abs(vocab_size) < limit and 0 < seq_len < limit and 0 < head_size < limit and n_heads % n_kv_heads == 0):
        raise ValueError("This is not a llama2.c checkpoint: the header makes no sense.")
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    rope = 2 * seq_len * (head_size // 2)
    if arch == "qwen35":
        tensors = qwen35_layout(*header, form)
        # a matrix of each layer is (layers * rows, row length) as the others are counted
        matrices = [(math.prod(shape[:-1]), shape[-1]) for _, shape, kind in tensors if kind]
        vectors = sum(math.prod(shape) for _, shape, kind in tensors if kind is False)
    elif arch in ("gpt2", "neox"):
        # the same tensors in the same order as gpt2_tensors() and llama2_convert.layout(arch=): q, k, v, o, the two
        # FFN matrices (no gate), and for GPT-2 the table of positions in place of the RoPE tables
        matrices = [(abs(vocab_size), dim)] + [(n_layers * dim, dim)] * 4 + [(n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim)]
        if arch == "gpt2":
            matrices.append((seq_len, dim))
            rope = 0
        # LayerNorm weights and biases (two per layer, one at the end), the biases of q, k, v, o and the FFN
        vectors = n_layers * (4 * dim + 3 * dim + dim + hidden_dim + dim) + 2 * dim
    else:
        # the same tensors in the same order as llama_tensors() and quantize.py: (rows, row length) of the matrices
        matrices = [(abs(vocab_size), dim), (n_layers * q_dim, dim), (n_layers * kv_dim, dim), (n_layers * kv_dim, dim),
                    (n_layers * dim, q_dim), (n_layers * hidden_dim, dim), (n_layers * dim, hidden_dim),
                    (n_layers * hidden_dim, dim)]
        vectors = 2 * n_layers * dim + dim + (n_layers * (q_dim + 2 * kv_dim) if form["bias"] else 0) \
            + (2 * n_layers * head_size if form["qk_norm"] else 0)
    if vocab_size < 0 and arch != "qwen35":
        matrices.append((abs(vocab_size), dim))
    floats = sum(rows * length for rows, length in matrices) + vectors + rope

    def group(length):
        size = 32
        while length % size:
            size //= 2
        return size

    # quantize.py: int8 values and a float32 scale per group; the vectors stay float32, the RoPE tables are left out
    int8 = sum(rows * length + 4 * (rows * length // group(length)) for rows, length in matrices) + 4 * vectors
    sizes = {28 + 4 * floats: "float32", 28 + 2 * floats: "float16", 28 + int8: "int8"}
    if all(length % 32 == 0 for _, length in matrices):
        # T98: 24 bytes and a float32 scale per group of 32 (only rows of whole groups can be int6)
        sizes.setdefault(28 + sum(rows * length // 32 * 28 for rows, length in matrices) + 4 * vectors, "int6")
    if size not in sizes:
        raise ValueError(f"This is not a llama2.c checkpoint: its header asks for {28 + 4 * floats} bytes as float32, "
                         f"{28 + 2 * floats} as float16 or {28 + int8} as int8, and the file has {size}.")
    return sizes[size]


def check_tokenizer(tokenizer, header):
    """ValueError unless tokenizer (a tokenizer.bin) holds exactly the vocabulary of the checkpoint with this header.

    The engine would read the first pieces of a larger vocabulary without complaint, and write nonsense.
    """
    vocab_size = abs(int(list(header)[5]))
    offset, pieces = 4, 0
    while offset + 8 <= len(tokenizer):
        if pieces == vocab_size and bytes(tokenizer[offset:offset + len(CHARSMAP)]) == CHARSMAP:
            # T216: a sentencepiece model's normalizer after the pieces, to the end
            (size,) = struct.unpack_from("<I", tokenizer, offset + len(CHARSMAP))
            offset += len(CHARSMAP) + 4 + size
            break
        _, length = struct.unpack_from("<fi", tokenizer, offset)
        if length < 0 or offset + 8 + length > len(tokenizer):
            raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
        offset += 8 + length
        pieces += 1
    if offset != len(tokenizer) or pieces == 0:
        raise ValueError("The smaller file is not a llama2.c tokenizer.bin.")
    if pieces != vocab_size:
        raise ValueError(f"This tokenizer.bin holds {pieces} pieces, but the checkpoint has a vocabulary of "
                         f"{vocab_size}: they do not belong together.")


# T108: how many tokens of a prompt forward_many() takes at once: forward.js's BATCH. The worker cannot answer a
# message (stop, a new model) while one call runs, and a block of 16 keeps that under a second on a 1.5B model.
# T147: forward.js says how many it takes (promptBlock: more where the GPU takes the prompt)
PROMPT_BLOCK = 16


class Llama:
    forward_many = None  # T108: forward.js's forwardMany(tokens, pos) for a prompt, where there is one
    prompt_block = staticmethod(lambda: PROMPT_BLOCK)  # T147: how many tokens forward_many() takes at once, now
    # T152: forward.js's generateMany(token, pos, history, count, temperature, topp, penalty, randoms, stops), which
    # runs count steps of generate() on the GPU (the forward pass and the sampling, with the random numbers drawn here)
    # and returns the tokens it sampled (a stop token last), or None where it did not (the CPU then takes the step);
    # token_block(): how many steps it takes at once now, 0 where the CPU is faster (or there is no GPU)
    generate_many = None
    token_block = staticmethod(lambda: 0)

    def __init__(self, checkpoint, tokenizer, dtype="float32", rope_theta=10000.0,
                 tokenizer_kind="bpe", nfkc=False, nfc=False, pretokenizer="gpt2", bias=False, arch="llama",
                 rotary=0, parallel_residual=False, bos=BOS, stop_tokens=(BOS,), kernels=None, specials=(),
                 disable=(), external=None, rope_scaling=None, ignore_merges=False, collapse=False,
                 unknown=None, qk_norm=False, head_dim=0, rms_norm_eps=RMS_EPS, interval=0, k_heads=0, v_heads=0,
                 k_head=0, v_head=0, conv=0, hadamard=0):
        """checkpoint: llama2.c "legacy" format, a 7 int header then the weights.

        dtype="float16" and dtype="int8" are this project's smaller variants (convert_hf.py, quantize.py), and
        dtype="int6" (T98) is int8 with six bits a value (pack6).
        arch="neox": GPT-NeoX, which is arch="gpt2" with RoPE over the first rotary values of every head
        (rotary=0 means all of them) and, when parallel_residual is on, the attention and the FFN both reading
        the same x instead of one after the other.
        arch="gpt2": LayerNorm instead of RMSNorm, GELU instead of SwiGLU (and no gate matrix), a learned table
        of positions instead of RoPE, and a bias after every projection. The tensors of the file differ with it,
        so it is llama2_convert.layout(arch=) that says what is there.
        bias=True: the checkpoint ends with a bias for q, k and v of every layer, which is added after those
        projections (Qwen2). The legacy header cannot say so, so the caller does, like the tokenizer settings.
        qk_norm=True (T124): after them come the RMSNorm weights of q and k (one head's size each, per layer), and
        every head of q and k is normalized with them before RoPE (Qwen3). The caller says so, like bias.
        head_dim: the size of a head where it is not dim / n_heads (T124: Qwen3 0.6B has 16 heads of 128 in a dim of
        1024): q and the attention's output are then n_heads * head_dim wide. rms_norm_eps: config.json's, the epsilon
        of every RMSNorm (the kernels take it too).
        arch="qwen35" (T229, Qwen3.5 and Prism's Bonsai 2): every interval-th layer is attention with an output gate
        (a sigmoid of a second projection of the input times the attention's output) over heads whose first rotary
        values RoPE turns; the others are Gated DeltaNet, whose k_heads, v_heads, k_head, v_head and conv say its
        sizes (FORM). hadamard: the block size of the rotation its matrices are stored in (see rotated()).
        rope_scaling: config.json's, for the RoPE tables that are not in the file (int8, float16); see
        rope_frequencies(). ignore_merges: a byte-level BPE takes a pre-tokenized piece that is in the vocabulary
        whole (Llama 3).
        bos starts every sequence; generation ends when the model emits one of stop_tokens.
        kernels is the path of simdkernel.so: the sampling runs on it (penalize, sample). The forward pass on the
        kernels is public/forward.js's (external, below); without external, NumPy computes the forward pass.
        disable: the optimizations to leave out, to measure what each one is worth (T52). Only what already has
        a fallback: "kernels" (NumPy does everything), "int8" (the weights are widened to float32 and the
        float32 kernel multiplies them), "relaxed" (matmul_q8 instead of matmul_q8r), "sampler" (NumPy
        samples) and "kv16" (an int8 model keeps its keys and values in float32 on a shared memory too, T110). Anything else is refused, so that a typo never quietly measures the wrong thing.
        external (T93): the weights are not in Python but in a WebAssembly memory of public/forward.js, which also
        runs the forward pass. checkpoint is then None, and external has size (of the file), read(offset, length)
        (a few bytes: the header, the final norm) and start(plan), which gets where every tensor is and returns an
        object with forward(token, pos, need_logits, logits) and backend. Python keeps the tokenizer, generate()
        and the sampling. The NumPy forward cannot run on weights it does not have: disable "kernels" without it.
        """
        if external is not None:
            head = external.read(0, 28)
            checkpoint = bytes(head.to_py() if hasattr(head, "to_py") else head)
            if "kernels" in tuple(str(name) for name in disable):
                raise ValueError("Without the kernels the weights have to be in Python: load them there instead.")
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads,
         self.n_kv_heads, vocab_size, self.seq_len) = struct.unpack_from("<7i", checkpoint, 0)
        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = vocab_size > 0
        self.vocab_size = abs(vocab_size)
        self.head_size = int(head_dim) or self.dim // self.n_heads
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        self.q_dim, kv_dim = self.n_heads * self.head_size, self.n_kv_heads * self.head_size

        disable = tuple(str(name) for name in disable)
        unnamed = [name for name in disable if name not in SWITCHES]
        if unnamed:
            raise ValueError(f"There is no optimization called {unnamed[0]!r}: {', '.join(SWITCHES)}.")
        self.disabled = disable
        # int6 (T98) is int8 with its values packed: from here on it is int8, except where the bytes are read
        six = str(dtype) == "int6"
        dtype = np.dtype(np.int8 if six else dtype)
        offset = 28
        # The int8 kernels work on groups of 32 only
        suitable = dtype != np.int8 or (dim % 32 == 0 and self.q_dim % 32 == 0 and kv_dim % 32 == 0 and hidden_dim % 32 == 0)
        kernels = load_kernels(kernels, "relaxed" in disable) if kernels and "kernels" not in disable and \
            (suitable or external is not None) else None
        # int8 kernels compute on the int8 weights directly: they are never widened, a quarter of the memory
        # (only forward.js computes on them: the NumPy forward widens every matrix)
        keep_int8 = external is not None and suitable and dtype == np.int8 and "int8" not in disable

        # only where it is: public/forward.js reads it (and widens what has to be widened) itself
        places = Places(dtype, six) if external is not None else None

        def take(*shape, matrix=True, widen=True):
            nonlocal offset
            count = math.prod(shape)
            if places is not None:
                tensor = places.take(*shape, matrix=matrix)
                offset = places.offset
                return tensor
            if dtype == np.int8 and matrix:
                # quantize.py: int8 values, then one float32 scale per group
                group = 32
                while shape[-1] % group:
                    group //= 2
                stored = count * 3 // 4 if six else count
                values = unpack6(np.frombuffer(checkpoint, dtype=np.uint8, count=stored, offset=offset)).reshape(-1) \
                    if six else np.frombuffer(checkpoint, dtype=np.int8, count=count, offset=offset)
                scales = np.frombuffer(checkpoint, dtype=np.float32, count=count // group, offset=offset + stored)
                offset += stored + scales.nbytes
                if not widen:
                    values, scales = values.reshape(*shape[:-1], -1, group), scales.reshape(*shape[:-1], -1, 1)
                    # With the kernels every tensor stays a view into the checkpoint buffer. Otherwise this is the
                    # embedding table next to widened copies: copy it, so that the buffer can be freed.
                    return (values, scales) if keep_int8 else (values.copy(), scales.copy())
                return (values.reshape(-1, group).astype(np.float32) * scales[:, None]).reshape(shape)
            # float32 weights are views into the checkpoint buffer: nothing is copied. float16 is widened.
            array = np.frombuffer(checkpoint, dtype=np.float32 if dtype == np.int8 else dtype, count=count, offset=offset)
            offset += array.nbytes
            if dtype == np.float16 and not widen:
                return array.reshape(shape).copy()
            return array.astype(np.float32, copy=dtype == np.int8 and not keep_int8).reshape(shape)

        self.arch, self.parallel_residual = arch, parallel_residual
        self.rms_norm_eps = float(rms_norm_eps)
        # how many values of each head RoPE turns: all of them unless the model says otherwise
        self.rotary = int(rotary) if rotary else self.head_size
        self.positions = None
        self.q_norm = self.k_norm = None
        self.ln_att_bias = self.ln_ffn_bias = self.ln_final_bias = None
        self.bo = self.b1 = self.b2 = None
        # a dict from Python, or a JavaScript object from the worker
        rope_scaling = rope_scaling.to_py() if hasattr(rope_scaling, "to_py") else rope_scaling
        rope = lambda width: rope_tables(self.seq_len, width, rope_theta, rope_scaling)
        self.form = {"interval": int(interval), "k_heads": int(k_heads), "v_heads": int(v_heads), "k_head": int(k_head),
                     "v_head": int(v_head), "conv": int(conv), "hadamard": int(hadamard), "head_dim": int(head_dim)}
        if arch == "qwen35":
            self.qwen35_tensors(take, shared_weights, keep_int8, dtype, rope, vocab_size)
        elif arch in ("gpt2", "neox"):
            self.gpt2_tensors(take, shared_weights, keep_int8, kv_dim, dtype, rope)
        else:
            self.llama_tensors(take, shared_weights, keep_int8, kv_dim, bias, dtype, rope, qk_norm)
        self.backend = "NumPy"
        if external is not None:
            if offset != int(external.size):
                raise ValueError(f"The checkpoint has {int(external.size)} bytes, and its header asks for {offset} "
                                 f"as {dtype.name}.")
            self.forward = self.external_forward(external, keep_int8, disable)
            if kernels and "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        else:
            # NumPy's forward pass; the forward pass on the kernels is forward.js's (external), since T93. A Qwen3.5
            # keeps keys and values for its layers of attention only
            attending = n_layers // int(interval) if arch == "qwen35" else n_layers
            self.key_cache = np.zeros((attending, self.n_kv_heads, min(KV_START, self.seq_len), self.head_size), dtype=np.float32)
            self.value_cache = np.zeros_like(self.key_cache)
            if kernels and "sampler" not in disable:
                self.penalize, self.sample = self.kernel_sampler(kernels)
        if (kernels or external is not None) and "sampler" in disable:
            self.backend += ", NumPy sampling"
        if disable:
            # the line has to say what the numbers are the numbers of
            self.backend += " (without " + ", ".join(name for name in SWITCHES if name in disable) + ")"
        self.tokenizer = Tokenizer(tokenizer, self.vocab_size, kind=tokenizer_kind, nfkc=nfkc, nfc=nfc,
                                   pretokenizer=pretokenizer, ignore_merges=ignore_merges, collapse=collapse,
                                   unknown=unknown)
        self.bos, self.stop_tokens = bos, {int(token) for token in stop_tokens}
        self.specials = tuple(str(special) for special in specials)  # see Tokenizer.encode()
        self.stats = {}
        self._run = 0

    def llama_tensors(self, take, shared_weights, keep_int8, kv_dim, bias, dtype, rope, qk_norm=False):
        """The tensors of a Llama (and of a Qwen2, which adds the q, k and v biases at the end, and of a Qwen3, which
        adds the norms of q and k after them), in file order."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        # With a separate classifier the embedding table is only ever read one row at a time, so an int8 or
        # float16 table stays as it is (a quarter or half of the memory) and forward() widens the row it needs.
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        self.rms_att_weight = take(n_layers, dim, matrix=False)
        self.wq = take(n_layers, self.q_dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wo = take(n_layers, dim, self.q_dim, widen=not keep_int8)
        self.rms_ffn_weight = take(n_layers, dim, matrix=False)
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.w3 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.rms_final_weight = take(dim, matrix=False)
        if dtype != np.int8:
            self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
            self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        # the q, k and v biases go last, so that a checkpoint without them is the file it always was
        self.bq = take(n_layers, self.q_dim, matrix=False) if bias else None
        self.bk = take(n_layers, kv_dim, matrix=False) if bias else None
        self.bv = take(n_layers, kv_dim, matrix=False) if bias else None
        if qk_norm:
            self.q_norm = take(n_layers, self.head_size, matrix=False)
            self.k_norm = take(n_layers, self.head_size, matrix=False)
        if dtype != np.float32:
            # half precision is too coarse for the rotation angles, and int8 files leave the RoPE tables out
            self.freq_cis_real, self.freq_cis_imag = (table.astype(np.float32) for table in rope(self.head_size))

    def qwen35_tensors(self, take, shared_weights, keep_int8, dtype, rope, vocab_size):
        """The tensors of a Qwen3.5 (T229), in the order of qwen35_layout()."""
        header = (self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads, vocab_size, self.seq_len)
        for name, shape, kind in qwen35_layout(*header, self.form):
            if kind is None:
                if dtype != np.int8:
                    setattr(self, name, take(*shape, matrix=False))
                continue
            if name == "token_embedding_table":
                widen = shared_weights and not keep_int8
            else:
                widen = not keep_int8
            setattr(self, name, take(*shape, matrix=bool(kind), widen=widen))
        if shared_weights:
            self.wcls = self.token_embedding_table
        if dtype != np.float32:
            # the angles of the rotated part only, in tables of a head's half (the rest is never read)
            tables = [np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32) for _ in range(2)]
            for table, values in zip(tables, rope(self.rotary)):
                table[:, :self.rotary // 2] = values
            self.freq_cis_real, self.freq_cis_imag = tables

    def gpt2_tensors(self, take, shared_weights, keep_int8, kv_dim, dtype, rope):
        """The tensors of a GPT-2 or a GPT-NeoX, in the order llama2_convert.layout() writes them. The two
        differ in one place: GPT-2 has a learned table of positions, GPT-NeoX the RoPE tables (left out of an
        int8 checkpoint, as everywhere)."""
        dim, hidden_dim, n_layers = self.dim, self.hidden_dim, self.n_layers
        vector = lambda n=dim: take(n_layers, n, matrix=False)
        self.token_embedding_table = take(self.vocab_size, dim, widen=shared_weights and not keep_int8)
        if self.arch == "neox":
            if dtype != np.int8:
                self.freq_cis_real = take(self.seq_len, self.head_size // 2, matrix=False)
                self.freq_cis_imag = take(self.seq_len, self.head_size // 2, matrix=False)
        else:
            self.positions = take(self.seq_len, dim, widen=True)
        self.rms_att_weight, self.ln_att_bias = vector(), vector()
        self.wq = take(n_layers, dim, dim, widen=not keep_int8)
        self.wk = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.wv = take(n_layers, kv_dim, dim, widen=not keep_int8)
        self.bq, self.bk, self.bv = vector(), vector(kv_dim), vector(kv_dim)
        self.wo = take(n_layers, dim, dim, widen=not keep_int8)
        self.bo = vector()
        self.rms_ffn_weight, self.ln_ffn_bias = vector(), vector()
        self.w1 = take(n_layers, hidden_dim, dim, widen=not keep_int8)
        self.b1 = vector(hidden_dim)
        self.w2 = take(n_layers, dim, hidden_dim, widen=not keep_int8)
        self.b2 = vector()
        self.rms_final_weight = take(dim, matrix=False)
        self.ln_final_bias = take(dim, matrix=False)
        self.wcls = self.token_embedding_table if shared_weights else take(self.vocab_size, dim, widen=not keep_int8)
        self.w3 = None
        if self.arch == "gpt2":
            # no rotation: the position is a row of a learned table, added to the embedding
            self.freq_cis_real = self.freq_cis_imag = np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32)
        elif dtype != np.float32:
            # the angles of the rotated part only, in a table of the same shape (the rest is never read)
            tables = [np.zeros((self.seq_len, self.head_size // 2), dtype=np.float32) for _ in range(2)]
            for table, values in zip(tables, rope(self.rotary)):
                table[:, :self.rotary // 2] = values
            self.freq_cis_real, self.freq_cis_imag = tables

    def embedding(self, token):
        if isinstance(self.token_embedding_table, tuple):
            values, scales = self.token_embedding_table
            return (values[token] * scales[token]).reshape(self.dim)
        return self.token_embedding_table[token].astype(np.float32)

    def external_forward(self, external, int8, disable):
        """forward() in public/forward.js (T93): Python hands over where every tensor is, and the few small
        arrays it computes itself (the RoPE tables of a checkpoint that leaves them out, the outlier channels of
        T92), and gets the logits back into one array of its own, which the sampling kernels then read."""
        tensors = {name: getattr(self, name).plan() for name in TENSOR_NAMES if isinstance(getattr(self, name, None), Tensor)}
        derived = {name: np.ascontiguousarray(getattr(self, name), dtype=np.float32).tobytes()
                   for name in ("freq_cis_real", "freq_cis_imag") if isinstance(getattr(self, name), np.ndarray)}
        channels = []
        if int8 and self.arch != "qwen35":  # (T229: a Qwen3.5's classifier may take its input rotated: none there)
            final = self.rms_final_weight
            raw = external.read(final.offset, self.dim * 4)
            weight = np.frombuffer(bytes(raw.to_py() if hasattr(raw, "to_py") else raw), dtype=np.float32)
            channels = [int(c) for c in outlier_channels(weight, min(OUTLIER_CHANNELS, self.dim))]
        plan = {"arch": self.arch, "dim": self.dim, "hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
                "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads, "head_size": self.head_size,
                "vocab_size": self.vocab_size, "seq_len": self.seq_len, "rotary": self.rotary,
                "parallel_residual": bool(self.parallel_residual), "kv_start": KV_START, "rms_norm_eps": self.rms_norm_eps,
                "shared_classifier": self.wcls is self.token_embedding_table, "int8": bool(int8),
                "relaxed": "relaxed" not in disable, "tensors": tensors, "derived": derived, "outliers": channels,
                # T110: the keys and values of an int8 model may be float16 (forward.js uses that on a shared memory)
                "half_kv": bool(int8) and "kv16" not in disable,
                # T229: a Qwen3.5's layers of Gated DeltaNet and Prism's rotation (FORM)
                "form": dict(self.form)}
        engine = external.start(plan)
        self.backend = str(engine.backend)
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        engine.bind(logits)
        self._external = (engine, logits)  # keep both alive: JS writes into the array
        run = engine.forward
        # T108: a prompt's tokens go through the layers several at a time, when forward.js offers that
        many = getattr(engine, "forwardMany", None)
        if many is not None:
            self.forward_many = lambda tokens, pos: many(list(tokens), pos)
            if getattr(engine, "promptBlock", None) is not None:
                self.prompt_block = lambda: int(engine.promptBlock)
        # T152: the steps of generate() on the GPU, where forward.js offers that
        on_gpu = getattr(engine, "generateMany", None)
        if on_gpu is not None:
            def generate_many(token, pos, history, count, temperature, topp, penalty, randoms, stops):
                ids = on_gpu(token, pos, list(history[-REPETITION_WINDOW:]), len(history), count, temperature, topp,
                            penalty, list(randoms), list(stops))
                return None if ids is None else [int(i) for i in ids]

            self.generate_many = generate_many
            self.token_block = lambda: int(engine.tokenBlock)

        def forward(token, pos, need_logits=True):
            run(token, pos, need_logits)
            return logits if need_logits else None

        return forward

    def release(self):
        """Let go of what JavaScript holds for this engine (the forward pass of forward.js and the array it fills).
        The worker calls it before it drops a model (T93). T205: what forward.js answers (a promise, settled once the
        GPU's worker let go of the device), for the worker to wait on before it reads the next model; else None."""
        external = getattr(self, "_external", None)
        if external is None:
            return None
        self._external = None
        return external[0].release()

    def forward(self, token, pos, need_logits=True):
        if self.arch == "qwen35":
            return self.forward_qwen35(token, pos, need_logits)
        n_kv_heads, head_size = self.n_kv_heads, self.head_size
        kv_mul = self.n_heads // n_kv_heads  # >1 with grouped-query attention
        if pos >= self.key_cache.shape[2]:
            # room for twice as many positions (see KV_START)
            positions = min(max(2 * self.key_cache.shape[2], pos + 1), self.seq_len)
            for name in ("key_cache", "value_cache"):
                cache = getattr(self, name)
                larger = np.zeros((*cache.shape[:2], positions, head_size), dtype=np.float32)
                larger[:, :, :cache.shape[2]] = cache
                setattr(self, name, larger)
        scale = np.float32(1.0 / math.sqrt(head_size))
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]
        neox, gpt2 = self.arch == "neox", self.arch == "gpt2"
        layer_norm = neox or gpt2
        # GPT-2 and GPT-NeoX normalize by the mean as well, and have a bias on every projection
        eps = self.rms_norm_eps
        norm = (lambda v, w, b: layernorm(v, w, b)) if layer_norm else (lambda v, w, b: rmsnorm(v, w, eps))
        if gpt2:
            turn = lambda v, c, s: v.reshape(-1, head_size)
        elif neox:
            # only the first self.rotary of every head are rotated, the rest go through untouched
            turn = lambda v, c, s: partial_rope(v.reshape(-1, head_size), c, s, self.rotary)
        else:
            turn = rope

        # Copy the token embedding into x, and (GPT-2) the row of this position
        x = self.embedding(token)
        if gpt2:
            x = x + self.positions[pos]

        # Forward all the layers
        for l in range(self.n_layers):
            # QKV matmuls for this position, RoPE on q and k, k and v go to the kv cache
            xb = norm(x, self.rms_att_weight[l], self.ln_att_bias[l] if layer_norm else None)
            qv, kv, vv = self.wq[l] @ xb, self.wk[l] @ xb, self.wv[l] @ xb
            if self.bq is not None:  # Qwen2 and GPT-2 add a bias to q, k and v
                qv, kv, vv = qv + self.bq[l], kv + self.bk[l], vv + self.bv[l]
            if self.q_norm is not None:  # Qwen3 normalizes every head of q and k
                qv, kv = head_norm(qv, self.q_norm[l], eps), head_norm(kv, self.k_norm[l], eps)
            q = turn(qv, cos, sin).reshape(n_kv_heads, kv_mul, head_size)
            self.key_cache[l, :, pos] = turn(kv, cos, sin)
            self.value_cache[l, :, pos] = vv.reshape(n_kv_heads, head_size)

            # Multihead attention over all timesteps so far, all heads at once
            keys = self.key_cache[l, :, :pos + 1]      # (n_kv_heads, pos + 1, head_size)
            values = self.value_cache[l, :, :pos + 1]
            att = (q @ keys.transpose(0, 2, 1)) * scale  # (n_kv_heads, kv_mul, pos + 1)
            att = np.exp(att - att.max(axis=-1, keepdims=True))
            att /= att.sum(axis=-1, keepdims=True)
            # Output projection and residual connection
            attended = self.wo[l] @ (att @ values).reshape(self.q_dim)
            if layer_norm:
                attended = attended + self.bo[l]
            # GPT-NeoX with use_parallel_residual: both branches read the x this layer began with
            before = x
            x = x + attended

            # FFN: w2(silu(w1(x)) * w3(x)), or GPT-2's w2(gelu(w1(x))), and residual connection
            xb = norm(before if self.parallel_residual else x, self.rms_ffn_weight[l],
                      self.ln_ffn_bias[l] if layer_norm else None)
            hb = self.w1[l] @ xb
            if layer_norm:
                x = x + self.w2[l] @ gelu(hb + self.b1[l]) + self.b2[l]
            else:
                hb = hb / (1.0 + np.exp(-hb)) * (self.w3[l] @ xb)
                x = x + self.w2[l] @ hb

        if not need_logits:
            return None
        # Final norm, then the classifier into logits (60% of all the multiply-adds of stories15M)
        return self.wcls @ norm(x, self.rms_final_weight, self.ln_final_bias)

    def forward_qwen35(self, token, pos, need_logits=True):
        """forward() of a Qwen3.5 (T229), after transformers' Qwen3_5ForCausalLM: every interval-th layer attention
        with an output gate, the others Gated DeltaNet (a causal conv, then a state of k_head x v_head per head of v
        that decays by exp(g) and learns every token's v by the delta rule), every layer then the SwiGLU FFN."""
        form, eps = self.form, self.rms_norm_eps
        interval, conv = form["interval"], form["conv"]
        k_heads, v_heads, k_head, v_head = form["k_heads"], form["v_heads"], form["k_head"], form["v_head"]
        k_dim = k_heads * k_head
        head_size, kv_mul = self.head_size, self.n_heads // self.n_kv_heads
        if pos == 0:
            # every sequence begins with nothing learnt: the conv's last inputs and the states of the delta rule
            linear = self.n_layers - self.n_layers // interval
            self.conv_state = np.zeros((linear, 2 * k_dim + v_heads * v_head, conv - 1), dtype=np.float32)
            self.ssm_state = np.zeros((linear, v_heads, k_head, v_head), dtype=np.float32)
        if pos >= self.key_cache.shape[2]:
            positions = min(max(2 * self.key_cache.shape[2], pos + 1), self.seq_len)
            for name in ("key_cache", "value_cache"):
                cache = getattr(self, name)
                larger = np.zeros((*cache.shape[:2], positions, head_size), dtype=np.float32)
                larger[:, :, :cache.shape[2]] = cache
                setattr(self, name, larger)
        block = form["hadamard"]
        # Prism's rotation (T229): a matrix stored in the rotated basis is fed its input rotated the same way
        turned = (lambda v, signs: hadamard(v * signs, block)) if block else (lambda v, signs: v)
        signs = lambda name: getattr(self, name, None)
        cos, sin = self.freq_cis_real[pos], self.freq_cis_imag[pos]
        x = self.embedding(token)
        if block:
            x = signs("signs_dim") * hadamard(x, block)  # the embedding's rows are stored rotated too
        full = linear = 0
        for l in range(self.n_layers):
            xb = rmsnorm(x, self.rms_att_weight[l], eps)
            xr = turned(xb, signs("signs_dim"))
            if (l + 1) % interval == 0:
                f = full
                full += 1
                q = head_norm(self.wq[f] @ xr, self.q_norm[f], eps)
                k = head_norm(self.wk[f] @ xr, self.k_norm[f], eps)
                gate = self.wg[f] @ xr
                q = partial_rope(q.reshape(-1, head_size), cos, sin, self.rotary).reshape(self.n_kv_heads, kv_mul, head_size)
                self.key_cache[f, :, pos] = partial_rope(k.reshape(-1, head_size), cos, sin, self.rotary)
                self.value_cache[f, :, pos] = (self.wv[f] @ xr).reshape(self.n_kv_heads, head_size)
                keys, values = self.key_cache[f, :, :pos + 1], self.value_cache[f, :, :pos + 1]
                att = (q @ keys.transpose(0, 2, 1)) * np.float32(1.0 / math.sqrt(head_size))
                att = np.exp(att - att.max(axis=-1, keepdims=True))
                att /= att.sum(axis=-1, keepdims=True)
                out = (att @ values).reshape(self.q_dim) / (1.0 + np.exp(-gate))
                x = x + self.wo[f] @ turned(out, signs("signs_q"))
            else:
                r = linear
                linear += 1
                mixed = self.wqkv[r] @ xr
                z = self.wz[r] @ xr
                beta = 1.0 / (1.0 + np.exp(-(self.wb[r] @ xb)))
                g = self.ssm_a[r] * softplus(self.wa[r] @ xb + self.ssm_dt[r])
                # the causal conv over this input and the last conv - 1, then SiLU
                window = np.concatenate([self.conv_state[r], mixed[:, None]], axis=1)
                self.conv_state[r] = window[:, 1:]
                mixed = (window * self.conv[r]).sum(axis=1)
                mixed = mixed / (1.0 + np.exp(-mixed))
                q = l2norm(mixed[:k_dim].reshape(k_heads, k_head)) / np.float32(math.sqrt(k_head))
                k = l2norm(mixed[k_dim:2 * k_dim].reshape(k_heads, k_head))
                v = mixed[2 * k_dim:].reshape(v_heads, v_head)
                # head h of v reads head h // (v_heads / k_heads) of q and k (transformers' repeat_interleave)
                q, k = np.repeat(q, v_heads // k_heads, axis=0), np.repeat(k, v_heads // k_heads, axis=0)
                state = self.ssm_state[r] * np.exp(g)[:, None, None]
                delta = (v - np.einsum("hkv,hk->hv", state, k)) * beta[:, None]
                state += k[:, :, None] * delta[:, None, :]
                self.ssm_state[r] = state
                out = np.einsum("hkv,hk->hv", state, q)
                # every head normalized, times SiLU of its part of z
                zs = z.reshape(v_heads, v_head)
                out = head_norm(out.reshape(-1), self.ssm_norm[r], eps).reshape(v_heads, v_head) * (zs / (1.0 + np.exp(-zs)))
                x = x + self.wout[r] @ turned(out.reshape(-1), signs("signs_v"))
            xb = turned(rmsnorm(x, self.rms_ffn_weight[l], eps), signs("signs_dim"))
            hb = self.w1[l] @ xb
            hb = hb / (1.0 + np.exp(-hb)) * (self.w3[l] @ xb)
            x = x + self.w2[l] @ turned(hb, signs("signs_hidden"))
        if not need_logits:
            return None
        return self.wcls @ turned(rmsnorm(x, self.rms_final_weight, eps), signs("signs_dim"))


    def kernel_sampler(self, kernels):
        """penalize() and sample() on the kernels: the same as the methods below, which stay for NumPy alone.

        Sorting the candidates of the nucleus was most of the time of a step for a small model with a large
        vocabulary, and a repetition penalty, which flattens the distribution, made it worse.
        """
        probabilities, order = np.empty(self.vocab_size, dtype=np.float32), np.empty(self.vocab_size, dtype=np.int32)
        recent = np.empty(REPETITION_WINDOW, dtype=np.int32)
        probabilities_p, order_p, recent_p = probabilities.ctypes.data, order.ctypes.data, recent.ctypes.data
        self._sampler_buffers = (probabilities, order, recent)  # keep them alive: the kernels only know addresses
        kernel_penalize, kernel_sample = kernels["penalize"], kernels["sample"]
        known = {}  # id(logits) -> (logits, address): forward() returns the same array every time

        def address(logits):
            entry = known.get(id(logits))
            if entry is None or entry[0] is not logits:
                if logits.dtype != np.float32 or not logits.flags.c_contiguous:
                    raise TypeError("the kernels need contiguous float32 logits")
                known.clear()
                entry = known[id(logits)] = (logits, logits.ctypes.data)
            return entry[1]

        seen = [None, 0]  # the list that recent[] mirrors, and its length then

        def penalize(logits, history, penalty):
            # recent[] is a ring of the latest tokens. generate() appends one token per step, and then one number
            # is written here: copying 64 of them from a list costs more than the kernel takes
            size = len(history)
            if seen[0] is history and size == seen[1] + 1:
                recent[(size - 1) % REPETITION_WINDOW] = history[-1]
            else:
                for position in range(max(size - REPETITION_WINDOW, 0), size):
                    recent[position % REPETITION_WINDOW] = history[position]
            seen[0], seen[1] = history, size
            kernel_penalize(address(logits), recent_p, min(size, REPETITION_WINDOW), penalty)

        def sample(logits, temperature, topp, rng):
            if temperature == 0.0:
                return Llama.greedy(logits)
            # the random number is drawn here, so that a seed gives the same text again
            token = kernel_sample(address(logits), logits.size, temperature, topp, rng.random(), probabilities_p, order_p)
            if token < 0:
                raise ValueError(NOT_FINITE)
            return token

        return penalize, sample

    def penalize(self, logits, history, penalty):
        """Make the tokens of the last steps less likely: tiny models love to loop."""
        recent = np.unique(history[-REPETITION_WINDOW:])
        logits[recent] = np.where(logits[recent] > 0, logits[recent] / penalty, logits[recent] * penalty)

    @staticmethod
    def greedy(logits):
        """Greedy argmax sampling: take the token with the highest probability. NumPy's argmax takes a NaN for the
        largest, so the logit it picks is finite exactly when the largest one is (T195)."""
        token = int(np.argmax(logits))
        if not math.isfinite(logits[token]):
            raise ValueError(NOT_FINITE)
        return token

    def sample(self, logits, temperature, topp, rng):
        if temperature == 0.0:
            return self.greedy(logits)
        # max() keeps a NaN (T195)
        best = logits.max()
        if not math.isfinite(best):
            raise ValueError(NOT_FINITE)
        nucleus = 0.0 < topp < 1.0
        if nucleus:
            # exp() over a vocabulary of 50000 or 100000 tokens costs as much as half a forward pass, and nearly all
            # of it goes to tokens that cannot be drawn: leave out, while still cheap, whatever is less than a ten
            # millionth as probable as the best token (together far below one percent of the probability mass)
            candidates = np.flatnonzero(logits >= best + temperature * math.log(1e-7))
            probabilities = np.exp((logits[candidates] - best) / temperature).astype(np.float64)
        else:
            candidates = np.arange(logits.size)
            probabilities = np.exp((logits - best) / temperature).astype(np.float64)
        probabilities /= probabilities.sum()
        if nucleus:
            # Top-p (nucleus) sampling: only the most probable tokens whose probabilities add up to topp.
            # Tokens below (1 - topp) / (n - 1) cannot be part of that set (llama2.c), so they need not be sorted.
            # That holds while one token at least stays: when all are below it (n * topp < 1, which the floor above
            # makes possible, T178), the others add up to less than (1 - topp), so the set is the most probable token
            # alone. It always stays
            cutoff = min((1.0 - topp) / max(probabilities.size - 1, 1), probabilities.max())
            likely = np.flatnonzero(probabilities >= cutoff)
            likely = likely[np.argsort(-probabilities[likely])]
            candidates, probabilities = candidates[likely], probabilities[likely]
            cumulative = np.cumsum(probabilities)
            cumulative = cumulative[:np.searchsorted(cumulative, topp) + 1]
        else:
            cumulative = np.cumsum(probabilities)
        # one random number on the cumulative distribution; Generator.choice() would cost a third of a millisecond
        chosen = np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")
        return int(candidates[min(chosen, cumulative.size - 1)])

    def generate(self, prompt="", steps=256, temperature=0.0, topp=0.9, repetition_penalty=1.0, seed=None, echo=True):
        """Yield the text piece by piece, as it is generated. echo=False leaves the prompt out of it (an instruction
        wrapped in a template, which nobody wants to read back)."""
        prompt_tokens = self.tokenizer.encode(prompt, self.specials) if prompt else []
        # Right now we cannot run for more than seq_len steps
        if steps <= 0 or steps > self.seq_len:
            steps = self.seq_len
        if len(prompt_tokens) >= steps:
            raise ValueError(f"The prompt is {len(prompt_tokens)} tokens long, but only {steps - 1} fit.")
        rng = np.random.default_rng(seed)
        # a character can be split over several tokens
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

        self._run += 1
        run = self._run
        token, count, sampled, forced = self.bos, 0, 0, 0
        history = [self.bos]
        start = sampling_start = time.perf_counter()
        first_token = None
        first = 0
        try:
            if self.forward_many is not None and len(prompt_tokens) > 1:
                # T108: the tokens of the prompt that make no logits (all but the last) go through the layers
                # prompt_block() at a time (T147: as many as the engine takes now); the text comes out as it would
                # one by one, only a block at once
                fed = [self.bos] + prompt_tokens[:-1]
                at = 0
                while at < len(fed):
                    # asked again every block: where the GPU gives up, the CPU's blocks are short again (T108)
                    block = fed[at:at + self.prompt_block()]
                    self.forward_many(block, at)
                    for pos in range(at, at + len(block)):
                        next_token = prompt_tokens[pos]
                        text = utf8.decode(self.tokenizer.decode(token, next_token, self.bos))
                        token = next_token
                        history.append(token)
                        count += 1
                        forced += 1
                        if text and echo:
                            yield text
                    at += len(block)
                first = len(fed)
                sampling_start = time.perf_counter()
            # T152: a step on the GPU (generate_many) samples there with the random numbers drawn here, several at
            # once (token_block()); a step on the CPU is the forward pass and the sampling here
            stops = sorted(self.stop_tokens)
            pos = first
            while pos < steps:
                if pos < len(prompt_tokens):
                    # Still processing the prompt: force the next token, and the logits are not needed
                    self.forward(token, pos, need_logits=False)
                    chosen = [prompt_tokens[pos]]
                    forced += 1
                    sampling_start = time.perf_counter()
                else:
                    chosen = None
                    many = min(self.token_block(), steps - pos)
                    if many > 0:
                        # a number for every step, drawn in the order the CPU draws them (none where greedy);
                        # those of the steps after a stop token go unused
                        randoms = [rng.random() for _ in range(many)] if temperature != 0.0 else []
                        chosen = self.generate_many(token, pos, history, many, temperature, topp, repetition_penalty,
                                                    randoms, stops)
                    if chosen is None:
                        logits = self.forward(token, pos)
                        if repetition_penalty != 1.0:
                            self.penalize(logits, history, repetition_penalty)
                        chosen = [self.sample(logits, temperature, topp, rng)]
                ended = False
                for next_token in chosen:
                    if pos >= len(prompt_tokens):
                        sampled += 1
                        if first_token is None:
                            first_token = time.perf_counter()
                        # The BOS token delimits sequences: the story is over
                        if next_token in self.stop_tokens:
                            ended = True
                            break
                    text = utf8.decode(self.tokenizer.decode(token, next_token, self.bos))
                    token = next_token
                    history.append(token)
                    count += 1
                    if text and (echo or pos >= len(prompt_tokens)):
                        yield text
                    pos += 1
                if ended:
                    break
            text = utf8.decode(b"", final=True)
            if text:
                yield text
        finally:
            # an abandoned generator that is finalized late must not overwrite the stats of a newer run
            if run == self._run:
                now = time.perf_counter()
                self.stats = {
                    "tokens": count,
                    "seconds": now - start,
                    # the prompt is not counted: its tokens skip the classifier, so they are much cheaper
                    "tokens_per_second": sampled / (now - sampling_start) if now > sampling_start else 0.0,
                    # The prompt costs a forward pass per token too, and it is what the reader waits for first.
                    # sampled counts the sampling steps, one more than "tokens" shows when a stop token ended it.
                    "sampled": sampled,
                    "prompt_tokens": forced,
                    "prompt_seconds": sampling_start - start,
                    "prompt_tokens_per_second": forced / (sampling_start - start) if sampling_start > start else 0.0,
                    "first_token_seconds": first_token - start if first_token is not None else 0.0,
                }
