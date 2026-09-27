# Encode -> decode round trips, on a synthetic vocabulary and on the real ones `make models` fetches.
import math
import re
import random
import unicodedata

import pytest

from conftest import checkpoint_vocab_size, detokenize, model_file, pack_tokenizer, tiny_tokenizer
from llama2_numpy import Tokenizer

TEXTS = [
    "hello world",
    "これからの流行りは、猫です",   # Japanese
    "sushi \U0001f363 and \U0001f363\U0001f363",                 # emoji, outside the vocabulary
    "\U00030EDE ಠ_ಠ",                                  # rare characters, byte fallback only
    "tabs\tand\nnewlines  and   spaces",
    "a",
    "éèê",
]


def roundtrip(tokenizer, text):
    tokens = tokenizer.encode(text)
    assert tokens, "encode produced nothing"
    return detokenize(tokenizer, tokens)


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
@pytest.mark.parametrize("text", TEXTS)
def test_synthetic_roundtrip(kind, text):
    tokenizer = tiny_tokenizer(kind=kind)
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
def test_nfkc_normalizes(kind):
    tokenizer = tiny_tokenizer(kind=kind, nfkc=True)
    assert roundtrip(tokenizer, "Ａ") == unicodedata.normalize("NFKC", "Ａ") == "A"


def test_byte_fallback_spells_out_unknown_characters():
    tokenizer = tiny_tokenizer(kind="bpe")
    tokens = tokenizer.encode("\U0001f363")
    # four bytes, none of which is a real piece
    assert [tokenizer.vocab[token] for token in tokens[1:]] == [b"<0xF0>", b"<0x9F>", b"<0x8D>", b"<0xA3>"]


def test_unigram_prefers_the_longest_well_scored_piece():
    tokenizer = tiny_tokenizer(kind="unigram")
    # " 流行り" scores better than " " + "流" + "行" + "り"
    assert [tokenizer.vocab[token] for token in tokenizer.encode("流行り")] == [" 流行り".encode("utf-8")]


def test_scores_below_the_unmatchable_threshold_are_never_used():
    tokenizer = tiny_tokenizer(kind="unigram")
    unmatchable = {i for i, score in enumerate(tokenizer.scores) if score <= Tokenizer.UNMATCHABLE}
    tokens = tokenizer.encode("hello world")
    assert not unmatchable.intersection(tokens)


# --------------------------------------------------------------------------- the real vocabularies

def real_tokenizer(name, vocab_size, kind="bpe", nfkc=False):
    return Tokenizer(model_file(name).read_bytes(), vocab_size, kind=kind, nfkc=nfkc)


@pytest.mark.parametrize("text", TEXTS)
def test_llama2_bpe_roundtrip(text):
    tokenizer = real_tokenizer("tokenizer.bin", checkpoint_vocab_size("stories15M.bin"))
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("text", TEXTS)
def test_tok512_bpe_roundtrip(text):
    tokenizer = real_tokenizer("tok512.bin", checkpoint_vocab_size("stories260K.bin"))
    assert roundtrip(tokenizer, text) == text


@pytest.mark.parametrize("text", TEXTS)
def test_tiny_lm_unigram_roundtrip(text):
    tokenizer = real_tokenizer("tiny-lm.tokenizer.bin", checkpoint_vocab_size("tiny-lm.bin"),
                               kind="unigram", nfkc=True)
    assert roundtrip(tokenizer, text) == unicodedata.normalize("NFKC", text)


def test_llama2_bpe_matches_sentencepiece_on_a_known_sentence():
    tokenizer = real_tokenizer("tokenizer.bin", checkpoint_vocab_size("stories15M.bin"))
    # the ids llama2.c prints for this prompt (vocabulary of Llama 2)
    assert tokenizer.encode("Once upon a time") == [9038, 2501, 263, 931]


def test_special_tokens_inside_a_prompt_become_their_token():
    tokenizer = tiny_tokenizer("bpe")
    end = tokenizer.index[b"</s>"]
    tokens = tokenizer.encode("hello</s>\nworld", ("</s>",))
    assert tokens.count(end) == 1
    before, after = tokens[:tokens.index(end)], tokens[tokens.index(end) + 1:]
    assert before == tokenizer.encode("hello")
    # no dummy space after a special token: "\nworld", not " \nworld"
    assert after == tokenizer.encode("x\nworld")[len(tokenizer.encode("x")):]
    # without being told, the engine spells the same characters out
    assert end not in tokenizer.encode("hello</s>\nworld")
    assert tokenizer.encode("hello", ("</s>",)) == tokenizer.encode("hello")


# ---------------------------------------------------------- the definitions, on made-up vocabularies (T200)
# encode_bpe() keeps the pairs in a heap from one merge to the next (T207), and encode_unigram() tries a piece at i only as far
# as the reach of text[i:i + 2]. Both must give what the plain definitions below give (the engine's own loops before
# T200), with ties, a piece written twice, unmatchable scores, NaN and characters the vocabulary lacks. A roundtrip
# does not see a wrong segmentation (any segmentation decodes to the text), and "Once upon a time" has four tokens.

def bpe_by_definition(tokenizer, text):
    """llama2.c's encode(): the characters' tokens (their bytes where the vocabulary lacks them), then the best pair
    merged again and again, the first of equal scores, and no pair that scores -1e10 or less"""
    tokens = []
    for char in text:
        piece = char.encode("utf-8")
        if piece in tokenizer.index:
            tokens.append(tokenizer.index[piece])
        elif tokenizer.unknown is not None:
            tokens += [] if tokens[-1:] == [tokenizer.unknown] else [tokenizer.unknown]
        else:
            tokens += [tokenizer.byte_tokens[byte] for byte in piece]
    while True:
        best = (-1e10, -1, -1)
        for i in range(len(tokens) - 1):
            id = tokenizer.index.get(tokenizer.vocab[tokens[i]] + tokenizer.vocab[tokens[i + 1]])
            if id is not None and tokenizer.scores[id] > best[0]:
                best = (tokenizer.scores[id], i, id)
        if best[1] == -1:
            return tokens
        tokens[best[1]:best[1] + 2] = [best[2]]


def unigram_by_definition(tokenizer, text):
    """the segmentation of the best total score, every piece up to the longest tried at every position"""
    longest = max(len(piece.decode("utf-8", "ignore")) for piece in tokenizer.vocab)
    n = len(text)
    best, back = [0.0] + [-math.inf] * n, [None] * (n + 1)
    for i in range(n):
        if best[i] == -math.inf:
            continue
        for j in range(i + 1, min(n, i + longest) + 1):
            id = tokenizer.index.get(text[i:j].encode("utf-8"))
            if id is not None and tokenizer.scores[id] > Tokenizer.UNMATCHABLE and best[i] + tokenizer.scores[id] > best[j]:
                best[j], back[j] = best[i] + tokenizer.scores[id], (i, [id])
        if back[i + 1] is None or back[i + 1][0] != i:  # a character the vocabulary lacks
            score = best[i] + tokenizer.unknown_score
            if score > best[i + 1]:
                spelled = [tokenizer.unknown] if tokenizer.unknown is not None else \
                    [tokenizer.byte_tokens[byte] for byte in text[i].encode("utf-8")]
                best[i + 1], back[i + 1] = score, (i, spelled)
    tokens, j = [], n
    while j > 0:
        i, ids = back[j]
        tokens[:0] = [] if tokenizer.unknown is not None and ids == [tokenizer.unknown] and tokens[:1] == ids else ids
        j = i
    return tokens


def made_up(seed, kind):
    """a vocabulary of a few characters (one to four bytes) and pieces of them, with scores that tie"""
    rng = random.Random(seed)
    letters = rng.sample(["a", "b", "c", " ", "é", "日", "本", "\U0001f600"], rng.randrange(2, 7))
    scores = [0.0, -1.0, -1.0, -2.0, -3.0, -1e10, -2e10, float("nan"), float("-inf")]
    rows = [(-1e9, b"<0x%02X>" % byte) for byte in range(256)]  # byte pieces, unmatchable as the converter writes them
    rows += [(0.0, letters[0].encode("utf-8"))] + [(rng.choice(scores), ch.encode("utf-8")) for ch in letters[1:]
                                                   if rng.random() < 0.8]
    for _ in range(rng.randrange(3, 50)):
        piece = "".join(rng.choices(letters, k=rng.randrange(2, 7))).encode("utf-8")
        rows.append((rng.choice(scores) if rng.random() < 0.5 else -float(rng.randrange(6)), piece))
    rows += [(rng.choice(scores), rows[rng.randrange(256, len(rows))][1]) for _ in range(3)]  # pieces written twice
    tokenizer = Tokenizer(pack_tokenizer(rows), len(rows), kind=kind, unknown=rng.choice([None, 256]))
    return tokenizer, letters + ["x"], rng


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
def test_made_up_vocabularies_follow_the_definition(kind):
    by_definition = bpe_by_definition if kind == "bpe" else unigram_by_definition
    for seed in range(300):
        tokenizer, letters, rng = made_up(seed, kind)
        encode = tokenizer.encode_bpe if kind == "bpe" else tokenizer.encode_unigram
        for _ in range(25):
            text = "".join(rng.choices(letters, k=rng.randrange(0, 30)))
            assert encode(text) == by_definition(tokenizer, text), (seed, text)


# ------------------------------------------------------------------ a sentencepiece model's normalizer (T126)

def sentencepiece(normalizer, spelled=False):
    """A sentencepiece unigram model with no byte pieces, as rinna's are, and piece 13 a word: byte + 3 for a
    newline, which is what the engine spelled one with before it read the normalizer (rinna's 13 is った).
    spelled: with a byte piece, as Llama's, Mistral's and tiny-lm's have them (256 of them there)."""
    from make_hf_fixture import field
    NORMAL, UNKNOWN, CONTROL, BYTE = 1, 2, 3, 6
    words = ["▁", "▁a", "▁b", "a", "b", "▁x", "x", "y", "z", "c", "った"]
    pieces = [("[UNK]", UNKNOWN), ("<s>", CONTROL), ("</s>", CONTROL)] + [(word, NORMAL) for word in words]
    pieces += [("<0x0A>", BYTE)] if spelled else []
    assert pieces[13][0] == "った"
    model = b"".join(field(1, field(1, text.encode()) + field(2, -1.0 - i / 10) + field(3, kind))
                     for i, (text, kind) in enumerate(pieces))
    return model + field(2, field(3, 1)) + field(3, normalizer), len(pieces)


def test_an_nmt_normalizer_makes_newlines_and_tabs_spaces_and_one_space_of_many():
    """The review of T126: rinna's three models are nmt_nfkc with remove_extra_whitespaces, and a newline was
    written as token 13 (った in japanese-gpt-1b) where sentencepiece writes ▁. The converter says so to the engine."""
    from make_hf_fixture import field
    from llama2_convert import sentencepiece_options, sentencepiece_pieces, tokenizer_bin
    model, size = sentencepiece(field(1, b"nmt_nfkc"))  # remove_extra_whitespaces unset: true, sentencepiece's default
    options = sentencepiece_options(model)
    assert options == {"tokenizer_kind": "unigram", "nfkc": True, "nmt": True, "collapse": True, "unknown": 0}
    data = tokenizer_bin(sentencepiece_pieces(model), size)
    tokenizer = Tokenizer(data, size, kind="unigram", nfkc=True, nmt=True, collapse=True, unknown=0)
    pieces = lambda text: [tokenizer.vocab[token].decode() for token in tokenizer.encode(text)]
    for text in ["a\nb", "a\tb", "a  b", " a\r\n\n b ", "a​b", "a\x01\nb"]:
        assert pieces(text) == [" a", " b"], text
    assert pieces("\n") == [] and pieces("x　y") == [" x", " ", "y"]
    # characters the vocabulary lacks: the unknown piece, one for a run of them, as sentencepiece writes them
    assert pieces("a\U00020BB7\U00020BB7b") == [" a", "[UNK]", "b"] and pieces("a \U00020BB7 b") == [" a", " ", "[UNK]", " b"]
    bpe = Tokenizer(data, size, kind="bpe", nfkc=True, nmt=True, collapse=True, unknown=0)  # rinna's 1B is BPE
    assert [bpe.vocab[token].decode() for token in bpe.encode("a\U00020BB7\U00020BB7\nb")] == [" a", "[UNK]", " b"]
    before = Tokenizer(data, size, kind="unigram", nfkc=True)
    assert 13 in before.encode("a\nb"), "without the normalizer's settings: byte + 3"


def test_an_identity_normalizer_without_collapsing_changes_nothing():
    """Llama's and Mistral's tokenizer.model: identity, remove_extra_whitespaces off; tiny-lm's: nfkc, off."""
    from make_hf_fixture import field
    from llama2_convert import sentencepiece_options
    for name, nfkc in ((b"identity", False), (b"nfkc", True)):
        model, _ = sentencepiece(field(1, name) + field(4, 0), spelled=True)
        assert sentencepiece_options(model) == {"tokenizer_kind": "unigram", "nfkc": nfkc}


def test_u2581_in_the_text_is_a_space():
    """T216: sentencepiece writes a space as U+2581 before it looks the pieces up, so a U+2581 the text has is a space
    to it (Llama 2, tiny-lm, llm-jp); the engine spelled it as a character the vocabulary lacks. remove_extra_whitespaces
    sees spaces only, so it does not collapse U+2581 unless the normalizer made it a space first (nmt: rinna's)."""
    from make_hf_fixture import field
    from llama2_convert import sentencepiece_pieces, tokenizer_bin
    for normalizer, settings in ((field(1, b"identity") + field(4, 0), {}), (field(1, b"nfkc") + field(4, 0), {"nfkc": True}),
                                 (field(1, b"nmt_nfkc"), {"nfkc": True, "nmt": True, "collapse": True, "unknown": 0}),
                                 (field(1, b"identity"), {"collapse": True})):
        model, size = sentencepiece(normalizer, spelled=True)
        data = tokenizer_bin(sentencepiece_pieces(model), size)
        collapsed = (lambda text: re.sub(" {2,}", " ", text).strip(" ")) if settings.get("collapse") else (lambda text: text)
        for kind in ("bpe", "unigram"):
            tokenizer = Tokenizer(data, size, kind=kind, **settings)
            plain = Tokenizer(data, size, kind=kind, **{**settings, "collapse": False})
            for text in ["a\u2581b", "\u2581a", "\u2581", "a\u2581\u2581b", "a \u2581b", "a  \u2581  b", "x\u2581",
                         "\u2581\u2581x\u2581 \u2581y", " \u2581 "]:
                # nmt makes U+2581 a space before the collapse; otherwise the collapse leaves it alone
                spaced = (collapsed(text.replace("\u2581", " ")) if settings.get("nmt")
                          else collapsed(text).replace("\u2581", " "))
                assert tokenizer.encode(text) == plain.encode(spaced), (settings, kind, text)
    llama = real_tokenizer("tokenizer.bin", checkpoint_vocab_size("stories15M.bin"))
    assert llama.encode("Once\u2581upon a\u2581time") == [9038, 2501, 263, 931]
    tiny = real_tokenizer("tiny-lm.tokenizer.bin", checkpoint_vocab_size("tiny-lm.bin"), kind="unigram", nfkc=True)
    assert tiny.encode("\u2581日本\u2581語") == tiny.encode(" 日本 語")


def test_the_conversion_hands_the_normalizer_to_the_engine():
    """T72's lesson: what the file does not say reaches the engine through the conversion's options"""
    import json
    import struct
    from conftest import synthetic_weights
    from make_hf_fixture import field
    from test_convert import hugging_face, safetensors_file
    import llama2_convert
    from llama2_numpy import Llama

    settings, weights = synthetic_weights()
    tensors, published = hugging_face(settings, weights, True)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    model, pieces = sentencepiece(field(1, b"nmt_nfkc"))
    model += b"".join(field(1, field(1, f"▁w{i}".encode()) + field(2, -9.0) + field(3, 1)) for i in range(settings["vocab_size"] - pieces))
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(published), model,
                                           "tokenizer.model", dtype="float32", max_seq_len=settings["seq_len"], start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    assert conversion.options["nmt"] and conversion.options["collapse"] and conversion.options["unknown"] == 0
    options = {key: value for key, value in conversion.options.items() if key != "dtype"}
    llama = Llama(bytes(conversion.checkpoint), bytes(conversion.tokenizer), **options)
    assert llama.tokenizer.encode("a\n\tb") == llama.tokenizer.encode("a b")
