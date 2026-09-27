# Byte-level BPE (GPT-2, SmolLM2, Qwen): the pre-tokenizers and the whole path tokenizer.json -> converter ->
# tokenizer.bin -> engine, against Hugging Face's own tokenizers. The vocabularies are trained here in a second,
# so nothing is downloaded and no file is checked in.
import json
import os
import random
import unicodedata

import pytest

from llama2_convert import tokenizer_bin, tokenizer_json_options, tokenizer_json_pieces
from llama2_numpy import Tokenizer, pretokenize
from conftest import CORPUS, TEXTS

tokenizers = pytest.importorskip("tokenizers", reason="pip install tokenizers to check against the real one")

GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def trained(pattern, digits):
    """A small byte-level BPE, and the tokenizer.json it saves."""
    from tokenizers import Tokenizer as Real, decoders, models, pre_tokenizers, trainers
    real = Real(models.BPE())
    steps = ([pre_tokenizers.Digits(individual_digits=True)] if digits else [])
    steps += ([pre_tokenizers.Split(tokenizers.Regex(pattern), behavior="isolated")] if pattern else [])
    steps += [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=not pattern)]
    real.pre_tokenizer = pre_tokenizers.Sequence(steps)
    real.decoder = decoders.ByteLevel()
    real.train_from_iterator([CORPUS] * 8, trainers.BpeTrainer(
        vocab_size=900, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return real, json.loads(real.to_str())


@pytest.mark.parametrize("name, pattern, digits", [
    ("gpt2", None, False), ("gpt2-digits", None, True), ("qwen", QWEN_PATTERN, False)])
@pytest.mark.parametrize("text", TEXTS)
def test_matches_the_real_tokenizer(name, pattern, digits, text):
    real, spec = trained(pattern, digits)
    options = tokenizer_json_options(spec)
    assert options["tokenizer_kind"] == "bytebpe" and options["pretokenizer"] == name
    vocab_size = real.get_vocab_size()
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size,
                     kind="bytebpe", pretokenizer=options["pretokenizer"])
    assert mine.encode(text) == real.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("name, pattern, digits", [
    ("gpt2", None, False), ("gpt2-digits", None, True), ("qwen", QWEN_PATTERN, False)])
def test_decodes_every_piece(name, pattern, digits):
    real, spec = trained(pattern, digits)
    vocab_size = real.get_vocab_size()
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size,
                     kind="bytebpe", pretokenizer=name)
    for id in range(vocab_size):
        # skip_special_tokens=False: the engine hands the page every piece, and stop tokens end the run instead
        assert mine.decode(0, id).decode("utf-8", "replace") == real.decode([id], skip_special_tokens=False)


@pytest.mark.parametrize("text", TEXTS)
def test_pretokenizers_follow_the_patterns(text):
    """The engine runs the patterns on the characters' classes, because re has no \\p{L} (T200)."""
    regex = pytest.importorskip("regex", reason="pip install regex to check the patterns themselves")
    assert pretokenize(text, "gpt2") == regex.findall(GPT2_PATTERN, text)
    assert pretokenize(text, "qwen") == regex.findall(QWEN_PATTERN, text)
    assert pretokenize(text, "gpt2-digits") == digits_then_gpt2(regex, text)


def digits_then_gpt2(regex, text):
    """SmolLM2's pre_tokenizer: Digits(individual_digits) splits off every character Rust's char::is_numeric calls a
    number (\\p{N}: ², Ⅱ and ① too, not only \\d), and GPT-2's pattern runs on each piece (T206)."""
    return [part for chunk in regex.findall(r"\p{N}|\P{N}+", text) for part in regex.findall(GPT2_PATTERN, chunk)]


# T200's review: a class of CharClasses or a translated pattern that is wrong shows only next to the characters it
# concerns, which TEXTS has few of. Characters of every class and every kind the patterns name, each next to each.
# (Not the letters new in the regex module's Unicode, where Python's unicodedata and the regex module disagree.)
# T206: \x1c to \x1f (spaces to str.isspace, not to \s) and ſ (which (?i:'s) folds to s) are in, as the real one has them.
PIECES = ["a", "b", "s", "t", "d", "m", "l", "r", "e", "v", "S", "T", "L", "D", "x", "'", "'s", "'LL", "'ve", "'T",
          " ", "  ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", "\xa0", " ", "　", "0", "7", "123",
          ".", ",", "!", "-", "_", "(", '"', "@", "é", "ß", "Ω", "я", "あ", "カ", "漢", "한", "ｱ", "Ａ", "１", "٣", "²", "Ⅻ",
          "①", "́", "‍", "、", "。", "\U0001f600", "\U00020bb7", "\U0001d7ce", "\U00010140", "’",
          "\x1c", "\x1f", "ſ", "'ſ", "'ſt"]


def test_pretokenizers_follow_the_patterns_on_random_texts():
    regex = pytest.importorskip("regex", reason="pip install regex to check the patterns themselves")
    patterns = {"gpt2": GPT2_PATTERN, "qwen": QWEN_PATTERN,
                "llama3": QWEN_PATTERN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")}
    rng = random.Random(200)
    for _ in range(3000):
        text = "".join(rng.choices(PIECES, k=rng.randrange(0, 16)))
        for name, pattern in patterns.items():
            assert pretokenize(text, name) == regex.findall(pattern, text), (name, text)
        assert pretokenize(text, "gpt2-digits") == digits_then_gpt2(regex, text), text


# T206: the classes against the real pre_tokenizers themselves (Oniguruma's \s, \p{L}, \p{N} and (?i), and Rust's
# char::is_numeric for Digits), at every code point but the surrogates. Each code point c goes into a few places that
# tell its class apart: next to letters, to punctuation, to digits, after a space, and after an apostrophe (for (?i)'s
# folds, of s, t, m, d and of the r, e, l of 're, 've, 'll).
def around(c):
    return f"a{c}a!{c}!1{c}1 {c}'{c}'{c}e'r{c}'{c}l\n"


def real_pretokenizer(name):
    from tokenizers import Regex, pre_tokenizers
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    if name == "gpt2":
        return byte_level
    if name == "gpt2-digits":
        return pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), byte_level])
    pattern = QWEN_PATTERN if name == "qwen" else QWEN_PATTERN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")
    return pre_tokenizers.Split(Regex(pattern), behavior="isolated")


# 17 to 29 s for each of the four on CI's runner, so only the full suite runs it (tests/suite.sh full, T193)
@pytest.mark.skipif(not os.environ.get("EVERY_CODE_POINT"), reason="EVERY_CODE_POINT=1: tests/suite.sh full runs it")
@pytest.mark.parametrize("name", ["gpt2", "gpt2-digits", "qwen", "llama3"])
def test_pretokenizers_split_every_character_as_the_real_ones_do(name):
    real = real_pretokenizer(name)

    def ends(text):
        """Where the real pieces end, and where the engine's do (the real offsets are the text's characters)."""
        theirs = [end for _, (_, end) in real.pre_tokenize_str(text)]
        mine, at = [], 0
        for piece in pretokenize(text, name):
            at += len(piece)
            mine.append(at)
        return theirs, mine

    def differs(code):
        theirs, mine = ends(around(chr(code)))
        return theirs != mine

    codes = [code for code in range(0x110000) if not 0xD800 <= code < 0xE000]
    wrong, alone = [], []
    for start in range(0, len(codes), 4096):
        chunk = codes[start:start + 4096]
        theirs, mine = ends("".join(around(chr(code)) for code in chunk))
        if theirs != mine:  # which of them
            culprits = [code for code in chunk if differs(code)]
            wrong += culprits
            # T206's review: a chunk that splits otherwise where no one of its code points does (the places of two
            # next to each other) fails too, not pass for want of a code point to name
            alone += [] if culprits else [f"U+{chunk[0]:04X} to U+{chunk[-1]:04X}"]
    assert not alone, f"{name}: these chunks split otherwise than the real one, no code point of them alone: {alone}"
    # Not the characters Python's Unicode has not assigned yet (Cn): the real tokenizers' Rust can know them (CI's
    # Python 3.14 and Rust's char::is_numeric of Unicode 17: Tolong Siki's digits U+11DE0 to U+11DE9, U+16FF4 to U+16FF6)
    unassigned = [code for code in wrong if unicodedata.category(chr(code)) == "Cn"]
    if unassigned:
        print(f"{name}: {len(unassigned)} code points unassigned in Python's Unicode "
              f"{unicodedata.unidata_version} split otherwise: " + ", ".join(f"U+{code:04X}" for code in unassigned[:20]))
    wrong = [code for code in wrong if code not in unassigned]
    shown = ", ".join(f"U+{code:04X} ({unicodedata.category(chr(code))})" for code in wrong[:40])
    assert not wrong, f"{name}: {len(wrong)} code points split otherwise than the real one: {shown}"


# T215: SmolLM2's vocabulary has no piece for 21 bytes (0x04, 0x06, 0x13, 0x14, 0x16, 0x1D, 0xC0, 0xC1, 0xF1, 0xF2,
# 0xF5 to 0xFF). Hugging Face's BPE, without byte_fallback or an unk_token, leaves such a byte out and merges its
# neighbours as if it were not there; the engine stopped on a KeyError. These are some of SmolLM2's, and 'z', which the
# corpus has: every piece and merge with one of them goes.
LACKING = "\x04\x1d\xc0\xf1z"


def lacking(pattern, digits, ignore_merges=False):
    """The trained vocabulary without the pieces of LACKING's bytes, as a real tokenizer and a tokenizer.json."""
    from llama2_numpy import BYTE_CHARS
    gone = set(LACKING.encode("utf-8").decode("latin-1").translate(BYTE_CHARS))
    _, spec = trained(pattern, digits)
    model = spec["model"]
    kept = [text for text, _ in sorted(model["vocab"].items(), key=lambda item: item[1]) if not gone & set(text)]
    specials = len(spec["added_tokens"])  # the trainer puts the special tokens first
    assert [token["id"] for token in spec["added_tokens"]] == list(range(specials))
    model["vocab"] = {text: id for id, text in enumerate(kept)}
    merges = [merge if isinstance(merge, list) else merge.split(" ") for merge in model["merges"]]
    model["merges"] = [merge for merge in merges if not gone & set("".join(merge))]
    model["ignore_merges"] = ignore_merges
    assert not model.get("byte_fallback") and model.get("unk_token") is None
    return tokenizers.Tokenizer.from_str(json.dumps(spec)), spec


@pytest.mark.parametrize("name, pattern, digits, ignore_merges", [
    ("gpt2", None, False, False), ("gpt2", None, False, True), ("gpt2-digits", None, True, False),
    ("qwen", QWEN_PATTERN, False, False)])
def test_leaves_out_the_bytes_the_vocabulary_lacks(name, pattern, digits, ignore_merges):
    real, spec = lacking(pattern, digits, ignore_merges)
    options = tokenizer_json_options(spec)
    vocab_size = real.get_vocab_size()
    mine = Tokenizer(tokenizer_bin(list(tokenizer_json_pieces(spec)), vocab_size), vocab_size,
                     kind="bytebpe", pretokenizer=options["pretokenizer"], ignore_merges=options["ignore_merges"])
    # each lacking character alone, between the letters of a word (whose neighbours then merge), at the ends of words
    # and runs of them; U+40000 is 0xF1 0x80 0x80 0x80 (the vocabulary has 0x80), U+0400 is 0xD0 0x80 (it has both)
    texts = [char for char in LACKING] + ["\U00040000", "\u0400"]
    texts += [f"{a}{char}{b}" for char in LACKING + "\U00040000" for a, b in
              (("Th", "e quick"), ("the", " end"), (" ", "dog"), ("1", "23"), ("it", "'s"), ("\n", "\n"), ("", ""))]
    texts += [LACKING * 3, "lazy dog, lazzy, zz z", "The" + LACKING + "quick", "a\x04\x04b\x1d\x1dc"]
    texts += [text[:k] + LACKING[k % len(LACKING)] + text[k:] for text in TEXTS for k in range(0, len(text) + 1, 3)]
    wrong = [text for text in texts if mine.encode(text) != real.encode(text, add_special_tokens=False).ids]
    assert not wrong, f"{len(wrong)} of {len(texts)} texts differ, as {wrong[0]!r}"


def test_refuses_what_the_engine_cannot_split():
    with pytest.raises(ValueError, match="does not know"):
        tokenizer_json_options({"model": {"type": "BPE"}, "pre_tokenizer": {"type": "Whitespace"}})
    with pytest.raises(ValueError, match="adds a space"):
        tokenizer_json_options({"model": {"type": "BPE"},
                                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True}})
