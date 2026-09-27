# Byte-level BPE (GPT-2, SmolLM2, Qwen): the pre-tokenizers and the whole path tokenizer.json -> converter ->
# tokenizer.bin -> engine, against Hugging Face's own tokenizers. The vocabularies are trained here in a second,
# so nothing is downloaded and no file is checked in.
import json
import random

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
    digits = [part for chunk in regex.findall(r"\d|\D+", text) for part in regex.findall(GPT2_PATTERN, chunk)]
    assert pretokenize(text, "gpt2-digits") == digits


# T200's review: a class of CharClasses or a translated pattern that is wrong shows only next to the characters it
# concerns, which TEXTS has few of. Characters of every class and every kind the patterns name, each next to each.
# (Not the characters where Python's unicodedata and the regex module disagree: \x1c to \x1f are spaces to
# str.isspace and not to \s, and letters new in the regex module's Unicode; nor ſ, which (?i:'s) folds to s.)
PIECES = ["a", "b", "s", "t", "d", "m", "l", "r", "e", "v", "S", "T", "L", "D", "x", "'", "'s", "'LL", "'ve", "'T",
          " ", "  ", "\t", "\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", "\xa0", " ", "　", "0", "7", "123",
          ".", ",", "!", "-", "_", "(", '"', "@", "é", "ß", "Ω", "я", "あ", "カ", "漢", "한", "ｱ", "Ａ", "１", "٣", "²", "Ⅻ",
          "①", "́", "‍", "、", "。", "\U0001f600", "\U00020bb7", "\U0001d7ce", "\U00010140", "’"]


def test_pretokenizers_follow_the_patterns_on_random_texts():
    regex = pytest.importorskip("regex", reason="pip install regex to check the patterns themselves")
    patterns = {"gpt2": GPT2_PATTERN, "qwen": QWEN_PATTERN,
                "llama3": QWEN_PATTERN.replace(r"|\p{N}|", r"|\p{N}{1,3}|")}
    rng = random.Random(200)
    for _ in range(3000):
        text = "".join(rng.choices(PIECES, k=rng.randrange(0, 16)))
        for name, pattern in patterns.items():
            assert pretokenize(text, name) == regex.findall(pattern, text), (name, text)
        digits = [part for chunk in regex.findall(r"\d|\D+", text) for part in regex.findall(GPT2_PATTERN, chunk)]
        assert pretokenize(text, "gpt2-digits") == digits, text


def test_refuses_what_the_engine_cannot_split():
    with pytest.raises(ValueError, match="does not know"):
        tokenizer_json_options({"model": {"type": "BPE"}, "pre_tokenizer": {"type": "Whitespace"}})
    with pytest.raises(ValueError, match="adds a space"):
        tokenizer_json_options({"model": {"type": "BPE"},
                                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True}})
