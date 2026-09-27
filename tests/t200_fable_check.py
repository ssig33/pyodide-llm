# Fable's independent reading of T200 (2026-09-27), run on CI only (tests.yml extra=): not for the main line.
#
# Compares encode() of main before T200 (tests/t200_fable/old_numpy.py, 50f1ac4), of the working copy
# (public/llama2_numpy.py), of T206 (t206_numpy.py, branch t206-onig) and of T207 (t207_numpy.py, branch t207-bpe-heap)
# on adversarial texts (every class of character next to every other, contractions in every case, ſ, K, the C0 and C1
# controls, every kind of space, NFKC compatibility characters, combining marks, emoji sequences, long runs), on the
# seven real vocabularies encode-bench.mjs takes; on made-up BPE and unigram vocabularies of its own (scores +inf, -0.0,
# a byte piece two pieces can make, a piece written twice whose first copy is unmatchable); and lone surrogates. Then
# the working copy and T206 against the real tokenizers (Hugging Face tokenizers, sentencepiece) on the same texts,
# with the smallest text of each difference.
#
#   pip install sentencepiece; python tests/t200_fable_check.py
import importlib.util
import json
import os
import random
import struct
import sys
import time
import unicodedata
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "public"))
import llama2_numpy as new  # noqa: E402
import llama2_convert as convert  # noqa: E402


def module(name):
    path = os.path.join(ROOT, "tests", "t200_fable", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old, t206, t207 = module("old_numpy"), module("t206_numpy"), module("t207_numpy")
MODULES = {"old": old, "new": new, "t206": t206, "t207": t207}
CACHE = os.path.join(ROOT, ".tmp", "t200-fable", "hf")
HF = "https://huggingface.co/{repo}/resolve/{rev}/{file}"


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    target = os.path.join(CACHE, name)
    if not os.path.exists(target):
        with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as f:
            f.write(r.read())
    return target


# ------------------------------------------------------------------------------------------------ the texts
ADVERSARIAL = [
    "", " ", "  ", "\n", "\r\n", "\n\n\n", " \n", "\n ", "\t", " \t ", "a", "A", "'", "'s", "'S", "'ſ", "'ſt", "'re", "'RE",
    "'rE", "it's", "IT'S", "it'S", "don't", "we'll", "WE'LL", "I'm", "I'M", "they've", "I'd", "x'y", "''", "'''", "'s's",
    "'ſ'ſ", "a'ſb", "hello world", "  hello   world  ", "hello\nworld", "hello\r\nworld", "hello \n world", "\n\nhello",
    "hello\n\n", "hello \n\n world", " \n \n ", "123", "1234567", " 123", "a1b2c3", "１２３", "٣٤٥", "²³", "Ⅻ", "①②", "𝟎𝟏",
    "𐄀", "1,234.56", "3.14e-10", "日本語のテキスト", " 日本語", "日本語 English 混在", "漢字カタカナひらがな", "ｶﾀｶﾅ", "㍿",
    "ﬁ", "ﬃ", "①", "Ⅸ", "㈱", "é", "́", "áb", "각", "가", "😀", "👍🏽", "👨‍👩‍👧‍👦", "🇯🇵",
    "‍", "​", "﻿", "﻿hello", "\x00", "a\x00b", "\x7f", "\x85", "a\x85b", "\xa0", "a\xa0b", "　",
    " ", " ", "᠎", " ", " ", "\x1c", "a\x1cb", "\x1c\x1d\x1e\x1f", " \x1c ", "\x0b", "\x0c",
    "\x0b\x0c", "", "￿", "\U0010ffff", "\U000e0001", "\U0001d7ce", "\U00016ff4", "\U00011de0", "\U00020bb7",
    "K", "İ", "ı", "ſ", "ß", "ẞ", "ǅ", "ǈ", "µ", "μ", "Ω", "ω", "ﬅ", "ﬆ", "<s>", "</s>", "<unk>", "<0x41>", "▁", "▁a",
    "a▁b", "Ġ", "Ċ", "Ġhello", "a" * 300, " " * 300, "あ" * 300, "1" * 300, "ab" * 200, "\n" * 200, "'s" * 100,
    "😀" * 100, "the quick brown fox jumps over the lazy dog", "The Quick Brown Fox", "THE QUICK BROWN FOX",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "http://example.com/path?a=1&b=2#frag",
    "user@example.com", "if (x < 0) { return -1; } // comment", "def f(x):\n    return x\n", "\tindented\n\t\tmore\n",
    "ｈｅｌｌｏ　ｗｏｒｌｄ", "Ａ１ａ", "アイウエオ", "가나다", "מה שלומך", "مرحبا بالعالم", "Привет, мир", "Ελληνικά", "हिन्दी",
    "ไทย", "中文测试", "改行\nを含む", "タブ\tを含む", "  前後に空白  ", "　全角空白　",
]
ALPHABET = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789") * 2 + [" "] * 30
            + ["\n"] * 4 + ["\r\n", "\r", "\t", "\x0b", "\x0c", "\x85", "\xa0", "　", " ", " ", "​",
                            "‍", "﻿", "\x1c", "\x1d", "\x1e", "\x1f", "\x00", "\x7f", "\x01", "᠎"]
            + list("'.,!?;:\"()[]{}<>-_=+*/\\|@#$%^&~`")
            + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", "'ſ", "'ſt", " '", "' "]
            + list("éèêëàâäïîôöùûüçñøåæœßÉÑÇ") + list("ſẞǅǈµμΩωKİı") + list("ⅫⅣ²³½①⑩０１９٣٤൧๑")
            + ["\U0001d7ce", "\U0001d7d0", "\U00011de0", "\U00016ff4"]
            + list("一二三の日本語カタカナひらがなｶﾀｶﾅﾞﾟ漢字ー、。「」・〜～") + list("가나다각") + list("абвгд") + list("αβγ")
            + list("مرحبا") + list("הא") + list("ไทย") + list("हिन्दी")
            + ["😀", "👍🏽", "👨‍👩‍👧‍👦", "🇯🇵", "𠮷", "\U00020bb7", "é", "́", "゙", "ﬁ", "㍿", "㈱", "ｈ", "Ａ"]
            + ["▁", "Ġ", "Ċ", "<", ">", "|", "<s>", "</s>", "<unk>", "<0x41>"])
SURROGATES = ["\ud800", "a\ud800", "\ud800a", "a\udfffb", "😀\ud83d", " \ud800 ", "😀", "日本\udc00語"]


def texts_of(seed):
    rng = random.Random(seed)
    texts = list(ADVERSARIAL)
    texts += ["".join(rng.choice(ALPHABET) for _ in range(rng.randrange(0, 60))) for _ in range(4000)]
    texts += ["".join(rng.choice(ALPHABET) for _ in range(rng.randrange(60, 400))) for _ in range(200)]
    plain = list("abcdefghij ") + list("日本語の") + ["\n", "'s", "12"]
    texts += ["".join(rng.choice(plain) for _ in range(2000)) for _ in range(6)]
    return texts


# ------------------------------------------------------------------------------------------- the tokenizers
def made(name, path, options=None):
    """tokenizer.bin, the options and the specials of a tokenizer file, as the page's converter makes them"""
    data = open(path, "rb").read()
    if name.endswith(".json"):
        parsed = json.loads(data)
        pieces = list(convert.tokenizer_json_pieces(parsed))
        specials = [t["content"] for t in parsed.get("added_tokens", []) if t.get("special")]
        return convert.tokenizer_bin(pieces, len(pieces)), convert.tokenizer_json_options(parsed), specials
    if name.endswith(".model"):
        pieces = list(convert.sentencepiece_pieces(data))
        return (convert.tokenizer_bin(pieces, len(pieces)), convert.sentencepiece_options(data),
                convert.sentencepiece_specials(data))
    return data, dict(options or {}), ["</s>", "<s>"]


def build(mod, data, options):
    count, offset = 0, 4
    while offset < len(data):
        offset += 8 + struct.unpack_from("<i", data, offset + 4)[0]
        count += 1
    keys = ("nfkc", "nfc", "pretokenizer", "ignore_merges", "nmt", "collapse", "unknown")
    return mod.Tokenizer(data, count, kind=options.get("tokenizer_kind", "bpe"), **{k: options[k] for k in keys if k in options})


def real_hf(path):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(path)
    return lambda text: tk.encode(text, add_special_tokens=False).ids


def real_sp(path):
    try:
        import sentencepiece as spm
    except ImportError:
        return None
    sp = spm.SentencePieceProcessor(model_file=path)
    return lambda text: sp.encode(text, out_type=int)


def outcome(f, text):
    try:
        return f(text)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}"


def shrink(text, differs):
    """the smallest text (by dropping characters) that still differs"""
    changed = True
    while changed:
        changed = False
        for i in range(len(text)):
            t = text[:i] + text[i + 1:]
            if differs(t):
                text, changed = t, True
                break
    return text


def t206_explained(text, options):
    """T206 changes the IDs only of texts with \\x1c-\\x1f, an apostrophe before ſ (qwen and llama3), or a number that is
    not \\d before/after Digits (gpt2-digits)."""
    if any("\x1c" <= c <= "\x1f" for c in text):
        return True
    if options.get("pretokenizer") in ("qwen", "llama3") and "'ſ" in text:
        return True
    if options.get("pretokenizer") == "gpt2-digits" and any(unicodedata.category(c)[0] == "N" and not c.isdecimal() for c in text):
        return True
    return False


def compare(label, data, options, specials, real, texts):
    toks = {name: build(mod, data, options) for name, mod in MODULES.items()}
    specials = [s for s in specials if s and s.encode("utf-8") in toks["old"].index][:4]
    print(f"\n== {label}: {len(toks['old'].vocab)} pieces, options {options}, specials {specials}")
    began = time.perf_counter()
    unexpected, t206_diffs, unexplained = [], [], []
    for text in texts:
        for named in ((), tuple(specials)):
            ids = {name: outcome(lambda t, tk=tk: tk.encode(t, named), text) for name, tk in toks.items()}
            if ids["new"] != ids["old"]:
                unexpected.append(("new", text, named, ids["old"], ids["new"]))
            if ids["t207"] != ids["old"]:
                unexpected.append(("t207", text, named, ids["old"], ids["t207"]))
            if ids["t206"] != ids["old"]:
                t206_diffs.append(text)
                if not t206_explained(text, options):
                    unexplained.append((text, named, ids["old"], ids["t206"]))
    print(f"  engine against engine: {len(texts)} texts x {2 if specials else 1} in {time.perf_counter() - began:.1f} s: "
          f"new == old and t207 == old: {'yes' if not unexpected else f'NO ({len(unexpected)})'}; "
          f"t206 differs from old on {len(t206_diffs)} texts, {len(unexplained)} of them not by the three T206 changes")
    for which, text, named, a, b in sorted(unexpected, key=lambda u: len(u[1]))[:5]:
        small = shrink(text, lambda t: outcome(lambda s: toks["old"].encode(s, named), t) != outcome(lambda s: toks[which].encode(s, named), t))
        print(f"  {which} != old: {small!r} (specials {named}): old {outcome(lambda s: toks['old'].encode(s, named), small)} "
              f"{which} {outcome(lambda s: toks[which].encode(s, named), small)}")
    for text, named, a, b in sorted(unexplained, key=lambda u: len(u[0]))[:5]:
        small = shrink(text, lambda t: outcome(lambda s: toks["old"].encode(s, named), t) != outcome(lambda s: toks["t206"].encode(s, named), t))
        print(f"  t206 != old, not explained: {small!r}: old {outcome(lambda s: toks['old'].encode(s, named), small)} "
              f"t206 {outcome(lambda s: toks['t206'].encode(s, named), small)}")
    # lone surrogates: the outcome of each
    table = {}
    for text in SURROGATES:
        for name, tk in toks.items():
            got = outcome(tk.encode, text)
            table.setdefault(name, []).append(got if isinstance(got, str) else f"{len(got)} ids")
    print("  lone surrogates (" + ", ".join(repr(t) for t in SURROGATES) + "):")
    for name, got in table.items():
        print(f"    {name}: {', '.join(got)}")
    # the real tokenizer
    if real is None:
        print("  real: none (sentencepiece not installed?)")
        return len(unexpected), len(unexplained)
    for name in ("new", "t206"):
        diffs = []
        for text in texts:
            theirs = outcome(real, text)
            mine = outcome(toks[name].encode, text)
            if theirs != mine:
                diffs.append(text)
        print(f"  {name} against the real tokenizer: {len(diffs)} of {len(texts)} texts differ")
        shown = set()
        for text in sorted(diffs, key=len)[:40]:
            small = shrink(text, lambda t: outcome(real, t) != outcome(toks[name].encode, t))
            if small in shown:
                continue
            shown.add(small)
            print(f"    {small!r}: real {outcome(real, small)} {name} {outcome(toks[name].encode, small)}")
            if len(shown) >= 12:
                break
    return len(unexpected), len(unexplained)


# ------------------------------------------------------------------ made-up vocabularies of Fable's own (BPE, unigram)
def pack(rows):
    out = [struct.pack("<i", max(len(text) for _, text in rows))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in rows]
    return b"".join(out)


def made_up(seed, kind):
    rng = random.Random(seed)
    letters = rng.sample(["a", "b", "c", " ", "<", "0", "x", "4", "1", ">", "é", "日", "\U0001f600"], rng.randrange(3, 10))
    scores = [0.0, -0.0, 1.0, -1.0, -1.0, -2.0, -5e9, -1e10, -2e10, float("inf"), float("nan"), float("-inf"), -1e8, -1e8 - 1]
    rows = [(-1e9, b"<0x%02X>" % byte) for byte in range(256)]
    rows += [(rng.choice(scores) if rng.random() < 0.5 else 0.0, ch.encode("utf-8")) for ch in letters if rng.random() < 0.85]
    for _ in range(rng.randrange(3, 60)):
        piece = "".join(rng.choices(letters, k=rng.randrange(2, 12))).encode("utf-8")
        rows.append((rng.choice(scores) if rng.random() < 0.6 else -float(rng.randrange(4)), piece))
    rows += [(rng.choice(scores), rows[rng.randrange(256, len(rows))][1]) for _ in range(4)]  # written twice
    if rng.random() < 0.5:  # a byte piece two pieces can make (its -1e9 is above encode_bpe's -1e10)
        rows += [(-1.0, b"<0x"), (-1.0, b"41>"), (-1.0, b"<"), (-1.0, b"0x41>")]
    rng.shuffle(rows)
    unknown = rng.choice([None, next(i for i, (_, t) in enumerate(rows) if t == b"<0x00>")])
    return pack(rows), {"tokenizer_kind": kind, **({"unknown": unknown} if unknown is not None else {})}, letters + ["z"], rng


def made_up_check():
    began, checked, bad = time.perf_counter(), 0, []
    for kind in ("bpe", "unigram"):
        for seed in range(400):
            data, options, letters, rng = made_up(seed, kind)
            toks = {name: build(mod, data, options) for name, mod in MODULES.items()}
            for _ in range(40):
                text = "".join(rng.choices(letters, k=rng.randrange(0, 40)))
                ids = {name: outcome(tk.encode, text) for name, tk in toks.items()}
                checked += 1
                for name in ("new", "t206", "t207"):
                    if ids[name] != ids["old"]:
                        bad.append((kind, seed, name, text, ids["old"], ids[name]))
    print(f"\n== made-up vocabularies of Fable's own: {checked} encodes in {time.perf_counter() - began:.1f} s, "
          f"{len(bad)} differ from old")
    for kind, seed, name, text, a, b in bad[:10]:
        print(f"  {kind} seed {seed} {name}: {text!r}: old {a} {name} {b}")
    return len(bad)


# ------------------------------------------------------------------------------------------------------- run
if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}, unicodedata {unicodedata.unidata_version}")
    try:
        import tokenizers
        print(f"tokenizers {tokenizers.__version__}")
    except ImportError:
        print("tokenizers: not installed")
    try:
        import sentencepiece
        print(f"sentencepiece {sentencepiece.__version__}")
    except ImportError:
        print("sentencepiece: not installed")
    files = {
        "rinna": fetch(HF.format(repo="rinna/japanese-gpt2-small", rev="f7fdefe2941d9629a7b2894564435e0e035df6a6", file="spiece.model"), "rinna-spiece.model"),
        "gpt2": fetch(HF.format(repo="openai-community/gpt2", rev="607a30d783dfa663caf39e06633721c8d4cfcd7e", file="tokenizer.json"), "gpt2-tokenizer.json"),
        "qwen3": fetch(HF.format(repo="Qwen/Qwen3-0.6B", rev="c1899de289a04d12100db370d81485cdf75e47ca", file="tokenizer.json"), "qwen3-tokenizer.json"),
        "llama3": fetch(HF.format(repo="unsloth/Llama-3.2-1B-Instruct", rev="5a8abab4a5d6f164389b1079fb721cfab8d7126c", file="tokenizer.json"), "llama3-tokenizer.json"),
        "tiny-lm": fetch(HF.format(repo="sbintuitions/tiny-lm", rev="main", file="spiece.model"), "tiny-lm-spiece.model"),
        "llm-jp": fetch(HF.format(repo="llm-jp/llm-jp-3-150m", rev="main", file="tokenizer.json"), "llm-jp-tokenizer.json"),
        "llama2": fetch("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model", "llama2-tokenizer.model"),
    }
    texts = texts_of(200)
    failures = 0
    failures += made_up_check()
    site = os.path.join(ROOT, "public", "models", "tokenizer.bin")
    cases = [
        ("bpe: Llama 2 (llama2.c's tokenizer.bin)", made("tokenizer.bin", site), real_sp(files["llama2"])),
        ("unigram: tiny-lm (spiece.model)", made("spiece.model", files["tiny-lm"]), real_sp(files["tiny-lm"])),
        ("unigram: llm-jp-3 150M (tokenizer.json)", made("tokenizer.json", files["llm-jp"]), real_hf(files["llm-jp"])),
        ("unigram: rinna japanese-gpt2-small (spiece.model)", made("spiece.model", files["rinna"]), real_sp(files["rinna"])),
        ("bytebpe: GPT-2", made("tokenizer.json", files["gpt2"]), real_hf(files["gpt2"])),
        ("bytebpe: Qwen3", made("tokenizer.json", files["qwen3"]), real_hf(files["qwen3"])),
        ("bytebpe: Llama 3.2", made("tokenizer.json", files["llama3"]), real_hf(files["llama3"])),
    ]
    # SmolLM2's pattern on GPT-2's vocabulary (engine against engine only)
    data, options, specials = made("tokenizer.json", files["gpt2"])
    cases.append(("bytebpe: GPT-2's vocabulary with gpt2-digits", (data, {**options, "pretokenizer": "gpt2-digits"}, specials), None))
    for label, (data, options, specials), real in cases:
        unexpected, unexplained = compare(label, data, options, specials, real, texts)
        failures += unexpected + unexplained
    print(f"\nt200-fable: {'no unexpected difference' if not failures else f'{failures} UNEXPECTED DIFFERENCES'}")
    sys.exit(1 if failures else 0)
