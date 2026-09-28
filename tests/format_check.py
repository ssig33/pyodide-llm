# format_check.py
# The format of every Hugging Face model of the list against the real one (T124, T127, T138): the IDs the page
# sends ([bos] + llama2_numpy.Tokenizer on the converter's tokenizer.bin, with the options the worker merges and
# filled()'s rules) against transformers' apply_chat_template on the files of the pinned revision. Only config.json,
# the tokenizer files and the safetensors headers (by Range) are fetched; no weights.
#
#   python3 tests/format_check.py [--hf] [--prompt <text> ...] <directory for the downloads> [model id ...]
#
# --prompt: one more prompt after PROMPTS (T221: what a visitor types, such as Qwen3's <tool_call>), as often as given.
# --hf: as ?hf= opens the repository, with what the converter makes alone (T143): the list's options and format are
# left out. A difference known for the list's format is then no error when it is gone.
#
# Needs the reference tools, which the page never uses: a venv with tests/requirements-reference.txt (docs/notes/dev-setup.md).
# The first BOS may differ (the page always starts with it, T131). The other differences known are in KNOWN, each
# taking out only its own difference, and anything else makes the exit status 1 (T145). What a card always passes besides the prompt
# (Swallow-MS's and llm-jp's system message) is in SYSTEM. Models with a sentencepiece tokenizer.model are compared
# through transformers' slow tokenizer, which is not the real one for every model: for those of SENTENCEPIECE the
# reference is the real template's text through the real sentencepiece instead.
import inspect
import json
import os
import re
import struct
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
import llama2_convert  # noqa: E402
import llama2_numpy  # noqa: E402

PROMPTS = ["これからの流行りを3つ挙げてください。", "What will be popular next? Name three things.",
           "  leading and trailing spaces  ", "trailing only \n", "改行\nを含む\n\n文", "A", "ＡＢＣ１２３ｶﾀｶﾅ①",
           "Hello  world\ttab", "<think> and <|im_end|> typed"]
# the system message a model card always passes with the prompt. llm-jp's instruct models: the page's format is the one
# their template writes with it (T144; without it the real one leaves out the first sentence of LLM_JP_INSTRUCT)
LLM_JP_SYSTEM = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
SYSTEM = {"hf-swallow-ms-7b-instruct": "あなたは誠実で優秀な日本人のアシスタントです。",
          **{id: LLM_JP_SYSTEM for id in ("hf-llm-jp-3-150m-instruct3", "hf-llm-jp-3-440m-instruct3",
                                          "hf-llm-jp-3-980m-instruct3", "hf-llm-jp-3.1-1.8b-instruct4")}}
# where transformers builds a slow tokenizer out of the tokenizer.model that is not the real one (2026-09-26,
# transformers 5.16.1: sarashina's split "挙げてください" and "next", CAT-Translate's spell everything a character at a time
# although they have a tokenizer.json): the real sentencepiece reads the real template's text instead (T144)
SENTENCEPIECE = {"hf-sarashina2.2-0.5b-instruct", "hf-sarashina2.2-1b-instruct", "hf-sarashina2.2-3b-instruct",
                 "hf-cat-translate-0.8b", "hf-cat-translate-1.4b"}
# T145: the differences known, each with what it takes out of the comparison (the rest still counts, so a new break
# in these models shows: two of them were 0/9 whatever else went wrong). One that no prompt needs any more is an error
# too, to be taken out of here.
KNOWN = {
    # the real tokenizer.json puts no dummy space before the first piece ("A"), the page does ("▁A"): T131, refused
    "hf-rakutenai-2.0-mini-instruct": {"why": "no dummy space before the first piece (T131)", "first_piece": True},
    # the page asks for the answer's channel at the end of the format (T132): the real text with it is the page's
    "hf-llm-jp-4-8b-instruct": {"why": "the page ends with <|channel|>final<|message|> (T132)",
                                "text": lambda text, prompt: text + "<|channel|>final<|message|>"},
    # the list's own ChatML has no system turn; the real template writes its default one
    "hf-smollm2-135m-instruct": {"why": "the list's ChatML, without the default system turn",
                                 "text": lambda text, prompt: re.sub(r"^<\|im_start\|>system\n.*?<\|im_end\|>\n", "", text, flags=re.S)},
    # tokenizer.model's NFKC where the real tokenizer.json has none (the page follows the former), and the template
    # strips the whole turn, which trims only the end of what was typed where the page trims both ends (T138). T192:
    # the real text with those two done to it must be the page's, so that another break in those prompts still shows
    # (the two prompts were taken out whole before)
    "hf-swallow-ms-7b-instruct": {"why": "NFKC of tokenizer.model, and the turn stripped as a whole (T138)",
                                  "text": lambda text, prompt: unicodedata.normalize("NFKC", trimmed(text, prompt))},
}
# The reference of a GGUF that has its own vocabulary: the original at the revision the list had before the GGUF
# (T136's first stage; T144). A GGUF with the original's vocabulary (hf.vocabulary, T136's second stage) says its own.
ORIGINALS = {"Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
             "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
             "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
             "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
             "Qwen/Qwen2.5-Coder-0.5B-Instruct": "ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
             "SakanaAI/TinySwallow-1.5B-Instruct": "91e9fcc30f56d224aea84356c4d850cc4c5a3260",
             "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
             "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
             "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674"}


def entries():
    script = "import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify(MODELS.filter((m) => m.hf))))"
    return json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))


def gguf(entry):
    return entry["hf"]["weights"].endswith(".gguf")


def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120).read()


def fetch(entry, directory):
    """The small files of the entry's repository at its revision (of the original's for a GGUF), and the headers of
    its safetensors or the beginning of its GGUF."""
    repo, revision = entry["hf"]["repo"], entry["hf"]["revision"]
    if gguf(entry):
        vocabulary = entry["hf"].get("vocabulary") or {"repo": entry["original"], "revision": ORIGINALS[entry["original"]]}
        head = directory / repo.replace("/", "__") / revision / f"{entry['hf']['weights']}.head"
        if not head.exists():
            head.parent.mkdir(parents=True, exist_ok=True)
            at = f"https://huggingface.co/{repo}/resolve/{revision}/{entry['hf']['weights']}"
            for size in (8 << 20, 32 << 20, 128 << 20):  # as the worker does: until the tensors' data begins
                data = get(at, {"Range": f"bytes=0-{size - 1}"})
                try:
                    llama2_convert.gguf_read(data)
                    break
                except llama2_convert.Incomplete:
                    continue
            head.write_bytes(data)
        repo, revision = vocabulary["repo"], vocabulary["revision"]
    folder = directory / repo.replace("/", "__") / revision
    folder.mkdir(parents=True, exist_ok=True)
    at = lambda name: f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    named = (entry["hf"].get("vocabulary") or {}).get("tokenizer") or entry["hf"].get("tokenizer") or "tokenizer.json"
    tokenizers = [named] if isinstance(named, str) else named
    # tokenizer.json and special_tokens_map.json too: transformers' reference reads them where they are
    for name in ["config.json", "tokenizer_config.json", "chat_template.jinja", "tokenizer.json",
                 "special_tokens_map.json", *tokenizers]:
        target = folder / name
        if target.exists() or (folder / f"{name}.missing").exists():
            continue
        try:
            data = get(at(name))
        except urllib.error.HTTPError:
            (folder / f"{name}.missing").touch()  # asked once; an empty file would look like an empty template
            continue
        target.write_bytes(data)
    if gguf(entry):
        return folder, head, tokenizers
    try:
        get(at("model.safetensors"), {"Range": "bytes=0-7"})
        shards = ["model.safetensors"]
    except urllib.error.HTTPError:
        index = json.loads(get(at("model.safetensors.index.json")))
        shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        target = folder / f"{shard}.header.json"
        if not target.exists():
            (size,) = struct.unpack("<Q", get(at(shard), {"Range": "bytes=0-7"}))
            target.write_bytes(get(at(shard), {"Range": f"bytes=8-{7 + size}"}))
    return folder, shards, tokenizers


class Sink:
    """The converter writes no weights here: only its options and its tokenizer are wanted."""

    def open(self, *args):
        pass

    def write(self, *args):
        pass


def sentencepiece_ids(model_file, text):
    """text as the real sentencepiece reads it where a tokenizer.json cuts it at the special tokens: the control
    pieces (<s>, </s>, <|user|> ...) stand for themselves, and every stretch between them is encoded (a user-defined
    piece sentencepiece keeps whole by itself)"""
    import sentencepiece
    model = sentencepiece.SentencePieceProcessor(model_file=str(model_file))
    specials = sorted({model.id_to_piece(i) for i in range(model.get_piece_size())
                       if model.is_control(i)} - {""}, key=len, reverse=True)
    ids, at = [], 0
    for found in re.finditer("|".join(map(re.escape, specials)), text):
        ids += model.encode(text[at:found.start()]) if found.start() > at else []
        ids.append(model.piece_to_id(found.group()))
        at = found.end()
    return ids + (model.encode(text[at:]) if at < len(text) else [])


def filled(template, prompt):
    """src/models.js's filled(), for today"""
    template = re.sub(r"\{date(?::([^}]*))?\}", lambda found: time.strftime(found.group(1) or "%Y-%m-%d"), template)
    return re.sub(r"\{prompt(:trim)?\}", lambda found: prompt.strip() if found.group(1) else prompt, template, count=1)


def conversion(entry, folder, shards, tokenizers):
    """What the worker builds: the joined headers, config.json, the first tokenizer the converter takes, the template.
    shards: for a GGUF, the file of its beginning"""
    if gguf(entry) and not entry["hf"].get("vocabulary"):
        return llama2_convert.Conversion.from_gguf(shards.read_bytes(), dtype="int8", sink=Sink())
    if gguf(entry):
        header, base = llama2_convert.gguf_weights(shards.read_bytes(), (folder / "config.json").read_text())
    elif shards == ["model.safetensors"]:
        header = (folder / "model.safetensors.header.json").read_text()
        base = 8 + len(header.encode())
    else:
        header, _ = llama2_convert.joined_shards([(folder / f"{shard}.header.json").read_text() for shard in shards])
        base = 0
    config = (folder / "tokenizer_config.json").read_text() if (folder / "tokenizer_config.json").exists() else ""
    has_template = bool(json.loads(config).get("chat_template")) if config else False
    jinja = folder / "chat_template.jinja"
    chat_template = jinja.read_text() if not has_template and jinja.exists() else None
    refusal = None
    for name in tokenizers:
        try:
            return llama2_convert.Conversion(header, base, (folder / "config.json").read_text(), (folder / name).read_bytes(),
                                             name, dtype="int8", start=base, tokenizer_config=config,
                                             chat_template=chat_template, sink=Sink())
        except (ValueError, FileNotFoundError) as error:
            refusal = error
    raise refusal


def main():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    arguments = sys.argv[1:]
    alone = "--hf" in arguments  # T143: the converter's options and format only, as ?hf= has them
    arguments = [argument for argument in arguments if argument != "--hf"]
    prompts = list(PROMPTS)
    while "--prompt" in arguments:
        at = arguments.index("--prompt")
        prompts.append(arguments[at + 1])
        del arguments[at:at + 2]
    directory, only = Path(arguments[0]), arguments[1:]
    accepted = set(inspect.signature(llama2_numpy.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    failed, stale = [], []
    for entry in entries():
        if only and entry["id"] not in only:
            continue
        folder, shards, tokenizers = fetch(entry, directory)
        made = conversion(entry, folder, shards, tokenizers)
        options = {**made.options, **({} if alone else entry.get("options", {}))}
        template = (None if alone else entry.get("template")) or made.options.get("template")
        if not template:
            print(f"none {entry['id']}: no format (what was typed is continued as it is)", flush=True)
            continue
        tokenizer = llama2_numpy.Tokenizer(made.tokenizer, abs(made.stream.header[5]), kind=options["tokenizer_kind"],
                                           **{key: value for key, value in options.items() if key in accepted})
        reference = AutoTokenizer.from_pretrained(folder)
        thinking = {"enable_thinking": False} if "(no thinking)" in entry["name"] else {}
        known = KNOWN.get(entry["id"], {})
        same, explained, diffs = 0, 0, []
        for prompt in prompts:
            page = [options["bos"]] + tokenizer.encode(filled(template, prompt), tuple(options.get("specials", ())))
            messages = ([{"role": "system", "content": SYSTEM[entry["id"]]}] if entry["id"] in SYSTEM else []) \
                + [{"role": "user", "content": prompt}]
            text = reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **thinking)
            if entry["id"] in SENTENCEPIECE:
                encoded = lambda text: sentencepiece_ids(folder / "tokenizer.model", text)
                real = encoded(text)
            else:
                # a text of KNOWN's as apply_chat_template(tokenize=True) encodes the text it renders
                encoded = lambda text: list(reference(text, add_special_tokens=False)["input_ids"])
                real = reference.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **thinking)
                real = list(real["input_ids"] if hasattr(real, "keys") else real)
            matches = lambda real: page == real or page[1:] == real
            if matches(real):
                same += 1
            elif "text" in known and matches(encoded(known["text"](text, prompt))):
                explained += 1
            elif known.get("first_piece") and first_piece(reference.convert_ids_to_tokens(page[1:]),
                                                          reference.convert_ids_to_tokens(real)):
                explained += 1
            else:
                diffs.append(f"{prompt!r}\n    page {reference.convert_ids_to_tokens(page)[:60]}"
                             f"\n    real {reference.convert_ids_to_tokens(real)[:60]}")
        where = "the list" if entry.get("template") and not alone else "the converter"
        verdict = "DIFF" if diffs else "known" if explained else "ok  "
        note = f", {explained} as known: {known['why']}" if explained else ""
        print(f"{verdict} {entry['id']}: {same}/{len(prompts)}{note} (format of {where})", flush=True)
        for diff in diffs[:3]:
            print("  ", diff)
        if diffs:
            failed.append(entry["id"])
        if known and not explained and not alone:
            print(f"STALE {entry['id']}: the difference known ({known['why']}) is gone: take it out of KNOWN")
            stale.append(entry["id"])
    if failed:
        print(f"{len(failed)} differ beyond what KNOWN says: {' '.join(failed)}")
    sys.exit(1 if failed or stale else 0)


def trimmed(text, prompt):
    """The real text with what was typed trimmed at both ends, as the page's {prompt:trim} writes it (T192): the real
    template strips the whole turn, which leaves the start of what was typed as it was."""
    for typed in (prompt, prompt.rstrip(), prompt.lstrip()):
        if typed and typed in text:
            return text.replace(typed, prompt.strip(), 1)
    return text


def first_piece(page, real):
    """The same pieces but the first, which the page writes with the dummy space the real one leaves out."""
    return len(page) == len(real) > 0 and page[1:] == real[1:] and page[0] == "▁" + real[0]


if __name__ == "__main__":
    main()
