# fixed_outputs.py
# Real models against the text they wrote when a person read it and found it sound (T86). Twice last week every
# test passed while the page wrote broken text (T72: v permuted as well, rotary never passed on), because the
# synthetic models of the unit tests cannot tell a sensible sentence from a wrong one. These can.
#
# One small model per architecture and source, converted the way the page converts it (llama2_convert.Conversion,
# fed the file in order) to float32, then 16 greedy tokens from the model's own prompt and template. float32
# because int8 changes its text with the order of additions (AGENTS.md); float32 with NumPy is the same, to the
# character, as float32 with the kernels. Where each model comes from, its prompt and template are read from
# src/models.js (with Node), so nothing here can drift from the list.
#
#   python3 tests/fixed_outputs.py <directory for the downloads> [--write] [--only=<id>,<id>]
#
# --write records what the models write now, into tests/fixtures/fixed-outputs.json: do that only after reading it.
# --only runs those models alone (a broken converter tried on the one model it should break, T204).
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
import llama2_convert  # noqa: E402
from llama2_convert import Conversion, Incomplete  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

FIXTURES = HERE / "fixtures" / "fixed-outputs.json"
NEW_TOKENS = 16
CHUNK = 8 << 20
# one per architecture and way in: GPT-NeoX, GPT-2, a Llama from a GGUF (T74), a Llama with a Unigram tokenizer.json,
# and a Qwen3 (T124: the norms of q and k, heads of 128 in a dim of 1024). Pythia and GPT-2 come from GGUFs since
# T136's third stage, so a GPT-2 from a safetensors file too (T204): rinna's, whose names begin with "transformer.",
# whose Conv1D matrices the plan transposes as they come, and whose sentencepiece model normalizes as nmt_nfkc.
MODELS = ["hf-pythia-70m", "hf-gpt2", "hf-japanese-gpt2-small", "hf-smollm2-135m-instruct", "hf-llm-jp-3-150m-instruct3",
          "hf-qwen3-0.6b"]


def entries():
    script = ("import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify("
              f"MODELS.filter((m) => {json.dumps(MODELS)}.includes(m.id)))))")
    found = {entry["id"]: entry for entry in json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))}
    return [found[id] for id in MODELS]


def fetch(entry, name, directory):
    hf = entry["hf"]
    target = directory / hf["repo"].replace("/", "--") / hf["revision"] / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{hf['repo']}/resolve/{hf['revision']}/{name}"
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):  # huggingface.co drops a connection now and then: three tries, a minute each
            try:
                with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
                    while block := response.read(CHUNK):
                        out.write(block)
                break
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt == 2:
                    raise  # a file the repository does not have (404) is not asked for again (T192: split models)
            except OSError:
                if attempt == 2:
                    raise
        partial.rename(target)
    return target


class File:
    """The converter's sink (llama2_convert.Writer): the float32 checkpoint goes into a file of its own, a memory map,
    never whole into memory (Qwen3 0.6B's is 2.4 GB, T124)."""

    def __init__(self, path):
        self.path = path

    def open(self, size, header, dtype, form):
        self.data = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


def converted(entry, directory):
    hf = entry["hf"]
    weights = fetch(entry, hf["weights"], directory)
    data = np.memmap(weights, dtype=np.uint8, mode="r")
    sink = File(weights.with_name("float32.bin"))
    # T136's second and third stages: a GGUF's weights with the vocabulary and config.json of the original
    vocabulary = hf.get("vocabulary")
    if hf["weights"].endswith(".gguf") and not vocabulary:
        size = 1 << 20
        while True:
            try:
                conversion = Conversion.from_gguf(bytes(data[:size]), dtype="float32", sink=sink)
                break
            except Incomplete:
                size *= 4
        first = conversion.base
    else:
        source = {"hf": vocabulary} if vocabulary else entry
        tokenizer = (vocabulary or hf)["tokenizer"]
        tokenizer = tokenizer if isinstance(tokenizer, str) else tokenizer[0]
        try:
            tokenizer_config = fetch(source, "tokenizer_config.json", directory).read_text()
        except OSError:
            tokenizer_config = ""
        config = fetch(source, "config.json" if vocabulary else hf["config"], directory).read_text()
        if vocabulary:
            size = 1 << 20
            while True:
                try:
                    header, first = llama2_convert.gguf_weights(bytes(data[:size]), config)
                    break
                except Incomplete:
                    size *= 4
        else:
            (length,) = np.frombuffer(bytes(data[:8]), dtype="<u8")
            header, first = bytes(data[8:8 + int(length)]).decode(), 8 + int(length)
        conversion = Conversion(header, first, config, fetch(source, tokenizer, directory).read_bytes(), tokenizer,
                                dtype="float32", tokenizer_config=tokenizer_config, sink=sink, start=first)
    for start in range(first, len(data), CHUNK):
        conversion.feed(bytes(data[start:start + CHUNK]))
    conversion.finish()
    sink.data.flush()
    return conversion, sink.path


def written(entry, conversion, checkpoint):
    """What the page would write with this model at temperature 0: the prompt in the model's template, the
    options of the conversion with those of src/models.js on top (as the worker merges them)."""
    options = {**conversion.options, **entry.get("options", {})}
    template = entry.get("template") or options.pop("template", None)
    options.pop("template", None)
    # filled() of src/models.js: {prompt:trim} trims (T138); the prompts here have no spaces to trim
    prompt = template.replace("{prompt:trim}", "{prompt}").replace("{prompt}", entry["prompt"]) if template else entry["prompt"]
    llama = Llama(np.memmap(checkpoint, dtype=np.uint8, mode="r"), conversion.tokenizer, kernels=None, **options)
    steps = len(llama.tokenizer.encode(prompt, llama.specials)) + NEW_TOKENS
    return "".join(llama.generate(prompt, steps=steps, temperature=0.0, echo=False))


def main():
    directory, write = Path(sys.argv[1]), "--write" in sys.argv
    only = [arg.split("=", 1)[1].split(",") for arg in sys.argv if arg.startswith("--only=")]
    if write and only:
        sys.exit("--write records every model: not with --only")
    expected = json.loads(FIXTURES.read_text()) if FIXTURES.exists() else {}
    got, failures = {}, []
    for entry in entries():
        if only and entry["id"] not in only[0]:
            continue
        text = written(entry, *converted(entry, directory))
        got[entry["id"]] = {"prompt": entry["prompt"], "text": text}
        same = expected.get(entry["id"], {}).get("text") == text
        print(f"{'ok  ' if same else 'DIFF'} {entry['id']}: {json.dumps(text, ensure_ascii=False)}", flush=True)
        if not same:
            failures.append(entry["id"])
    if write:
        FIXTURES.parent.mkdir(exist_ok=True)
        FIXTURES.write_text(json.dumps(got, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {FIXTURES}")
    elif failures:
        print(f"FAILED: {', '.join(failures)} wrote something else than {FIXTURES.name} says")
        sys.exit(1)


if __name__ == "__main__":
    main()
