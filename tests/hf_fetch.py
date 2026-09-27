# hf_fetch.py: the files of a Hugging Face model of src/models.js, at the revision the list pins, for measurements in
# CI that need the original (T100): the weights, config.json, the tokenizer and tokenizer_config.json. Prints what to
# hand to tests/perplexity_prepare.py: the directory, or the .gguf file. For a GGUF that takes the vocabulary of its
# original (T136's second stage, hf.vocabulary): the directory of the original's files, with the GGUF linked into it
# (T145: int4.yml and draft.yml named the originals instead).
#
#   python3 tests/hf_fetch.py <model id, or hf:<owner>/<repository>@<revision> (as tests/e2e.mjs takes it)> <directory>
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

from fixed_outputs import HERE, fetch

model_id, directory = sys.argv[1], Path(sys.argv[2])
script = f"import('./src/models.js').then(({{ MODELS }}) => console.log(JSON.stringify(MODELS.find((m) => m.id === {json.dumps(model_id)}))))"
if model_id.startswith("hf:"):
    # a repository the list does not name as it is (the original of a model the list takes from a GGUF, T98)
    repo, _, revision = model_id[3:].partition("@")
    entry = {"hf": {"repo": repo, "revision": revision or "main", "weights": "model.safetensors",
                    "config": "config.json", "tokenizer": "tokenizer.json"}}
else:
    entry = json.loads(subprocess.check_output(["node", "-e", script], cwd=HERE.parent))
hf = entry["hf"]
try:
    weights = fetch(entry, hf["weights"], directory)
except urllib.error.HTTPError as error:
    # T192: a model without a model.safetensors is split over several files (Qwen3 1.7B and up). Its index says which,
    # read as the page's worker reads it (shardsOf()): the files named in weight_map, in the order of their names.
    # perplexity_prepare.py joins them as the page does (llama2_convert.joined_shards()).
    if error.code != 404 or hf["weights"].endswith(".gguf"):
        raise
    index = fetch(entry, f"{hf['weights']}.index.json", directory)
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if not shards:
        raise
    for name in shards:
        weights = fetch(entry, name, directory)
gguf, vocabulary = hf["weights"].endswith(".gguf"), hf.get("vocabulary")
if gguf and not vocabulary:
    print(weights)  # T74: the GGUF says its configuration and vocabulary itself
    sys.exit(0)
# the configuration and the tokenizer: the model's, or for a GGUF those of the original it takes them from
source = {"hf": vocabulary} if vocabulary else entry
folder = directory / source["hf"]["repo"].replace("/", "--") / source["hf"]["revision"]  # where fetch() puts them
tokenizer = (vocabulary or hf)["tokenizer"]
for name in ["config.json" if vocabulary else hf["config"],
             *([tokenizer] if isinstance(tokenizer, str) else tokenizer[:1]), "tokenizer_config.json"]:
    try:
        fetch(source, name, directory)
    except OSError:
        pass  # tokenizer_config.json is optional
if vocabulary:
    link = folder / hf["weights"]
    if not link.exists():
        link.symlink_to(weights.resolve())
print(folder)
