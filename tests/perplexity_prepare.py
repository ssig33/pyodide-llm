# perplexity_prepare.py
# A Hugging Face model converted the way the page converts it (llama2_convert.Conversion, fed the file in order),
# for tests/perplexity.mjs and tests/perplexity_native.py (T85): <out>.bin, <out>.tokenizer.bin and <out>.json,
# the options the page would give Llama(). convert_hf.py does not say those options; the page's path does.
#
#   python3 tests/perplexity_prepare.py <directory with config.json, model.safetensors and the tokenizer | a .gguf> <out> [int8|float32]
#
# A directory with config.json, the tokenizer and a .gguf (tests/hf_fetch.py makes it for T136's second stage): the
# GGUF's weights with the original's vocabulary and configuration, as the page reads them (llama2_convert.gguf_weights).
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
from llama2_convert import Conversion, Incomplete, gguf_weights, joined_shards  # noqa: E402

CHUNK = 8 << 20

directory, out = Path(sys.argv[1]), sys.argv[2]
dtype = sys.argv[3] if len(sys.argv) > 3 else "int8"


class File:
    """The converter's sink (llama2_convert.Writer): the checkpoint goes straight into <out>.bin, a memory map, never
    whole into memory (a float32 Qwen3 0.6B is 2.4 GB, T124)."""

    def open(self, size, header, dtype, form):
        self.data = np.memmap(f"{out}.bin", dtype=np.uint8, mode="w+", shape=(size,))

    def write(self, offset, raw):
        self.data[offset:offset + raw.size] = raw


sink = File()
pieces = None  # the files to feed and where, where there is more than one (T192)
if directory.suffix == ".gguf":
    # T74: one file holds the weights, the configuration and the vocabulary; its header is read first, as the page does
    data = np.memmap(directory, dtype=np.uint8, mode="r")
    size = 1 << 20
    while True:
        try:
            conversion = Conversion.from_gguf(bytes(data[:size]), dtype=dtype, sink=sink)
            break
        except Incomplete:
            size *= 2
    first = conversion.base
else:
    tokenizer = next(p for p in (directory / n for n in ("tokenizer.json", "spiece.model", "tokenizer.model")) if p.exists())
    config = (directory / "config.json").read_text()
    weights = sorted(directory.glob("*.gguf"))
    if weights:
        # T136's second stage: the header of the GGUF as a safetensors one, once config.json agrees with it
        data = np.memmap(weights[0], dtype=np.uint8, mode="r")
        size = 1 << 20
        while True:
            try:
                header, base = gguf_weights(bytes(data[:size]), config)
                break
            except Incomplete:
                size *= 2
        conversion = Conversion(header, base, config, tokenizer.read_bytes(), tokenizer.name, dtype=dtype, start=base,
                                sink=sink)
        first = base
    else:
        # one model.safetensors, or (T192) the shards its index names (tests/hf_fetch.py fetches them), joined as the
        # page's worker joins them (T105): one header over their data one after another, each shard fed from its base
        index = directory / "model.safetensors.index.json"
        names = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
                 if not (directory / "model.safetensors").exists() and index.exists() else ["model.safetensors"])
        shards = []
        for name in names:
            shard = np.memmap(directory / name, dtype=np.uint8, mode="r")
            size = struct.unpack("<Q", bytes(shard[:8]))[0]
            shards.append((shard, bytes(shard[8:8 + size]).decode(), 8 + size))
        if len(shards) == 1:
            header, base = shards[0][1], shards[0][2]
            pieces = [(shards[0][0], base, len(shards[0][0]))]
        else:
            header, lengths = joined_shards([text for _, text, _ in shards])
            base = 0
            pieces = [(shard, begin, begin + length) for (shard, _, begin), length in zip(shards, lengths)]
        conversion = Conversion(header, base, config, tokenizer.read_bytes(), tokenizer.name, dtype=dtype, start=base,
                                sink=sink)
for data, begin, end in pieces or [(data, first, len(data))]:
    for start in range(begin, end, CHUNK):
        conversion.feed(bytes(data[start:min(start + CHUNK, end)]))
conversion.finish()
sink.data.flush()
Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
options = {key: value for key, value in conversion.options.items() if key != "template"}
Path(f"{out}.json").write_text(json.dumps(options))
print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes, options {options}")
