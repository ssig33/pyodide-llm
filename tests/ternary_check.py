# T230: Prism's two packings of one model, read here: every tensor of the PTQ1_0 GGUF must be the one of the PQ2_0 GGUF,
# value for value (the same ternary weights and float16 scales, packed two ways), and the converter's readers must agree
# with tests/gguf_check.py's. Over HTTP ranges, a tensor at a time (Bonsai 2's two files are 13 GB), for CI:
#
#   python3 tests/ternary_check.py <URL of the PTQ1_0 GGUF> <URL of the PQ2_0 GGUF> [every n-th tensor = 1]
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import llama2_convert  # noqa: E402
from gguf_check import BYTES, TYPE_NAMES, WIDEN  # noqa: E402


def fetch(url, start, stop):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) == stop - start:
                return data
        except OSError:
            if attempt == 4:
                raise
    raise OSError(f"{url}: {stop - start} bytes asked for, fewer came")


def header(url):
    for size in (1 << 24, 1 << 26):
        try:
            return llama2_convert.gguf_read(fetch(url, 0, size))
        except llama2_convert.Incomplete:
            continue
    raise ValueError(f"{url}: the header is longer than 64 MiB")


def values(url, base, info, readers):
    count = int(np.prod(info["shape"]))
    raw = fetch(url, base + info["offset"], base + info["offset"] + int(count * BYTES[info["type"]]))
    return [reader(raw) for reader in readers]


def main():
    dense, packed = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    (m1, t1, b1), (m2, t2, b2) = header(dense), header(packed)
    assert set(t1) == set(t2), "the two files hold other tensors"
    ours = {142: llama2_convert.pq2_0, 143: llama2_convert.ptq1_0, 0: lambda raw: np.frombuffer(raw, np.float32),
            30: llama2_convert.bfloat16}
    theirs = lambda kind: (lambda raw: WIDEN[kind][2](np.frombuffer(raw, np.uint8)).reshape(-1)) if kind in WIDEN else ours[kind]
    compared, differing, kinds = 0, [], {}
    for i, name in enumerate(sorted(t1)):
        if i % every:
            continue
        a, b = t1[name], t2[name]
        assert a["shape"] == b["shape"], name
        first, first_ref = values(dense, b1, a, [ours[a["type"]], theirs(a["type"])])
        second, second_ref = values(packed, b2, b, [ours[b["type"]], theirs(b["type"])])
        same = np.array_equal(first, second) and np.array_equal(first, first_ref) and np.array_equal(second, second_ref)
        kinds[f"{TYPE_NAMES.get(a['type'])} / {TYPE_NAMES.get(b['type'])}"] = kinds.get(f"{TYPE_NAMES.get(a['type'])} / {TYPE_NAMES.get(b['type'])}", 0) + 1
        compared += 1
        if not same:
            differing.append(name)
            print(f"{name}: DIFFERS")
    print(json.dumps({"compared": compared, "differing": differing, "kinds": kinds}))
    sys.exit(1 if differing else 0)


main()
