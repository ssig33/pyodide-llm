# gpu_answer.py
# NumPy's answer for tests/gpu-check.mjs (T135): the keys and values of every layer at every position of a prompt,
# and the logits of its last token. Pyodide imports it for the models of this directory and the made-up ones; a model
# too large for Pyodide (T153: Qwen3 0.6B widened to float32 is 2.4 GB) has it from the native Python (T183):
#
#   python3 tests/gpu_answer.py <prefix> <count> <text>     prints {"exact": answer, "half": answer} as JSON
#
# where <prefix>.bin, <prefix>.tokenizer.bin and <prefix>.json (the options) are what tests/gpu_prepare.py writes.
#
# half (T183, E16 of T153's review): the same with the cache rounded to float16 as it is written, as the page's is
# (T110) and the GPU's attention reads it: what is left against it is the GPU's arithmetic, not the cache's.
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
from llama2_numpy import Llama  # noqa: E402


class Half(np.ndarray):
    """A cache that rounds to float16 what is written into it (Llama.forward writes a position of a layer at a time)"""

    def __setitem__(self, index, value):
        super().__setitem__(index, np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32))

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # what is computed from it is a plain array (the attention's scores, its output): only the cache rounds
        array = np.asarray(array).view(np.ndarray)
        return array[()] if return_scalar else array


def answer(data, vocabulary, text, count, options, half=False):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens"""
    numpy = Llama(data, vocabulary, **options)
    if text:
        tokens = ([numpy.bos] + list(numpy.tokenizer.encode(text)))[:count]
    else:
        tokens = [numpy.bos] + [int(t) for t in np.random.default_rng(1).integers(3, numpy.vocab_size, count - 1)]
    assert len(tokens) == count, f"the text is {len(tokens)} tokens long"
    if half:
        # room for every position from the start, so that the cache never grows (growing makes a plain array again)
        for name in ("key_cache", "value_cache"):
            cache = getattr(numpy, name)
            setattr(numpy, name, np.zeros((*cache.shape[:2], count, cache.shape[3]), dtype=np.float32).view(Half))
    for pos, token in enumerate(tokens[:-1]):
        numpy.forward(token, pos, need_logits=False)
    logits = numpy.forward(tokens[-1], count - 1)
    n = count - 1
    kv = lambda cache: np.ascontiguousarray(np.asarray(cache)[:, :, :n, :].transpose(0, 2, 1, 3).reshape(numpy.n_layers, n, -1), dtype=np.float32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    return {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
            "header": list(struct.unpack_from("<7i", data, 0))}


if __name__ == "__main__":
    prefix, count, text = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    data = np.memmap(f"{prefix}.bin", dtype=np.uint8, mode="r")
    vocabulary = Path(f"{prefix}.tokenizer.bin").read_bytes()
    options = json.loads(Path(f"{prefix}.json").read_text())
    print(json.dumps({"exact": answer(data, vocabulary, text, count, options),
                      "half": answer(data, vocabulary, text, count, options, half=True)}))
