# T229: the small Qwen3.5 of tests/qwen35_model.py as files tests/forward-check.mjs reads (<out>.bin,
# <out>.tokenizer.bin, <out>.json), to hold public/forward.js to the NumPy forward. Its matrices a third of
# qwen35_model's (random weights of that size move the logits by far more than int8's activations do), in float32 and
# int8, and the same model stored as Prism stores Bonsai 2 (tests/test_qwen35.py's rotated(): a GGUF with the rotation).
#
#   python3 tests/qwen35_prepare.py <directory>
import json
import sys
from pathlib import Path

import numpy as np
from qwen35_model import weights
from test_qwen35 import conversion, from_gguf, gguf, rotated

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
tame = {name: (w / 3 if w.ndim == 2 and "norm" not in name and "embed" not in name else w).astype(np.float32)
        for name, w in weights().items()}
folded, metadata = rotated(tame)
made = {"qwen35-float32": conversion(tame), "qwen35-int8": conversion(tame, dtype="int8"),
        "qwen35-rotated-int8": from_gguf(gguf(folded, more=metadata, grouped_out=True), dtype="int8", with_config=True)}
for name, conversion in made.items():
    (out / f"{name}.bin").write_bytes(bytes(conversion.checkpoint))
    (out / f"{name}.tokenizer.bin").write_bytes(bytes(conversion.tokenizer))
    options = {key: value for key, value in conversion.options.items() if key not in ("template", "specials")}
    (out / f"{name}.json").write_text(json.dumps(options))
    print(name, len(bytes(conversion.checkpoint)), "bytes")
