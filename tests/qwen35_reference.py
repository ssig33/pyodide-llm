# T229: what transformers' Qwen3_5ForCausalLM computes for the small model of tests/qwen35_model.py, written to
# tests/fixtures/qwen35-logits.json for tests/test_qwen35.py (which needs neither PyTorch nor transformers). By hand,
# in a venv with torch and transformers (docs/notes/dev-setup.md):
#
#   .venv/bin/python tests/qwen35_reference.py
import json
from pathlib import Path

import torch
import transformers
from qwen35_model import CONFIG, TOKENS, weights
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

config = Qwen3_5TextConfig(**{key: value for key, value in CONFIG.items() if key != "model_type"})
model = Qwen3_5ForCausalLM(config).eval()
state = {name: torch.from_numpy(value) for name, value in weights().items()}
missing, unexpected = model.load_state_dict(state, strict=False)
assert not unexpected and not [name for name in missing if "rotary" not in name], (missing, unexpected)
with torch.no_grad():
    # the whole sequence at once (the chunked delta rule), and token by token with the cache (the recurrent one)
    whole = model(torch.tensor([TOKENS])).logits[0]
    cache, steps = None, []
    for token in TOKENS:
        out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        steps.append(out.logits[0, -1])
    steps = torch.stack(steps)
print("whole against step by step:", float((whole - steps).abs().max()))
Path(__file__).with_name("fixtures").joinpath("qwen35-logits.json").write_text(json.dumps({
    "transformers": transformers.__version__, "torch": torch.__version__, "tokens": TOKENS,
    "logits": [[round(float(v), 6) for v in row] for row in steps]}) + "\n")
