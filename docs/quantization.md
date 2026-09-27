# Quantization

The larger models are stored in int8, and in 6 bits where int8 does not fit. This page says how, and what it costs
in quality. Perplexity is lower for better predictions; "+1%" means 1% worse than the original.

## int8

Weights are split into groups of 32, each with one float32 scale (the largest value of the group maps to 127).
The kernels multiply the int8 values as they are, without widening them to float32. For llm-jp-3-150m that took
the WebAssembly heap from 897 MB to 283 MB.

The weights alone are hard to tell from the original:

| model | original | int8 |
|---|---:|---:|
| stories15M | - | +0.04% |
| tiny-lm | 91.3 | 91.1 |
| llm-jp-3-150m | 22.76 | 22.69 |

The most likely token agrees about 98% of the time. Greedy output leaves the original's text part way through but
stays coherent. int4 cost +16.8% and was rejected.

## Activations: 8 or 7 bits

The kernels also quantize the input of each matrix product, per group of 32. Where the browser has relaxed SIMD
(Chrome, Edge, Firefox), they use 7 bits, which makes int8 about 30% faster. Measured on the first 1,499 tokens
of three Japanese Wikipedia articles (`node tests/perplexity.mjs`):

| llm-jp-3-150m | perplexity |
|---|---:|
| float16 original | 29.976 |
| int8 weights, NumPy | 29.907 |
| + 8-bit activations | 29.965 |
| + 7-bit activations | 30.094 (+0.39%) |

tiny-lm: +0.01% with 8 bits, +0.34% with 7.

Other architectures (1,500 tokens of English Wikipedia; rinna on Japanese):

| model | original float32 | int8, NumPy | + 8-bit activations | + 7-bit activations |
|---|---:|---:|---:|---:|
| rinna/japanese-gpt2-small | 33.253 | +0.05% | +0.11% | +0.20% |
| EleutherAI/pythia-160m | 47.795 | +1.17% | +1.50% | +2.23% |
| openai-community/gpt2 | 35.340 | +2.06% | +4.73% | +17.2% (before the fix below) |

**GPT-2 and 7 bits.** The loss came from one place: the input of the classifier. GPT-2's final LayerNorm
multiplies a few channels 12 to 17 times, and one such channel in a group of 32 rounds the other 31 away. The
engine now takes the 8 channels with the largest norm weights out before quantizing, and multiplies them in
float32 separately. It does this only where the largest norm weight is at least 4 times the median (GPT-2: 13.9;
the other models 1.1 to 1.9). GPT-2's 7-bit row went from 41.431 to 36.127 (+2.2% against the original). The
remaining difference comes from the int8 weights of the classifier.

## 6 bits

Where int8 does not fit (Safari, which has no 64-bit memory, and devices that report little memory), the page
stores the weights in 6 bits. Measured on five models (SmolLM2 135M and 360M, Qwen2.5 0.5B, Pythia 410M,
llm-jp-3 440M):

- 4 bits: +10 to +48% in every form tried (llama.cpp's Q4_0: +12 to +15%). Too much.
- 5 bits (groups of 32, float16 scale): up to +9.94%.
- **6 bits: up to +3.37%.** The owner accepted a loss of 3 to 5%, so 6 bits it is.
- Larger models lose less: llm-jp-3 980M +1.02%, Llama 3.2 1B +1.37%, Qwen2.5 1.5B +1.68%. Their greedy text is
  often the same as the original's, word for word.

A 6-bit value is stored as an int8 whose lowest 2 bits are 0, with a scale a quarter as large. So only the packing
(32 values in 24 bytes) is new, and everything else is the int8 path. A 6-bit file is 7/9 of the int8 file.

6 bits is slower: about 0.45 times int8 on one thread (stories15M, 243 against 490 tok/s), 0.37 to 0.53 times on 4
threads in CI. So the page uses 6 bits only where int8 does not fit. Where the browser has 64-bit memory, a model
too large for 32 bits stays int8: Llama 3.2 3B wrote 1.9 tok/s in 6 bits on 32-bit memory and 4.1 tok/s in int8
on 64-bit memory (Chromium in CI). `?bits=6` or `?bits=8` chooses by hand.

## Models from GGUF

Some models of the list are fetched as a Q8_0 GGUF (llama.cpp's int8 with a float16 scale per 32 values). Q8_0
turns back into int8 without loss. Each GGUF was compared with its original tensor by tensor before it went into
the list (`tests/gguf_check.py`). In 6 bits, a GGUF is quantized twice, which adds about 3% to the error of the
weights (Qwen2.5 0.5B: 0.02375 against 0.02308); on 1,500 tokens the perplexity could not tell the two apart.

## Other small effects

- A BOS token at the start: the page always starts with one, while some models are used without it. The
  difference was within ±3%, in either direction depending on the text, so it was kept. One model broke without
  the right BOS (DeepSeek-R1 Distill Qwen 1.5B), and its entry now names it.
- RMSNorm's epsilon: 1e-5 for all models until Qwen3 0.6B showed +0.12% with it; the converter now passes the
  model's own value.
