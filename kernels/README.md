# SIMD kernels

`kernel.ts` and `kernel_relaxed.ts` (AssemblyScript) are the numeric core of the site: float32 and int8 matmul,
activation quantization, rmsnorm, RoPE, attention, SwiGLU, the residual add, LayerNorm and GELU (for GPT-2 models,
T65), and sampling (repetition penalty, softmax,
top-p). `public/llama2_numpy.py` loads
them with `ctypes.CDLL` (`load_kernels`) and drives them from Python (`Llama.kernel_forward`): Python keeps
sequencing the layers, NumPy keeps owning the memory, the kernels get addresses and work in place. Nothing is
copied and there is no JavaScript glue. When they cannot be loaded, NumPy does the math as before.

`make kernels` compiles them into `public/simdkernel.so` and `public/simdkernel_relaxed.wasmlib` (a few KB, not
committed). `build.py` needs no Emscripten: it compiles with AssemblyScript and prepends the `dylink.0` section
that makes a module an Emscripten side module. Verified to load in Pyodide 0.29.4 and 314.0.7.

## Measured (2026-09-19, phone-class ARM CPU, one thread)

## What each optimization is worth (T52)

`?without=kernels,int8,relaxed,sampler` leaves any of them out, one at a time, so that the contributions come
apart. Measured on the development machine (Node on Pyodide, 64 tokens, int8 models, the same seed):

| | tiny-lm | llm-jp-3-150m |
|---|---|---|
| NumPy only (`?without=kernels`) | 48.6 tok/s | 9.6 tok/s |
| + the kernels, int8 widened to float32 | 187.1 | 37.9 |
| + int8 kept as int8 | 248.1 | 65.4 |
| + relaxed SIMD (7-bit activations) | 315.3 | 84.3 |
| + sampling in the kernel | **415.6** | **92.8** |

In Chromium on the same machine, tiny-lm goes 44.8 (`?without=kernels`) -> 82.5 -> 175.8 -> 193.0 -> 334.6 tok/s.

So the kernels are worth 8.6x and 9.7x over NumPy, and none of the four steps is negligible. The status line
says what ran, ending in "(without ...)" when something was left out: a number whose conditions are unclear is
worth nothing.

| model | NumPy | kernels |
|---|---:|---:|
| stories15M float32, Pyodide in Node | 53 tok/s | 200 tok/s, the same text |
| stories15M int8, Pyodide in Node | 55 | 351 |
| tiny-lm int8, greedy, Pyodide in Node | 56 | 422 |
| llm-jp-3-150m int8, Pyodide in Node | 9.3 tok/s, 897 MB of WASM heap | 81 tok/s, 283 MB |
| Chromium: stories15M float32 / int8 | 50 | 186 / 296 |
| Chromium: tiny-lm int8, sampled with a repetition penalty | 43 | 271-296 (171 while NumPy still did the sampling, 252-274 before T54) |
| Chromium: llm-jp-3-150m int8, sampled, 256 tokens | 8.5 | 75-79 (47 while NumPy still did the sampling, 61-66 before T54) |
| Chromium: stories3_5M / stories260K float32 (grouped-query attention) | 141 / 268 | 402 / 951 |

For comparison: native llama2.c with `gcc -Ofast` runs stories15M at 214 tok/s on the same machine. Firefox loads the
kernels too (what looked like a Firefox 5-7x slower on the test machine was Playwright's debugger, see below). Every
other browser is tested on the runners of GitHub Actions (below). Overheads inside Pyodide: one ctypes call 3-12 us, `array.ctypes.data` about 4 us (addresses are
taken once), one small NumPy call 1.5-2.5 us; a token takes about 100 kernel calls.

## Every browser the runners offer (2026-09-19, T44 and T58)

`.github/workflows/browsers.yml` runs `tests/e2e.mjs` against the deployed site on five systems, in Playwright's
own Chromium, Firefox and WebKit and in the Chrome and Edge that the runner has installed: one job per system,
its browsers one after the other, by hand (`gh workflow run browsers.yml`) or once a week. **All 23 combinations
ran all four models.** tok/s over 256 tokens
(`*`: the model stopped by itself before 190 tokens, so the number is less steady). The runners are shared virtual
machines with different CPUs, some of them much faster than the machine of the other tables: read the table for
"does it run, and on which kernel", and across browsers on the same system, not across systems.

| system | browser | CPU of the runner | int8 runs on | llm-jp-3-150m | tiny-lm | stories15M | stories260K |
|---|---|---|---|---:|---:|---:|---:|
| Linux, x86-64 | chrome 152.0.7977.82 | AMD EPYC 7763 | `matmul_q8r` | 88 | 413* | 502 | 2409 |
| Linux, x86-64 | chromium 148.0.7778.96 | AMD EPYC 7763 | `matmul_q8r` | 107 | 418 | 518 | 2044 |
| Linux, x86-64 | firefox 150.0.2 | AMD EPYC 7763 | `matmul_q8r` | 13 | 63* | 57 | 295 |
| Linux, x86-64 | msedge 152.0.4191.66 | Intel Xeon 6973P-C | `matmul_q8r` | 80* | 552* | 826 | 2809 |
| Linux, x86-64 | webkit 26.4 | AMD EPYC 7763 | `matmul_q8` | 52 | 261* | 288 | 1636 |
| Linux, ARM64 | chromium 148.0.7778.0 | unknown | `matmul_q8r` | 105 | 490* | 571 | 2295 |
| Linux, ARM64 | firefox 150.0.2 | unknown | `matmul_q8r` | 8 | 42* | 40 | 320 |
| Linux, ARM64 | webkit 26.4 | unknown | `matmul_q8` | 72 | 420* | 441 | 2495 |
| Windows, x86-64 | chrome 152.0.7977.83 | Intel Xeon Platinum 8573C | `matmul_q8r` | 117 | 574* | 672 | 2344 |
| Windows, x86-64 | chromium 148.0.7778.96 | AMD EPYC 7763 | `matmul_q8r` | 95 | 448 | 416 | 2186 |
| Windows, x86-64 | firefox 150.0.2 | AMD EPYC 7763 | `matmul_q8r` | 13 | 58* | 52 | 275 |
| Windows, x86-64 | msedge 152.0.4191.66 | AMD EPYC 7763 | `matmul_q8r` | 95 | 397* | 478 | 2209 |
| Windows, x86-64 | webkit 26.4 | AMD EPYC 9V45 | `matmul_q8` | 82 | 310* | 403 | 2191 |
| Windows, ARM64 | chrome 153.0.8010.37 | Cobalt 100 | `matmul_q8r` | 100 | 505* | 523 | 2102 |
| Windows, ARM64 | chromium 148.0.7778.96 | Cobalt 100 | `matmul_q8r` | 35 | 148 | 160 | 638 |
| Windows, ARM64 | firefox 150.0.2 | Cobalt 100 | `matmul_q8r` | 9 | 42* | 39 | 230 |
| Windows, ARM64 | msedge 153.0.4234.32 | Cobalt 100 | `matmul_q8r` | 100 | 514 | 512 | 2072 |
| Windows, ARM64 | webkit 26.4 | Cobalt 100 | `matmul_q8` | 25 | 103* | 111 | 443 |
| macOS, Apple M1 | chrome 152.0.7977.83 | Apple M1 (Virtual) | `matmul_q8r` | 157 | 363* | 838 | 2883 |
| macOS, Apple M1 | chromium 148.0.7778.96 | Apple M1 (Virtual) | `matmul_q8r` | 156 | 579* | 745 | 2822 |
| macOS, Apple M1 | firefox 150.0.2 | Apple M1 (Virtual) | `matmul_q8r` | 10 | 56* | 56 | 392 |
| macOS, Apple M1 | msedge 152.0.4191.66 | Apple M1 (Virtual) | `matmul_q8r` | 155 | 477* | 640 | 1738 |
| macOS, Apple M1 | webkit 26.4 | Apple M1 (Virtual) | `matmul_q8` | 137 | 664 | 660 | 3938 |

- **WebKit has no relaxed SIMD**, on any system: the int8 models fall back to `matmul_q8`, which had never met a
  real browser before T44. It is Playwright's WebKit, close to Safari, not Safari itself.
- **The Firefox rows of this table are not Firefox's speed** (T59). Playwright drives its patched Firefox through the
  debugger, and SpiderMonkey compiles the WebAssembly of a debugged page with the baseline compiler only. A bare
  SIMD loop with nothing of Pyodide runs at 0.43 G multiply-adds per second there against 5.7 in Chromium (scalar
  code 0.07 against 1.2); forcing the baseline compiler changes nothing, and with the optimizing compiler alone
  that Firefox says "no WebAssembly compiler available". Inside the kernels it is 13 to 15 times slower, while a
  ctypes call costs 24 us and is 4% of a token. **The Firefox that the runners have installed, driven by Selenium
  and geckodriver (`tests/stock-firefox.mjs`), is as fast as the Chromium family on every system:**

  | system | Playwright's Firefox 150, llm-jp-3-150m | installed Firefox 155 | Chromium family on that system |
  |---|---:|---:|---:|
  | Linux, x86-64 | 13 tok/s, ready in 24 s | 106 tok/s, ready in 5 s | 80-107 |
  | Linux, ARM64 | 8 | 68 | 105 |
  | Windows, x86-64 | 13 | 85 | 95-117 |
  | Windows, ARM64 | 9 | 65 | 100 |
  | macOS, Apple M1 | 13 | 132 | 155-157 |

  So never read a speed off Playwright's Firefox, here or on the development machine; it is good for "does it
  run". Whether a Firefox that a person opens is as fast as the one Selenium opens was not checked, but nothing in
  Marionette suggests otherwise.
- On Windows for ARM, Playwright's Chromium and WebKit are x86-64 builds that Windows emulates, a third as fast
  as the native Chrome and Edge next to them.
- The first run found a bug of the page, not of a browser: the real Edge reports the 404 of `/favicon.ico` as a
  console error, which the test counts as a failure. The page now carries its icon as a data URI.

## Where the time of one token goes (2026-09-19, T53)

`node tests/profile.mjs [node|chromium|firefox] [model ...]` measures this inside Pyodide. A browser's clock is too
coarse for single kernel calls, so every number is a difference of totals: the same model runs its forward pass
with the real kernels, with kernels that only count, and with kernels that do nothing (what Python and NumPy cost
around the calls); a kernel called on nothing (n = 0) costs exactly one ctypes call. Positions 16 to 79.

#### One token, Pyodide 314.0.7 in Node v24.19.0

| model | calls | per call | kernels: layers | kernels: classifier | ctypes calls | Python and NumPy | sampling | token | tok/s | read per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm-jp-3-150m (SIMD kernels, int8, relaxed SIMD) | 231 | 2.9 us | 6.29 ms (52%) | 4.70 ms (39%) | 0.67 ms (6%) | 0.10 ms (1%) | 0.35 ms (3%) | 12.10 ms | 83 | 114 MB (classifier 57) |
| tiny-lm (SIMD kernels, int8, relaxed SIMD) | 79 | 2.9 us | 0.61 ms (26%) | 1.32 ms (55%) | 0.23 ms (9%) | 0.04 ms (2%) | 0.20 ms (8%) | 2.40 ms | 416 | 18 MB (classifier 15) |
| stories260K (SIMD kernels, float32) | 77 | 2.6 us | 0.15 ms (37%) | 0.03 ms (8%) | 0.20 ms (47%) | 0.03 ms (7%) | 0.01 ms (2%) | 0.42 ms | 2390 | 1 MB (classifier 0) |

#### Matmul throughput by matrix size (row length 768), Pyodide 314.0.7 in Node v24.19.0

| rows | int8 weights | matmul_q8 | matmul_q8r (relaxed SIMD) | float32 weights | matmul_f32 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0 MB | 2.32 G MAC/s (2.6 GB/s) | 2.38 G MAC/s (2.7 GB/s) | 0.0 MB | 2.36 G MAC/s (9.4 GB/s) |
| 64 | 0.0 MB | 5.04 G MAC/s (5.7 GB/s) | 5.89 G MAC/s (6.6 GB/s) | 0.2 MB | 4.30 G MAC/s (17.2 GB/s) |
| 512 | 0.4 MB | 7.86 G MAC/s (8.8 GB/s) | 10.67 G MAC/s (12.0 GB/s) | 1.6 MB | 5.08 G MAC/s (20.3 GB/s) |
| 4096 | 3.1 MB | 8.00 G MAC/s (9.0 GB/s) | 10.68 G MAC/s (12.0 GB/s) | 12.6 MB | 5.45 G MAC/s (21.8 GB/s) |
| 32768 | 25.2 MB | 8.67 G MAC/s (9.8 GB/s) | 11.64 G MAC/s (13.1 GB/s) | 100.7 MB | 5.42 G MAC/s (21.7 GB/s) |
| 98304 | 75.5 MB | 8.74 G MAC/s (9.8 GB/s) | 11.97 G MAC/s (13.5 GB/s) | 302.0 MB | 4.34 G MAC/s (17.3 GB/s) |

#### One token, Pyodide 314.0.7 in chromium 148.0.7778.0

| model | calls | per call | kernels: layers | kernels: classifier | ctypes calls | Python and NumPy | sampling | token | tok/s | read per token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm-jp-3-150m (SIMD kernels, int8, relaxed SIMD) | 231 | 3.1 us | 6.39 ms (52%) | 4.58 ms (38%) | 0.72 ms (6%) | 0.10 ms (1%) | 0.40 ms (3%) | 12.19 ms | 82 | 114 MB (classifier 57) |
| tiny-lm (SIMD kernels, int8, relaxed SIMD) | 79 | 3.0 us | 0.62 ms (25%) | 1.35 ms (54%) | 0.24 ms (10%) | 0.05 ms (2%) | 0.22 ms (9%) | 2.48 ms | 403 | 18 MB (classifier 15) |
| stories260K (SIMD kernels, float32) | 77 | 2.8 us | 0.17 ms (36%) | 0.04 ms (8%) | 0.21 ms (47%) | 0.03 ms (6%) | 0.01 ms (2%) | 0.46 ms | 2181 | 1 MB (classifier 0) |

#### Matmul throughput by matrix size (row length 768), Pyodide 314.0.7 in chromium 148.0.7778.0

| rows | int8 weights | matmul_q8 | matmul_q8r (relaxed SIMD) | float32 weights | matmul_f32 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0 MB | 2.31 G MAC/s (2.6 GB/s) | 2.36 G MAC/s (2.7 GB/s) | 0.0 MB | 2.40 G MAC/s (9.6 GB/s) |
| 64 | 0.0 MB | 5.05 G MAC/s (5.7 GB/s) | 5.89 G MAC/s (6.6 GB/s) | 0.2 MB | 4.21 G MAC/s (16.9 GB/s) |
| 512 | 0.4 MB | 7.85 G MAC/s (8.8 GB/s) | 10.48 G MAC/s (11.8 GB/s) | 1.6 MB | 4.91 G MAC/s (19.7 GB/s) |
| 4096 | 3.1 MB | 8.09 G MAC/s (9.1 GB/s) | 10.52 G MAC/s (11.8 GB/s) | 12.6 MB | 3.97 G MAC/s (15.9 GB/s) |
| 32768 | 25.2 MB | 8.70 G MAC/s (9.8 GB/s) | 11.14 G MAC/s (12.5 GB/s) | 100.7 MB | 5.14 G MAC/s (20.5 GB/s) |
| 98304 | 75.5 MB | 8.55 G MAC/s (9.6 GB/s) | 11.89 G MAC/s (13.4 GB/s) | 302.0 MB | 4.69 G MAC/s (18.8 GB/s) |

Inside a worker of Chromium, as the page runs it, the numbers are the same (llm-jp-3-150m 81 tok/s).

What this says:

- **The matrix products are nine tenths of a token of llm-jp-3-150m** (layers 52%, the classifier over 99584
  tokens 38%). ctypes calls are 6%, Python and NumPy around them 1%, sampling 3%. One call per layer instead of 19
  (T47) can win at most those 7%; for tiny-lm at most 12%; only a model as small as stories260K, which nobody
  waits for, would double.
- **The int8 product is not bound by memory bandwidth.** Its throughput does not fall when the matrix leaves the
  caches (10.7 G multiply-adds per second at 0.4 MB, 12.0 at 75 MB), and it moves 13 GB/s while the float32
  kernel moves 20 GB/s through the same memory. So it is the arithmetic of `matmul_q8r` that limits the default
  model, and a faster inner loop would show: up to about 1.5x before bandwidth becomes the limit.
- **Attention is slow for what it does.** A token of llm-jp-3-150m takes 12.6 ms at position 8 and 16.3 ms at
  position 248 (Node): 3.7 ms for some 3 million multiply-adds and 24000 exponentials, several times the cost
  per multiply-add of the matrix products. This is why a run of 256 tokens reached 70 tok/s in Node and 61-66 in
  the page, not the 82 of the table above. (T54 has since rewritten attention, see below.)

## What T54 made of that (2026-09-19)

The tables above are from before these changes. Old and new kernels were compared in one process, taking turns
(a run of its own is noisier than the differences): a token of llm-jp-3-150m went from 13.3 to 12.1 ms at position
8, from 15.2 to 12.4 at position 120 and from 17.9 to 12.8 at position 240 (1.09x, 1.23x, 1.39x); tiny-lm 1.08x to
1.20x. In the page, 256 tokens of llm-jp-3-150m: 61-66 -> 75-79 tok/s.

- **Attention walks the cache row by row, all heads of a position at once** (the scratch is now `nh * (pos + 1)`
  floats). A head at a time meant `nh` strided passes over a cache that does not fit the CPU's caches, 12 layers of
  it. With four accumulators for the scores, four exponentials at a time, and the values of four positions added
  per load and store of the output, one call at position 248 went from 109 to 67 us on a cache that fits, and the
  3.7 ms that position 248 added to a token became 0.7 ms.
- **`matmul_q8r`: four groups at a time** (their scales multiplied as one vector, two accumulators taking turns, the
  bias correction as a dot product of its own): 9.8-11.0 -> 11.2-12.5 G multiply-adds per second. That is where it
  ends: a loop of nothing but `i32x4.relaxed_dot_i8x16_i7x16_add_s` with four accumulators reaches 13.0 (9.7 with
  one), so the kernel is within a tenth of what the instruction gives under V8 on this CPU. (Whether V8 turns it
  into one SDOT was not looked up; the speed suggests it does not.)
- Tried and taken back: two rows at a time, sharing the loads of the activations, was slower (10.4 / 8.8 / 11.1
  against 11.2 / 12.1 / 12.5: too many live vectors). The same unrolling in `matmul_q8`, the path without relaxed
  SIMD, changed nothing (7.6-8.3 before and after: its widening multiplies are the cost).
- **The text of an int8 model changed, and that is no error.** Both versions of `matmul_q8r` agree with exact
  integer arithmetic to 1e-6 on the real weights, but a different order of float additions moves a last bit, the
  next layer rounds an activation to the neighbouring 7-bit step, and twelve layers later a logit of
  llm-jp-3-150m differs by up to 1.0. The kernels have always been that far from the NumPy path, which does not
  quantize activations (old kernels 1.7, new ones 1.6, largest difference over 99584 logits). tiny-lm did not
  flip anywhere in the same test (7.6e-6), and the float32 models still write NumPy's text letter for letter.
  What the 7-bit activations cost in perplexity: +0.4%, see the int8 section (T55).

## The whole context (2026-09-19, T56)

llm-jp-3-150m used to be cut from 4096 to 512 tokens when the site was built, for the sake of the KV cache
(12 layers x positions x 512 x 4 bytes x 2: 200 MB at 4096). Now it keeps its context, and the cache starts with
room for 256 positions and doubles when the text gets there, one layer at a time, so that growing never needs a
second copy of all of it. Pyodide in Node, a token with its logits:

| context | heap with the model | position 8 | 248 | 500 | 1000 | 2000 | 3000 | 4070 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512, reserved at once (before) | 302 MB | 90 tok/s | 84 | | | | | |
| 1024, reserved at once | 302 MB | 90 | 84 | 75 | 63 | | | |
| 2048, reserved at once | 362 MB | 86 | 81 | 74 | 64 | 49 | | |
| 4096, reserved at once | 499 MB | 82 | 83 | 77 | 67 | 51 | 41 | 35 |

In the page (Chromium), with the growing cache and the stop tokens switched off: 4096 tokens in 90 s, 45 tok/s
over all of it, the heap 302 MB until position 1024, then 362 MB, and 522 MB from position 2048 to the end (20 MB
more than reserving everything at once: what the smaller arrays left behind). A text of a few hundred tokens,
which is what the model usually writes before it stops by itself, costs what it cost before.

## int8

Weights as `quantize.py` writes them: int8 values in groups of 32 with one float32 scale per group; they stay
views into the downloaded buffer and are never widened. Per matmul the input is quantized once
(`quantize_x`, shared by q/k/v and by w1/w3) and multiplied by `matmul_q8`. With relaxed SIMD, `matmul_q8r`
multiplies int8 by 7-bit unsigned values: activations get a bias of 64, and `corrections = -64 * sum of the
group` (an int32 a group, T197; before it the float32 `scale * sum`) takes it out again, added to the group's
integer sum before that is scaled. The relaxed module is loaded in `try/except`: a browser without relaxed SIMD
(shipping Safari) refuses to compile it, which arrives in Python as `OSError` (verified by switching the
feature off in Firefox), and int8 then runs on `matmul_q8`.

What quantizing the activations costs (2026-09-19, T55; `node tests/perplexity.mjs [model]`, 1499 tokens from the
beginnings of three Japanese Wikipedia articles, contexts of 512 tokens, Pyodide in Node):

| computation | llm-jp-3-150m | tiny-lm |
|---|---:|---:|
| float16 original, no quantization at all | 29.976 | not measured |
| int8 weights, NumPy, activations not quantized | 29.907 (-0.23%) | 88.327 |
| int8 weights, kernels, 8-bit activations (`matmul_q8`) | 29.965 (-0.04%) | 88.339 (+0.01% against NumPy) |
| int8 weights, kernels, 7-bit activations (`matmul_q8r`) | 30.094 (+0.39%) | 88.628 (+0.34% against NumPy) |

The percentages of llm-jp-3-150m are against its original. So the seventh bit costs about half a percent of
perplexity and buys a third more speed (11.2-12.5 against 7.6-8.3 G multiply-adds per second); the largest
difference between two logits, 1.6 as noted above, sounds worse than it is. Nothing to fix.

Except for GPT-2 (2026-09-24, T92). openai-community/gpt2 lost 17% of perplexity to the seventh bit (41.4 against
36.1 with NumPy, 1500 tokens of English Wikipedia), and all of it in the classifier's input: its final LayerNorm
has a weight that blows a few channels up 12 to 17 times (316 against a median of 0.3). One such channel sets
the scale of its group of 32, and the other 31 round to nothing. The fix is `add_columns`: the 8 channels with the
largest norm weight are taken out of the vector before `quantize_x`, and their columns of the classifier, widened
to float32 once at load, are multiplied separately in one SIMD pass over the logits (vocab_size × 8 multiply-adds
next to vocab_size × dim). GPT-2 is back to 36.127 (+0.16% against NumPy); its speed did not measurably change
(old and new in one process, 4 rounds: 57-69 against 61-70 tok/s). Only a norm whose largest weight is 4 times
its median gets it (`OUTLIER_RATIO`; GPT-2 13.9, every other model of the list 1.1-1.9): tiny-lm gains nothing
from it (+0.31% either way) and would pay about 3% of speed. `simdkernel.so` is 6307 → 6572 bytes.

## Rules that are easy to break

- **No static data.** The side module has no relocations, so a data segment would be written over Pyodide's own
  memory. That rules out AssemblyScript's std math (`Mathf.exp` uses tables), strings and asserts; `build.py`
  refuses a module with a data section. `fexp` in `kernel.ts` is the table-free replacement.
- **Keep every array alive whose address a kernel gets.** The kernels only know numbers: a scratch array that
  no Python object refers to any more is freed, and the kernel then writes into whatever lives there next. It
  shows up as a rare `memory access out of bounds` (`_kernel_buffers`, `_sampler_buffers` in the engine).
- Mutable globals are fine (they are no data segment): `sample` keeps the state of its partial sort in three.
- `sample` sorts only as far as the nucleus reaches and gets its random number from Python, so a seed gives the
  same text again. With the same random number it picks the token that `Llama.sample` (NumPy) picks.
- The relaxed module must not be named `*.so` if it ever ships inside a wheel: Pyodide pre-loads every `.so`.
- `attention` needs a scratch of `heads * (pos + 1)` floats (the engine allocates `heads * seq_len`), since T54.
- `attention` expects the KV cache as `[seq][kv_heads * head_size]` per layer (the NumPy forward uses another
  layout) and handles grouped-query attention and any head size. int8 models whose row lengths are not
  multiples of 32 run on NumPy.
- Chromium compiles at most 8 MB synchronously on the main thread; these modules are a few KB, and the site
  runs Pyodide in a worker anyway.
- The ABI this relies on is Emscripten's dynamic linking, not the CPython C API, so one binary has served every
  Pyodide version so far. A future Emscripten could break it, which is why NumPy stays as the fallback and why
  `tests/smoke.mjs` checks both paths against the latest Pyodide before every deployment.
- `?kernel=off` in the page URL forces NumPy, to compare.
