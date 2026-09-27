# Performance

Speed in tokens per second (tok/s), with where each number was measured. The machines differ, so numbers from two
machines are not compared with each other.

- **The first development machine** (until 2026-09-26): a phone-class ARM CPU (Cortex-A78 ×4 + A55 ×4), 6.6 GB,
  no swap.
- **The second development machine** (from 2026-09-26): a cloud ARM server (Ampere A1, Neoverse-N1 ×2), 11.9 GB.
- **CI**: GitHub's runners, shared virtual machines. Their speed differs from run to run, so they are not used to
  compare systems.
- **The owner's devices**: an Android phone (Xiaomi 13T Pro), an iPhone, an ARM Chromebook.

## From pure Python to kernels

stories15M, greedy, one thread, the first development machine (2026-09-19):

| implementation | tok/s |
|---|---:|
| pure Python (`llama2.py`) | 0.26 |
| NumPy in Pyodide (Node), float32 | 53 |
| NumPy in Pyodide (Node), int8 | 55 |
| SIMD kernels in Pyodide (Node), float32 | 200 |
| SIMD kernels in Pyodide (Node), int8 | 351 |
| native llama2.c, `gcc -Ofast`, for reference | 214 |

Pyodide's NumPy is built without BLAS and without SIMD, so it runs like scalar WebAssembly. Replacing it with
SciPy's OpenBLAS was only 1.15 times faster.

What each step adds, with `?without=kernels,int8,relaxed,sampler` (Node's Pyodide, 64 tokens, the first
development machine):

| | tiny-lm | llm-jp-3-150m |
|---|---:|---:|
| NumPy only | 48.6 | 9.6 |
| + the kernels, int8 widened to float32 | 187.1 | 37.9 |
| + int8 kept as int8 | 248.1 | 65.4 |
| + relaxed SIMD (7-bit activations) | 315.3 | 84.3 |
| + sampling in a kernel | 415.6 | 92.8 |

In Chromium, tiny-lm goes 44.8 → 82.5 → 175.8 → 193.0 → 334.6 tok/s.

## In the browser

Chromium, the first development machine, with the kernels (`?kernel=off` gives the NumPy column):

| model | NumPy | kernels |
|---|---:|---:|
| stories260K float32 | 268 | 951 |
| stories3_5M float32 | 141 | 402 |
| stories15M float32 / int8 | 50 | 186 / 296 |
| tiny-lm int8, sampled with a repetition penalty | 43 | 271-296 |
| llm-jp-3-150m int8, sampled, 256 tokens | 8.5 | 75-79 |

Firefox is as fast as the Chromium family (llm-jp-3-150m: 106 against 107 tok/s on the same Linux runner),
measured in the Firefox that the runners have installed, through Selenium. The Firefox that Playwright drives looks
8 to 15 times slower because Playwright drives it through the debugger, and a debugged page gets its WebAssembly
from the baseline compiler only. WebKit has no relaxed SIMD, so int8 runs on the plain SIMD kernel there. Per
browser and system: [kernels/README.md](../kernels/README.md).

## Where a token goes

`node tests/profile.mjs`, the JavaScript forward pass, one thread, the first development machine:

| model | ms per token | matrix products | of which the classifier |
|---|---:|---:|---:|
| llm-jp-3-150m | 9.2 | 92% | 46% |
| tiny-lm | 1.6 | 92% | 74% |
| stories15M | 1.6 | 88% | 53% |

Attention grows with the position: in llm-jp-3-150m it is 2% of a token at position 16, 48% at 2000, 65% at 4000.

## Long texts

By default a model writes until it stops or its context is full; llm-jp-3-150m has 4096 tokens. Its KV cache grows
with the text (302 MB of heap for a short text, 522 MB at the end), and the speed falls with the position: about
85 tok/s at position 8, 65 at 1000, 50 at 2000, 35 at 4070 (Node). The tables above are about 256 tokens.

## Threads

A token of an int8 model reads every weight once, one multiply-add per byte. So threads help until memory
bandwidth is full:

- On the first development machine, 1.2 to 1.3 times (the int8 kernel already reads about 15 GB/s on one thread;
  the ceiling was about 20).
- On machines with more bandwidth, more: on CI's Linux ARM runner (4 vCPUs), 4 threads were 1.6 to 1.9 times one
  thread (tiny-lm and llm-jp-3-150m). In Chromium on CI's Linux x64: Qwen2.5 0.5B 27.8 → 41.1 tok/s, Qwen2.5 1.5B
  9.3 → 16.1, llm-jp-3 980M 15 → 28.5.
- Rows are handed out in chunks that the threads take in turn, so that a slow core does not hold the others back.
  On the first development machine, 8 threads were no faster than 4. The page searches for the best number on each device.
- A prompt is run in blocks of up to 16 tokens, so that each chunk of weights is used for all of them while it is
  in the cache. On one thread that is 1.16 times; with threads it is more (llm-jp-3-150m's prompt, 4 threads,
  2.91 times, the first development machine).
- With threads, the KV cache is kept in float16: half the reads, which is faster when memory is the limit (4
  threads at position 4000: 44 → 52-55 tok/s) and slower on one thread (38 → 21). Its quality cannot be told from
  float32 (perplexity 29.957 against 30.094).

The owner's Android (`/benchmark/`): one thread reads 12.9 to 18.9 GB/s for a token, 4 threads 27.6 to 28.7 GB/s;
a prompt on 4 threads reaches 30.8 to 43.9 G multiply-adds per second.

How close each kernel is to the ceiling of the machine (on the second development machine):
[notes/t158-cpu-audit-2026-09-27.md](notes/t158-cpu-audit-2026-09-27.md) (in Japanese).

## Larger models

Chromium on CI's Linux runner, from the click to "ready" (download and conversion included) and the speed:

| model | ready | tok/s | heap |
|---|---:|---:|---:|
| llm-jp-3 150M instruct3 | 13 s | 106 | |
| llm-jp-3 440M instruct3 | 28 s | 35 | |
| TinyLlama 1.1B Chat | 38 s | 12 | |
| llm-jp-3 980M instruct3 | 53 s | 15 | |
| Qwen2.5 3B Instruct (int8, 32-bit memory) | 67.4 s | 8.1 | 4072 MB |
| Llama 3.2 3B Instruct (int8, 64-bit memory) | 64.4 s | 7.5 | 4266 MB |
| sarashina2.2 3B Instruct (int8, 64-bit memory) | 73.7 s | 7.7 | 4409 MB |
| Qwen2.5 7B Instruct | 162.5 s | 3.9 | 9716 MB |
| llm-jp-4 8B instruct | 219.6 s | 3.6 | 11029 MB |
| Llama 3.1 Swallow 8B Instruct | 254.7 s | 3.4 | 10317 MB |

The first four rows were measured before the threads; the 3B to 8B rows with 4 threads. These runners have a fast
line; from Japan, huggingface.co delivered 7 to 9 MB/s, so a 1B model takes about 5 minutes to fetch, and the
conversion about 5 seconds.

## Downloading

- The site's model in parts over several connections: 167 MB took 20.4 s in one stream and 11.2 s split.
- Hugging Face: the size of each part matters more than the number of connections (Qwen2.5 0.5B from CI: 23 s with
  4 MiB parts, 11.3 s with 8 MiB, 8.3 s with 16 MiB). On a slower line, the line is the limit.
- The model download starts before Pyodide has loaded and usually ends first: in Chromium, tiny-lm's 33 MB were in
  after 3.2 s and the page was ready at 10.2 s, waiting for Pyodide and NumPy.
- A converted model is kept in the browser: llm-jp-3 440M was ready in 4.4 s the second time, 18.4 s the first
  (Chromium).

## Not measured

- Safari on a Mac, and iPhones other than the owner's.
- The speed of the GPU path on real devices: see [webgpu.md](webgpu.md).
