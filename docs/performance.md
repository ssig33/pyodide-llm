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

Around the forward pass, Python chooses each token (`node tests/overhead.mjs`, the CI's x86-64 and arm64 runners,
one thread, 128 tokens): what a generated token costs outside `forward.js` is 0.31-0.35 ms for llm-jp-3-150m (4% of
the token) and 0.24-0.35 ms for tiny-lm (16-20%), most of it the sampling kernel (measured before that kernel's
speed-up below). A call from Python into
JavaScript costs 1.3-1.7 µs. The first code after a forward pass is slower than the same code warm, since the pass
has read the weights through the caches. With the threads a page uses, the forward pass is 1.5-1.9 times shorter on
those runners and the outside stays, so it is an estimated 5-8% and 26-27% there (not measured).

## The kernels, 2026-09-27

Changes to the CPU kernels, each timed against the kernel before it in the same process, taking turns. The CI's
x86-64 runners draw a different CPU from run to run, so the CPU is named where it mattered. None of these has been
timed on the owner's devices yet.

| change | measured on | times as fast |
|---|---|---:|
| a prompt's int8 matrix product in tiles of 4 rows × 4 tokens | CI arm64 (Neoverse-N2), 1 and 4 threads | 1.17-1.33 |
| the same, llm-jp-3-150m's prompt in blocks of 16 tokens | CI arm64, 1 thread / 4 threads | 1.25-1.27 / 1.09-1.15 |
| the same, 4 to 6 tokens of narrow rows | CI x86-64 (AMD EPYC 9V45), 1 thread | 0.88-0.99 |
| the sampling kernel, llm-jp-3-150m / tiny-lm | CI x86-64 (EPYC 7763) | 2.35 / 2.07 |
| the same | CI arm64 | 1.54 / 1.28 |
| 6-bit weights widened in 8 instructions instead of 15 | CI x86-64 / arm64 | 1.64 / 1.26-1.30 |
| Safari's int8 kernel (no relaxed SIMD) | the second development machine, 1 thread | 1.17-1.21 |
| SwiGLU and GELU four values at a time | the second development machine, 1 thread | 4.12 / 4.24 |
| one token's int8 kernel scales a group once, not four times | CI arm64 (Neoverse-N2) / x86-64 (EPYC 9V45) | 1.04-1.12 / 0.88-0.95 |

The last row is slower on AMD's Zen 5 with one thread, where the kernel before it already read 87 to 94% as fast as
a loop that only reads; with 4 threads it is 0.99 to 1.01 there. The owner chose to keep it. The prompt's tiles
reduce only the reads: the bookkeeping of each group stays, so they stay far from what a loop of dot products
alone reaches.

## The tokenizer

Before the first token, Python encodes the prompt. `node tests/encode-bench.mjs` compares the engine's `encode()`
with an earlier version in one Pyodide (CI's x86-64 runner, an AMD EPYC 7763, and its arm64 runner; ms for a text
of about 64 tokens, x86-64 / arm64):

| vocabulary | before 2026-09-27 | now |
|---|---:|---:|
| Llama 2 (llama2.c's BPE) | 6.06 / 6.58 | 1.09 / 1.12, then 0.55 on x86-64 with the heap below |
| tiny-lm (Unigram) | 1.02 / 1.07 | 0.55 / 0.58 |
| llm-jp-3-150m (Unigram) | 3.09 / 3.33 | 1.34 / 1.43 |

rinna's sentencepiece model is 3.0 times as fast, Llama 3.2's byte-level BPE 2.2 times, GPT-2's and Qwen3's 1.08
to 1.12 times. llama2.c's BPE used to grow with the square of the text; it now takes its pairs from a heap, and
1040 tokens take 9.9 ms instead of 185.8 (EPYC 7763), 10.6 instead of 172.5 (arm64); before both changes, 1000
tokens took 1.7 s. In exchange, loading a Unigram tokenizer takes 11 to 38 ms longer, once per model, and it keeps
more in Pyodide's memory (llm-jp-3-150m: 9.1 → 15.2 MiB). These speed-ups did not change any ID.

Since 2026-09-28 a sentencepiece model normalizes with its own map (tiny-lm's, rinna's and Swallow-MS's; the others
have none), walked in Python. That costs back part of the speed-up above: tiny-lm's text of about 64 tokens takes
0.28 → 0.39 ms on an Intel Xeon 6973P-C and 0.59 → 0.82 ms on the arm64 runner (1000 tokens: 4.9 → 6.6 and
10.4 → 13.6 ms), rinna's and Swallow-MS's are 0.68 to 0.74 times as fast as before, and the tokenizers without a map
(llm-jp-3-150m, Llama 2, the byte-level ones) are as fast as before. The map adds about 1 MiB to tiny-lm's tokenizer
in Pyodide (7.8 → 8.7 MiB).

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
  threads at position 4000: 44 → 52-55 tok/s) and slower on one thread (38 → 21), on the first development machine.
  Its quality cannot be told from float32 (perplexity 29.957 against 30.094). Since T160 a float16 number is widened
  in four instructions instead of seven (at position 4000, 1.14 to 1.35 times the tokens per second on the CI
  runners), yet on those runners a float32 cache is still as fast or faster at 1, 2 and 4 threads, and 1.15 to 1.44
  times faster for a model with grouped-query attention (Qwen2.5 0.5B at position 2000), which widens each key and
  value once for every query head that shares it. So since T160 such a model keeps its cache in float32 with threads
  too (Qwen2.5 0.5B at position 2000, against float16: 1.45 to 1.52 times on one thread, 1.16 to 1.31 on four, CI's
  x86-64 and arm64), unless float32 would not fit a 32-bit memory (Qwen2.5 3B) or the model needs a 64-bit memory
  anyway (Llama 3.2 3B, the 7B and 8B models): those keep float16, and so does a model with a key for every head.

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
- The kernels and the tokenizer of 2026-09-27 on the owner's devices.
- Which side the page chooses on real devices, and how fast the GPU path is there: see [webgpu.md](webgpu.md).
