# Pyodide LLM

Run language models in your browser using Python and WebAssembly! (Until September 2026 this project was called pyodide-llama-py.)

This project leverages [Pyodide](https://pyodide.org/) to run a Python implementation of the Llama architecture (`llama2_numpy.py`, a NumPy port of `llama2.py`) directly in the web browser. It is an experiment in how far Python on WebAssembly can go, not a product. The default model is [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm), the lightest one that writes Japanese (33 MB); [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) writes far more coherent Japanese and English at a quarter of the speed and five times the download, and is one click away; the TinyStories models from the [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) project can be selected as well.

## Features

- **Pure Browser-based Inference:** No backend server required for inference.
- **Python in WebAssembly:** Python sequences the transformer layers, and small WASM SIMD kernels, loaded with `ctypes` and working in place on NumPy memory, do the math: 75 tokens/s for the 150M parameter model and 300 to 950 for the small ones, about a thousand times faster than the original pure Python loops (NumPy alone reaches 50). See [Measurements](#measurements).
- **Streaming Output:** Pyodide runs in a Web Worker and every token is shown as soon as it is generated, so the page never freezes. While a text is being written, the send button stops it.
- **Settings You Can See:** every answer says which temperature and seed wrote it and how fast; the button left of the prompt changes them, and the seed under an answer is a button that fixes it, so that two models can be compared on the same seed.
- **Models Straight from Hugging Face:** the last group of the model list is not hosted here. The page fetches `model.safetensors` from huggingface.co, converts it to int8 in your browser with the same Python code that builds the site's models, and keeps the result for the next visit ([how](#models-from-hugging-face)).
- **Your Own Model:** a llama2.c checkpoint from your disk runs without being uploaded ([how](#your-own-model)).
- **Several Models:** Japanese / English models (tiny-lm with 29M parameters by default, because a public page should not make a phone fetch 171 MB unasked; llm-jp-3 with 150M writes far better Japanese and is one click away, and the page remembers what you chose), and TinyStories models from 260K to 42M parameters. `?model=<id>` selects one directly.

## Live Demo

You can try the live demo on GitHub Pages (if configured):
[https://takano32.github.io/pyodide-llm/](https://takano32.github.io/pyodide-llm/)

## Getting Started

### Prerequisites

- Node.js 24 LTS (the page is built with [Astro](https://astro.build/))
- Python 3 with NumPy (`make` converts the tiny-lm checkpoint with `convert_hf.py` and quantizes with `quantize.py`)
- Docker (optional)

### Running Locally with Makefile

1. Clone the repository:
   ```bash
   git clone https://github.com/takano32/pyodide-llm.git
   cd pyodide-llm
   ```

2. Run the application:
   ```bash
   make run
   ```
   This will download the model files (about 1 GB) and convert tiny-lm, install dependencies, and start the Astro dev server at `http://localhost:8080/pyodide-llm/`.

### Running Locally with Docker

1. Build and run the Docker container:
   ```bash
   docker build -t pyodide-llm .
   docker run -p 8080:8080 pyodide-llm
   ```
2. Open `http://localhost:8080/pyodide-llm/` in your browser.

## Models from Hugging Face

The models of the group "From Hugging Face, converted in this browser" live on huggingface.co, not on this site.
When you choose one, the worker fetches `model.safetensors` in parts of 8 MiB over six connections, hands them to
`public/llama2_convert.py` in the order of the file, and that converts every tensor as it arrives and writes it as
int8 to its place in a buffer of the final size: the download is never held as a whole, and the peak is the
converted model plus a few dozen megabytes. The result goes into a cache of the browser, so the next visit fetches
and converts nothing; About lists what is kept and deletes it. A download of more than 500 MB asks first.

What can be on that list is narrow: a plain Llama architecture, one safetensors file, and a tokenizer the engine
reads (a Unigram `tokenizer.json` or a sentencepiece model). The instruction-tuned models get what you type inside
their chat template, for one turn; this page keeps no conversation. Measured in Chromium on the development
machine: llm-jp-3-150m-instruct3, 305 MB fetched and converted in 38 s, then ready in 8 s from the cache, 81
tok/s; llm-jp-3-440m-instruct3, 0.9 GB in 97 s, 686 MB of heap, 29 tok/s. The models of a billion parameters
are checked in CI, which has the memory and a fast line (Chromium on a Linux runner): llm-jp-3-980m-instruct3 is
ready 53 s after the click (2.0 GB fetched and converted) and writes 15 tok/s, TinyLlama 1.1B Chat is ready in
38 s and writes 12 tok/s.

A model that is in no list can be named in the URL: `?hf=<owner>/<repository>` (optionally `&revision=`,
`&template=` with `{prompt}` in it) converts a Hugging Face repository, and refuses in words what it cannot run;
`?checkpoint=<url>&tokenizer=<url>` reads files in llama2.c's format from any server that answers cross-origin
range requests, for example
`?checkpoint=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin&tokenizer=https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin`.

## Your Own Model

The folder button next to the model list (or dropping the files on the page) opens a model from your own disk.
The files are read where they are: nothing is uploaded, and nothing is requested from the network. Choose them
together:

- a checkpoint in llama2.c's legacy format, for example `stories110M.bin` of the
  [TinyLlamas](https://huggingface.co/karpathy/tinyllamas), or what `convert_hf.py` and `quantize.py` write.
  float32, float16 and int8 are told apart by the header and the file size;
- its `tokenizer.bin` (llama2.c's format; it must hold exactly the vocabulary of the checkpoint);
- optionally a `.json` with whatever differs from llama2.c's conventions, shaped like an entry of `src/models.js`:

```json
{ "name": "tiny-lm", "options": { "tokenizer_kind": "unigram", "nfkc": true, "stop_tokens": [1, 2] },
  "generation": { "steps": 256, "temperature": 0.7, "topp": 0.9, "repetition_penalty": 1.3 }, "prompt": "これからの流行りは" }
```

WebAssembly addresses 32 bits and a phone gives a tab far less, so a checkpoint of more than 1 GB asks first.

The files Hugging Face publishes work as well: choose `model.safetensors`, `config.json` and the tokenizer
(`tokenizer.json` of the Unigram kind, or a sentencepiece `tokenizer.model`) together, and the page converts them
in the browser, with the same Python code that builds the models of this site (`public/llama2_convert.py`). It
reads the weights a few megabytes at a time and writes int8 directly, so llm-jp-3-150m (305 MB of bfloat16)
takes 7 seconds and no more memory than the converted model itself, and then writes, seed for seed, what the
site's own copy writes. A `.json` next to them may say `{"conversion": {"dtype": "float16", "max_seq_len": 1024}}`
(the defaults are int8 and a context of at most 4096 tokens). Only plain Llama models are accepted.

## How it Works

1. **Pyodide Initialization:** The browser resolves the latest Pyodide release at page load and loads that runtime from the CDN, so there is no version to bump by hand. Append `?pyodide=<version>` to the URL to force a specific version.
2. **Environment Setup:** A Web Worker (`public/worker.js`) loads Pyodide, NumPy, `public/llama2_numpy.py` and the SIMD kernels (`kernels/`, compiled by `make kernels`; `?kernel=off` runs on NumPy alone). The chat-like page itself is `src/pages/index.astro`, and the model list is `src/models.js`.
3. **Model Loading:** The selected model checkpoint and its tokenizer are downloaded while Pyodide is still loading, in parts of 8 MiB over several connections at once (about 1.8x as fast as one stream), straight into one preallocated buffer while a progress bar shows the download. The larger models are distributed as int8 (3.5x smaller; measured perplexity cost on stories15M: +0.04%) and widened to float32 once, and their unquantized originals can be selected for comparison; float32 weights of the small models are NumPy views into the buffer, nothing is copied.
4. **Inference:** When you press the send button (or Ctrl / Cmd + Enter; Enter alone breaks the line), the prompt is sent to the worker, where a Python generator yields the text token by token; each piece is posted back and appended to the output.

No binary is committed to this repository: `make models` downloads the model files when the site is deployed (or for `make run`). llm-jp-3 and tiny-lm are published in Hugging Face format, so `convert_hf.py` converts them, with nothing but NumPy, into the llama2.c checkpoint and tokenizer formats that `llama2_numpy.py` reads, and `quantize.py` turns the larger checkpoints into int8. Their tokenizers are sentencepiece unigram models, which `llama2_numpy.py` encodes with a Viterbi search (the Llama 2 vocabulary of the TinyStories models uses llama2.c's pair merging).

## Measurements

All numbers below were measured on 2026-09-19 on one phone-class ARM CPU (Cortex-A78 x4 + A55 x4), a single
thread, no swap. An overview of the whole project in Japanese, and the survey of the other browser ports that it
started from, are in this [gist](https://gist.github.com/takano32/196c6f93979ad44f98cee5712fdd3901).

Tokens per second, stories15M, greedy:

| implementation | tok/s |
|---|---:|
| pure Python (`llama2.py`) | 0.26 |
| NumPy in Pyodide (Node), float32 | 53 |
| NumPy in Pyodide (Node), int8 | 55 |
| SIMD kernels in Pyodide (Node), float32 | 200 |
| SIMD kernels in Pyodide (Node), int8 | 351 |
| native llama2.c, `gcc -Ofast`, for reference | 214 |

Plain JavaScript reached 38 tok/s and a standalone WASM SIMD build 170-190 on the same machine. Threads did not
help at this model size, not even with native OpenMP, so the site does not use them.

In Chromium, per model (`?kernel=off` gives the NumPy column):

| model | NumPy | kernels |
|---|---:|---:|
| stories260K float32 | 268 | 951 |
| stories3_5M float32 | 141 | 402 |
| stories15M float32 / int8 | 50 | 186 / 296 |
| tiny-lm int8, sampled with a repetition penalty | 43 | 271-296 |
| llm-jp-3-150m int8, sampled, 256 tokens | 8.5 | 75-79 |

The tiny-lm and llm-jp rows are the current defaults. They were 171 and 47 tok/s while NumPy still did the
sampling, and 252-274 and 61-66 before attention learned to walk its cache row by row (T54). A weekly workflow runs the deployed site on Linux, Windows (both on x86-64 and ARM) and macOS, in Playwright's
Chromium, Firefox and WebKit and in the installed Chrome and Edge: all 23 combinations run all four models
([table](kernels/README.md#every-browser-the-runners-offer-2026-09-19-t44-and-t58)). WebKit, Safari's engine, has no
relaxed SIMD, so int8 runs on the plain SIMD kernel there. Firefox is as fast as the Chromium family (llm-jp-3-150m: 106 against 107
tok/s on the same Linux runner), measured in the Firefox that the runners have installed, through Selenium. The
Firefox that Playwright drives looks 8 to 15 times slower, because Playwright drives it through the debugger, and
a debugged page gets its WebAssembly from the baseline compiler only: those numbers say nothing about Firefox.
Safari itself, and a real iPhone, were not measured.

Where a token of llm-jp-3-150m goes (`node tests/profile.mjs`, the same in Node and in Chromium): the matrix
products of the layers 52%, the classifier over 99584 tokens 38%, the ctypes calls 6%, sampling 3%, Python and
NumPy around the calls 1%. The interpreter is no longer what limits it. Nor is memory bandwidth: the int8 product
is as fast on a matrix of 75 MB as on one that fits the cache, so its arithmetic is the limit
(details in [kernels/README.md](kernels/README.md)).

By default a model writes until it stops by itself or its context is full, and llm-jp-3-150m has its whole
context of 4096 tokens: the KV cache grows with the text (302 MB of heap for a short text, 522 MB at the very
end), and the speed falls with the position, from 80 tok/s to 35 at position 4070, 45 tok/s over all 4096 tokens.
The numbers in the tables are about 256 tokens.

Memory, llm-jp-3-150m int8: the kernels multiply the int8 weights as they are instead of widening them to
float32, which takes the WASM heap from 897 MB to 283 MB.

Quantization to int8 (groups of 32, one float32 scale per group) is not distinguishable from the original in
perplexity, while int4 is:

| model | original | int8 |
|---|---:|---:|
| stories15M | - | +0.04% |
| tiny-lm | 91.3 | 91.1 |
| llm-jp-3-150m | 22.76 | 22.69 |

That table is about the weights alone. The kernels also quantize the activations, to 8 bits, or to 7 where
the browser has relaxed SIMD: on another text, llm-jp-3-150m has a perplexity of 29.98 as the float16 original,
29.91 with int8 weights, 29.97 with 8-bit activations and 30.09 (+0.4%) with 7-bit ones (`node tests/perplexity.mjs`).

The most likely token agrees about 98% of the time; greedy output diverges from the original part way through
but stays coherent. int4 cost +16.8% perplexity and was rejected.

Downloading the model in parts of 8 MiB over several connections at once is about 1.8x faster than one stream:
167 MB took 20.4 s in one stream and 11.2 s split, against the production CDN.

## Acknowledgments

- [Pyodide](https://pyodide.org/) for the Python WebAssembly runtime.
- [llama2.py](https://github.com/tairov/llama2.py) by tairov for the pure Python Llama 2 implementation.
- [llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) by LLM-jp (Apache License 2.0) for the default Japanese / English model.
- [tiny-lm](https://huggingface.co/sbintuitions/tiny-lm) by SB Intuitions (MIT License) for the Japanese / English model; its license is deployed next to the converted file.
- [TinyLlamas](https://huggingface.co/karpathy/tinyllamas) by Andrej Karpathy and [ellishg/tinyllamas](https://huggingface.co/ellishg/tinyllamas) for the compact TinyStories checkpoints.
- [llama2.c](https://github.com/karpathy/llama2.c) for the inspiration and model format.

## License

[Mozilla Public License 2.0](LICENSE), the same as Pyodide's. Some files carry code from other projects under their own licenses, and keep those notices where the code is: `public/llama2_numpy.py` (tairov/llama2.py and karpathy/llama2.c, MIT) and `public/shaders.js` (llama.cpp and ONNX Runtime, MIT; TensorFlow.js, Apache-2.0). The models are not in this repository and each keeps its own license (see the model list).
