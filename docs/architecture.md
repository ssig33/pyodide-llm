# How it works

This page describes the parts of the site and how they talk to each other. The decisions behind them, with dates
and measurements, are in [AGENTS.md](../AGENTS.md) (in Japanese).

## The parts

```mermaid
flowchart TD
  page["the page: src/pages/index.astro"]
  worker["Web Worker: public/worker.js<br/>resolves Pyodide, downloads and converts models, keeps them (public/kept.js)"]
  engine["the engine: public/llama2_numpy.py<br/>checkpoint, tokenizer, sampling, generate()"]
  convert["the converter: public/llama2_convert.py<br/>safetensors and GGUF to the llama2.c format, int8 or 6 bits"]
  forward["public/forward.js<br/>the forward pass, on its own WebAssembly memory"]
  kernels["kernels/: WebAssembly SIMD kernels"]
  helper["public/helper.js: software threads"]
  gpu["public/gpu.js and public/shaders.js: the GPU worker and its WGSL shaders"]
  page -- "the model, the prompt, the settings" --> worker
  worker -- "each piece of text" --> page
  worker --> engine
  worker --> convert
  engine -- "tensor positions; one token or a block of the prompt" --> forward
  forward --> kernels
  forward --> helper
  forward -- "blocks of a prompt" --> gpu
```

| Part | What it does |
|---|---|
| `src/pages/index.astro` | The chat page. It draws what the worker reports; the URL holds the state (`?model=`, `?hf=`, `?bits=`, `?without=` and others). |
| `src/models.js` | The model list: files, sizes, engine options, generation settings, chat templates. The first entry is the default. |
| `public/worker.js` | Loads Pyodide and NumPy, downloads the model in parts while Pyodide loads, and runs `generate()`. |
| `public/llama2_numpy.py` | The engine. Reads llama2.c's legacy format (float32, float16, int8, 6 bits), the tokenizers (llama2.c's BPE, sentencepiece Unigram, byte-level BPE), the architectures (Llama, Qwen2, Qwen3, GPT-2, GPT-NeoX), and samples. Without the kernels, NumPy does the arithmetic. |
| `public/llama2_convert.py` | Converts a Hugging Face model (or a Q8_0 GGUF) as its file arrives, and writes it into the model's memory. The same code builds the site's models (`convert_hf.py`, `quantize.py`) and converts in the browser. It also reads a model's chat template (a small part of Jinja). |
| `public/forward.js` | One token's forward pass, and a block of prompt tokens, in JavaScript. It calls the same kernels in the same order as the Python engine would. |
| `kernels/` | The SIMD kernels: int8 and float32 matrix products, activation quantization, RMSNorm, LayerNorm, RoPE, attention, SwiGLU, GELU, and the sampling (repetition penalty, softmax, top-p). |
| `public/helper.js`, `public/jobs.js` | Software threads, and the work they share. |
| `public/gpu.js`, `public/shaders.js` | The GPU worker and all WGSL shaders. |
| `public/coi.js` | The Service Worker: adds the headers for threads, keeps Pyodide and NumPy for offline use. |
| `public/benchmark/`, `src/pages/benchmark.astro` | `/benchmark/`, the measurements of one device in one report. |

## Why the forward pass is in JavaScript

Until September 2026, Python ran the layers in order and called the kernels through `ctypes`, working in place on
NumPy memory. Two things moved the loop out of Python:

- Each `ctypes` call cost about 3 µs, and calls between layers added up (6% of a token of llm-jp-3-150m). Moving the
  loop of layers to JavaScript made int8 models 1.15 to 1.26 times faster.
- Pyodide does not expose its own `WebAssembly.Memory`, so JavaScript could not run the kernels on Python's
  memory. The weights now live in a separate WebAssembly memory that `forward.js` owns, and Python only tells it
  where each tensor is. A converted model is written straight into that memory.

Python still does everything around the arithmetic: reading and converting models, the tokenizer, the chat
template, the sampling (through a kernel), and the text that goes back to the page. NumPy remains the fallback
when the kernels cannot be loaded (`?without=kernels`), and the reference that the tests compare with.

## Threads

A page on GitHub Pages cannot send the headers (COOP and COEP) that allow `SharedArrayBuffer`. The Service Worker
(`public/coi.js`) adds them to every response, so after one reload the page is cross-origin isolated and can share
memory between workers. Where that fails, the page runs on one thread.

With shared memory, the forward pass is split among software threads: each matrix product is cut into chunks of
rows that the threads take in turn, and attention is split by head. The page searches for the fastest number of
threads on each device and writes its decision to the console (lines starting with `threads:`).

## Memory

- WebAssembly has 32-bit addresses (4 GB). Where the browser has 64-bit memory (Chrome and Firefox), a model that
  does not fit runs on it, about 10% slower. Where it does not (Safari), the page stores the weights in 6 bits.
- What the forward pass needs besides the weights (the KV cache for the whole context, its growth, the
  activations) is computed from the header before the model is loaded (`footprint()` in `forward.js`).
- The model's WebAssembly memory is reused when another model is chosen: Chromium would not create a third
  WebAssembly memory on one page.

## Downloading and converting

- The site's models are split into parts of 8 MiB and fetched over 8 connections while Pyodide loads.
- Hugging Face models are fetched in parts of 8 or 16 MiB over 6 connections, in the order of the file, and each
  tensor is converted when it arrives. The download is never held as a whole: the peak is about the converted
  model.
- Several of the models in the list are fetched as a Q8_0 GGUF with the vocabulary and the configuration of the
  original repository: half the download, the same weights to within the rounding of the scales.
- Converted models are kept in the browser (OPFS, or the Cache API), so the next visit fetches nothing.

## The GPU

Where the browser has WebGPU in a worker, `forward.js` starts a GPU worker for each model. It copies the int8
weights of the layers to the GPU, checks each shader against JavaScript on this device, times them, and then runs
the blocks of a prompt (up to 64 tokens) on the GPU where that was measured faster than the CPU. The keys and
values it computes are written back into the CPU's cache, and the tokens that follow run on the CPU. See
[webgpu.md](webgpu.md).
