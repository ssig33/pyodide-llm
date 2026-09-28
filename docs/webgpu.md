# The GPU (WebGPU)

Where the browser has WebGPU in a worker, the page uses the GPU without any option, and checks on each device
whether that is faster than the CPU. Where it is not, the page stays on the CPU. This page says what runs on the
GPU today, how the page chooses, and what has been measured.

## What runs on the GPU today

**The blocks of a prompt**, up to 64 tokens at a time, and **the tokens the model writes**, 4 at a time (the
forward pass and the sampling of each on the GPU, the ids read back once), for models of Llama's shape. Each goes to
the GPU only where the device measures it faster than the CPU.

The reason is the first measurement (2026-09-26, on the owner's three devices): moving the generation of one token
to the GPU as it was on the CPU was slower everywhere. For Llama 3.2 1B, the GPU's speed divided by the CPU's was
0.97 on the Android phone, 0.69 on the iPhone (Safari) and 0.53 on the ARM Chromebook. The phones' GPUs read large
matrices about 3 times as fast as one CPU core, but each token carried a fixed cost of about 18 to 20 ms (about 240
dispatches, small matrices, reading the result back). A prompt is different: one read of the weights serves many
tokens, and the fixed cost is shared among them.

## How it works

1. For each model, `forward.js` starts a GPU worker (`public/gpu.js`). It copies the int8 weights and scales of
   the layers and their norms to the GPU. Where the tokens of the answer can run on the GPU (step 5), the
   classifier, the embeddings and the RoPE table go there as well; otherwise they stay on the CPU.
2. It compiles the candidate shaders for the matrix products and checks each against JavaScript on a small matrix
   on this device (subgroup order, float16 rounding and driver errors only show there). Then it times the ones
   that are right on the first layer of the model, and takes the fastest.
3. It times whole blocks of 16 and 64 tokens on the GPU. The page times the CPU on real prompts. For each block, it
   runs on the GPU only if the GPU is expected to take less than 0.95 of the CPU's time. Short prompts stay on the
   CPU, because the GPU's fixed cost is not shared among enough tokens.
4. The GPU writes the keys and values of each layer back into the CPU's cache (in float16; widened to float32 where
   the CPU keeps them so, for most models with grouped-query attention), so either side can go on from any position.
5. For the tokens the model writes, the GPU runs one token's layer in a fused form (the matrices of the one-token
   benchmark below: llama.cpp's `mul_mat_vec` or ONNX Runtime's DP4A), checks each form against JavaScript on the
   model's own first layer and classifier, and takes the fastest. Four tokens go in one submission, each sampled on
   the GPU with the random number the CPU drew for it. The page times a token on each side and gives the tokens to
   the GPU where it is faster by more than 5%. Models with biases (Qwen2), per-head norms (Qwen3), LayerNorm (GPT-2,
   GPT-NeoX) or matrices whose parts do not start on the device's binding alignment keep their tokens on the CPU,
   and so, as a precaution, does a browser that does not say how much memory the device has (Safari, Firefox):
   with the answer on the GPU the classifier and the embeddings go there too (for llm-jp-3-150m the GPU's share
   grows from about 73 to 189 MB, estimated), and such a browser gives no way to tell whether that fits.
6. Every 8 answers, the page measures the side it did not choose again, on part of a prompt and on the first
   tokens of an answer, in case the device has warmed up or cooled down.

The page is ready without waiting for the GPU: until the GPU is ready, prompts run on the CPU. The shape it chose
is remembered per model and device, so the next visit compiles two shaders instead of all of them. When another
model is chosen, the page waits (up to 5 seconds) for the GPU worker to let go of its buffers and its device
before it loads the next one, so that the two do not hold memory at the same time.

The status line says what happens. Before the page has timed both sides it says "WebGPU where it is faster than the
CPU". Then it says "prompts and answers on WebGPU" where the GPU is faster for both, "prompts on WebGPU, answers on
the CPU (faster here)" where the CPU writes the answer faster, and "answers on the CPU" where the GPU cannot write
the answer for this model (the reason is in the console). The prompts can also say "prompts of 29 tokens and more on
WebGPU", "prompts on the CPU (faster here than WebGPU)", or "prompts on the CPU (reason)" where the GPU is not used
at all.

## Models too large to hold twice

A WebAssembly memory cannot shrink, so the page cannot load a model on both sides, measure, and then drop the CPU's
copy. It decides before the weights arrive. If the weights on both sides fit in half of the memory the device
reports, the page keeps both and measures, as above. Chromium reports at most 8, which is read as "8 GB or more";
there both copies may take up to 6.5 GiB, so the models of the list up to 2B keep both, and those of 3B and more do
not.

Otherwise a Llama-shaped int8 model whose tokens the GPU can write goes on the GPU alone: the layers' matrices and the
embedding and classifier go to the GPU as they are converted or read, and only the norms stay in the CPU's memory. The
GPU embeds the prompt's tokens itself and keeps the keys and values; nothing but the chosen tokens comes back. Nothing
then runs on the CPU. On a device that reports 8 there is no limit to its size, as there is none for the CPU alone (a
7B model takes about 9.2 GB there); on a smaller device it must fit in half of the memory. A browser that does not
report the memory (Safari, Firefox) does not put a model on the GPU alone, because it keeps the answer on the CPU.

With the model on the GPU alone, the page cannot time the CPU, so it estimates the CPU from `/benchmark/`. If the
CPU section has been run on this device, a written token takes the time to read the model's weights at the speed
that section measured, and a token of a prompt takes the model's multiply-adds at the section's prompt speed. The
GPU's side is what the GPU worker timed as it started. The two are weighed by how the page has been used recently
(the tokens of prompts against the tokens written, as many of each until it knows). If the CPU would take less than
0.95 of the GPU's time, the page loads the model again on the CPU and remembers this for the model on this device,
browser version and shaders, so the next visit loads it on the CPU at once. A new run of the CPU section, another
browser version or new shaders make the page weigh the two again. Without a run of `/benchmark/` the model stays on
the GPU. If the GPU fails, the page also loads the model again on the CPU.

A model converted from Hugging Face is kept as it arrives. 6-bit models are not put on the GPU alone: the GPU holds
them widened to int8, which is larger. On the GPU alone, Llama 3.2 3B takes about 4.1 GB in all (4.09 GB on the GPU
and 0.02 GB on the CPU's side) against 4.7 GB on the CPU alone, and Llama 3.1 Swallow 8B about 9.6 GB against 10.8 GB
(estimates for 4096 positions, without the page and Pyodide themselves; not measured).

## Where the GPU is not used

- The browser has no WebGPU in a worker (Firefox so far, including Firefox 156 on Android).
- The adapter is a fallback that runs on the CPU (SwiftShader, lavapipe): it would never be faster, and compiling
  the shaders took 2 to 4 minutes.
- The page is not cross-origin isolated (no shared memory between the workers).
- The model's weights are float32 (not int8 or 6-bit). Qwen2's biases, Qwen3's per-head norms of q and k, and
  GPT-2 and GPT-NeoX (LayerNorm, GELU, the biases, learned positions, partial RoPE, the parallel residual) run on
  the GPU as well. A model in 64-bit memory (over 4 GB) goes as one in 32-bit memory, and a matrix larger than a
  buffer the device binds goes in pieces of rows. 6-bit weights are widened to int8 on the GPU as they are
  uploaded, so they take as much GPU memory as int8. The page chooses 6 bits only where memory is short (a device
  that reports less than 8 GB, or Safari for a model past 4 GB), and there the rule below keeps the model on the
  CPU: today a 6-bit model reaches the GPU in practice only when it is asked for (`?bits=6`).
- The weights would not fit twice (in WebAssembly memory for the CPU and again on the GPU; on phones and Apple
  devices both are the same memory), and the model cannot go on the GPU alone (above): Qwen2, Qwen3, GPT-2,
  GPT-NeoX and 6-bit models. Twice must fit in half of `navigator.deviceMemory`, or in 6.5 GiB where Chromium
  reports 8. A browser that does not report it (Safari, Firefox) is taken as 4 GB, so the 1B models stay on the CPU
  there. The prompts of such a model still go to the GPU where its layers alone fit in the room the CPU's copy
  leaves (Qwen2.5 3B's do not).

## The shaders and where they come from

The shapes are taken from public implementations, and each file keeps their notices (`public/shaders.js`):

| What | Source |
|---|---|
| Prompt matrix products, tiles in registers (32×32 and 64×64, float16) | llama.cpp's WebGPU backend (MIT). The float32 variant is ours. |
| Prompt matrix products, vec4 tiles | TensorFlow.js's `matmul_packed_webgpu.ts` (Apache-2.0) |
| Prompt matrix products with packed int8 dot products (DP4A) | ONNX Runtime Web's MatMulNBits (MIT) |
| Attention | llama.cpp's `flash_attn_tile` (MIT). The path for devices without subgroups is ours. |
| RMSNorm, and Qwen3's per-head norms of q and k | llama.cpp's `rms_norm_mul` (MIT) |
| Qwen2's biases of q, k and v, GPT-2's and GPT-NeoX's biases | llama.cpp's `binary` ADD (MIT) |
| LayerNorm of GPT-2 and GPT-NeoX | llama.cpp's `row_norm` NORM (MIT), with the weight and the bias in the same dispatch as llama.cpp's Metal `kernel_norm_mul_add_f32` has them (MIT) |
| GELU of GPT-2 and GPT-NeoX | llama.cpp's `unary` GELU (MIT) |
| One token's matrix × vector (benchmark only) | llama.cpp's `mul_mat_vec`, ONNX Runtime's MatMulNBits (MIT) |
| The sampling split over chunks of the vocabulary, in 4 dispatches (benchmark only) | MLC LLM's two-stage softmax (`chunk_lse`, `softmax_with_chunked_sum`; Apache-2.0) and llama.cpp's `argmax`, `soft_max` and `cumsum` (MIT). The order across chunks, the penalty applied only by the chunk that holds the token, and the draw in two stages without the nucleus are ours. |
| One token's layer in 5 dispatches instead of 14 | built on llama.cpp's `mul_mat_vec` |
| One token's layer on packed int8 dot products, the vector quantized before each matrix | ONNX Runtime's DP4A MatMulNBits for small M (MIT), with the fused writes of the line above. The norm and its quantizing in one dispatch take their form from vLLM's `rms_norm_per_block_quant` (Apache-2.0; no lines copied). |
| 6-bit weights widened to int8 as they are uploaded | ours (the packing is this project's); four values a 32-bit word by byte masks and shifts, the form of llama.cpp's Q6_K in CUDA and Metal (MIT; no lines copied) |
| The device's ceilings (benchmark only) | the loops of clpeak (GPL-3.0): the shapes only, no lines copied |
| Sampling on the GPU and several tokens a submission: the repetition penalty, softmax, top-p and the draw, the next token's row of the embedding | the penalty of MLC LLM (Apache-2.0); llama.cpp's `argmax`, `soft_max`, `cumsum` and `get_rows` (MIT); top-p without sorting from MLC LLM's `top_p_pivot` (Apache-2.0). Carrying the state from one token to the next, and the draw by the same pivots, are ours. |

## Measured

On the owner's Android phone (Xiaomi 13T Pro, Chrome 153, Arm Valhall GPU), in the `/benchmark/` reports of
2026-09-27, each number from one run:

- **The device's ceilings**: 1146 GFLOPS in float32, 1650 in float16, 4684 G operations per second in packed int8
  dot products, 242 GB/s reading workgroup memory and 39.9 GB/s reading a buffer.
- **A prompt** of 64 tokens: ONNX Runtime's DP4A tile of 64×64 took 0.80 ms per token (305 GFLOPS), 7.1 times the
  first shader (`batched`, 43 GFLOPS) and 5.1 times the CPU on 4 threads (4.03 ms per token). At 16 tokens it was
  1.2 times the CPU. TensorFlow.js's vec4 tile reached 68 GFLOPS, and llama.cpp's float16 tiles 10 to 15 GFLOPS
  (why they are so slow on this GPU is not known).
- **One token's matrix × vector**: ONNX Runtime's DP4A form for small M read Llama 3.2 1B's feed-forward matrix (w1) at
  38.6 GB/s, 96.8% of the buffer read; llama.cpp's `mul_mat_vec` read it at 9.2 to 10.8 GB/s.
- **One token's layer** of Llama 3.2 1B's shape: 3.36 ms in the fastest form (DP4A, with the norms in their own
  dispatches), that is 20.4 GB/s of weights, half the buffer read.
- **Fixed costs**: an empty dispatch 36 µs, waiting for a submission 8.76 ms, reading 4 bytes back 7.87 ms.

From these, one written token of Llama 3.2 1B on this GPU comes to about 70 ms, against about 58 ms on the CPU's 4
threads, so the page is expected to keep the answer on the CPU there (an estimate from the parts: 16 layers, the
classifier, the sampling and a quarter of a submission's wait), while a prompt of 64 tokens was 5 times as fast on
the GPU. The page's own choice on this phone has not been reported yet. The sampling alone took 5.9 ms in one
workgroup, on nearly flat logits; the benchmark now also times it split over chunks of the vocabulary.

**Not measured yet**: the page's choice on real devices (which side, from how many tokens, how long the GPU takes
to get ready), the iPhone and the PC with the current shaders, where the time of a layer goes, and the sampling in
chunks. The numbers from CI and from the development machine come from fallback adapters (a CPU doing the GPU's
work) and say only that the shaders are right, not how fast a GPU is.

## Correctness

- `tests/gpu-check.mjs` runs a prompt of 150 tokens on the CPU and on the GPU, and compares the keys and values
  written back and the logits of the last token with NumPy. It runs every shader shape, and the attention without
  subgroups, in Chromium's SwiftShader and in Node with Dawn and Mesa's lavapipe.
- It then writes 8 tokens greedy on the GPU (4, one on the CPU, 3 more) and checks the ids against NumPy's
  (or a near tie), the keys and values written back, a stop token and a sampled token.
- Deliberately broken shaders (a wrong causal mask, a RoPE sign, a GQA head mapping, a missing quantization step
  and others) fail these checks.
- The sampling on the GPU picks the token the CPU's sampling picks for the same logits and random number (or, where
  a float32 sum moves a border, one next to it: within 1e-4 of the probability mass), in the benchmark's check; the JavaScript it
  is held to is held to the CPU's kernel in `tests/smoke.mjs`.
- On the device, the page checks each shader against JavaScript before it uses it.

## Next

In order, with the owner's numbers: the sampling in chunks in the page, and whatever the breakdown of a layer shows to be slow. A seed gives the same text again on the same device and the
same path, but not across the CPU and the GPU, whose forward passes differ in the last digits (the CPU rounds the
activations to 7 or 8 bits). Either side takes the same random number for the same token; but every 8 answers, the
first tokens of one go to the side not chosen, so that answer may differ from an earlier one with the same seed. The
tasks are in
[TODO.md](../TODO.md) (T156, T191, T202 and T208, in Japanese).

## Try it yourself

Open [/benchmark/](https://takano32.github.io/pyodide-llm/benchmark/) and press the GPU section's button. It
measures the device's ceilings, a prompt through each shader, one token's matrix × vector, the fused layer and
where its time goes, the sampling alone, and tokens generated on the GPU read back one at a time or several at
once, and can open the report as a GitHub issue. Its model section times the page's own path: prompts and answers
as the page chooses, on the CPU only and on the GPU only.
