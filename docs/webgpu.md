# The GPU (WebGPU)

Where the browser has WebGPU in a worker, the page uses the GPU without any option, and checks on each device
whether that is faster than the CPU. Where it is not, the page stays on the CPU. This page says what runs on the
GPU today, how the page chooses, and what has been measured.

## What runs on the GPU today

**The blocks of a prompt**, up to 64 tokens at a time. The tokens the model writes still run on the CPU.

The reason is the first measurement (2026-09-26, on the owner's three devices): moving the generation of one token
to the GPU as it was on the CPU was slower everywhere. For Llama 3.2 1B, the GPU's speed divided by the CPU's was
0.97 on the Android phone, 0.69 on the iPhone (Safari) and 0.53 on the ARM Chromebook. The phones' GPUs read large
matrices about 3 times as fast as one CPU core, but each token carried a fixed cost of about 18 to 20 ms (about 240
dispatches, small matrices, reading the result back). A prompt is different: one read of the weights serves many
tokens, and the fixed cost is shared among them.

## How it works

1. For each model, `forward.js` starts a GPU worker (`public/gpu.js`). It copies the int8 weights and scales of
   the layers and their norms to the GPU. The embeddings and the classifier stay on the CPU.
2. It compiles the candidate shaders for the matrix products and checks each against JavaScript on a small matrix
   on this device (subgroup order, float16 rounding and driver errors only show there). Then it times the ones
   that are right on the first layer of the model, and takes the fastest.
3. It times whole blocks of 16 and 64 tokens on the GPU. The page times the CPU on real prompts. For each block, it
   runs on the GPU only if the GPU is expected to take less than 0.95 of the CPU's time. Short prompts stay on the
   CPU, because the GPU's fixed cost is not shared among enough tokens.
4. The GPU writes the keys and values of each layer back into the CPU's cache (in float16), and the rest (the last
   token of the prompt and everything the model writes) runs on the CPU.
5. Every 8 answers, the page measures the side it did not choose again, on part of a prompt, in case the device
   has warmed up or cooled down.

The page is ready without waiting for the GPU: until the GPU is ready, prompts run on the CPU. The shape it chose
is remembered per model and device, so the next visit compiles two shaders instead of all of them.

The status line says what happens: "prompts on WebGPU", "prompts of 29 tokens and more on WebGPU", "prompts on the
CPU (faster here than WebGPU)", or "prompts on the CPU (reason)".

## Where the GPU is not used

- The browser has no WebGPU in a worker (Firefox so far, including Firefox 156 on Android).
- The adapter is a fallback that runs on the CPU (SwiftShader, lavapipe): it would never be faster, and compiling
  the shaders took 2 to 4 minutes.
- The page is not cross-origin isolated (no shared memory between the workers).
- The model's weights are float32 (not int8 or 6-bit). Qwen2's biases, Qwen3's per-head norms of q and k, and
  GPT-2 and GPT-NeoX (LayerNorm, GELU, the biases, learned positions, partial RoPE, the parallel residual) run on
  the GPU as well. A model in 64-bit memory (over 4 GB) goes as one in 32-bit memory, and a matrix larger than a
  buffer the device binds goes in pieces of rows. 6-bit weights are to be widened to int8 on the GPU as they are
  uploaded; until the shader for that is in, a 6-bit model stays on the CPU.
- The weights would not fit twice: today they are kept in WebAssembly memory for the CPU and again on the GPU. On
  phones and Apple devices both are the same memory. If the total is more than half of `navigator.deviceMemory`,
  the model stays on the CPU. Chromium reports at most 8, which is read as "8 GB or more"; a browser that does not
  report it (Safari, Firefox) is taken as 4 GB, so the 1B models stay on the CPU there.

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
| One token's layer in 5 dispatches instead of 14 (benchmark only) | built on llama.cpp's `mul_mat_vec` |
| One token's layer on packed int8 dot products, the vector quantized before each matrix (benchmark only) | ONNX Runtime's DP4A MatMulNBits for small M (MIT), with the fused writes of the line above. The norm and its quantizing in one dispatch take their form from vLLM's `rms_norm_per_block_quant` (Apache-2.0; no lines copied). |
| The device's ceilings (benchmark only) | the loops of clpeak (GPL-3.0): the shapes only, no lines copied |
| Sampling on the GPU and several tokens a submission (benchmark only): the repetition penalty, softmax, top-p and the draw, the next token's row of the embedding | the penalty of MLC LLM (Apache-2.0); llama.cpp's `argmax`, `soft_max`, `cumsum` and `get_rows` (MIT); top-p without sorting from MLC LLM's `top_p_pivot` (Apache-2.0). Carrying the state from one token to the next, and the draw by the same pivots, are ours. |

## Measured

On the owner's Android phone (Chrome 153, Arm Valhall GPU), before the tiled shaders: a prompt through the first
GPU shader (`batched`) took 30.08, 6.22 and 5.51 ms per token at 1, 16 and 64 tokens, about 44 GFLOPS at 64
tokens. The CPU with 4 threads reached about 88 G operations per second on the same work, twice the GPU. That
shader used each read of the weights for 8 tokens only, and the tiled shaders above replaced it.

**Not measured yet**: the tiled shaders, the attention, the ceilings and the page's choice on the owner's devices.
The numbers from CI and from the development machine come from fallback adapters (a CPU doing the GPU's work) and
say only that the shaders are right, not how fast a GPU is.

## Correctness

- `tests/gpu-check.mjs` runs a prompt of 150 tokens on the CPU and on the GPU, and compares the keys and values
  written back and the logits of the last token with NumPy. It runs every shader shape, and the attention without
  subgroups, in Chromium's SwiftShader and in Node with Dawn and Mesa's lavapipe.
- Deliberately broken shaders (a wrong causal mask, a RoPE sign, a GQA head mapping, a missing quantization step
  and others) fail these checks.
- The sampling on the GPU picks the token the CPU's sampling picks for the same logits and random number (or, where
  a float32 sum moves a border, one next to it: within 1e-4 of the probability mass), in the benchmark's check; the JavaScript it
  is held to is held to the CPU's kernel in `tests/smoke.mjs`.
- On the device, the page checks each shader against JavaScript before it uses it.

## Next

In order: generation on the GPU where the device measures it faster (several tokens a submission, sampled on the
GPU with the CPU's random numbers, is in the benchmark: a seed gives the same text again on the same device and the
same path, but not across the CPU and the GPU, whose forward passes differ in the last digits (the CPU rounds the
activations to 7 or 8 bits)); then the shader that widens 6-bit weights; and keeping the weights once
instead of twice. The tasks are in [TODO.md](../TODO.md) (T151 to T157, in Japanese).

## Try it yourself

Open [/benchmark/](https://takano32.github.io/pyodide-llm/benchmark/) and press the GPU section's button. It
measures the device's ceilings, a prompt through each shader, one token's matrix × vector, the fused layer, and
tokens generated on the GPU read back one at a time or several at once, and can open the report as a GitHub issue.
