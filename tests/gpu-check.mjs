// T135: the prompt on the GPU (public/gpu.js) against the CPU's (public/forward.js) and NumPy's (llama2_numpy.py), in
// a real browser, as WebGPU is nowhere else: Playwright's Chromium, whose WebGPU without a GPU is SwiftShader (the CPU
// in the GPU's place: its speed means nothing, its numbers are right or not).
//
//   node tests/gpu-check.mjs [model id | synthetic | made-up ...] [--engine chromium|chrome|msedge] [--forms <part,part>]
//   node tests/gpu-check.mjs ... --engine dawn --webgpu <the npm package webgpu's directory>
//
// T147: --engine dawn runs the same in Node on Dawn (the npm package webgpu, not a dependency of this project: install
// it under .tmp/) with the Vulkan of the machine, Mesa's lavapipe on the development machine: shader-f16 and subgroups,
// which SwiftShader lacks (AGENTS.md). The harness and the GPU's worker are then worker threads.
//
// Node reads each model with Pyodide as the page does and records two things: the plan that forward.js gets from
// Python (where every tensor is), and NumPy's answer for a prompt of 150 tokens: the keys and values of every layer at
// every position the prompt's blocks fill, and the logits of its last token. NumPy multiplies the int8 weights
// widened to float32 by float32 activations, which is what the GPU does; forward.js on the CPU quantizes the
// activations as well (7 bits with relaxed SIMD), so it is farther from both by design. The browser then runs
// forward.js in a worker (it waits in Atomics.wait, which a page may not) on the same memory: the prompt through
// forwardMany() and its last token through forward(), once on the CPU and with the GPU's worker once for the shaders
// it chooses and once for every tiled shader of the matrices (T147), the prompt in blocks of 16 and then all at once.
// The cache starts at 8 positions, so that both grow within the prompt (8 to 256). Checked:
//   - the GPU took every token of the prompt (gpuTokens), the same again from position 0, and a block that begins
//     past the keys and values it holds went to the CPU;
//   - the keys and values it wrote back into the cache, against NumPy's: the worst row (a layer's keys or values of
//     one position) no more than the line of the shader's arithmetic (GPU_LINE and the next, see there);
//   - T147: a request forward.js gave up on (a GPU that answers late) writes nothing (the made-up model only);
//   - T187: the scale of the first layer's keys and values against NumPy's (SCALE_LINE, see there);
//   - the logits of the prompt's last token (the CPU's in both runs, on the GPU's keys and values in one): NumPy's
//     most likely token or a near tie (their KL divergence from NumPy's is printed, not held to a line: see there).
// "synthetic": a made-up int8 model with grouped-query attention (4 heads, 2 of keys and values; none of the models
// of this directory has it). T153: "synthetic-qwen2", the same with biases of q, k and v (Qwen2's; T187: drawn around
// 0, 0.3·N, as GPT-2's below) and an epsilon of 1e-6; "synthetic-qwen3", the norms of every head of q and k (Qwen3's), heads of 32 where dim / heads is 16 (q and
// the attention's output 128 wide, dim 64), and an epsilon of 0.5, near mean(x²) (T150: an epsilon far below it
// hides a wrong one); both of three layers. T154: "synthetic-gpt2", GPT-2's form (LayerNorm with biases, a bias after
// every matrix, an FFN of two matrices and GELU, learned positions and no RoPE), and "synthetic-neox", GPT-NeoX's (the
// same with RoPE on the first quarter of every head, as Pythia's rotary_pct 0.25, and the parallel residual), both of
// three layers and 4 heads of keys and values (neither has grouped-query attention), their biases drawn around 0 (the
// review of T154: a bias of 1 ± 0.03 is nearly the same number everywhere, and LayerNorm's mean takes most of it out);
// "synthetic-neox-256", GPT-NeoX with Pythia 1B's heads of 256 (dim 512, 2 heads), 64 of them turned. T155:
// "synthetic-6bit", int6 weights (T98's, which the GPU widens to int8 as it takes them: the shader is shaders.js's
// WIDEN_SIX), and "synthetic-wide", int8 in a 64-bit memory with the checkpoint 4 GiB up (T101: every address past
// 2^32; the pages below are never touched), both of dim 128 and hidden 320 with the GPU's matrices in pieces of
// 20480 bytes at most (gpuForce.pieceBytes: w1 and w3 in three, w2 in two, the last of w1's rows short), as a matrix
// past a buffer of the device goes. The others are the models of this directory (make models kernels), or <prefix>.json: a
// model tests/perplexity_prepare.py converted (<prefix>.bin, <prefix>.tokenizer.bin, and the options in <prefix>.json;
// gpu-prompt.yml's input real= fetches and converts models of src/models.js so, T183), whose NumPy answer comes from
// the native Python ($PYTHON, python3 by default).
//
// T152: a generation's steps on the GPU (forward.js's generateMany, gpu.js's fused layer of a token and SAMPLE), after
// the prompt of a run whose GPU took them (the first, and one a form of a token's layer this adapter can make: each
// forced): NumPy continues the prompt greedy for GEN tokens (its logits of each step, and the keys and values of their
// positions); the GPU is asked for 4 steps greedy from the prompt's last token, the CPU then takes one step (the
// forward pass on the keys and values the GPU wrote back), and the GPU 3 more (the CPU's position goes up to it first).
// Checked: the ids are NumPy's, or where one is not, a near tie (T187's: NumPy gives it at least half the probability
// of its most likely; the steps after it are not compared, their inputs differ; the CPU's step feeds NumPy's id to the
// GPU's next ones whatever it chose); the keys and values the GPU wrote back of its first 4 positions in forward.js's
// cache against NumPy's, a layer at a time (T187), to the line of the prompt's shaders of the same arithmetic (float32 GPU_LINE or K × E16;
// DP4A K_PACKED × Q8's distance) or the run's own prompt's distance, the larger (the steps' attention reads the
// prompt's keys and values; the next 3 read the CPU's step's too, its 7-bit activations, and are held to their ids); a
// stop token (NumPy's second) ends the steps after it; a penalty of 100 on NumPy's first id (greedy) gives the largest
// of NumPy's logits penalized so (or a near tie); two sampled steps at temperature 2 with random numbers of 0.02 and
// 0.98 give two tokens of NumPy's nucleus, not the same (the random numbers reach the GPU: where either lands is not
// held to NumPy's, the GPU's logits being others; the draw itself is SAMPLE's, checked on the device, gpu.js); and a
// step after the prompt went through the CPU while the GPU's own cache held the keys and values of other tokens
// (a prompt of them through the GPU first): the CPU's most likely token on its own keys and values, or a near tie of
// its logits (the CPU's keys and values went up).
//
// T183: what a person reads to judge it, in the log of CI (the development machine does not run WebGPU's tests): E16,
// how far NumPy's answer moves when nothing but its cache is rounded to float16 (answer(half=True), T153's review), and
// for each model a table of the keys and values by layer, a column E16 and one a run (the CPU's, and the GPU's lettered
// A, B...) against NumPy's, another against NumPy's with its cache in float16, a line a run with its form, its line and
// the ratio to it, how many E16 it is and its most likely token, and the seconds of every step. T187: Q8 (NumPy's
// with the inputs of the layers' matrices in 8 bits as well) beside E16 and a table against it, the first layer's
// scale, the logits' KL divergence and the difference of the ten most likely (in place of the logits' ratio to the
// CPU's).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";
import * as wgsl from "../public/shaders.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
// T147: --forms <part,part>: only the matrices' shaders whose names hold one of these (all of them by default)
const only = option("--forms", "");
const webgpu = option("--webgpu", "");
// T153: the made-up models of another form (see above). Three layers: a layer's vectors are read at l × their size,
// which a second layer alone would not tell from 0 + size
const SYNTHETIC = { "synthetic": [{}, {}], "synthetic-qwen2": [{ layers: 3, bias: true }, { bias: true, rms_norm_eps: 1e-6 }],
  "synthetic-qwen3": [{ layers: 3, qk_norm: true, head_dim: 32 }, { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  "synthetic-gpt2": [{ layers: 3, kv_heads: 4, arch: "gpt2" }, { arch: "gpt2" }],
  "synthetic-neox": [{ layers: 3, kv_heads: 4, arch: "neox" }, { arch: "neox", rotary: 4, parallel_residual: true }],
  "synthetic-neox-256": [{ dim: 512, hidden: 1024, layers: 3, heads: 2, kv_heads: 2, arch: "neox" }, { arch: "neox", rotary: 64, parallel_residual: true }],
  // T155: [form, options, the run's: the GPU's pieces, a 64-bit memory]
  "synthetic-6bit": [{ dim: 128, hidden: 320, layers: 3, six: true }, { dtype: "int6" }, { force: { pieceBytes: 20480 } }],
  "synthetic-wide": [{ dim: 128, hidden: 320, layers: 3 }, {}, { force: { pieceBytes: 20480 }, wide: true }] };
// the models to check: by default every made-up one and the site's three; "made-up" stands for every made-up one (T193:
// gpu-prompt.yml's suites name them so, and a made-up model added above joins them)
const ids = (args.length ? args : ["made-up", "stories15M", "tiny-lm", "llm-jp-3-150m"])
  .flatMap((id) => (id === "made-up" ? Object.keys(SYNTHETIC) : [id]));
// T147: 150 tokens, so that the GPU's blocks of 64 are two and a part (the tiles' ends), and the caches grow to 256
const COUNT = 150, KV_START = 8;
// T152: the greedy steps NumPy takes after the prompt
const GEN = 8;
// The worst row of the keys and values against NumPy's, by what the matrices' shader computes in (T147, measured on
// Dawn's lavapipe and this machine's SwiftShader, 149 tokens: two blocks of 64 and a part). The CPU's forward.js: 4.1e-2
// to 3.2e-1 (its 7-bit activations; the made-up model's random weights the most). A GPU that is wrong lands far past
// the lines (broken on purpose: RoPE at the next position 1.13 to 1.16, no causal mask 1.86 to 6.97, the keys written
// back in the order [token][layer] 2.50 to 4.62, T135; T147's six in TODO.md).
//   float32 (TF.js's tiles, llama.cpp's f32): 8.1e-4 to 2.4e-3 for the models here, 3.90e-3 for the made-up one: the
//     float16 of the cache (2^-11 = 4.9e-4 of a value) and the attention on float16 keys and values from the first
//     layer on, as the CPU's (T135's GPU read its own float32 ones: 4.5e-4 to 4.7e-4 at 39 tokens), growing with the
//     positions (the made-up model 1.56e-3 at 39 tokens). The line: 8e-3.
//   f16 (llama.cpp's f16: every weight × scale and activation rounded to 11 bits): 9.5e-4 to 6.1e-3. The line: 1.5e-2.
//   8 bits (ORT's DP4A: the activations quantized as the CPU's matmul_q8 takes them, 8 bits where the CPU's relaxed
//     SIMD takes 7): held to Q8 (T187, below).
// T153: those lines hold for the first layer of every model and for all layers of a shallow one. Over the layers of a
// deep one the float16 of the cache grows by itself, the more so with Qwen's large activations: E16 (NumPy's answer
// with nothing but its cache rounded to float16, against NumPy's) is its measure. A GPU that computes as it should is a
// few E16 away, whatever the CPU's forward.js does (a line of a quarter of the CPU's error loosened the made-up models'
// lines 1.3 to 10 times, and a GPU that used layer 0's biases in every layer passed: T153's review). So every layer of
// a float32 or f16 form is held to the shader's line or K × E16, whichever is larger, and the first layer to the
// shader's line alone. K (CI's Dawn on lavapipe, run 36308518805, 149 tokens): the float32 shaders are 0.9 to 1.6 E16
// away (Qwen3 0.6B the most), llama.cpp's f16 0.9 to 3.2 and 5.5 (Qwen3 0.6B with the f16 attention), while a GPU that
// reads layer 0's biases in every layer (synthetic-qwen2) is 13.9 E16 away on float32 and 14.4 on f16: K is 4 and 8.
// T187: a layer at a time, the line of layer l max(shader's line, K × E16 of layer l): against the largest E16 of all
// layers, a deep model's early layers had a line of the late ones (Qwen3 0.6B's DP4A 2.78 where its layers 0 to 7 read
// 0.026 to 0.28 by their own). A GPU as it should be went to 0.75 of the lines a layer (runs 36325111865 and
// 36325960845, and on lavapipe's subgroups of 16, 36325111922: Qwen3 0.6B's f16 with the f16 attention at layer 12; the made-up models 0.66
// at most). The made-up Qwen2 with its biases around 0 (T187): a GPU that reads layer 0's biases in every layer is 158
// E16 away on float32 and f16, 39.6 and 19.8 times its layer's line (run 36325111772).
const GPU_LINE = 8e-3, HALF_LINE = 1.5e-2;
const K = { float32: 4, f16: 8 };
// T187: none of the lines is a ratio to the CPU's error (T147 to T153 held DP4A to 0.75 of it and the logits to 1.5
// times it: a correct DP4A went to 0.98 of its line on Qwen2.5 0.5B, one that read layer 0's biases in every layer
// passed at 0.55, and GPT-2's logits went past 1.5 on keys and values that were NumPy's own; they moved wherever the
// CPU's arithmetic did). Each is held to a yardstick of its own. The numbers are CI's Dawn on lavapipe (runs
// 36314674862 and 36314674849, 149 tokens, the made-up models, Qwen2.5 0.5B, Qwen3 0.6B, llm-jp-3 150M, SmolLM2 360M,
// and with T154 GPT-2 124M and Pythia 160M; TODO.md's T187):
//   - the packed shaders (8-bit activations) to Q8: NumPy's answer with the inputs of the layers' matrices quantized
//     as quantize_x does it (8 bits, 32 a group, half to even) and its cache in float16, which is ORT's DP4A's
//     arithmetic. Their first layer against Q8's: 8.1e-5 to 2.9e-3 (Qwen3 0.6B the most), the line GPU_LINE (a
//     LayerNorm's mean over n − 1: 1.57e-2 and 1.93e-2). Past the first layer the 8-bit roundings that fall on the
//     other side of a boundary part them from Q8 as far as Q8 is from NumPy's (a GPU as it should be 0.25 to 2.6 Q8
//     from Q8), so all their layers are held against NumPy's: 0.49 to 1.40 times Q8's distance from it (GPT-2 124M
//     the most, its outlier channels). A layer at a time (the review of T187), the line of layer l is K_PACKED × Q8's
//     distance at layer l: the review read a GPU as it should be at 2.0 times it on lavapipe's subgroups of 16 from its
//     tables' two digits (Qwen2.5 0.5B's layer 17, Qwen3 0.6B's layer 27), so K_PACKED is 3; the runs read 1.44 at
//     most (0.48 of the line, SmolLM2 360M's layer 13, run 36325960845; 1.17 on subgroups of 16, run 36325111922), the
//     made-up models 1.02. A bias read from
//     layer 0 in every layer went under a line of 2 on the made-up Qwen2 while its biases were near 1 in every layer
//     (0.98), and goes past its layer's line 2.34 times (7.02 Q8 over all layers) with biases drawn around 0 (0.3·N; a
//     correct one 1.01 Q8, 0.34 of its lines), and 2534 times on Qwen2.5 0.5B (run 36325111772).
//   - the scale of the first layer's keys and values (the norm's scale wrong by a little, as a mean of squares or a
//     variance over n − 1 (1 + 1/2n: 7.8e-3 at dim 64, 5.2e-4 at 960) or a wrong epsilon, stays under the rows' lines,
//     and the fused norm's quantized integers do not see it at all, T175): the least-squares factor s of the GPU's
//     against NumPy's (the packed shaders' against Q8's, whose own scale moves by up to 4.4e-4), sum(g·w) / sum(w·w),
//     which the rounding of single values leaves at 1 within its noise over thousands of values. |s − 1|: 3.8e-9 to
//     3.3e-5 as it should be; RMSNorm over n − 1 5.2e-4 to 1.0e-2, LayerNorm's variance over n − 1 5.4e-4 to 7.4e-3.
//     The line SCALE_LINE: n − 1 goes past it up to a dim of 5000. Only where f32 rounds to f16 to the nearest, as
//     SwiftShader and lavapipe do (the review, 2026-09-27): WGSL (15.7.6) leaves the direction open, and a GPU that
//     rounds toward zero shrinks every value, which does not average out. A throwaway branch that truncates the
//     cache's pack2x16float and the f16 tiles' operands (run 36320147208) read s − 1 = −3.4e-4 to −3.9e-4 on the
//     float32 and packed shaders and −9.5e-4 to −1.1e-3 on the f16 ones, every model, with every other line held: on
//     such a GPU this line fails a GPU as it should be (then hold the scale against NumPy's rounded toward zero too,
//     and the f16 shaders to about 2e-3).
//   - the logits of the prompt's last token (on the CPU in every run, on the GPU's keys and values in the GPU's): the
//     most likely token NumPy's, or one NumPy gives at least half the probability of its most likely (a near tie:
//     T147's stories15M at 149 tokens). Their KL divergence from NumPy's is printed and held to no line (the review,
//     2026-09-27): the CPU computes that last token with its 7-bit activations, so the KL is the CPU's arithmetic,
//     not the GPU's. The CPU alone is 9.1e-5 to 9.8e-2 on the made-up models (synthetic-6bit the most), and on keys and
//     values that are 1.0 E16 (TF.js) synthetic-wide read 0.22 to 0.27 on every job of run 36320963317, where a line
//     of 0.05 failed it. A wrong bias on Qwen2.5 0.5B reads 9.3 to 11.7, and fails the keys and values' lines 1569
//     times over and the most likely token as well. Printed beside it: the largest difference of the logits of
//     NumPy's ten most likely tokens over NumPy's largest |logit| (the old measure took all of them: GPT-2's went past
//     on a token some 50 logits below the top).
const K_PACKED = 3, SCALE_LINE = 1e-4, TIE = Math.LN2;

// ---- Node: the plans and NumPy's answers
const { pyodide: py } = await pyodideWithEngine();
const PYTHON = `
import base64, gc, struct, numpy as np, llama2_numpy, llama2_convert
from llama2_numpy import Llama, external_tensors

def synthetic(dim=64, hidden=128, layers=2, heads=4, kv_heads=2, vocab=320, seq_len=256, seed=0, six=False, **form):
    """A made-up int8 checkpoint and its tokenizer.bin, as quantize.py writes one: grouped-query attention. form
    (T153): llama2_numpy.FORM's bias, qk_norm and head_dim, whose vectors are drawn as the norms' are. six (T155):
    int6 (T98), as llama2_convert's Writer writes it: every matrix's packed values, then its scales"""
    rng = np.random.default_rng(seed)
    header = (dim, hidden, layers, heads, kv_heads, vocab, seq_len)
    out = [struct.pack("<7i", *header)]
    # T154: GPT-2's and GPT-NeoX's vectors (Llama.gpt2_tensors' order) are the weights of the LayerNorms at 0, 6 and 10,
    # and biases, which are drawn around 0 as the matrices are. T187: so are Llama's biases of q, k and v (Qwen2's, the
    # vectors 3 to 5 after the three norms): around 1 as a norm (1 ± 0.03), the biases of every layer were nearly the
    # same, and DP4A's 8 bits hid a GPU that read layer 0's in every layer (0.98 of its line)
    gpt = form.get("arch", "llama") in ("gpt2", "neox")
    is_bias = (lambda i: i not in (0, 6, 10)) if gpt else (lambda i: bool(form.get("bias")) and 3 <= i < 6)
    vectors = 0
    for shape, is_matrix in llama2_convert.layout(*header, **form):
        if is_matrix is None:
            continue  # the RoPE tables: an int8 file leaves them out
        values = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        if not is_matrix:
            bias = is_bias(vectors)
            vectors += 1
            out.append((values if bias else 1.0 + values * 0.1).astype(np.float32).tobytes())
            continue
        if six:
            q, scales = llama2_numpy.quantize6(values.reshape(-1, shape[-1]))
            out += [llama2_numpy.pack6(q).tobytes(), scales.tobytes()]
            continue
        q, scales = llama2_convert.quantize(values.reshape(-1, shape[-1]))
        out += [q.tobytes(), scales.tobytes()]
    pieces = [f"<{i}>".encode() for i in range(vocab)]
    tokenizer = struct.pack("<i", max(map(len, pieces))) + b"".join(struct.pack("<fi", 0.0, len(p)) + p for p in pieces)
    return b"".join(out), tokenizer

class Half(np.ndarray):
    """T153: a cache that keeps what is written into it rounded to float16 (the GPU's and the CPU's cache, T110)"""
    def __setitem__(self, key, value):
        super().__setitem__(key, np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32))

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # T183: what is computed from the cache is a plain array (the attention's scores and output, then the residual
        # stream), or its later writes round as well: without this, stories15M's keys at 40 tokens were 7.8e-3 from
        # the review's engine that rounds the cache where it writes it (.tmp/t153-review/ref16), with it the same
        array = np.asarray(array).view(np.ndarray)
        return array[()] if return_scalar else array

class Q8(np.ndarray):
    """T187: a layer's matrix whose product takes its input quantized to 8 bits a group of 32, as the CPU's quantize_x
    makes it for matmul_q8 and the GPU's QUANTIZE for ORT's DP4A: the largest |value| / 127 the group's scale, each
    value times 1 / scale rounded half to even, in float32"""
    def __matmul__(self, x):
        x = np.asarray(x, dtype=np.float32)
        groups = x.reshape(-1, 32)
        scale = np.abs(groups).max(axis=1, keepdims=True) / np.float32(127)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale > 0)
        return np.asarray(self).view(np.ndarray) @ (np.rint(groups * inverse) * scale).reshape(x.shape)

def answer(data, vocabulary, text, count, options, half=False, q8=False):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens. half (T153): the cache rounded to float16, and nothing else (made whole
    at first, so that it never grows into an array of another class). q8 (T187): the inputs of the layers' matrices
    quantized to 8 bits (Q8)"""
    numpy = Llama(data, vocabulary, **options)
    if q8:
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            if getattr(numpy, name, None) is not None:
                setattr(numpy, name, np.asarray(getattr(numpy, name)).view(Q8))
    if half:
        numpy.key_cache = np.zeros((numpy.n_layers, numpy.n_kv_heads, numpy.seq_len, numpy.head_size), dtype=np.float32).view(Half)
        numpy.value_cache = np.zeros_like(numpy.key_cache).view(Half)
    if text:
        tokens = ([numpy.bos] + list(numpy.tokenizer.encode(text)))[:count]
    else:
        tokens = [numpy.bos] + [int(t) for t in np.random.default_rng(1).integers(3, numpy.vocab_size, count - 1)]
    assert len(tokens) == count, f"the text is {len(tokens)} tokens long"
    for pos, token in enumerate(tokens[:-1]):
        numpy.forward(token, pos, need_logits=False)
    logits = numpy.forward(tokens[-1], count - 1)
    n = count - 1
    kv = lambda cache, at=0, m=n: np.ascontiguousarray(cache[:, :, at:at + m, :].transpose(0, 2, 1, 3).reshape(numpy.n_layers, m, -1), dtype=np.float32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    out = {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
           "header": list(struct.unpack_from("<7i", data, 0))}
    if half:
        return out
    # T152: GEN greedy steps after the prompt: the ids, the logits of each step, the keys and values of their positions
    greedy, rows = [], [np.array(logits, dtype=np.float32)]
    for step in range(${GEN}):
        greedy.append(int(np.argmax(rows[-1])))
        if step < ${GEN} - 1:
            rows.append(np.array(numpy.forward(greedy[-1], count + step), dtype=np.float32))
    return {**out, "greedy": greedy, "greedyLogits": b64(np.stack(rows)), "greedyKeys": b64(kv(numpy.key_cache, n, ${GEN})),
            "greedyValues": b64(kv(numpy.value_cache, n, ${GEN}))}

def answers(data, vocabulary, text, count, options):
    """answer() and beside it: T153, the keys and values of answer(half=True); T187, its logits, and the keys, values
    and logits of answer(half=True, q8=True) (the arithmetic of ORT's DP4A on the GPU: 8-bit inputs, the cache in
    float16)"""
    # (one engine at a time: Llama holds cycles, and Qwen3 0.6B widened to float32 is 2.4 GB)
    exact = answer(data, vocabulary, text, count, options)
    gc.collect()
    rounded = answer(data, vocabulary, text, count, options, half=True)
    gc.collect()
    eight = answer(data, vocabulary, text, count, options, half=True, q8=True)
    gc.collect()
    return {**exact, "keys16": rounded["keys"], "values16": rounded["values"], "logits16": rounded["logits"],
            "keys8": eight["keys"], "values8": eight["values"], "logits8": eight["logits"]}
`;
py.runPython(PYTHON);

// each text three times over: long enough for COUNT tokens of every model here
const TEXTS = {
  english: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. " +
    "One day, she saw a big red ball under a tree. She ran to the ball and kicked it high into the sky, and everyone laughed.",
  japanese: "富士山は静岡県と山梨県にまたがる活火山で、標高三七七六メートルの日本最高峰である。古くから信仰の対象とされ、" +
    "多くの和歌や絵画に描かれてきた。二〇一三年には世界文化遺産に登録され、毎年夏には多くの登山者が山頂を目指す。",
};
const cases = [];
const directory = path.join(root, ".tmp", "gpu-check");
fs.mkdirSync(directory, { recursive: true });
for (const id of ids) {
  let options, text, run = {};
  const began = performance.now();
  if (SYNTHETIC[id]) {
    const [form, engineOptions] = SYNTHETIC[id];
    run = SYNTHETIC[id][2] ?? {};
    py.globals.set("FORM", py.toPy(form));
    py.runPython(`data, vocabulary = synthetic(**FORM)`);
    options = { dtype: "int8", ...engineOptions };
  } else if (id.endsWith(".json")) {
    // T153: NumPy's answer in the native Python ($PYTHON, python3 by default): Qwen3 0.6B widened to float32 is 2.4
    // GB, which with the file's copies went past Pyodide's 4 GB and a 7.5 GB scope of the development machine
    const prefix = id.slice(0, -".json".length);
    options = JSON.parse(fs.readFileSync(id, "utf8"));
    text = TEXTS.english.repeat(3);
    const native = spawnSync(process.env.PYTHON ?? "python3", ["-c", `import sys, json\nsys.path.insert(0, ${JSON.stringify(path.join(root, "public"))})\n${PYTHON}
data, vocabulary = open(sys.argv[1] + ".bin", "rb").read(), open(sys.argv[1] + ".tokenizer.bin", "rb").read()
print(json.dumps(answers(data, vocabulary, sys.argv[2], ${COUNT}, json.load(open(sys.argv[1] + ".json")))))`, prefix, text],
      { encoding: "utf8", maxBuffer: 1 << 30, stdio: ["ignore", "pipe", "inherit"] });
    if (native.status !== 0) throw new Error(`NumPy's answer for ${id} failed (${native.status ?? native.signal})`);
    const numpySeconds = (performance.now() - began) / 1000;
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${prefix}.tokenizer.bin`));
    py.runPython(`vocabulary = open("tokenizer.bin", "rb").read()`);
    cases.push({ ...caseOf(id, options, JSON.parse(native.stdout), new Uint8Array(fs.readFileSync(`${prefix}.bin`))), numpySeconds });
    continue;
  } else {
    const entry = MODELS.find((m) => m.id === id);
    if (!entry) throw new Error(`no model ${id} in src/models.js`);
    py.FS.writeFile("model.bin", fs.readFileSync(root + entry.checkpoint));
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
    py.runPython(`data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()`);
    options = entry.options;
    text = (/日本語/.test(entry.note) ? TEXTS.japanese : TEXTS.english).repeat(3);
  }
  if (options.dtype !== "int8" && options.dtype !== "int6") throw new Error(`${id} is ${options.dtype}: the GPU takes int8 and int6 weights`);
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("TEXT", text ?? "");
  const reference = py.runPython(`answers(data, vocabulary, TEXT, ${COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  const numpySeconds = (performance.now() - began) / 1000;
  cases.push({ ...caseOf(id, options, reference, py.runPython("data").toJs()), numpySeconds, ...run });
}
// the plan forward.js gets from Python (the vocabulary in Pyodide's globals), recorded: Llama(external=) with a start()
// that keeps it, and the case the browser runs
function caseOf(id, options, reference, bytes) {
  const began = performance.now();
  let plan;
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("recorder", {
    size: bytes.length,
    read: (offset, length) => bytes.slice(offset, offset + length),
    start: (given) => {
      plan = given.toJs({ dict_converter: Object.fromEntries });
      return { backend: "recorded", bind() {}, forward() {}, release() {} };
    },
  });
  py.runPython(`Llama(None, vocabulary, kernels="simdkernel.so", external=recorder, **OPTIONS).release()`);
  plan.kv_start = KV_START;
  // T156: a Llama of int8 whose steps the GPU takes goes on the GPU alone too: where its tensors are from its header
  let places;
  if ((options.arch ?? "llama") === "llama" && !options.bias && !options.qk_norm && options.dtype === "int8") {
    py.globals.set("HEADER", py.toPy([...new Int32Array(bytes.slice(0, 28).buffer)]));
    places = py.runPython(`external_tensors(HEADER, OPTIONS["dtype"], OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  }
  for (const [name, value] of Object.entries(plan.derived)) plan.derived[name] = Buffer.from(value).toString("base64");
  const name = path.basename(id), file = path.join(directory, `${name}.bin`);
  fs.writeFileSync(file, bytes);
  return { id, plan, places, headDim: options.head_dim ?? 0, arch: options.arch ?? "llama", dtype: options.dtype, checkpoint: `/case/${name}.bin`, file, reference,
    planSeconds: (performance.now() - began) / 1000 };
}

// ---- the browser: a page that is cross-origin isolated (its own headers), a worker that runs forward.js
const HARNESS = /* js */ `
const search = "?v=gpu-check";
const { compileKernels, createForward, weightsMemory, footprint, external, gpuOnlyWeights, gpuOnlyPlan } = await import("/public/forward.js" + search);
const { GPU_DONE, GPU_FAILED, GPU_BEAT, GPU_WANTED } = await import("/public/jobs.js" + search);
const fetched = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const b64 = (floats) => {
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let text = "";
  for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
};
const openGpu = () => new Worker("/public/gpu.js" + search, { type: "module" });
// T147: every tiled shader of the matrices this adapter can make (each forced in a run of its own), and the one the
// GPU's worker chooses by timing them (the first run)
const wgsl = await import("/public/shaders.js" + search);
const adapter = await navigator.gpu?.requestAdapter();
const forms = !adapter ? [] : wgsl.promptForms({ half: adapter.features.has("shader-f16"), subgroups: adapter.features.has("subgroups"),
  packed: navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product"),
  memory: adapter.limits.maxComputeWorkgroupStorageSize, threads: adapter.limits.maxComputeInvocationsPerWorkgroup })
  .filter((form) => !form.none).map((form) => form.name).filter((name) => !ONLY.length || ONLY.some((part) => name.includes(part)));
// T152: the forms of a token's layer this adapter can make (gpu.js's TOKEN_FORMS), each forced in a run of its own
const features = navigator.gpu?.wgslLanguageFeatures;
const tokenForms = !adapter ? [] : ["llama.cpp, fused (T150)",
  ...(adapter.features.has("subgroups") && features?.has("subgroup_id") ? ["llama.cpp, fused (T150), subgroups"] : []),
  ...(features?.has("packed_4x8_integer_dot_product") ? ["DP4A, fused (T175)", "DP4A, fused (T175), the norms apart"] : [])];
try {
  const narrow = compileKernels(await fetched("/public/simdkernel_shared.wasm"), await fetched("/public/simdkernel_relaxed_shared.wasm"));
  const results = [];
  for (const c of await (await fetch("/cases.json")).json()) {
    const plan = c.plan;
    for (const name of Object.keys(plan.derived)) plan.derived[name] = Uint8Array.from(atob(plan.derived[name]), (ch) => ch.charCodeAt(0));
    const checkpoint = await fetched(c.checkpoint), size = checkpoint.length, tokens = c.reference.tokens, n = tokens.length - 1;
    // T155: a 64-bit memory (c.wide) with its kernels, the checkpoint 4 GiB up as threads-check's --high puts it
    const high = c.wide ? 2 ** 32 : 0;
    const kernels = c.wide ? compileKernels(await fetched("/public/simdkernel_shared64.wasm"), await fetched("/public/simdkernel_relaxed_shared64.wasm"), true) : narrow;
    const after = footprint(c.reference.header, size, { dtype: c.dtype, halfKV: true, gpu: true, kvStart: plan.kv_start, head_dim: c.headDim, arch: c.arch });
    const { memory, base: low } = weightsMemory(size + high, { shared: true, wide: Boolean(c.wide), after });
    const base = low + high;
    new Uint8Array(memory.buffer, base, size).set(checkpoint);
    // T148: SwiftShader and lavapipe are fallback adapters, which the page refuses: the tests take them (fallback),
    // and give the GPU every block it can take (always: a fallback adapter is far slower than the CPU). T155: the
    // case's own (the pieces of its matrices)
    const TESTS = { fallback: true, always: true, ...c.force };
    // T152: the steps of a generation (see the head of this file), where the GPU took them
    const generation = (engine) => {
      // (T152's review: and the most this adapter binds, which a classifier may pass, Llama 3.2 1B's 262.7 MB on lavapipe's 128 MiB)
      if (!engine.tokenBlock) return { why: engine.gpuTokensWhyNot ?? "the GPU took no steps", planned: engine.gpuTokensPlanned,
        binds: Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize) };
      const greedy = c.reference.greedy, history = (ids) => [...tokens, ...ids];
      const ask = (token, pos, before, count, settings = [0, 0.9, 1], randoms = [], stops = []) => {
        const h = history(before);
        return engine.generateMany(token, pos, h.slice(-64), h.length, count, ...settings, randoms, stops);
      };
      const out = { form: engine.gpuReady?.tokens, pieces: engine.gpuReady?.tablePieces, first: ask(tokens[n], n, [], 4) };
      if (out.first && out.first.every((id, i) => id === greedy[i])) {
        // the CPU's step, on the keys and values the GPU wrote back; then the GPU's, from the CPU's position up
        engine.forward(greedy[3], n + 4);
        out.cpuStep = Array.from(engine.logits()).reduce((best, x, i, xs) => (x > xs[best] ? i : best), 0);
        out.second = ask(greedy[4], n + 5, greedy.slice(0, 5), 3);
      }
      const rows = engine.keysAndValues(n, 8);
      Object.assign(out, { keys: b64(rows.keys), values: b64(rows.values) });
      out.stopped = ask(tokens[n], n, [], 4, [0, 0.9, 1], [], [greedy[1]]);
      // the penalty: NumPy's first id in the history, 100 greedy
      out.penaltyHistory = [...tokens.slice(0, -1), greedy[0], tokens[n]];
      out.penalized = engine.generateMany(tokens[n], n, out.penaltyHistory.slice(-64), out.penaltyHistory.length, 1, 0, 0.9, 100, [], []);
      out.sampled = [0.02, 0.98].map((random) => ask(tokens[n], n, [], 1, [2, 0.999, 1], [random]));
      // (T152's review) 4 sampled steps with a penalty in one submission against the same 4 asked one at a time: the
      // GPU's arithmetic is the same, so the ids are to the bit (the random numbers taken in their order, and the
      // penalty's window handed over from the history at every request as the GPU carries it within a submission)
      const drawn = [2, 0.999, 1.3], numbers = [0.3, 0.7, 0.1, 0.9];
      out.together = ask(tokens[n], n, [], 4, drawn, numbers);
      out.apart = [];
      for (let i = 0; i < 4; i++) out.apart.push(...(ask(i ? out.apart[i - 1] : tokens[n], n + i, out.apart.slice(0, i), 1, drawn, [numbers[i]]) ?? [NaN]));
      // the CPU's keys and values going up: a prompt of other tokens through the GPU (its own cache then holds theirs),
      // the prompt through the CPU, and the GPU's step from its last token, which must see the CPU's of every position.
      // Twice, over two others (the review: the ids alone let a position left out pass, the GPU's own keys and values of
      // the same token being near the CPU's): the keys and values the step wrote back are then to the bit the same
      const upload = (other) => {
        engine.forwardMany(other, 0);
        engine.gpuSide = "cpu";
        engine.forwardMany(tokens.slice(0, -1), 0);
        engine.gpuSide = null;
        // (the CPU's logits of the prompt's last token on the CPU's keys and values, the answer: NumPy's is not, the
        // CPU's keys and values being up to 46% from NumPy's on the made-up models)
        engine.forward(tokens[n], n);
        const logits = b64(engine.logits().slice()), id = ask(tokens[n], n, [], 1), rows = engine.keysAndValues(n, 1);
        return { logits, id, kv: b64(rows.keys) + b64(rows.values) };
      };
      // (the two others differ at every position, the last as well: the first try's two ended on the same token, whose
      // first layer's keys and values are the token's alone, and an upload that left the last position out passed on a
      // model of two layers)
      const reversed = tokens.slice(0, -1).reverse();
      const [one, two] = [reversed, [...reversed.slice(1), reversed[0]]].map(upload);
      Object.assign(out, { uploadedLogits: one.logits, uploaded: one.id, uploadedTwice: two.id, uploadedSame: one.kv === two.kv });
      return out;
    };
    const run = async (gpu, gpuForce, gpuRemembered, steps = false) => {
      const started = performance.now();
      const engine = createForward({ memory, base, size, kernels, plan, gpu, gpuForce: { ...TESTS, ...gpuForce }, gpuRemembered });
      const note = gpu ? await engine.gpu : undefined;
      const readySeconds = (performance.now() - started) / 1000;
      const ready = engine.gpuReady;
      const began = performance.now();
      // blocks of 16 at first (T108's, as Python handed them before T147): each block sees the keys and values the
      // GPU keeps of the ones before it, and the caches grow between them
      for (let at = 0; at < n; at += 16) engine.forwardMany(tokens.slice(at, Math.min(at + 16, n)), at);
      const promptMs = performance.now() - began, gpuTokens = engine.gpuTokens;
      engine.forward(tokens[n], n);
      const logits = engine.logits().slice(), { keys, values } = engine.keysAndValues(0, n);
      const out = { note, ready, form: engine.gpuForm, attention: engine.gpuAttention, promptMs, gpuTokens, logits: b64(logits), keys: b64(keys), values: b64(values) };
      // T152: a generation's steps on the GPU, where this run takes them (steps)
      if (gpu && steps) out.steps = { ...generation(engine), forced: gpuForce.tokens, cut: Boolean(gpuForce.tablePieceBytes) && plan.vocab_size >= 192 };
      if (gpu) {
        // the same prompt again from position 0, all of it at once (one block of the GPU's): the GPU's keys and values
        // of the first run are written over
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, -1), 0);
        engine.forward(tokens[n], n);
        const again = engine.keysAndValues(0, n);
        out.again = { gpuTokens: engine.gpuTokens, logits: b64(engine.logits().slice()), keys: b64(again.keys), values: b64(again.values) };
        // a block that begins past what the GPU holds (the CPU wrote position n) goes to the CPU
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, 2), n + 1);
        out.past = { gpuTokens: engine.gpuTokens };
      }
      // T205: the GPU's worker says it let go of its device before the next model is read (false: not within 5 s)
      out.ended = await engine.release();
      // T183: the seconds of the run, and of those until the GPU's worker said it was ready (its shaders compiled)
      return { ...out, seconds: (performance.now() - started) / 1000, readySeconds };
    };
    // T147: a GPU that answers late. forward.js gives its request up (stalledMs 1: the GPU's worker counts every 250
    // ms) and runs the block itself; the stop it posts never reaches the worker (a worker that went on before it read
    // the stop). When the worker is done (its count stands for a second) it must have answered nothing: forward.js
    // wanted the request no more, and the memory could be the next model's by then
    const late = async () => {
      let inner;
      const stopLost = () => {
        inner = openGpu();
        return { postMessage: (data) => data.type !== "stop" && inner.postMessage(data), set onmessage(f) { inner.onmessage = f; },
          set onerror(f) { inner.onerror = f; } };
      };
      const engine = createForward({ memory, base, size, kernels, plan, gpu: stopLost, gpuForce: TESTS, stalledMs: 1 });
      const note = await engine.gpu;
      const words = new Int32Array(memory.buffer, 0, GPU_WANTED + 1);
      engine.forwardMany(tokens.slice(0, -1), 0);
      // T148: the CPU did the blocks the GPU gave up (the same numbers as the CPU's own run: its blocks of 16)
      const cpuKeys = b64(engine.keysAndValues(0, n).keys);
      const gpuTokens = engine.gpuTokens;
      for (let beat = -1, still = 0, waited = 0; still < 1000 && waited < 120000; waited += 100) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        still = Atomics.load(words, GPU_BEAT) === beat ? still + 100 : 0;
        beat = Atomics.load(words, GPU_BEAT);
      }
      const out = { note, gpuTokens, keys: cpuKeys, done: Atomics.load(words, GPU_DONE), failed: Atomics.load(words, GPU_FAILED) };
      engine.release();
      inner.terminate();
      return out;
    };
    // T156: the model on the GPU alone, as the worker puts one it cannot hold twice (forward.js's gpuOnlyWeights): the
    // checkpoint streamed in stretches of odd sizes (a stretch ends in the middle of a word, a matrix, a layer), the
    // layers' matrices to the GPU's worker and the rest into memory packed without them. The prompt and 4 greedy steps
    // (none on the CPU); then a step on the CPU, which must throw (the weights are not here). And a GPU that fails as
    // it opens (pieces of 4096 bytes, fewer than the device's alignment of rows: gpu.js's piecesOf refuses): the bytes
    // still counted as taken, the GPU lost with the reason as the engine starts, a step on the CPU throws
    const direct = async (force) => {
      const worker = openGpu();
      const weights = gpuOnlyWeights({ memory, base, size, tensors: c.places, worker });
      worker.postMessage({ type: "open", plan: gpuOnlyPlan(c.reference.header, c.places, force), flow: weights.flow });
      const began = performance.now(), stretches = [1234567, 777, 3, 65536, 1 << 20];
      for (let at = 0, k = 0; at < size; at += stretches[k++ % stretches.length]) {
        weights.write(at, checkpoint.subarray(at, Math.min(size, at + stretches[k % stretches.length])));
        await weights.room();
      }
      await weights.drained();
      let lostWhy = null;
      const outside = external({ memory, base, size, kernels, gpu: () => worker, gpuForce: force, halfKeys: true,
        direct: { place: weights.place, stored: weights.stored, onLost: (why) => { lostWhy = why; } } });
      const engine = outside.start(structuredClone(plan));
      const note = await engine.gpu;
      const out = { note, stored: weights.stored };
      if (!lostWhy) {
        for (let at = 0; at < n; at += 16) engine.forwardMany(tokens.slice(at, Math.min(at + 16, n)), at);
        const { keys, values } = engine.keysAndValues(0, n);
        Object.assign(out, { keys: b64(keys), values: b64(values), gpuTokens: engine.gpuTokens, status: engine.gpuStatus,
          first: engine.generateMany(tokens[n], n, tokens.slice(-64), tokens.length, 4, 0, 0.9, 1, [], []) });
      }
      out.lost = lostWhy;  // (before the step on the CPU below, which loses the GPU: the worker would load it again)
      try {
        engine.forward(tokens[n], n);
      } catch (error) {
        out.threw = String(error?.message ?? error);
      }
      out.seconds = (performance.now() - began) / 1000;
      engine.release();
      return out;
    };
    const gpu = [];
    // (the forced ones untimed, T153: a block of 64 tokens of Qwen3 0.6B took more than the 180 s of a step on lavapipe)
    for (const form of [undefined, ...forms]) gpu.push(await run(openGpu, form ? { matrices: form, quick: true } : {}, undefined, !form));
    // T152: every form of a token's layer, forced, where the model's steps go to the GPU (not Qwen's, GPT-2's, GPT-NeoX's
    // yet); in one piece each (a token's layer reads a matrix whole: the first run's pieces, T155, left the steps on
    // the CPU)
    // T209: the classifier and the embedding in pieces of about a third of the table (as a table past what the device
    // binds: Llama 3.2 3B's on the owner's Android), so that EMBED's and the classifier's pieces are what the steps read
    if (gpu[0].steps?.planned !== false) {
      const tablePieceBytes = Math.ceil((plan.vocab_size * plan.dim) / 3);
      for (const form of tokenForms) gpu.push(await run(openGpu, { matrices: forms[0], quick: true, tokens: form, pieceBytes: Infinity, tablePieceBytes }, undefined, true));
    }
    // the attention without subgroups or f16 (the lanes of the workgroup stand for a subgroup), where the adapter
    // has them and so chose the other
    if (forms.length) gpu.push(await run(openGpu, { matrices: forms[0], attention: "llama.cpp flash attention tiles", quick: true }));
    // T148 (the made-up model only): a fallback adapter refused as the page refuses it, before anything is compiled;
    // the shaders of the first run remembered for this adapter (its key), which are then the only ones compiled, and
    // not for another key
    let refused, remembered;
    if (c.id === "synthetic") {
      const began = performance.now(), engine = createForward({ memory, base, size, kernels, plan, gpu: openGpu });
      refused = { note: await engine.gpu, seconds: (performance.now() - began) / 1000, ended: await engine.release() };
      // T205: a model let go while its GPU is still getting ready (gpu.js's start() stops at its next step and says so)
      const starting = createForward({ memory, base, size, kernels, plan, gpu: openGpu, gpuForce: TESTS });
      const stopped = performance.now();
      refused.whileStarting = { ended: await starting.release(), seconds: (performance.now() - stopped) / 1000 };
      const first = gpu[0].ready ?? {}, kept = { key: first.key, matrices: first.matrices, attention: first.attention };
      remembered = { same: (await run(openGpu, {}, kept)).ready, other: (await run(openGpu, {}, { ...kept, key: kept.key + "|another" })).ready };
    }
    const cpu = await run(undefined), lateRun = c.id === "synthetic" ? await late() : undefined;
    // T156: the model on the GPU alone, last (it packs the memory the others read the checkpoint whole from)
    let alone;
    const tokenRun = gpu.find((r) => r.steps?.forced === tokenForms[0] && r.steps?.first);
    if (c.places && tokenRun && !c.wide && !c.force) {
      const force = { ...TESTS, matrices: forms[0], quick: true, tokens: tokenForms[0], pieceBytes: Infinity, tablePieceBytes: Math.ceil((plan.vocab_size * plan.dim) / 3) };
      alone = { right: await direct(force), pieces: await direct({ ...force, pieceBytes: 4096 }), keys: tokenRun.keys, values: tokenRun.values, first: tokenRun.steps.first };
    }
    results.push({ id: c.id, cpu, gpu, late: lateRun, refused, remembered, alone });
  }
  postMessage({ results, forms });
} catch (error) {
  postMessage({ error: String(error?.stack ?? error) });
}
`;
// an icon of its own: a browser without one asks for /favicon.ico (a 404 in the console of the real Chrome)
const PAGE = `<!doctype html><meta charset="utf-8"><title>gpu-check</title><link rel="icon" href="data:,"><script type="module">
const worker = new Worker("/harness.js", { type: "module" });
worker.onmessage = ({ data }) => { window.__gpuCheck = data; };
worker.onerror = (event) => { window.__gpuCheck = { error: event.message ?? "the harness did not start" }; };
</script>`;
const types = { ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".html": "text/html; charset=utf-8" };
const server = http.createServer((req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  const send = (type, body) => {
    res.writeHead(200, { ...headers, "Content-Type": type });
    res.end(body);
  };
  const found = cases.find((c) => c.checkpoint === pathname);
  if (pathname === "/") return send(types[".html"], PAGE);
  if (pathname === "/harness.js") return send(types[".js"], `const ONLY = ${JSON.stringify(only ? only.split(",") : [])};\n${HARNESS}`);
  // (T152: and NumPy's greedy ids after the prompt, which the harness feeds the GPU's steps)
  if (pathname === "/cases.json") return send(types[".json"], JSON.stringify(cases.map(({ file, reference: { tokens, header, greedy }, ...c }) => ({ ...c, reference: { tokens, header, greedy } }))));
  if (found) return send("application/octet-stream", fs.readFileSync(found.file));
  const file = path.join(root, "public", pathname.replace(/^\/public\//, ""));
  if (!pathname.startsWith("/public/") || !fs.existsSync(file)) {
    res.writeHead(404, headers);
    return res.end();
  }
  send(types[path.extname(file)] ?? "application/octet-stream", fs.readFileSync(file));
}).listen(0);

const lines = [];
const outcome = engine === "dawn" ? await inDawn() : await inBrowser();
server.close();
if (lines.length) console.log(lines.join("\n"));
// T147: the list of forms comes from an adapter of its own, which SwiftShader now and then does not give (AGENTS.md)
if (!outcome.error && !outcome.forms?.length) outcome.error = "no tiled shader to force: the harness got no GPU adapter (run it again)";
if (outcome.error) {
  console.error(`FAILED\n- ${outcome.error}`);
  process.exit(1);
}

async function inBrowser() {
  const playwright = await import("playwright-core");
  // Chromium's WebGPU without a GPU: SwiftShader (as tests/bench-check.mjs has it)
  const WEBGPU = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"];
  const browser = await playwright.chromium.launch({ ...(engine === "chromium" ? {} : { channel: engine }), args: WEBGPU });
  const page = await browser.newPage();
  page.on("console", (message) => lines.push(`[${message.type()}] ${message.text()}`));
  page.on("pageerror", (error) => lines.push(`[pageerror] ${error.message}`));
  await page.goto(`http://localhost:${server.address().port}/`);
  // SwiftShader compiles a tiled shader in 10 to 90 s (T147): every shader of the made-up model takes some minutes
  await page.waitForFunction(() => window.__gpuCheck, null, { timeout: 5400000 });
  const result = await page.evaluate(() => window.__gpuCheck);
  await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
  return result;
}

// T147: the harness in a worker thread of Node, WebGPU from Dawn (the npm package webgpu at --webgpu), the GPU's
// worker (public/gpu.js) in a worker thread of its own, the modules from their files and the rest from the server
async function inDawn() {
  const { Worker } = await import("node:worker_threads");
  if (!webgpu) throw new Error("--engine dawn wants --webgpu <the directory of the npm package webgpu>");
  const dir = path.join(directory, "dawn");
  fs.mkdirSync(dir, { recursive: true });
  const prelude = `import { parentPort, Worker as NodeWorker } from "node:worker_threads";
import { create, globals } from ${JSON.stringify(pathToFileURL(path.resolve(webgpu, "index.js")).href)};
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, "navigator", { value: { gpu: create([]) }, configurable: true });
globalThis.self = globalThis;
globalThis.postMessage = (data) => parentPort.postMessage(data);
const origin = "http://localhost:${server.address().port}", nativeFetch = fetch;
globalThis.fetch = (url, init) => nativeFetch(new URL(url, origin), init);
console.log = console.info = console.warn = (...parts) => parentPort.postMessage({ line: parts.join(" ") });
`;
  const gpuFile = path.join(dir, "gpu-worker.mjs"), harnessFile = path.join(dir, "harness.mjs");
  // the GPU's worker: its messages wait until gpu.js has set onmessage (a module worker's port opens at its first await)
  fs.writeFileSync(gpuFile, `${prelude}
const waiting = [];
globalThis.onmessage = null;  // gpu.js sets it, a module's plain assignment
parentPort.on("message", (data) => (globalThis.onmessage ? globalThis.onmessage({ data }) : waiting.push(data)));
globalThis.close = () => process.exit(0);
await import(${JSON.stringify(pathToFileURL(path.join(root, "public", "gpu.js")).href + "?v=gpu-check")});
waiting.splice(0).forEach((data) => globalThis.onmessage({ data }));
`);
  fs.writeFileSync(harnessFile, `${prelude}
globalThis.Worker = class {
  constructor() {
    this.worker = new NodeWorker(${JSON.stringify(gpuFile)});
    this.worker.on("message", (data) => (data.line !== undefined ? parentPort.postMessage(data) : this.onmessage?.({ data })));
    this.worker.on("error", (error) => this.onerror?.({ message: String(error?.stack ?? error) }));
  }
  postMessage(data) { this.worker.postMessage(data); }
  terminate() { this.worker.terminate(); }
};
const ONLY = ${JSON.stringify(only ? only.split(",") : [])};
${HARNESS.replaceAll('import("/public/', `import(${JSON.stringify(pathToFileURL(path.join(root, "public")).href + "/")} + "`)}
`);
  return new Promise((resolve) => {
    const worker = new Worker(harnessFile);
    worker.on("message", (data) => {
      if (data.line !== undefined) return lines.push(`[log] ${data.line}`);
      resolve(data);
      worker.terminate();
    });
    worker.on("error", (error) => resolve({ error: String(error?.stack ?? error) }));
  });
}

// ---- the comparisons
const floats = (b64) => {
  const bytes = Buffer.from(b64, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
// the worst row (width values) of got against want: the largest difference over the row's largest value
function worstRow(got, want, width) {
  let worst = 0;
  for (let at = 0; at < want.length; at += width) {
    let largest = 0, difference = 0;
    for (let i = at; i < at + width; i++) {
      largest = Math.max(largest, Math.abs(want[i]));
      difference = Math.max(difference, Math.abs(got[i] - want[i]));
    }
    worst = Math.max(worst, difference / (largest || 1));
  }
  return worst;
}
const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);
// T187: the log-probabilities of logits (float64)
function logSoftmax(logits) {
  const top = logits.reduce((m, x) => Math.max(m, x), -Infinity);
  let sum = 0;
  for (const x of logits) sum += Math.exp(x - top);
  const log = top + Math.log(sum);
  return Float64Array.from(logits, (x) => x - log);
}
// the KL divergence (nats) of got's softmax from want's (want's log-probabilities given)
function divergence(p, got) {
  const q = logSoftmax(got);
  let kl = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > -80) kl += Math.exp(p[i]) * (p[i] - q[i]);
  return Math.max(kl, 0);
}
// the least-squares factor of got against want, less 1: a scale that is wrong by a little shows here, where the
// rounding of every value (a noise) averages out
function scaleOff(got, want) {
  let gw = 0, ww = 0;
  for (let i = 0; i < want.length; i++) { gw += got[i] * want[i]; ww += want[i] * want[i]; }
  return gw / ww - 1;
}
// the status of a GPU that takes the prompts (T152: what it says of the tokens after it)
// (T152: "prompts and answers on WebGPU", or "prompts on WebGPU, answers on the CPU")
const promptsOnGpu = (note) => /^prompts (and answers )?on WebGPU($|, )/.test(note ?? "");
let failed = false;
for (const { id, cpu, gpu: runs, late, refused, remembered, alone } of outcome.results) {
  const c = cases.find((entry) => entry.id === id), ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (c.headDim || dim / heads) * kvHeads;
  const exact = { keys: ref.keys, values: ref.values };
  // T187: the worst row of every layer (the lines are held a layer at a time)
  const byLayer = (got, want = exact) => Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(got.keys), layer(want.keys), kvDim), worstRow(layer(got.values), layer(want.values), kvDim));
  });
  const kv = (run, want = exact) => Math.max(worstRow(floats(run.keys), floats(want.keys), kvDim), worstRow(floats(run.values), floats(want.values), kvDim));
  // T153: the first layer's alone (a wrong step shows there already; the float16 of the cache grows over the layers)
  const first = (b64) => floats(b64).subarray(0, n * kvDim);
  const firstKv = (run, want = exact) => Math.max(worstRow(first(run.keys), first(want.keys), kvDim), worstRow(first(run.values), first(want.values), kvDim));
  // T187: the first layer's scale against NumPy's, the larger of the keys' and the values'
  const scale = (run, want = exact) => [scaleOff(first(run.keys), first(want.keys)), scaleOff(first(run.values), first(want.values))]
    .reduce((worst, x) => (Math.abs(x) > Math.abs(worst) ? x : worst), 0);
  const cpuKv = kv(cpu);
  // T153: E16, NumPy's answer with its cache in float16 against NumPy's. T187: Q8, with the matrices' inputs in 8 bits
  const half = { keys: ref.keys16, values: ref.values16 }, eight = { keys: ref.keys8, values: ref.values8 };
  const e16 = kv(half), q8 = kv(eight), q8First = firstKv(eight), e16s = byLayer(half), q8s = byLayer(eight);
  const want = floats(ref.logits), largest = want.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  const p = logSoftmax(want), top = argmax(want);
  const tenBest = [...want.keys()].sort((i, j) => want[j] - want[i]).slice(0, 10);
  const logits = (b64) => {
    const got = floats(b64);
    return { kl: divergence(p, got), ten: tenBest.reduce((m, i) => Math.max(m, Math.abs(got[i] - want[i])), 0) / largest, token: argmax(got) };
  };
  // the most likely token NumPy's, or one NumPy gives at least half the probability of its own (a near tie)
  const near = (token) => token === top || p[token] >= p[top] - TIE;
  const cpuLogits = logits(cpu.logits);
  const measures = { e16, q8, cpuLogits, logits, scale, kl16: divergence(p, floats(ref.logits16)), kl8: divergence(p, floats(ref.logits8)) };
  console.log(`${id} (${layers} layers, ${heads} heads, ${kvHeads} of keys and values, ${n} tokens): the CPU's keys and values ` +
    `${cpuKv.toExponential(2)} from NumPy's (the first layer's scale ${scale(cpu).toExponential(1)}), logits KL ${cpuLogits.kl.toExponential(2)}, ` +
    `most likely ${cpuLogits.token} ${cpuLogits.token === top ? "as NumPy's" : `(NumPy's ${top})`}, the prompt ${(cpu.promptMs / n).toFixed(2)} ms a token; ` +
    `E16 (NumPy's cache in float16) ${e16.toExponential(2)} (KL ${measures.kl16.toExponential(1)}, scale ${scale(half).toExponential(1)}); ` +
    `Q8 (and its matrices' inputs in 8 bits) ${q8.toExponential(2)}, the first layer ${q8First.toExponential(2)} (KL ${measures.kl8.toExponential(1)}, scale ${scale(eight).toExponential(1)})`);
  if (late) {
    // T147: a request forward.js gave up on is answered by nothing
    const tried = promptsOnGpu(late.note) && late.gpuTokens === 0;
    // T148: and the CPU did those blocks itself: its keys are the CPU's own run's, to the bit
    const cpuDid = late.keys === cpu.keys;
    const wrong = tried && (late.done !== 0 || late.failed !== 0 || !cpuDid);
    console.log(`  a GPU that answers late: ${!tried ? `not tried (${late.note}, ${late.gpuTokens} tokens on the GPU)` : wrong ? `it answered (done ${late.done}, failed ${late.failed}) or the CPU did not do its blocks (${cpuDid ? "it did" : "it did not"}) — FAILED` : "given up, it wrote nothing, and the CPU did the blocks"}`);
    failed ||= wrong;
  }
  if (refused) {
    // T148: refused before anything was compiled (SwiftShader compiles a shader in 10 to 90 s): within a few seconds
    const right = /a fallback adapter/.test(refused.note) && refused.seconds < 30 && refused.ended;
    console.log(`  a fallback adapter without the tests' leave: ${refused.note} in ${refused.seconds.toFixed(1)} s, ` +
      `${refused.ended ? "its worker ended" : "its worker did not say it ended"}${right ? "" : " — FAILED"}`);
    failed ||= !right;
    // T205: stopped while it got ready, it ends and says so (within forward.js's 5 s)
    const { whileStarting: w } = refused;
    console.log(`  a GPU let go while it got ready: ${w.ended ? "its worker ended" : "its worker did not say it ended"} in ${w.seconds.toFixed(1)} s${w.ended ? "" : " — FAILED"}`);
    failed ||= !w.ended;
  }
  if (alone) {
    // T156: on the GPU alone the same shaders on the same weights write the same keys and values and ids as the run
    // with every byte in memory, to the bit; a step on the CPU throws; a GPU that fails as it opens is lost at once,
    // and says why
    const { right: r, pieces: p } = alone;
    const same = r.keys === alone.keys && r.values === alone.values && JSON.stringify(r.first) === JSON.stringify(alone.first);
    const good = !r.lost && r.gpuTokens === n && same && /^prompts and answers on WebGPU/.test(r.status ?? "") && /on it alone/.test(r.threw ?? "")
      && Boolean(p.lost) && /on it alone/.test(p.threw ?? "");
    console.log(`  on the GPU alone (${(r.stored / 1e6).toFixed(2)} MB here): ${r.lost ? `lost (${r.lost})` : `${r.gpuTokens} tokens of the prompt, keys and values ` +
      `${r.keys === alone.keys && r.values === alone.values ? "the same to the bit" : "NOT the same"} as with the weights here, steps ${JSON.stringify(r.first)} ` +
      `(${JSON.stringify(alone.first)}), "${r.status}", a step on the CPU ${r.threw ? "threw" : "ran"}`}; a GPU that fails as it opens: ${p.lost ? `lost (${p.lost})` : "not lost"}, ` +
      `a step on the CPU ${p.threw ? "threw" : "ran"}${good ? "" : " — FAILED"}`);
    failed ||= !good;
  }
  if (remembered) {
    const right = remembered.same?.remembered === true && remembered.other?.remembered === false;
    console.log(`  the shaders remembered: for this adapter ${remembered.same?.remembered ? "taken" : "not taken"} (${remembered.same?.matrices}), ` +
      `for another ${remembered.other?.remembered ? "taken" : "not taken"}${right ? "" : " — FAILED"}`);
    failed ||= !right;
  }
  for (const gpu of runs) {
    const failures = [];
    if (!promptsOnGpu(gpu.note)) failures.push(`the GPU did not take it: ${gpu.note}`);
    if (gpu.gpuTokens !== n || gpu.again?.gpuTokens !== n) failures.push(`the GPU took ${gpu.gpuTokens} and ${gpu.again?.gpuTokens} of ${n} tokens`);
    if (gpu.ended === false) failures.push("the GPU's worker did not say it ended within 5 s of the release (T205)");
    if (gpu.past?.gpuTokens !== 0) failures.push(`a block past the GPU's keys and values went to the GPU (${gpu.past?.gpuTokens} tokens)`);
    const kind = /DP4A/.test(gpu.form ?? "") ? "packed" : /f16/.test(gpu.form ?? "") ? "f16" : "float32";
    // T187: the packed shaders' first layer and scale against Q8, their own arithmetic, and all their layers against
    // NumPy's by Q8's distance from it (see the lines); the others against NumPy's
    const packed = kind === "packed", [firstStick, name] = packed ? [eight, "Q8"] : [exact, "NumPy's"];
    const shaderLine = kind === "f16" ? HALF_LINE : GPU_LINE;
    // T187: a line a layer (the review: the largest of all layers against the largest of E16's or Q8's let a deep
    // model's early layers go 10 times their own E16 or Q8)
    const lines = e16s.map((e16l, l) => (packed ? K_PACKED * q8s[l] : Math.max(shaderLine, K[kind] * e16l)));
    const gpuKv = kv(gpu), againKv = gpu.again ? kv(gpu.again) : NaN, gpuFirst = firstKv(gpu, firstStick);
    const gpuLayers = byLayer(gpu), againLayers = gpu.again ? byLayer(gpu.again) : lines.map(() => NaN);
    const ratios = lines.map((line, l) => Math.max(gpuLayers[l], againLayers[l]) / line);
    // the worst layer (NaN the worst of all)
    const at = ratios.reduce((worst, x, l) => (!(x <= ratios[worst]) ? l : worst), 0);
    Object.assign(gpu, { line: lines[at], ratio: ratios[at], layer: at });
    if (!(ratios[at] <= 1)) {
      failures.push(`the keys and values of the GPU's layer ${at} are ${gpuLayers[at].toExponential(2)} and ${againLayers[at].toExponential(2)} from NumPy's (its line ${lines[at].toExponential(2)})`);
    }
    if (!(gpuFirst <= shaderLine)) failures.push(`the first layer's keys and values are ${gpuFirst.toExponential(2)} from ${name} (line ${shaderLine.toExponential(2)})`);
    const gpuScale = scale(gpu, firstStick), againScale = gpu.again ? scale(gpu.again, firstStick) : NaN;
    if (!(Math.abs(gpuScale) <= SCALE_LINE) || !(Math.abs(againScale) <= SCALE_LINE)) {
      failures.push(`the first layer's keys and values are scaled by 1 + ${gpuScale.toExponential(2)} and 1 + ${againScale.toExponential(2)} of ${name} (line ${SCALE_LINE})`);
    }
    const gpuLogits = logits(gpu.logits), againLogits = logits(gpu.again.logits);
    if (!near(gpuLogits.token) || !near(againLogits.token)) failures.push(`another most likely token (${gpuLogits.token}, ${againLogits.token}) than NumPy's ${top}, and not a near tie`);
    console.log(`  ${gpu.form ?? "no form"}, ${gpu.attention ?? "no attention"}: keys and values ${gpuKv.toExponential(2)} (all at once ${againKv.toExponential(2)}, ` +
      `the first layer ${gpuFirst.toExponential(2)} from ${name}, the CPU's ${firstKv(cpu).toExponential(2)}; ${(gpuKv / e16).toFixed(1)} E16, ${(gpuKv / q8).toFixed(2)} Q8; ` +
      `layer ${at} ${ratios[at].toFixed(2)} of its line), ` +
      `the first layer's scale ${gpuScale.toExponential(1)} (${againScale.toExponential(1)}), logits KL ${gpuLogits.kl.toExponential(2)} (${againLogits.kl.toExponential(2)}), ` +
      `the prompt ${(gpu.promptMs / n).toFixed(2)} ms a token (${gpu.note})` +
      (failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""));
    failed ||= failures.length > 0;
    // (evaluated, and said, whatever came before)
    const stepsFine = gpu.steps ? stepsRight(c, gpu.steps, { e16s, q8s, kvDim, prompt: gpuLayers }) : true;
    failed ||= !stepsFine;
  }
  layerTables(c, cpu, runs, measures);
}
process.exit(failed ? 1 : 0);

// T213: the first matrix a token's layer would bind inside a joined buffer (q, k and v as one, gate and up as one,
// in gpu.js's tokensLayout's order; each after the ones before it) whose values or scales would not start where a device binds a buffer, or null where every
// one does (the model's steps then belong on the GPU). The device's alignment is WebGPU's default of 256 bytes
// (gpu.js asks for no other), the scales 4 bytes a group of 32: stories15M's k starts at 288 × 288 weights, whose
// scales, 10368 bytes, are not a multiple of 256 (T150's (b))
function unbound(c) {
  const [dim, hidden, , heads, kvHeads] = c.reference.header, head = c.headDim || dim / heads, ALIGN = 256, GROUP = 32;
  const joined = { wk: head * heads * dim, wv: head * (heads + kvHeads) * dim, w3: hidden * dim };
  for (const [name, values] of Object.entries(joined)) if (values % ALIGN || (values / GROUP) * 4 % ALIGN) return name;
  return null;
}

// T152: the steps of a generation on the GPU against NumPy's greedy continuation (see the head of this file): a line
// "  a token by <form>: ..." and whether it is right
function stepsRight(c, steps, { e16s, q8s, kvDim, prompt }) {
  const ref = c.reference, greedy = ref.greedy, vocab = floats(ref.logits).length;
  const rows = floats(ref.greedyLogits), logitsOf = (i) => rows.subarray(i * vocab, (i + 1) * vocab);
  const failures = [], said = [];
  // a model whose steps stay on the CPU (not asked of the GPU), or the run in pieces (T155: a token's layer reads a
  // matrix whole): said, and right
  if (steps.why) {
    // (T152's review: or a classifier larger than what the adapter binds, a table being one piece; the rows of the
    // logits hold its size: vocabulary × dim)
    const table = floats(ref.logits).length * ref.header[0];
    const right = steps.planned === false || (c.force?.pieceBytes && !steps.forced && /past a buffer/.test(steps.why)) ||
      (/^the (classifier|embedding) is past a buffer/.test(steps.why ?? "") && table > steps.binds) ||
      steps.why === `${unbound(c)} would not start where this GPU binds a buffer`;
    console.log(`  a token: on the CPU (${steps.why})${right ? "" : " — FAILED"}`);
    return right;
  }
  // whether NumPy gives id at least half the probability of want (T187's near tie: log-probabilities within TIE)
  const near = (logits, want, id) => id === want || logSoftmax(logits)[id] >= logSoftmax(logits)[want] - TIE;
  // the GPU's ids against NumPy's, step after step from `at`, up to the first that is not NumPy's (then a near tie; the
  // steps after it have other inputs). keep: the positions whose inputs were NumPy's are compared (theirs, [position
  // - n] each)
  const positions = [];
  const run = (ids, at, count, keep = true) => {
    if (!ids || ids.length !== count) {
      failures.push(`${count} steps from ${at} came back as ${JSON.stringify(ids)}`);
      return 0;
    }
    let same = 0;
    for (; same < count && ids[same] === greedy[at + same]; same++);
    if (keep) for (let i = 0; i <= Math.min(same, count - 1); i++) positions.push(at + i);
    if (same < count && !near(logitsOf(at + same), greedy[at + same], ids[same])) {
      failures.push(`step ${at + same} is ${ids[same]}, NumPy's ${greedy[at + same]}, and not a near tie`);
    }
    return same;
  };
  // (the keys and values of the second run's positions are not held to a line: they are the attention's over the
  // CPU's step between, its 7-bit activations, 1.02e-2 to 1.30e-2 on llm-jp-3 150M's float forms, run 36324147634)
  const first = run(steps.first, 0, 4), second = first === 4 ? run(steps.second, 5, 3, false) : 0;
  // the step after the prompt went through the CPU (the GPU's cache held another's): the CPU's most likely token on
  // the same keys and values, or a near tie of the CPU's logits
  const cpuLogits = floats(steps.uploadedLogits ?? ""), cpuTop = argmax(cpuLogits), uploaded = steps.uploaded?.[0];
  const cpuP = cpuLogits.length ? logSoftmax(cpuLogits) : [], gap = cpuP[cpuTop] - cpuP[uploaded];
  // (the CPU's logits carry its 7-bit activations at that token, the GPU's not: a quarter of the probability, not half.
  // synthetic-wide's GPU took a token 0.72 below the CPU's on every form, run 36326266335; the GPU without the CPU's
  // keys and values, see TODO.md's T152)
  if (!cpuLogits.length || !(gap <= 2 * TIE)) failures.push(`after a prompt on the CPU the step is ${uploaded}, the CPU's ${cpuTop}, and not a near tie (${gap?.toFixed(2)} below it in log-probability)`);
  said.push(`${first}${first === 4 ? ` and ${second}` : ""} greedy ids as NumPy's${first === 4 && second === 3 ? "" : " (then a near tie)"}` +
    `, after a prompt on the CPU ${uploaded}${uploaded === cpuTop ? " as the CPU's" : ` (the CPU's ${cpuTop}, ${gap.toFixed(2)} below it)`}` +
    (first === 4 ? `, the CPU's step between them ${steps.cpuStep === greedy[4] ? "NumPy's" : `${steps.cpuStep} (NumPy's ${greedy[4]})`}` : ""));
  // the keys and values the GPU wrote back of the positions whose inputs were NumPy's (not the CPU's step's: the CPU's)
  const layers = ref.header[2];
  const rowsOf = (b64) => {
    const all = floats(b64), out = new Float32Array(layers * positions.length * kvDim);
    for (let l = 0; l < layers; l++) {
      positions.forEach((p, k) => out.set(all.subarray((l * GEN + p) * kvDim, (l * GEN + p + 1) * kvDim), (l * positions.length + k) * kvDim));
    }
    return out;
  };
  // a layer at a time (T187's lines: the largest of all layers against the largest E16 or Q8 let a deep model's early
  // layers go 10 times their own; the review of T152): the line of layer l is the prompt's shaders' of the same
  // arithmetic at layer l (T187: DP4A by Q8's distance), or the run's own prompt's distance at layer l where that is
  // larger (the steps' attention reads its keys and values)
  const [gotK, gotV, wantK, wantV] = [steps.keys, steps.values, ref.greedyKeys, ref.greedyValues].map(rowsOf);
  const rowsAt = (all, l) => all.subarray(l * positions.length * kvDim, (l + 1) * positions.length * kvDim);
  const packed = /DP4A/.test(steps.form ?? "");
  const offs = Array.from({ length: layers }, (_, l) => Math.max(worstRow(rowsAt(gotK, l), rowsAt(wantK, l), kvDim), worstRow(rowsAt(gotV, l), rowsAt(wantV, l), kvDim)));
  const lines = offs.map((_, l) => Math.max(packed ? K_PACKED * q8s[l] : Math.max(GPU_LINE, K.float32 * e16s[l]), prompt[l]));
  const ratios = offs.map((off, l) => off / lines[l]);
  const at = ratios.reduce((worst, x, l) => (!(x <= ratios[worst]) ? l : worst), 0);
  if (!(ratios[at] <= 1)) failures.push(`the keys and values of the steps' layer ${at} are ${offs[at].toExponential(2)} from NumPy's (its line ${lines[at].toExponential(2)})`);
  said.push(`keys and values of ${positions.length} positions ${Math.max(...offs).toExponential(2)}, layer ${at} ${ratios[at].toFixed(2)} of its line ${lines[at].toExponential(2)}`);
  // (T152's review) 4 sampled steps in one submission, and one at a time: the same ids to the bit
  if (JSON.stringify(steps.together) !== JSON.stringify(steps.apart)) failures.push(`4 sampled steps together ${JSON.stringify(steps.together)}, one at a time ${JSON.stringify(steps.apart)}`);
  said.push(`4 sampled steps ${JSON.stringify(steps.together)}${JSON.stringify(steps.together) === JSON.stringify(steps.apart) ? " one at a time too" : ""}`);
  // (T152's review) the CPU's keys and values went up whole: over two other caches the step wrote the same bits
  if (!steps.uploadedSame || JSON.stringify(steps.uploaded) !== JSON.stringify(steps.uploadedTwice)) {
    failures.push(`after a prompt on the CPU over two other caches of the GPU, the step wrote other keys and values (${JSON.stringify(steps.uploaded)}, ${JSON.stringify(steps.uploadedTwice)})`);
  }
  // a stop token: NumPy's second id ends the steps after it
  const until = greedy.indexOf(greedy[1]) + 1, wanted = greedy.slice(0, until);
  if (first >= until && JSON.stringify(steps.stopped) !== JSON.stringify(wanted)) failures.push(`with ${greedy[1]} a stop token the steps were ${JSON.stringify(steps.stopped)}, not ${JSON.stringify(wanted)}`);
  said.push(`a stop token after ${steps.stopped?.length} steps`);
  // the penalty (100 on NumPy's first id, greedy): the largest of NumPy's logits penalized so, or a near tie
  const penalized = logitsOf(0).slice();
  wgsl.penalizeLikeCpu(penalized, steps.penaltyHistory ?? [], 100);
  const wantPenalized = wgsl.argmaxLikeCpu(penalized), gotPenalized = steps.penalized?.[0];
  if (!near(penalized, wantPenalized, gotPenalized)) failures.push(`with a penalty of 100 the step is ${gotPenalized}, NumPy's ${wantPenalized}, and not a near tie`);
  said.push(`penalized ${gotPenalized}${gotPenalized === wantPenalized ? " as NumPy's" : ` (NumPy's ${wantPenalized})`}`);
  // sampled at temperature 2 with a random number of 0.02 and one of 0.98: two tokens of NumPy's nucleus, and not
  // the same (the random numbers reach the GPU; its logits are not NumPy's, so where either lands is not held to it)
  const walk = wgsl.walkLikeCpu(logitsOf(0), 2, 0.999), [low, high] = steps.sampled ?? [];
  if (!(walk.tokens.includes(low?.[0]) && walk.tokens.includes(high?.[0]) && low[0] !== high[0])) {
    failures.push(`sampled at 0.02 and 0.98: ${low?.[0]} and ${high?.[0]}, of NumPy's nucleus of ${walk.tokens.length}, which are to differ`);
  }
  said.push(`sampled ${low?.[0]} and ${high?.[0]}`);
  // T209: the tables were cut where the run asked for it (a vocabulary of 192 rows or more is 3 pieces of 64)
  if (steps.cut && !(steps.pieces > 1)) failures.push(`the tables in ${steps.pieces} piece, not cut`);
  console.log(`  a token by ${steps.form}${steps.pieces > 1 ? ` (the tables in ${steps.pieces} pieces)` : ""}: ${said.join(", ")}${failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""}`);
  return !failures.length;
}

// T183: what a person reads in the log of CI to judge the numbers (Markdown: gpu-prompt.yml puts the log in the run's
// summary too). A table of the keys and values by layer against NumPy's, with E16's column (NumPy's with its cache in
// float16) and T187's Q8 (and its matrices' inputs in 8 bits) beside the runs', where a layer that jumps shows; the same
// against NumPy's with its cache in float16 (what is left is the arithmetic, not the cache); a line a run: its form, its
// line and the ratio to it, how many E16 and Q8, the first layer's scale, its logits' KL and the largest difference of
// NumPy's ten most likely, its most likely token; the seconds of every step. The runs are the GPU's worker's first (the
// forms it chose), then one a form of the matrices (T147), then the attention without subgroups.
function layerTables(c, cpu, runs, { e16, q8, cpuLogits, logits, scale }) {
  const ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (c.headDim || dim / heads) * kvHeads;
  const exact = { keys: ref.keys, values: ref.values }, half = { keys: ref.keys16, values: ref.values16 };
  const eight = { keys: ref.keys8, values: ref.values8 };
  const byLayer = (got, want) => Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(got.keys), layer(want.keys), kvDim), worstRow(layer(got.values), layer(want.values), kvDim));
  });
  const e = (x) => x.toExponential(1), letter = (i) => String.fromCharCode(65 + i);
  const columns = [["CPU", cpu], ...runs.map((run, i) => [letter(i), run])];
  const tables = [[`against NumPy's (E16: NumPy's with its cache in float16; Q8: and the matrices' inputs in 8 bits)`, exact,
    [["E16", half], ["Q8", eight]]], ["against NumPy's with its cache in float16", half, []],
    ["against Q8 (T187: what the packed shaders are held to)", eight, []]];
  for (const [title, want, more] of tables) {
    const table = [...more, ...columns].map(([, got]) => byLayer(got, want));
    console.log(`\nkeys and values of ${c.id} by layer, ${title}: the worst row, its largest difference over its largest value\n`);
    console.log(`| layer | ${[...more, ...columns].map(([name]) => name).join(" | ")} |`);
    console.log(`|---:|${table.map(() => "---:").join("|")}|`);
    for (let l = 0; l < layers; l++) console.log(`| ${l} | ${table.map((column) => e(column[l])).join(" | ")} |`);
    console.log(`| all | ${table.map((column) => e(Math.max(...column))).join(" | ")} |`);
  }
  console.log("");
  const all = (run) => Math.max(...byLayer(run, exact));
  const said = (x) => `logits KL ${e(x.kl)}, the ten most likely ${e(x.ten)}, most likely ${x.token}`;
  console.log(`- CPU: forward.js on the CPU: ${e(all(cpu))}, ${(all(cpu) / e16).toFixed(1)} E16, ${(all(cpu) / q8).toFixed(2)} Q8, ` +
    `the first layer's scale ${e(scale(cpu))}, ${said(cpuLogits)}`);
  runs.forEach((run, i) => {
    console.log(`- ${letter(i)}: ${run.form ?? "no form"}, ${run.attention ?? "no attention"}: ${e(all(run))} ` +
      `(all at once ${run.again ? e(all(run.again)) : "-"}), the worst layer ${run.layer} at ${run.ratio.toFixed(2)} of its line ${e(run.line)}, ` +
      `${(all(run) / e16).toFixed(1)} E16, ${(all(run) / q8).toFixed(2)} Q8, ${e(Math.max(...byLayer(run, half)))} from NumPy's with its cache in float16, ` +
      `${e(Math.max(...byLayer(run, eight)))} from Q8, the first layer's scale ${e(scale(run))} (against Q8 ${e(scale(run, eight))}), ${said(logits(run.logits))}${logits(run.logits).token === cpuLogits.token ? " as the CPU's" : ""}`);
  });
  const seconds = { "NumPy (with its cache in float16, and Q8)": c.numpySeconds, "the plan": c.planSeconds, CPU: cpu.seconds };
  runs.forEach((run, i) => { seconds[`${letter(i)} (its GPU ready in ${run.readySeconds.toFixed(1)})`] = run.seconds; });
  console.log(`\nseconds of ${c.id}: ${Object.entries(seconds).map(([step, s]) => `${step} ${s.toFixed(1)}`).join(", ")}\n`);
}
