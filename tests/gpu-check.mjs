// T135: the prompt on the GPU (public/gpu.js) against the CPU's (public/forward.js) and NumPy's (llama2_numpy.py), in
// a real browser, as WebGPU is nowhere else: Playwright's Chromium, whose WebGPU without a GPU is SwiftShader (the CPU
// in the GPU's place: its speed means nothing, its numbers are right or not).
//
//   node tests/gpu-check.mjs [model id | synthetic ...] [--engine chromium|chrome|msedge] [--forms <part,part>]
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
//   - the logits of the prompt's last token (the CPU's in both runs, on the GPU's keys and values in one): the same
//     most likely token as the run on the CPU, and no farther from NumPy's than LOGITS_LINE times that run (see there).
// "synthetic": a made-up int8 model with grouped-query attention (4 heads, 2 of keys and values; none of the models
// of this directory has it). T153: "synthetic-qwen2", the same with biases of q, k and v (Qwen2's) and an epsilon of
// 1e-6; "synthetic-qwen3", the norms of every head of q and k (Qwen3's), heads of 32 where dim / heads is 16 (q and
// the attention's output 128 wide, dim 64), and an epsilon of 0.5, near mean(x²) (T150: an epsilon far below it
// hides a wrong one); both of three layers. T154: "synthetic-gpt2", GPT-2's form (LayerNorm with biases, a bias after
// every matrix, an FFN of two matrices and GELU, learned positions and no RoPE), and "synthetic-neox", GPT-NeoX's (the
// same with RoPE on the first quarter of every head, as Pythia's rotary_pct 0.25, and the parallel residual), both of
// three layers and 4 heads of keys and values (neither has grouped-query attention). The others are the models of this directory (make models kernels), or <prefix>.json: a
// model tests/perplexity_prepare.py converted (<prefix>.bin, <prefix>.tokenizer.bin, and the options in <prefix>.json;
// gpu-prompt.yml's input real= fetches and converts models of src/models.js so, T183), whose NumPy answer comes from
// the native Python ($PYTHON, python3 by default).
//
// T183: what a person reads to judge it, in the log of CI (the development machine does not run WebGPU's tests): E16,
// how far NumPy's answer moves when nothing but its cache is rounded to float16 (answer(half=True), T153's review), and
// for each model a table of the keys and values by layer, a column E16 and one a run (the CPU's, and the GPU's lettered
// A, B...) against NumPy's, another against NumPy's with its cache in float16, a line a run with its form, its line and
// the ratio to it, how many E16 it is, its logits against the CPU's and its most likely token, and the seconds of
// every step.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
// T147: --forms <part,part>: only the matrices' shaders whose names hold one of these (all of them by default)
const only = option("--forms", "");
const webgpu = option("--webgpu", "");
const ids = args.length ? args : ["synthetic", "synthetic-qwen2", "synthetic-qwen3", "synthetic-gpt2", "synthetic-neox", "stories15M", "tiny-lm", "llm-jp-3-150m"];
// T153: the made-up models of another form (see above). Three layers: a layer's vectors are read at l × their size,
// which a second layer alone would not tell from 0 + size
const SYNTHETIC = { "synthetic": [{}, {}], "synthetic-qwen2": [{ layers: 3, bias: true }, { bias: true, rms_norm_eps: 1e-6 }],
  "synthetic-qwen3": [{ layers: 3, qk_norm: true, head_dim: 32 }, { qk_norm: true, head_dim: 32, rms_norm_eps: 0.5 }],
  "synthetic-gpt2": [{ layers: 3, kv_heads: 4, arch: "gpt2" }, { arch: "gpt2" }],
  "synthetic-neox": [{ layers: 3, kv_heads: 4, arch: "neox" }, { arch: "neox", rotary: 4, parallel_residual: true }] };
// T147: 150 tokens, so that the GPU's blocks of 64 are two and a part (the tiles' ends), and the caches grow to 256
const COUNT = 150, KV_START = 8;
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
//     SIMD takes 7): 0.30 to 0.56 of the CPU's. The line: 0.75 of the CPU's (a ratio to the CPU's error: measure it
//     again where the CPU's arithmetic changes, T159, T165).
// T153: those lines hold for the first layer of every model and for all layers of a shallow one. Over the layers of a
// deep one the float16 of the cache grows by itself, the more so with Qwen's large activations: E16 (NumPy's answer
// with nothing but its cache rounded to float16, against NumPy's) is its measure. A GPU that computes as it should is a
// few E16 away, whatever the CPU's forward.js does (a line of a quarter of the CPU's error loosened the made-up models'
// lines 1.3 to 10 times, and a GPU that used layer 0's biases in every layer passed: T153's review). So every layer of
// a float32 or f16 form is held to the shader's line or K × E16, whichever is larger, and the first layer to the
// shader's line alone. K (CI's Dawn on lavapipe, run 36308518805, 149 tokens): the float32 shaders are 0.9 to 1.6 E16
// away (Qwen3 0.6B the most), llama.cpp's f16 0.9 to 3.2 and 5.5 (Qwen3 0.6B with the f16 attention), while a GPU that
// reads layer 0's biases in every layer (synthetic-qwen2) is 13.9 E16 away on float32 and 14.4 on f16: K is 4 and 8.
// The packed shaders stay on their ratio to the CPU's (their 8-bit activations are 19 to 47 E16 away: E16 is not
// their measure; T147's weakness, TODO.md)
const GPU_LINE = 8e-3, HALF_LINE = 1.5e-2, PACKED_LINE = 0.75;
const K = { float32: 4, f16: 8 };
// The logits of the prompt's last token, the largest difference from NumPy's over the largest of NumPy's: the CPU's
// own run 1.6e-2 to 5.6e-2, the one on the GPU's keys and values 0.75 to 1.04 times that (closer: its keys and values
// are NumPy's but for the float16); broken on purpose 0.27 to 1.13, 4.8 times the CPU's and more. The line: no more
// than 1.5 times the CPU's, and the same most likely token.
const LOGITS_LINE = 1.5;

// ---- Node: the plans and NumPy's answers
const { pyodide: py } = await pyodideWithEngine();
const PYTHON = `
import base64, struct, numpy as np, llama2_numpy, llama2_convert
from llama2_numpy import Llama

def synthetic(dim=64, hidden=128, layers=2, heads=4, kv_heads=2, vocab=320, seq_len=256, seed=0, **form):
    """A made-up int8 checkpoint and its tokenizer.bin, as quantize.py writes one: grouped-query attention. form
    (T153): llama2_numpy.FORM's bias, qk_norm and head_dim, whose vectors are drawn as the norms' are"""
    rng = np.random.default_rng(seed)
    header = (dim, hidden, layers, heads, kv_heads, vocab, seq_len)
    out = [struct.pack("<7i", *header)]
    for shape, is_matrix in llama2_convert.layout(*header, **form):
        if is_matrix is None:
            continue  # the RoPE tables: an int8 file leaves them out
        values = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        if not is_matrix:
            out.append((1.0 + values * 0.1).astype(np.float32).tobytes())
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

def answer(data, vocabulary, text, count, options, half=False):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens. half (T153): the cache rounded to float16, and nothing else (made whole
    at first, so that it never grows into an array of another class)"""
    numpy = Llama(data, vocabulary, **options)
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
    kv = lambda cache: np.ascontiguousarray(cache[:, :, :n, :].transpose(0, 2, 1, 3).reshape(numpy.n_layers, n, -1), dtype=np.float32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    return {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
            "header": list(struct.unpack_from("<7i", data, 0))}

def answers(data, vocabulary, text, count, options):
    """answer() and, T153, the keys and values of answer(half=True) beside it"""
    exact = answer(data, vocabulary, text, count, options)
    rounded = answer(data, vocabulary, text, count, options, half=True)
    return {**exact, "keys16": rounded["keys"], "values16": rounded["values"]}
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
  let options, text;
  const began = performance.now();
  if (SYNTHETIC[id]) {
    const [form, engineOptions] = SYNTHETIC[id];
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
  if (options.dtype !== "int8") throw new Error(`${id} is ${options.dtype}: the GPU takes int8 weights`);
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("TEXT", text ?? "");
  const reference = py.runPython(`answers(data, vocabulary, TEXT, ${COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  const numpySeconds = (performance.now() - began) / 1000;
  cases.push({ ...caseOf(id, options, reference, py.runPython("data").toJs()), numpySeconds });
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
  for (const [name, value] of Object.entries(plan.derived)) plan.derived[name] = Buffer.from(value).toString("base64");
  const name = path.basename(id), file = path.join(directory, `${name}.bin`);
  fs.writeFileSync(file, bytes);
  return { id, plan, headDim: options.head_dim ?? 0, arch: options.arch ?? "llama", checkpoint: `/case/${name}.bin`, file, reference, planSeconds: (performance.now() - began) / 1000 };
}

// ---- the browser: a page that is cross-origin isolated (its own headers), a worker that runs forward.js
const HARNESS = /* js */ `
const search = "?v=gpu-check";
const { compileKernels, createForward, weightsMemory, footprint } = await import("/public/forward.js" + search);
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
try {
  const kernels = compileKernels(await fetched("/public/simdkernel_shared.wasm"), await fetched("/public/simdkernel_relaxed_shared.wasm"));
  const results = [];
  for (const c of await (await fetch("/cases.json")).json()) {
    const plan = c.plan;
    for (const name of Object.keys(plan.derived)) plan.derived[name] = Uint8Array.from(atob(plan.derived[name]), (ch) => ch.charCodeAt(0));
    const checkpoint = await fetched(c.checkpoint), size = checkpoint.length, tokens = c.reference.tokens, n = tokens.length - 1;
    const { memory, base } = weightsMemory(size, { shared: true, after: footprint(c.reference.header, size, { dtype: "int8", halfKV: true, gpu: true, kvStart: plan.kv_start, head_dim: c.headDim, arch: c.arch }) });
    new Uint8Array(memory.buffer, base, size).set(checkpoint);
    // T148: SwiftShader and lavapipe are fallback adapters, which the page refuses: the tests take them (fallback),
    // and give the GPU every block it can take (always: a fallback adapter is far slower than the CPU)
    const TESTS = { fallback: true, always: true };
    const run = async (gpu, gpuForce, gpuRemembered) => {
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
      engine.release();
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
    const gpu = [];
    // (the forced ones untimed, T153: a block of 64 tokens of Qwen3 0.6B took more than the 180 s of a step on lavapipe)
    for (const form of [undefined, ...forms]) gpu.push(await run(openGpu, form ? { matrices: form, quick: true } : {}));
    // the attention without subgroups or f16 (the lanes of the workgroup stand for a subgroup), where the adapter
    // has them and so chose the other
    if (forms.length) gpu.push(await run(openGpu, { matrices: forms[0], attention: "llama.cpp flash attention tiles", quick: true }));
    // T148 (the made-up model only): a fallback adapter refused as the page refuses it, before anything is compiled;
    // the shaders of the first run remembered for this adapter (its key), which are then the only ones compiled, and
    // not for another key
    let refused, remembered;
    if (c.id === "synthetic") {
      const began = performance.now(), engine = createForward({ memory, base, size, kernels, plan, gpu: openGpu });
      refused = { note: await engine.gpu, seconds: (performance.now() - began) / 1000 };
      engine.release();
      const first = gpu[0].ready ?? {}, kept = { key: first.key, matrices: first.matrices, attention: first.attention };
      remembered = { same: (await run(openGpu, {}, kept)).ready, other: (await run(openGpu, {}, { ...kept, key: kept.key + "|another" })).ready };
    }
    results.push({ id: c.id, cpu: await run(undefined), gpu, late: c.id === "synthetic" ? await late() : undefined, refused, remembered });
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
  if (pathname === "/cases.json") return send(types[".json"], JSON.stringify(cases.map(({ file, reference: { keys16, values16, ...reference }, ...c }) => ({ ...c, reference }))));
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
let failed = false;
for (const { id, cpu, gpu: runs, late, refused, remembered } of outcome.results) {
  const c = cases.find((entry) => entry.id === id), ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (c.headDim || dim / heads) * kvHeads;
  const kv = (run) => Math.max(worstRow(floats(run.keys), floats(ref.keys), kvDim), worstRow(floats(run.values), floats(ref.values), kvDim));
  // T153: the first layer's alone (a wrong step shows there already; the float16 of the cache grows over the layers)
  const first = (b64) => floats(b64).subarray(0, n * kvDim);
  const firstKv = (run) => Math.max(worstRow(first(run.keys), first(ref.keys), kvDim), worstRow(first(run.values), first(ref.values), kvDim));
  const cpuKv = kv(cpu);
  // T153: E16, NumPy's answer with its cache in float16 against NumPy's
  const e16 = Math.max(worstRow(floats(ref.keys16), floats(ref.keys), kvDim), worstRow(floats(ref.values16), floats(ref.values), kvDim));
  const want = floats(ref.logits), largest = want.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  const logitsError = (b64) => floats(b64).reduce((m, x, i) => Math.max(m, Math.abs(x - want[i])), 0) / largest;
  const cpuLogits = logitsError(cpu.logits), best = argmax(floats(cpu.logits));
  console.log(`${id} (${layers} layers, ${heads} heads, ${kvHeads} of keys and values, ${n} tokens): the CPU's keys and values ` +
    `${cpuKv.toExponential(2)} from NumPy's, logits ${cpuLogits.toExponential(2)}, most likely ${best} ` +
    `${argmax(want) === best ? "as NumPy's" : `(NumPy's ${argmax(want)})`}, the prompt ${(cpu.promptMs / n).toFixed(2)} ms a token; ` +
    `E16 (NumPy's cache in float16) ${e16.toExponential(2)}`);
  if (late) {
    // T147: a request forward.js gave up on is answered by nothing
    const tried = late.note === "prompts on WebGPU" && late.gpuTokens === 0;
    // T148: and the CPU did those blocks itself: its keys are the CPU's own run's, to the bit
    const cpuDid = late.keys === cpu.keys;
    const wrong = tried && (late.done !== 0 || late.failed !== 0 || !cpuDid);
    console.log(`  a GPU that answers late: ${!tried ? `not tried (${late.note}, ${late.gpuTokens} tokens on the GPU)` : wrong ? `it answered (done ${late.done}, failed ${late.failed}) or the CPU did not do its blocks (${cpuDid ? "it did" : "it did not"}) — FAILED` : "given up, it wrote nothing, and the CPU did the blocks"}`);
    failed ||= wrong;
  }
  if (refused) {
    // T148: refused before anything was compiled (SwiftShader compiles a shader in 10 to 90 s): within a few seconds
    const right = /a fallback adapter/.test(refused.note) && refused.seconds < 30;
    console.log(`  a fallback adapter without the tests' leave: ${refused.note} in ${refused.seconds.toFixed(1)} s${right ? "" : " — FAILED"}`);
    failed ||= !right;
  }
  if (remembered) {
    const right = remembered.same?.remembered === true && remembered.other?.remembered === false;
    console.log(`  the shaders remembered: for this adapter ${remembered.same?.remembered ? "taken" : "not taken"} (${remembered.same?.matrices}), ` +
      `for another ${remembered.other?.remembered ? "taken" : "not taken"}${right ? "" : " — FAILED"}`);
    failed ||= !right;
  }
  for (const gpu of runs) {
    const failures = [];
    if (gpu.note !== "prompts on WebGPU") failures.push(`the GPU did not take it: ${gpu.note}`);
    if (gpu.gpuTokens !== n || gpu.again?.gpuTokens !== n) failures.push(`the GPU took ${gpu.gpuTokens} and ${gpu.again?.gpuTokens} of ${n} tokens`);
    if (gpu.past?.gpuTokens !== 0) failures.push(`a block past the GPU's keys and values went to the GPU (${gpu.past?.gpuTokens} tokens)`);
    const kind = /DP4A/.test(gpu.form ?? "") ? "packed" : /f16/.test(gpu.form ?? "") ? "f16" : "float32";
    const shaderLine = kind === "packed" ? PACKED_LINE * cpuKv : kind === "f16" ? HALF_LINE : GPU_LINE;
    const line = kind === "packed" ? shaderLine : Math.max(shaderLine, K[kind] * e16);
    const firstLine = kind === "packed" ? PACKED_LINE * firstKv(cpu) : shaderLine;
    const gpuKv = kv(gpu), againKv = gpu.again ? kv(gpu.again) : NaN, gpuFirst = firstKv(gpu);
    gpu.line = line;
    if (!(gpuKv <= line) || !(againKv <= line)) {
      failures.push(`the keys and values of the GPU are ${gpuKv.toExponential(2)} and ${againKv.toExponential(2)} from NumPy's (line ${line.toExponential(2)})`);
    }
    if (!(gpuFirst <= firstLine)) failures.push(`the first layer's keys and values are ${gpuFirst.toExponential(2)} from NumPy's (line ${firstLine.toExponential(2)})`);
    const gpuLogits = logitsError(gpu.logits), againLogits = logitsError(gpu.again.logits);
    if (!(gpuLogits <= LOGITS_LINE * cpuLogits) || !(againLogits <= LOGITS_LINE * cpuLogits)) {
      failures.push(`the logits on the GPU's keys and values are ${gpuLogits.toExponential(2)} and ${againLogits.toExponential(2)} from NumPy's, the CPU's ${cpuLogits.toExponential(2)}`);
    }
    // the most likely token: the CPU's, or one whose logit NumPy puts no farther below its largest than the CPU's run
    // is from NumPy's (a near tie, T147: stories15M at 149 tokens, where the GPU's logits were nearer NumPy's)
    const near = (b64) => { const i = argmax(floats(b64)); return i === best || want[argmax(want)] - want[i] <= LOGITS_LINE * cpuLogits * largest; };
    if (!near(gpu.logits) || !near(gpu.again.logits)) failures.push("another most likely token than on the CPU, and not a near tie");
    console.log(`  ${gpu.form ?? "no form"}, ${gpu.attention ?? "no attention"}: keys and values ${gpuKv.toExponential(2)} (all at once ${againKv.toExponential(2)}, ` +
      `the first layer ${gpuFirst.toExponential(2)}, the CPU's ${firstKv(cpu).toExponential(2)}; ${(gpuKv / e16).toFixed(1)} E16), ` +
      `logits ${gpuLogits.toExponential(2)} (${againLogits.toExponential(2)}), the prompt ${(gpu.promptMs / n).toFixed(2)} ms a token (${gpu.note})` +
      (failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""));
    failed ||= failures.length > 0;
  }
  layerTables(c, cpu, runs, { e16, cpuLogits, logitsError, best });
}
process.exit(failed ? 1 : 0);

// T183: what a person reads in the log of CI to judge the numbers (Markdown: gpu-prompt.yml puts the log in the run's
// summary too). A table of the keys and values by layer against NumPy's, with E16's column (NumPy's with its cache in
// float16) beside the runs', where a layer that jumps shows; the same against NumPy's with its cache in float16 (what is
// left is the arithmetic, not the cache); a line a run: its form, its line and the ratio to it, how many E16, its
// logits against the CPU's and its most likely token; the seconds of every step. The runs are the GPU's worker's first
// (the forms it chose), then one a form of the matrices (T147), then the attention without subgroups.
function layerTables(c, cpu, runs, { e16, cpuLogits, logitsError, best }) {
  const ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (c.headDim || dim / heads) * kvHeads;
  const exact = { keys: ref.keys, values: ref.values }, half = { keys: ref.keys16, values: ref.values16 };
  const byLayer = (got, want) => Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(got.keys), layer(want.keys), kvDim), worstRow(layer(got.values), layer(want.values), kvDim));
  });
  const e = (x) => x.toExponential(1), letter = (i) => String.fromCharCode(65 + i);
  const columns = [["CPU", cpu], ...runs.map((run, i) => [letter(i), run])];
  const tables = [[`against NumPy's (E16: NumPy's with its cache in float16)`, exact, [["E16", half]]],
    ["against NumPy's with its cache in float16", half, []]];
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
  console.log(`- CPU: forward.js on the CPU: ${e(all(cpu))}, ${(all(cpu) / e16).toFixed(1)} E16, logits ${e(cpuLogits)} from NumPy's, most likely ${best}`);
  runs.forEach((run, i) => {
    const worst = Math.max(all(run), run.again ? all(run.again) : 0), token = argmax(floats(run.logits));
    console.log(`- ${letter(i)}: ${run.form ?? "no form"}, ${run.attention ?? "no attention"}: ${e(all(run))} ` +
      `(all at once ${run.again ? e(all(run.again)) : "-"}), line ${e(run.line)}, ${(worst / run.line).toFixed(2)} of the line, ` +
      `${(all(run) / e16).toFixed(1)} E16, ${e(Math.max(...byLayer(run, half)))} from NumPy's with its cache in float16, ` +
      `logits ${(logitsError(run.logits) / cpuLogits).toFixed(2)} of the CPU's, most likely ${token}${token === best ? " as the CPU's" : ""}`);
  });
  const seconds = { "NumPy (and with its cache in float16)": c.numpySeconds, "the plan": c.planSeconds, CPU: cpu.seconds };
  runs.forEach((run, i) => { seconds[`${letter(i)} (its GPU ready in ${run.readySeconds.toFixed(1)})`] = run.seconds; });
  console.log(`\nseconds of ${c.id}: ${Object.entries(seconds).map(([step, s]) => `${step} ${s.toFixed(1)}`).join(", ")}\n`);
}
