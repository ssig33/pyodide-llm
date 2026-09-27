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
// of this directory has it). The others are the models of this directory (make models kernels).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
// T147: --forms <part,part>: only the matrices' shaders whose names hold one of these (all of them by default)
const only = option("--forms", "");
const webgpu = option("--webgpu", "");
const ids = args.length ? args : ["synthetic", "stories15M", "tiny-lm", "llm-jp-3-150m"];
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
const GPU_LINE = 8e-3, HALF_LINE = 1.5e-2, PACKED_LINE = 0.75;
// The logits of the prompt's last token, the largest difference from NumPy's over the largest of NumPy's: the CPU's
// own run 1.6e-2 to 5.6e-2, the one on the GPU's keys and values 0.75 to 1.04 times that (closer: its keys and values
// are NumPy's but for the float16); broken on purpose 0.27 to 1.13, 4.8 times the CPU's and more. The line: no more
// than 1.5 times the CPU's, and the same most likely token.
const LOGITS_LINE = 1.5;

// ---- Node: the plans and NumPy's answers
const { pyodide: py } = await pyodideWithEngine();
py.runPython(`
import base64, struct, numpy as np, llama2_numpy, llama2_convert
from llama2_numpy import Llama

def synthetic(dim=64, hidden=128, layers=2, heads=4, kv_heads=2, vocab=320, seq_len=256, seed=0):
    """A made-up int8 checkpoint and its tokenizer.bin, as quantize.py writes one: grouped-query attention"""
    rng = np.random.default_rng(seed)
    header = (dim, hidden, layers, heads, kv_heads, vocab, seq_len)
    out = [struct.pack("<7i", *header)]
    for shape, is_matrix in llama2_convert.layout(*header):
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

def answer(data, vocabulary, text, count, options):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens"""
    numpy = Llama(data, vocabulary, **options)
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
`);

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
  if (id === "synthetic") {
    py.runPython(`data, vocabulary = synthetic()`);
    options = { dtype: "int8" };
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
  const reference = py.runPython(`answer(data, vocabulary, TEXT, ${COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  // the plan forward.js gets from Python, recorded: Llama(external=) with a start() that keeps it
  let plan;
  const bytes = py.runPython("data").toJs();
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
  const file = path.join(directory, `${id}.bin`);
  fs.writeFileSync(file, bytes);
  cases.push({ id, plan, checkpoint: `/case/${id}.bin`, file, reference });
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
    const { memory, base } = weightsMemory(size, { shared: true, after: footprint(c.reference.header, size, { dtype: "int8", halfKV: true, gpu: true, kvStart: plan.kv_start }) });
    new Uint8Array(memory.buffer, base, size).set(checkpoint);
    // T148: SwiftShader and lavapipe are fallback adapters, which the page refuses: the tests take them (fallback),
    // and give the GPU every block it can take (always: a fallback adapter is far slower than the CPU)
    const TESTS = { fallback: true, always: true };
    const run = async (gpu, gpuForce, gpuRemembered) => {
      const engine = createForward({ memory, base, size, kernels, plan, gpu, gpuForce: { ...TESTS, ...gpuForce }, gpuRemembered });
      const note = gpu ? await engine.gpu : undefined;
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
      return out;
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
    for (const form of [undefined, ...forms]) gpu.push(await run(openGpu, { matrices: form }));
    // the attention without subgroups or f16 (the lanes of the workgroup stand for a subgroup), where the adapter
    // has them and so chose the other
    if (forms.length) gpu.push(await run(openGpu, { matrices: forms[0], attention: "llama.cpp flash attention tiles" }));
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
  if (pathname === "/cases.json") return send(types[".json"], JSON.stringify(cases.map(({ file, ...c }) => c)));
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
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (dim / heads) * kvHeads;
  const kv = (run) => Math.max(worstRow(floats(run.keys), floats(ref.keys), kvDim), worstRow(floats(run.values), floats(ref.values), kvDim));
  const cpuKv = kv(cpu);
  const want = floats(ref.logits), largest = want.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  const logitsError = (b64) => floats(b64).reduce((m, x, i) => Math.max(m, Math.abs(x - want[i])), 0) / largest;
  const cpuLogits = logitsError(cpu.logits), best = argmax(floats(cpu.logits));
  console.log(`${id} (${layers} layers, ${heads} heads, ${kvHeads} of keys and values, ${n} tokens): the CPU's keys and values ` +
    `${cpuKv.toExponential(2)} from NumPy's, logits ${cpuLogits.toExponential(2)}, most likely ${best} ` +
    `${argmax(want) === best ? "as NumPy's" : `(NumPy's ${argmax(want)})`}, the prompt ${(cpu.promptMs / n).toFixed(2)} ms a token`);
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
    const line = /DP4A/.test(gpu.form ?? "") ? PACKED_LINE * cpuKv : /f16/.test(gpu.form ?? "") ? HALF_LINE : GPU_LINE;
    const gpuKv = kv(gpu), againKv = gpu.again ? kv(gpu.again) : NaN;
    if (!(gpuKv <= line) || !(againKv <= line)) {
      failures.push(`the keys and values of the GPU are ${gpuKv.toExponential(2)} and ${againKv.toExponential(2)} from NumPy's (line ${line.toExponential(2)})`);
    }
    const gpuLogits = logitsError(gpu.logits), againLogits = logitsError(gpu.again.logits);
    if (!(gpuLogits <= LOGITS_LINE * cpuLogits) || !(againLogits <= LOGITS_LINE * cpuLogits)) {
      failures.push(`the logits on the GPU's keys and values are ${gpuLogits.toExponential(2)} and ${againLogits.toExponential(2)} from NumPy's, the CPU's ${cpuLogits.toExponential(2)}`);
    }
    // the most likely token: the CPU's, or one whose logit NumPy puts no farther below its largest than the CPU's run
    // is from NumPy's (a near tie, T147: stories15M at 149 tokens, where the GPU's logits were nearer NumPy's)
    const near = (b64) => { const i = argmax(floats(b64)); return i === best || want[argmax(want)] - want[i] <= LOGITS_LINE * cpuLogits * largest; };
    if (!near(gpu.logits) || !near(gpu.again.logits)) failures.push("another most likely token than on the CPU, and not a near tie");
    console.log(`  ${gpu.form ?? "no form"}, ${gpu.attention ?? "no attention"}: keys and values ${gpuKv.toExponential(2)} (all at once ${againKv.toExponential(2)}), ` +
      `logits ${gpuLogits.toExponential(2)} (${againLogits.toExponential(2)}), the prompt ${(gpu.promptMs / n).toFixed(2)} ms a token (${gpu.note})` +
      (failures.length ? ` — FAILED\n    - ${failures.join("\n    - ")}` : ""));
    failed ||= failures.length > 0;
  }
}
process.exit(failed ? 1 : 0);
