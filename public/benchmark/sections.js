// The sections of /benchmark/ (T134) that need neither Pyodide nor a GPU, in a module worker of their own: the page
// starts one for a section and ends it after, so that no WebAssembly memory of one section outlives it (T96: Chromium
// refused a page its third memory; a phone has little room for the next section's).
//
//   { step: "device" }                          what this browser can do: SIMD, relaxed SIMD, 64-bit memories, shared
//                                               memory, WebGPU and OPFS in a worker, the storage the browser grants
//   { step: "cpu", threads }                    the forward pass of forward.js, the one the model page runs, on a
//                                               made-up int8 model of Llama 3.2 1B's width (random weights, two
//                                               layers), one token at a time at each count of software threads, and
//                                               a prompt's tokens 16 at a time (T108), as the GPU section does them
//   { step: "line", site, hf, rates, seconds }  the line: a part of the site's model and a range of a model on
//                                               huggingface.co, how long until the first byte and how fast; then
//                                               huggingface.co read no faster than each of rates (MB/s)
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T109). While a step runs, it tells the page each stage it starts
// ({stage, at, of}, T177): the page shows it, and it answers none of the page's questions.
const search = new URL(import.meta.url).search;  // ?v=<commit>: every file of the same deployment
const at = (name) => new URL(`../${name}${search}`, import.meta.url);
const stage = (name, at, of) => postMessage({ stage: name, at, of });

self.onmessage = async ({ data }) => {
  try {
    const result = data.step === "device" ? await device() : data.step === "cpu" ? await cpu(data.threads)
      : await line(data);
    postMessage({ step: data.step, result });
  } catch (error) {
    postMessage({ step: data.step, error: `${error?.name ?? "Error"}: ${error?.message ?? error}` });
  }
};

// ---- the kernels, as the model's worker compiles them (public/worker.js): the plain build, and the shared one where
// the page is cross-origin isolated. null where this browser cannot compile them (no WebAssembly SIMD)
async function kernelsOf(forward, kind) {
  const fetched = await Promise.all([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`].map((name) =>
    fetch(at(name)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
  if (!fetched[0]) return null;
  try {
    return forward.compileKernels(fetched[0], fetched[1]);
  } catch {
    return null;
  }
}

async function device() {
  const forward = await import(at("forward.js"));
  const plain = await kernelsOf(forward, "plain");
  const found = {
    crossOriginIsolated: self.crossOriginIsolated,
    sharedMemory: typeof SharedArrayBuffer !== "undefined",
    simd: Boolean(plain),
    relaxedSimd: Boolean(plain?.relaxed),
    memory64: forward.memory64(),
    webgpu: Boolean(self.navigator?.gpu),
    opfs: Boolean(navigator.storage?.getDirectory),
    syncHandle: typeof FileSystemFileHandle !== "undefined" && typeof FileSystemFileHandle.prototype.createSyncAccessHandle === "function",
    waitAsync: typeof Atomics.waitAsync === "function",
    cores: navigator.hardwareConcurrency ?? null,
    memoryGB: navigator.deviceMemory ?? null,
  };
  try {
    const { usage, quota } = await navigator.storage.estimate();
    Object.assign(found, { usage, quota });
  } catch {
    // no navigator.storage here: said by opfs above
  }
  return found;
}

// ---- the CPU. A made-up model of Llama 3.2 1B's width (dim 2048, FFN 8192, 32 heads of 64, 8 of them for keys and
// values) with two layers and a vocabulary of 32000: 211 MB of int8 weights and their scales, which is what a token
// reads, whatever the numbers in them. The forward pass is forward.js's own (createForward, the kernels of
// simdkernel_*.wasm, the software threads of helper.js): what differs from the model page is only that the plan of the
// tensors comes from here instead of from Python.
const SHAPE = { dim: 2048, hidden: 8192, layers: 2, heads: 32, kvHeads: 8, vocab: 32000, seqLen: 64 };
const GROUP = 32;
const TOKENS = 12, WARM = 3;
// T108: a prompt's tokens go through the layers BATCH (16) at a time, without logits
const PROMPT = Array.from({ length: 16 }, (_, i) => 1 + i), PROMPT_RUNS = 5;

function madeUpModel() {
  const { dim, hidden, layers, heads, kvHeads, vocab, seqLen } = SHAPE;
  const headSize = dim / heads, kvDim = kvHeads * headSize;
  const tensors = {};
  let offset = 28;  // after the legacy header, as in a checkpoint file
  const int8 = (name, shape) => {
    const count = shape.reduce((a, b) => a * b, 1);
    tensors[name] = { kind: "int8", offset, shape, group: GROUP, scales: offset + count };
    offset += count + (count / GROUP) * 4;
  };
  const f32 = (name, shape) => {
    tensors[name] = { kind: "f32", offset, shape };
    offset += shape.reduce((a, b) => a * b, 1) * 4;
  };
  int8("token_embedding_table", [vocab, dim]);
  f32("rms_att_weight", [layers, dim]);
  int8("wq", [layers, dim, dim]);
  int8("wk", [layers, kvDim, dim]);
  int8("wv", [layers, kvDim, dim]);
  int8("wo", [layers, dim, dim]);
  f32("rms_ffn_weight", [layers, dim]);
  int8("w1", [layers, hidden, dim]);
  int8("w2", [layers, dim, hidden]);
  int8("w3", [layers, hidden, dim]);
  f32("rms_final_weight", [dim]);
  // RoPE's tables, which Python computes for an int8 file (derived: bytes of float32)
  const cos = new Float32Array(seqLen * headSize / 2), sin = new Float32Array(seqLen * headSize / 2);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < headSize / 2; i++) {
      const angle = pos / 10000 ** ((2 * i) / headSize);
      cos[pos * headSize / 2 + i] = Math.cos(angle);
      sin[pos * headSize / 2 + i] = Math.sin(angle);
    }
  }
  const plan = { arch: "llama", dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads,
    head_size: headSize, vocab_size: vocab, seq_len: seqLen, rotary: headSize, parallel_residual: false, kv_start: seqLen,
    rms_norm_eps: 1e-5, shared_classifier: true, int8: true, relaxed: true, tensors, outliers: [], half_kv: true,
    derived: { freq_cis_real: new Uint8Array(cos.buffer), freq_cis_imag: new Uint8Array(sin.buffer) } };
  return { plan, size: offset, header: [dim, hidden, layers, heads, kvHeads, vocab, seqLen] };
}

// random bytes for the int8 values, a scale that keeps the activations near 1, and norms of 1
function fillWeights(memory, base, { plan }) {
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const noise = new Uint8Array(1 << 20);
  for (let at = 0; at < noise.length; at += 65536) crypto.getRandomValues(noise.subarray(at, at + 65536));
  for (const t of Object.values(plan.tensors)) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (t.kind === "int8") {
      for (let at = 0; at < count; at += noise.length) U.set(noise.subarray(0, Math.min(noise.length, count - at)), base + t.offset + at);
      F.fill(1 / (64 * Math.sqrt(t.shape.at(-1))), (base + t.scales) / 4, (base + t.scales) / 4 + count / GROUP);
    } else {
      F.fill(1, (base + t.offset) / 4, (base + t.offset) / 4 + count);
    }
  }
}

// a software thread of forward.js, as the model's worker starts one (public/worker.js's spawnThread)
const spawn = (data) => new Promise((resolve, reject) => {
  const worker = new Worker(at("helper.js"), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => reject(new Error(event.message ?? "a software thread did not start"));
  worker.postMessage(data);
});

async function cpu(counts = [1, 2, 4]) {
  const forward = await import(at("forward.js"));
  const shared = self.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined";
  const kernels = shared ? await kernelsOf(forward, "shared") : await kernelsOf(forward, "plain");
  if (!kernels) return { none: "no WebAssembly SIMD in this browser: the model page runs on NumPy here" };
  // T177: making the model, then each count of software threads (one only without shared memory)
  const stages = 1 + counts.filter((asked) => shared || asked === 1).length;
  let done = 0;
  stage("making the model", ++done, stages);
  const model = madeUpModel();
  const after = forward.footprint(model.header, model.size, { dtype: "int8", relaxed: Boolean(kernels.relaxed), halfKV: shared });
  const { memory, base } = forward.weightsMemory(model.size, { shared, after });
  fillWeights(memory, base, model);
  const engine = forward.createForward({ memory, base, size: model.size, kernels, plan: model.plan, spawn: shared ? spawn : undefined });
  const rows = [];
  try {
    for (const asked of counts) {
      // without shared memory there are no software threads: one row, of one thread
      if (!shared && asked > 1) continue;
      stage(`${asked} software thread${asked > 1 ? "s" : ""}`, ++done, stages);
      const threads = await engine.setThreads(asked);
      if (threads !== asked) {
        rows.push({ asked, threads, none: "the browser did not start that many software threads" });
        continue;
      }
      for (let i = 0; i < WARM; i++) engine.forward(1, i, true);
      const times = [];
      for (let i = 0; i < TOKENS; i++) {
        const began = performance.now();
        engine.forward(1 + i, (WARM + i) % SHAPE.seqLen, true);
        times.push(performance.now() - began);
      }
      times.sort((a, b) => a - b);
      const ms = times[times.length >> 1];
      const logits = engine.logits();
      // a prompt of 16 tokens at positions 0 to 15, over and over (the keys and values of those positions are rewritten)
      engine.forwardMany(PROMPT, 0);
      const blocks = [];
      for (let i = 0; i < PROMPT_RUNS; i++) {
        const began = performance.now();
        engine.forwardMany(PROMPT, 0);
        blocks.push(performance.now() - began);
      }
      blocks.sort((a, b) => a - b);
      rows.push({ asked, threads, msPerToken: ms, GBps: model.size / (ms / 1000) / 1e9,
                  promptMsPerToken: blocks[blocks.length >> 1] / PROMPT.length, finite: logits.every(Number.isFinite) });
    }
  } finally {
    engine.release();
  }
  return { backend: engine.backend, shared, megabytes: model.size / 1e6, rows };
}

// ---- the line. A fetch read to its end, or read no faster than rate (MB/s): each piece is taken only when the bytes
// so far are due at that rate. Chromium and Firefox then stop taking from the connection, and the server's sending
// slows with it (the connection's flow control): what is paced is this page's download, not the whole line of the
// device. WebKit does not stop taking (T134's review): there only this page's reading is paced.
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
// A fetch that brings nothing for STALL_MS is given up (as the model's worker gives up on Pyodide, T118), and while
// bytes come the page hears of it once a second: its own deadline is for a section that says nothing at all.
const STALL_MS = 30_000;
// range: false for a part of the site's model, which GitHub Pages sends gzipped and answers a range of with a piece of
// the gzip stream (AGENTS.md): the whole part, read to its end
async function measure(url, { bytes, rate, range = true }) {
  const began = performance.now();
  const abort = new AbortController();
  let stalled = false, told = began, timer;
  const alive = () => {
    clearTimeout(timer);
    timer = setTimeout(() => { stalled = true; abort.abort(); }, STALL_MS);
    if (performance.now() - told > 1000) {
      told = performance.now();
      postMessage({ alive: true });
    }
  };
  alive();
  let res, headers, got = 0, first;
  try {
    res = await fetch(url, { headers: range ? { Range: `bytes=0-${bytes - 1}` } : {}, cache: "no-store", signal: abort.signal });
    if (!res.ok) throw new Error(`${url.split("?")[0]}: ${res.status}`);
    headers = performance.now() - began;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      alive();
      if (done) break;
      first ??= performance.now();
      got += value.length;
      // the last piece waits too: WebKit takes the whole range whether it is read or not and hands it over in large
      // pieces, so there only the reading is paced, and a last piece taken at once made the rate look many times higher
      if (rate) {
        const due = first + (Math.min(got, bytes) / (rate * 1e6)) * 1000;
        if (due > performance.now()) await sleep(due - performance.now());
      }
      if (got >= bytes) {
        // a server that sends the whole file for a range (WebKit and Hugging Face's CDN, T112) is read no further
        reader.cancel().catch(() => {});
        break;
      }
    }
  } catch (error) {
    throw stalled ? new Error("no bytes came for 30 s") : error;
  } finally {
    clearTimeout(timer);
  }
  const ended = performance.now();
  return { status: res.status, headersMs: headers, firstByteMs: (first ?? ended) - began, bytes: Math.min(got, bytes),
           MBps: Math.min(got, bytes) / 1e6 / ((ended - (first ?? began)) / 1000) };
}

async function line({ site, hf, rates = [], seconds = 4 }) {
  const out = { site: null, hf: null, paced: [] };
  const stages = 2 + rates.length;
  stage("this site: a part of the model", 1, stages);
  try {
    out.site = await measure(site.url, site);
  } catch (error) {
    out.site = { error: String(error.message ?? error) };
  }
  stage("huggingface.co: a range of a model", 2, stages);
  try {
    out.hf = await measure(hf.url, { bytes: hf.bytes });
  } catch (error) {
    out.hf = { error: String(error.message ?? error) };
    return out;
  }
  for (const [i, rate] of rates.entries()) {
    stage(`huggingface.co no faster than ${rate} MB/s`, 3 + i, stages);
    // a line slower than the rate asked for cannot be paced to it: said as such, not measured for ever
    if (out.hf.MBps < rate) {
      out.paced.push({ rate, slower: true });
      continue;
    }
    try {
      out.paced.push({ rate, ...(await measure(hf.url, { bytes: Math.round(rate * 1e6 * seconds), rate })) });
    } catch (error) {
      out.paced.push({ rate, error: String(error.message ?? error) });
    }
  }
  return out;
}
