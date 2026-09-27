// The sections of /benchmark/ (T134) that need neither Pyodide nor a GPU, in a module worker of their own: the page
// starts one for a section and ends it after, so that no WebAssembly memory of one section outlives it (T96: Chromium
// refused a page its third memory; a phone has little room for the next section's).
//
//   { step: "device" }                          what this browser can do: SIMD, relaxed SIMD, 64-bit memories, shared
//                                               memory, WebGPU and OPFS in a worker, the storage the browser grants
//   { step: "cpu", threads }                    the forward pass of forward.js, the one the model page runs, on a
//                                               made-up int8 model of Llama 3.2 1B's width (random weights, two
//                                               layers), one token at a time at each count of software threads, and
//                                               a prompt's tokens 16 at a time (T108), as the GPU section does them;
//                                               and the CPU's ceilings beside them (T163)
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
  // T163: the ceilings' loops on the same memory (no memory more: T96); their failure leaves the forward pass alone
  let ceilings = null, ceilingsFailed;
  try {
    ceilings = await ceilingsOf(memory, base, model.size, shared);
  } catch (error) {
    ceilingsFailed = `${error?.name ?? "Error"}: ${error?.message ?? error}`;
  }
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
      // T163: the ceilings right after the row they are held against (a phone that warms up slows both alike): reading
      // at the same count, and the compute loops after the row of one thread
      if (ceilings) ceilings.read.push({ threads, ...(await attempt(() => ceilings.reading(threads))) });
      if (ceilings && threads === 1) {
        const none = { none: "no relaxed SIMD in this browser" };
        ceilings.dot = ceilings.relaxed ? await attempt(ceilings.dotting) : none;
        ceilings.dotRegisters = ceilings.relaxed ? await attempt(ceilings.dottingRegisters) : none;
        ceilings.fma = await attempt(ceilings.fmaing);
      }
    }
  } finally {
    engine.release();
    ceilings?.stop();
  }
  const { read, dot, dotRegisters, fma } = ceilings ?? {};
  return { backend: engine.backend, shared, megabytes: model.size / 1e6, tokenMegabytes: tokenBytes(model, kernels) / 1e6,
           layerWeights: layerWeights(model), rows, ceilings: ceilings ? { read, dot, dotRegisters, fma } : { error: ceilingsFailed } };
}

// what a token reads: the checkpoint, and with relaxed SIMD matmul_q8r's corrections too (a float32 for each group of
// every int8 matrix: 23.4 MB here). The GB/s column stays the checkpoint's (T157 turns it into other models' speeds);
// its share of reading alone counts these bytes (T163's review). Safari's matmul_q8 reads no corrections.
const tokenBytes = ({ plan, size }, kernels) => size + (kernels.relaxed ? Object.values(plan.tensors)
  .filter((t) => t.kind === "int8").reduce((sum, t) => sum + (t.shape.reduce((a, b) => a * b, 1) / t.group) * 4, 0) : 0);

// the weights of the layers' matrices: a multiply-add each for every token of a prompt (no classifier there)
const layerWeights = ({ plan }) => ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
  .reduce((sum, name) => sum + plan.tensors[name].shape.reduce((a, b) => a * b, 1), 0);

// ---- T163: the CPU's ceilings, loops of one kind of instruction (kernels/ceilings.ts, kernels/ceilings_relaxed.ts,
// the forms of T158's audit): reading alone (GB/s) at each count of software threads, over the made-up model's weights
// (the bytes a token reads); relaxed_dot with its two loads from L1 and f32 multiply + add on registers (G MAC/s), one
// thread. Timed as the GPU section's ceilings are (T168): passes doubled until one run takes PAIR_MS, then PAIRS runs of
// n and of 2n passes, the median of the differences; a ratio of 2n to n past STEADY takes the pairs once more, then is
// unsteady. The watch on a loop the compiler cut short: its checksum of n passes must be the one computed here, it
// must take PAIR_MS by MOST_PASSES, and 2n passes must take at least SHORT times n (a loop whose passes were dropped
// takes the same time for any count: T163's review). cutShort: the section is then wrong, as for logits not finite.
const PAIR_MS = 20, PAIRS = 5, STEADY = [1.6, 2.2], SHORT = 1.3, MOST_PASSES = 2 ** 30;
const cutShort = (message) => Object.assign(new Error(message), { cutShort: true });
const middle = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
// run(n): the ms of n passes; right(n): whether the loop's checksum of n passes is the one computed here
function paired(run, right) {
  run(1);
  let n = 1, ms;
  while ((ms = run(n)) < PAIR_MS && n < MOST_PASSES) n *= 2;
  if (ms < PAIR_MS) throw cutShort(`${n} passes took ${ms.toFixed(3)} ms: the loop was compiled away`);
  if (!right(n)) throw cutShort(`the checksum of ${n} passes is not the loop's work: the compiler cut it short`);
  for (let tries = 0; ; tries++) {
    const differences = [], ratios = [];
    for (let i = 0; i < PAIRS; i++) {
      const once = run(n), twice = run(2 * n);
      differences.push(twice - once);
      ratios.push(twice / once);
    }
    const ratio = middle(ratios), took = middle(differences), steady = took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1];
    if (steady) return { ms: took, passes: n, ratio };
    if (tries) {
      if (!(took > 0) || ratio < SHORT) throw cutShort(`${2 * n} passes took ${ratio.toFixed(2)} times as long as ${n}: the loop does not do its passes`);
      return { ms: took, passes: n, ratio, unsteady: true };
    }
  }
}
const attempt = async (measure) => {
  try {
    return await measure();
  } catch (error) {
    return { error: `${error?.name ?? "Error"}: ${error?.message ?? error}`, ...(error?.cutShort ? { cutShort: true } : {}) };
  }
};
const timed = (loop) => (n) => {
  const began = performance.now();
  loop(n);
  return performance.now() - began;
};
const rate = (work, r) => ({ rate: (work * r.passes) / (r.ms / 1000) / 1e9, passes: r.passes, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) });
// the checksum of the int loops: the lane j of their sum is n times one pass's (lanes[j]), wrapped to 32 bits
const xorOf = (lanes, n) => lanes.reduce((x, lane) => x ^ Math.imul(n, lane), 0);

// the control words of the reading threads, on a SharedArrayBuffer of their own (not a WebAssembly memory: T96): as
// forward.js cuts a matrix's rows (T93), the weights are read a CHUNK at a time, each thread taking the next one from
// NEXT until TOTAL (passes × chunks), so that a slow core (a little one of a big.LITTLE phone) takes fewer instead of
// holding the others back (T163's review: equal parts read at the slowest thread's speed)
const GO = 0, DONE = 1, COUNT = 2, TOTAL = 3, STOP = 4, NEXT = 5, CHUNK = 1 << 20, STARTING_MS = 30_000;

// The loops on the memory of the made-up model, its weights at base (size bytes): nothing new is reserved (T96).
// dotting() makes 4 KB of them 7-bit for its loop (see ceilings_relaxed.ts) and puts them back after.
async function ceilingsOf(memory, base, size, shared) {
  const kind = shared ? "shared" : "plain";
  const [plain, relaxed] = await Promise.all([`ceilings_${kind}.wasm`, `ceilings_relaxed_${kind}.wasm`].map((name) =>
    fetch(at(name)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
  if (!plain) throw new Error("the ceilings' loops could not be fetched");
  const module = new WebAssembly.Module(plain);
  const loops = new WebAssembly.Instance(module, { env: { memory } }).exports;
  let dot = null, dotRegisters = null;
  try {
    if (relaxed) ({ dot, dotRegisters } = new WebAssembly.Instance(new WebAssembly.Module(relaxed), { env: { memory } }).exports);
  } catch {
    // no relaxed SIMD here (Safari): the row says so
  }
  const chunks = Math.floor(size / CHUNK), bytes = chunks * CHUNK;
  const control = shared ? new Int32Array(new SharedArrayBuffer(8 * 4)) : null;
  const readers = [];
  let unstarted;  // a reading thread that did not start: no count past it can be read
  const started = async (count) => {
    if (unstarted && count > readers.length) throw unstarted;
    while (readers.length < count - 1) {
      const worker = new Worker(at("benchmark/reader.js"), { type: "module" });
      readers.push(worker);  // stop() ends it even where it never answers
      let timer;
      await Promise.race([new Promise((resolve, reject) => {
        worker.onmessage = resolve;
        worker.onerror = (event) => reject(new Error(event.message ?? "a reading thread did not start"));
        worker.postMessage({ module, memory, control: control.buffer, base, chunk: CHUNK, chunks, share: readers.length });
      }), new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`a reading thread did not start in ${STARTING_MS / 1000} seconds`)), STARTING_MS);
      })]).catch((error) => {
        unstarted = error;
        throw error;
      }).finally(() => clearTimeout(timer));
    }
  };
  // one pass's sums of the int32 lanes of the weights read, for the checksum of the reading loop
  const lanes = [0, 0, 0, 0];
  const ints = new Int32Array(memory.buffer, base, bytes / 4);
  for (let i = 0; i < ints.length; i++) lanes[i & 3] = (lanes[i & 3] + ints[i]) | 0;
  return {
    relaxed: Boolean(dot),
    read: [],
    // reading alone with count threads, CHUNKs taken in turn (this worker and the reading threads); the checksum is of
    // this worker's loop over all of them (the reading threads run the same loop)
    async reading(count) {
      await started(count);
      const r = paired((n) => {
        const began = performance.now();
        if (count === 1) {
          loops.read(base, bytes, n);
        } else {
          Atomics.store(control, DONE, 0);
          Atomics.store(control, NEXT, 0);
          Atomics.store(control, TOTAL, n * chunks);
          Atomics.store(control, COUNT, count);
          Atomics.add(control, GO, 1);
          Atomics.notify(control, GO);
          for (let k; (k = Atomics.add(control, NEXT, 1)) < n * chunks;) loops.read(base + (k % chunks) * CHUNK, CHUNK, 1);
          for (let done; (done = Atomics.load(control, DONE)) < count - 1;) Atomics.wait(control, DONE, done);
        }
        return performance.now() - began;
      }, (n) => loops.read(base, bytes, n) === xorOf(lanes, n));
      const { rate: GBps, ...rest } = rate(bytes, r);
      return { GBps, ...rest };
    },
    dotting() {
      const w = new Int8Array(memory.buffer, base, 4096), x = new Uint8Array(memory.buffer, base + 4096, 4096);
      const kept = x.slice();
      try {
        for (let i = 0; i < x.length; i++) x[i] &= 0x7f;
        const sums = [0, 0, 0, 0];
        for (let i = 0; i < w.length; i++) sums[(i >> 2) & 3] += w[i] * x[i];
        const { rate: GMACs, ...rest } = rate(4096, paired(timed((n) => dot(base, n)), (n) => dot(base, n) === xorOf(sums, n)));
        return { GMACs, ...rest };
      } finally {
        x.set(kept);
      }
    },
    dottingRegisters() {
      const loop = (n) => dotRegisters(n, 3, 5, 1);
      const { rate: GMACs, ...rest } = rate(128, paired(timed(loop), (n) => loop(n) === dotRegistersSum(n)));
      return { GMACs, ...rest };
    },
    fmaing() {
      const loop = (n) => loops.fma(n, 1, 0.9999, 1e-7);
      const { rate: GMACs, ...rest } = rate(32, paired(timed(loop), (n) => loop(n) === fmaSum(n)));
      return { GMACs, ...rest };
    },
    stop() {
      if (control) {
        Atomics.store(control, STOP, 1);
        Atomics.add(control, GO, 1);
        Atomics.notify(control, GO);
      }
      readers.forEach((worker) => worker.terminate());
    },
  };
}

// ceilings.ts's fma(n, 1, 0.9999, 1e-7) in float32, one lane (the four are alike) of the 8 accumulators, which start at
// 0 to 7, rounded as WebAssembly rounds each step
function fmaSum(n) {
  const f = Math.fround, d = f(1e-7), w = [1, 1.0009765625, 1.001953125, 1.0029296875];
  let x = f(0.9999), y = f(x + 0.25);
  const a = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      a[k] = f(a[k] + f(w[k] * x));
      a[4 + k] = f(a[4 + k] + f(w[k] * y));
    }
    x = f(x - d);
    y = f(y - d);
  }
  const lane = f(f(f(a[0] + a[1]) + f(a[2] + a[3])) + f(f(a[4] + a[5]) + f(a[6] + a[7])));  // as the loop sums them
  return f(f(f(lane + lane) + lane) + lane);
}

// ceilings_relaxed.ts's dotRegisters(n, 3, 5, 1): a lane of each dot adds 4 × a × 5, a running 3, 4, 5, … as int8
// (wrapping at 127); the 8 accumulators start at lanes 1000k + j, and the checksum sums the 4 lanes of all 8
function dotRegistersSum(n) {
  const int8 = (v) => ((v + 128) % 256 + 256) % 256 - 128;
  // the sum of a over n passes: whole turns of 256 values add -128 each
  let sum = Math.floor(n / 256) * -128;
  for (let i = 0; i < n % 256; i++) sum += int8(3 + i);
  const start = 4 * 28000 + 8 * 6;  // the accumulators' lanes: 1000k + j, k 0 to 7, j 0 to 3
  return (start + Math.imul(32 * 20, sum | 0)) | 0;
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
