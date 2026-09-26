// gpu.js (T135, T147): the model page's GPU, in a worker of its own. forward.js (in the model's worker) makes it where
// the page asked for it (?gpu=on) and the model is one it takes, and hands it the blocks of a prompt (up to plan.batch
// tokens) one at a time, waiting in Atomics.wait for the answer: the model's worker cannot wait for a promise while
// Python calls forwardMany(), and this one can (T94's design; T134's bridge measured 12 to 95 µs a round trip).
//
//   { type: "start", memory, plan }  the layers of the model onto the GPU, read from the shared memory of forward.js
//                                    (plan: what forward.js says of them, addresses in that memory). Says
//                                    { type: "progress" } every few seconds meanwhile (a shader may take a minute to
//                                    compile where the CPU stands in for the GPU: T147), then { type: "ready",
//                                    adapter, bytes, seconds, form, attention, forms } or { type: "unusable", reason }
//   { type: "prompt", serial, count, pos }
//                                    the rows of count tokens at positions pos, pos + 1, ... (embedded by forward.js,
//                                    at plan.rows) through every layer, and their keys and values of every layer into
//                                    plan.staging as float16. The answer is in the control area: words.failed, then
//                                    words.done = serial, while words.beat counts up meanwhile; a failure also says
//                                    { type: "failed", reason }. Nothing is written where words.wanted is no longer
//                                    serial: forward.js gave the request up (T147: a worker that went on late must not
//                                    write into a memory that may hold the next model by then)
//   { type: "stop" }                 every buffer and the device let go, and the worker ends
//
// A block goes as one submission: per layer the RMSNorm, the matrices of q, k and v (each weight read once for the
// block's tokens, by the tiled shader chosen on this device: see chooseMatrices), RoPE with the keys and values into the
// GPU's own cache (float16), the attention (llama.cpp's flash attention with tiles: every token sees the positions up to
// its own), the output matrix added to the residual, the RMSNorm, the gate and the up matrices, SwiGLU, the down matrix
// added. The last layer stops at its keys and values: nothing of a prompt's token after them is used. Then the block's
// keys and values of every layer are copied out and read back. The activations are float32 (quantized to 8 bits first
// for the packed shaders, as the CPU's matmul_q8 takes them), the weights int8 widened or multiplied as int8.
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T109).

const shaders = import(new URL(`shaders.js${new URL(import.meta.url).search}`, import.meta.url));

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;
// a tiled shader is timed on the model's first gate matrix by a whole block, once to warm up and then TIMES times (the
// median); on a fallback adapter (the CPU in the GPU's place: its times are no GPU's) once
const TIMES = 3;

let model = null;  // what is on the GPU for the model: the device, the plan, the buffers, the pipelines, the cache
let starting = false, stopping = false, lost = null;

onmessage = ({ data }) => {
  if (data.type === "start") start(data.memory, data.plan);
  else if (data.type === "prompt") prompt(data);
  else if (data.type === "stop") stop();
};

// how often a worker that puts a model on the GPU says it is still at it (forward.js gives one up that says nothing
// for 30 s: a worker the browser ended. A device that hangs is lost, by the browser's watchdog, and its promises fail)
const ALIVE_MS = 5000;

async function start(memory, plan) {
  const began = performance.now();
  starting = true;
  const alive = setInterval(() => postMessage({ type: "progress" }), ALIVE_MS);
  try {
    const wgsl = await shaders;
    if (!self.navigator?.gpu) return unusable("no WebGPU in a worker here");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (stopping) return end();
    if (!adapter) return unusable("no GPU adapter here");
    // the largest buffer of the layers: one of a layer's matrices
    const largest = Math.max(...Object.values(plan.matrices).map(({ rows, n }) => rows * n));
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    if (largest > limit) return unusable(`a matrix of ${megabytes(largest)} is more than a buffer of this GPU (${megabytes(limit)})`);
    // shader-f16 and subgroups where the adapter has them (a device refuses a feature it lacks), and the adapter's
    // workgroup memory and threads (the tiles and the attention size themselves by them)
    const { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup } = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
      requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup } });
    device.lost.then((info) => { lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    const info = adapter.info ?? {};
    model = { device, memory, plan, wgsl, owned: [], limit, info, fallback: Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter) };
    if (stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    const bytes = await upload(model);
    if (stopping) return end();
    await prepare(model);
    await chooseAttention(model);
    if (stopping) return end();
    await chooseMatrices(model);
    if (stopping) return end();
    bindLayers(model);
    grow(model, Math.min(plan.kvStart, plan.seqLen));
    await device.queue.onSubmittedWorkDone();
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full).message})`);
    if (lost) return unusable(lost);
    postMessage({ type: "ready", adapter: describe(adapter), bytes, seconds: (performance.now() - began) / 1000,
      form: model.form.name, attention: model.attention.name, forms: model.forms });
  } catch (error) {
    unusable(String(error?.message ?? error));
  } finally {
    clearInterval(alive);
    starting = false;
  }
}

const megabytes = (bytes) => `${Math.round(bytes / 2 ** 20)} MiB`;
function describe(adapter) {
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}

function unusable(reason) {
  postMessage({ type: "unusable", reason });
  end();
}
function stop() {
  stopping = true;
  if (!starting) end();  // else start() ends at its next step
}
// every buffer and the device let go (T94: a model changed for another leaves nothing on the GPU), and this worker ends
function end() {
  if (model) {
    model.owned.forEach((buffer) => buffer.destroy());
    model.cache?.owned.forEach((buffer) => buffer.destroy());
    model.device.destroy();
    model = null;
  }
  self.close();
}

// ---- buffers
function buffer(m, bytes, usage = STORAGE, owned = m.owned) {
  const made = m.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  owned.push(made);
  return made;
}
function uniform(m, data, owned = m.owned) {
  const made = buffer(m, data.byteLength, UNIFORM | COPY_DST, owned);
  m.device.queue.writeBuffer(made, 0, data);
  return made;
}
// bytes of the shared memory at address onto the GPU, a CHUNK at a time through a copy
let scratch;
function copyIn(m, target, address, bytes) {
  scratch ??= new Uint8Array(CHUNK);
  for (let done = 0; done < bytes; done += CHUNK) {
    const n = Math.min(CHUNK, bytes - done);
    scratch.set(new Uint8Array(m.memory.buffer, address + done, n));
    m.device.queue.writeBuffer(target, done, scratch, 0, n);
  }
}
async function readBack(m, source, bytes) {
  const target = m.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: MAP_READ | COPY_DST });
  try {
    const encoder = m.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, target, 0, Math.ceil(bytes / 4) * 4);
    m.device.queue.submit([encoder.finish()]);
    await target.mapAsync(MAP_READ);
    return target.getMappedRange().slice(0, bytes);
  } finally {
    target.destroy();
  }
}
// what fn does on the GPU, a validation error of it thrown (a pipeline or a bind group the device refused)
async function validated(m, fn) {
  m.device.pushErrorScope("validation");
  let result, failure;
  try {
    result = await fn();
  } catch (error) {
    failure = error;
  }
  const invalid = await m.device.popErrorScope();
  if (failure) throw failure;
  if (invalid) throw new Error(invalid.message);
  return result;
}
const pipelineOf = (m, code, constants) => m.device.createComputePipelineAsync({ layout: "auto",
  compute: { module: m.device.createShaderModule({ code }), entryPoint: "main", constants } });
const bind = (m, pipeline, buffers) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
function dispatch(pass, pipeline, group, x, y = 1, z = 1) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y, z);
}

// every layer's matrices (values and scales, as the checkpoint holds them) and the weights of its two norms
async function upload(m) {
  const { plan } = m, group = m.wgsl.GROUP;
  let bytes = 0;
  m.matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n }]) => [name, { rows, n, layers: [] }]));
  for (let l = 0; l < plan.layers; l++) {
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      const [valuesAt, scalesAt] = matrix.layers[l], valueBytes = matrix.rows * matrix.n, scaleBytes = (valueBytes / group) * 4;
      const values = buffer(m, valueBytes, STORAGE | COPY_DST), scales = buffer(m, scaleBytes, STORAGE | COPY_DST);
      copyIn(m, values, valuesAt, valueBytes);
      copyIn(m, scales, scalesAt, scaleBytes);
      m.matrices[name].layers.push([values, scales]);
      bytes += valueBytes + scaleBytes;
    }
    // what was written waits in memory until the GPU takes it: let it, before more comes
    await m.device.queue.onSubmittedWorkDone();
    if (stopping) return bytes;
  }
  const normBytes = plan.layers * plan.dim * 4;
  m.norms = {};
  for (const [name, address] of Object.entries(plan.norms)) {
    m.norms[name] = buffer(m, normBytes, STORAGE | COPY_DST);
    copyIn(m, m.norms[name], address, normBytes);
    bytes += normBytes;
  }
  return bytes;
}

// the pipelines of a layer's small steps and the buffers of a block's activations
async function prepare(m) {
  const { plan, wgsl } = m, B = plan.batch;
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  [m.norm, m.rope, m.swiglu, m.quantize] = await Promise.all([wgsl.RMSNORM, wgsl.ROPE, wgsl.SWIGLU, wgsl.QUANTIZE].map((code) => pipelineOf(m, code)));
  // a block's tokens, each array dense: token t's row at t times its width
  m.x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  m.xb = buffer(m, B * Math.max(plan.dim, qDim) * 4, STORAGE | COPY_DST);
  m.q = buffer(m, B * qDim * 4);
  m.k = buffer(m, B * kvDim * 4);
  m.v = buffer(m, B * kvDim * 4);
  m.gate = buffer(m, B * plan.hidden * 4, STORAGE | COPY_DST);
  m.up = buffer(m, B * plan.hidden * 4);
  // the packed shaders' input: the activations of a matrix quantized, 8 bits a value and a float32 scale a group
  m.xq = buffer(m, B * widest);
  m.xs = buffer(m, B * (widest / wgsl.GROUP) * 4);
  m.angles = buffer(m, B * plan.headSize * 4, STORAGE | COPY_DST);
  m.step = buffer(m, 16, UNIFORM | COPY_DST);
  m.readback = buffer(m, 2 * plan.layers * B * kvDim * 2, MAP_READ | COPY_DST);
  // what a request writes into them, from the shared memory
  m.rows = new Float32Array(B * plan.dim);
  m.turns = new Float32Array(B * plan.headSize);
  m.ropeShape = uniform(m, new Uint32Array([plan.heads, plan.kvHeads, plan.headSize, plan.turned]));
  m.swigluGroup = bind(m, m.swiglu, [m.gate, m.up, uniform(m, new Uint32Array([plan.hidden, 0, 0, 0])), m.step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's and the attention's, dim wide) and gate (SwiGLU's)
  const quantizing = (from, n) => ({ group: bind(m, m.quantize, [from, m.xq, m.xs, uniform(m, new Uint32Array([n, n, 0, 0])), m.step]),
    x: Math.ceil(n / wgsl.GROUP / 64) });
  m.quantizeXb = quantizing(m.xb, plan.dim);
  m.quantizeGate = quantizing(m.gate, plan.hidden);
}

// ---- the attention: llama.cpp's flash attention with tiles, with f16 in the workgroup's memory where there is
// shader-f16 and its subgroups where there are (and subgroup_id), else the same without them; each checked first
// against JavaScript (checkAttention), the next one tried where it is wrong
async function chooseAttention(m) {
  const { device, plan, wgsl } = m;
  const half = device.features.has("shader-f16");
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: device.limits.maxComputeInvocationsPerWorkgroup,
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  const tried = [];
  const options = [{ half, subgroups }, { half: false, subgroups: false }].filter((o, i) => i === 0 || o.half !== half || o.subgroups !== subgroups);
  for (const option of options) {
    const shape = wgsl.flashShape({ headSize: plan.headSize, ...option, ...limits });
    const name = `llama.cpp flash attention tiles${option.half ? ", f16" : ""}${option.subgroups ? ", subgroups" : ""}`;
    if (plan.force.attention && plan.force.attention !== name) continue;
    if (shape.none) {
      tried.push(`${name}: ${shape.none}`);
      continue;
    }
    try {
      const pipeline = await validated(m, () => pipelineOf(m, wgsl.flashTile(shape)));
      const wrong = await checkAttention(m, pipeline);
      if (!wrong) {
        m.attention = { name, pipeline, shape };
        return;
      }
      tried.push(`${name}: ${wrong}`);
    } catch (error) {
      tried.push(`${name}: ${error?.message ?? error}`);
    }
  }
  throw new Error(`no attention is right on this GPU (${tried.join("; ") || `none named ${plan.force.attention}`})`);
}

// The attention on made-up numbers against JavaScript's: 9 tokens at positions 61 to 69 (two tiles of 4 and one
// token of a third), 4 heads of q on 2 of keys and values, 70 positions (a KV_TILE of 64 and a part of one) of keys and
// values as float16 of random bits between 2^-3 and 4 in size. Each token's output against its softmax over the
// positions up to its own, no more than LINE of the largest |value| of its head: f16 weights in the workgroup's
// memory are within 2^-10 of theirs (each rounded either way, WGSL leaves the direction to the device) and so the
// output within about 1e-3 of the largest value (the sum of the weights is float32), while a wrong mask, head or
// tile is off by a tenth and more.
const LINE = 4e-3;
async function checkAttention(m, pipeline) {
  const { plan } = m, size = plan.headSize, heads = 4, kvHeads = 2, tokens = 9, pos = 61, positions = pos + tokens;
  const kvDim = kvHeads * size, qDim = heads * size, scale = 1 / Math.sqrt(size);
  const q = new Float32Array(tokens * qDim).map(() => Math.random() * 2 - 1);
  const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((12 + ((Math.random() * 5) | 0)) << 10) | ((Math.random() * 1024) | 0));
  const keys = new Uint16Array(positions * kvDim).map(halfBits), values = new Uint16Array(positions * kvDim).map(halfBits);
  const owned = [];
  const make = (data, usage = STORAGE | COPY_DST) => {
    const made = buffer(m, data.byteLength, usage, owned);
    m.device.queue.writeBuffer(made, 0, data);
    return made;
  };
  try {
    const out = buffer(m, tokens * qDim * 4, STORAGE | COPY_SRC, owned);
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([heads, kvHeads]);
    new Float32Array(params, 8, 1)[0] = scale;
    const group = bind(m, pipeline, [make(q), make(keys), make(values), out, uniform(m, params, owned),
      uniform(m, new Uint32Array([tokens, pos, 0, 0]), owned)]);
    const encoder = m.device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, group, heads * Math.ceil(tokens / m.wgsl.FLASH_Q_TILE));
    pass.end();
    m.device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBack(m, out, tokens * qDim * 4));
    const k = Float32Array.from(keys, halfToFloat), v = Float32Array.from(values, halfToFloat);
    let worst = 0;
    for (let t = 0; t < tokens; t++) {
      for (let h = 0; h < heads; h++) {
        const kv = Math.floor(h / (heads / kvHeads)) * size, row = t * qDim + h * size, seen = pos + t + 1;
        const scores = new Float64Array(seen);
        for (let p = 0; p < seen; p++) {
          for (let d = 0; d < size; d++) scores[p] += q[row + d] * scale * k[p * kvDim + kv + d];
        }
        const most = Math.max(...scores);
        let sum = 0, largest = 0;
        for (let p = 0; p < seen; p++) sum += (scores[p] = Math.exp(scores[p] - most));
        for (let p = 0; p < positions; p++) for (let d = 0; d < size; d++) largest = Math.max(largest, Math.abs(v[p * kvDim + kv + d]));
        for (let d = 0; d < size; d++) {
          let want = 0;
          for (let p = 0; p < seen; p++) want += scores[p] * v[p * kvDim + kv + d];
          worst = Math.max(worst, Math.abs(got[row + d] - want / sum) / largest);
        }
      }
    }
    return worst <= LINE ? null : `its output is ${worst.toExponential(2)} of the largest value from JavaScript's (line ${LINE})`;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}
function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// ---- the matrices: the tiled shaders of T146 that this device can make (shaders.js's promptForms), each checked on a
// small matrix against JavaScript (checkForm: subgroups, f16 rounding and drivers differ from device to device, and
// only the device can say), the right ones timed on the model's own gate matrix by a whole block, and the fastest
// taken. forms: what each came to (ms, or why not), for the console. plan.force.matrices (tests only): that form alone, untimed.
async function chooseMatrices(m) {
  const { device, plan, wgsl } = m;
  const packed = Boolean(navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product"));
  let forms = wgsl.promptForms({ half: device.features.has("shader-f16"), subgroups: device.features.has("subgroups"), packed,
    memory: device.limits.maxComputeWorkgroupStorageSize, threads: device.limits.maxComputeInvocationsPerWorkgroup });
  if (plan.force.matrices) forms = forms.filter((form) => form.name === plan.force.matrices);
  m.forms = [];
  let best = null;
  for (const form of forms) {
    if (form.none) {
      m.forms.push({ name: form.name, none: form.none });
      continue;
    }
    try {
      const pipeline = await validated(m, () => pipelineOf(m, form.code, form.constants));
      const tiled = { ...form, pipeline };
      const wrong = await checkForm(m, tiled);
      if (wrong) {
        m.forms.push({ name: form.name, none: `wrong: ${wrong}` });
        continue;
      }
      const ms = plan.force.matrices ? 0 : await timeForm(m, tiled);
      m.forms.push({ name: form.name, ms });
      if (!best || ms < best.ms) best = { ...tiled, ms };
    } catch (error) {
      m.forms.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!best) throw new Error(`no tiled shader is right on this GPU (${m.forms.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.matrices}`})`);
  m.form = best;
}

// a matrix of rows × n (w and s, buffers) by the tokens of from into to (added where add): the bind group of form
const productGroup = (m, form, w, s, rows, n, from, to, add, owned = m.owned, io = m) => bind(m, form.pipeline,
  [w, s, form.packed ? io.xq : from, to, uniform(m, new Uint32Array([rows, n / 4, n / m.wgsl.GROUP, 0, n, rows, add ? 1 : 0, 0]), owned),
   io.step, ...(form.packed ? [io.xs] : [])]);
// the workgroups of form for rows by count tokens: the tiles numbered over x, then y (as T146's shaders number them)
function multiply(m, pass, form, group, rows, count) {
  const tiles = Math.ceil(rows / form.tile.rows) * Math.ceil(count / form.tile.tokens);
  const across = Math.min(tiles, m.device.limits.maxComputeWorkgroupsPerDimension);
  dispatch(pass, form.pipeline, group, across, Math.ceil(tiles / across));
}

// T146's check (public/benchmark/gpu.js's checkTiled) of a form: 300 rows of 544 (17 groups of 32: a part of a tile
// of rows everywhere), 11 and 70 tokens (a part of a tile of tokens; two or one and a part), and 11 tokens whose x and
// y are wider than the product (xStride 608, yStride 320), each product twice into the same y (the second added:
// shape.add) against JavaScript's (shaders.js's tiledOff). The reason it is wrong, or null
async function checkForm(m, form) {
  const { device, wgsl } = m, rows = 300, n = 544, perRow = n / wgsl.GROUP;
  const w = new Int8Array(rows * n).map(() => (Math.random() * 256) | 0), s = new Float32Array(rows * perRow).map(() => Math.random() * 0.01);
  for (const { tokens, wider } of [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }]) {
    const xStride = n + wider, yStride = rows + (wider ? 20 : 0), owned = [];
    const x = new Float32Array(tokens * xStride).map(() => (Math.random() - 0.5) * 2);
    const make = (data, usage = STORAGE | COPY_DST | COPY_SRC) => {
      const made = buffer(m, data.byteLength, usage, owned);
      device.queue.writeBuffer(made, 0, data);
      return made;
    };
    try {
      const io = { xq: buffer(m, tokens * xStride, STORAGE | COPY_SRC, owned), xs: buffer(m, tokens * (xStride / wgsl.GROUP) * 4, STORAGE | COPY_SRC, owned),
        step: uniform(m, new Uint32Array([tokens, 0, 0, 0]), owned) };
      const wb = make(w), sb = make(s), xb = make(x), y = make(new Float32Array(tokens * yStride));
      const shape = (add) => uniform(m, new Uint32Array([rows, n / 4, perRow, 0, xStride, yStride, add ? 1 : 0, 0]), owned);
      const group = (add) => bind(m, form.pipeline, [wb, sb, form.packed ? io.xq : xb, y, shape(add), io.step, ...(form.packed ? [io.xs] : [])]);
      const groups = await validated(m, () => [group(false), group(true)]);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      if (form.packed) {
        dispatch(pass, m.quantize, bind(m, m.quantize, [xb, io.xq, io.xs, uniform(m, new Uint32Array([n, xStride, 0, 0]), owned), io.step]),
          Math.ceil(n / wgsl.GROUP / 64), tokens);
      }
      groups.forEach((g) => multiply(m, pass, form, g, rows, tokens));
      pass.end();
      device.queue.submit([encoder.finish()]);
      const got = new Float32Array(await readBack(m, y, tokens * yStride * 4));
      const xq = form.packed ? new Int8Array(await readBack(m, io.xq, tokens * xStride)) : null;
      const xs = form.packed ? new Float32Array(await readBack(m, io.xs, tokens * (xStride / wgsl.GROUP) * 4)) : null;
      const { wrong } = wgsl.tiledOff({ w, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: form.half });
      if (wrong) return `${wrong} (${tokens} tokens${wider ? ", wider x and y" : ""})`;
    } finally {
      owned.forEach((b) => b.destroy());
    }
  }
  return null;
}

// ms of the model's first gate matrix (w1 of layer 0: hidden × dim, the widest kind of a layer's matrices) by a block
// of plan.batch tokens with form, its quantization included for a packed one
async function timeForm(m, form) {
  const { device, plan } = m, { rows, n, layers } = m.matrices.w1, owned = [];
  try {
    device.queue.writeBuffer(m.step, 0, new Uint32Array([plan.batch, 0, 0, 0]));
    const group = await validated(m, () => productGroup(m, form, ...layers[0], rows, n, m.xb, m.gate, false, owned));
    const once = async () => {
      const began = performance.now(), encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      if (form.packed) dispatch(pass, m.quantize, m.quantizeXb.group, m.quantizeXb.x, plan.batch);
      multiply(m, pass, form, group, rows, plan.batch);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    await once();
    const times = [];
    for (let i = 0; i < (m.fallback ? 1 : TIMES); i++) times.push(await once());
    return times.sort((a, b) => a - b)[times.length >> 1];
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// every layer's bind groups but those of the cache, with the form chosen
function bindLayers(m) {
  const { plan } = m, form = m.form;
  const product = (name, l, from, to, add = false) => {
    const { rows, n, layers } = m.matrices[name];
    return { group: productGroup(m, form, ...layers[l], rows, n, from, to, add), rows };
  };
  m.layers = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    m.layers.push({
      attentionNorm: bind(m, m.norm, [m.x, m.norms.attention, m.xb, normShape, m.step]),
      q: product("wq", l, m.xb, m.q), k: product("wk", l, m.xb, m.k), v: product("wv", l, m.xb, m.v),
      o: product("wo", l, m.xb, m.x, true),
      ffnNorm: bind(m, m.norm, [m.x, m.norms.ffn, m.xb, normShape, m.step]),
      gate: product("w1", l, m.xb, m.gate), up: product("w3", l, m.xb, m.up),
      down: product("w2", l, m.gate, m.x, true),
    });
  }
}

// The GPU's own keys and values (float16, a pair to a u32), per layer [positions][kvDim], grown as the CPU's cache
// grows (doubling from plan.kvStart: a prompt seldom needs the whole context) and kept from block to block; with them
// the bind groups that read them
function grow(m, needed) {
  const { device, plan } = m, old = m.cache;
  const kvDim = plan.kvHeads * plan.headSize, row = kvDim * 2;
  const capacity = Math.min(Math.max(2 * (old?.capacity ?? 0), needed), plan.seqLen);
  if (capacity * row > m.limit) throw new Error(`the keys of ${capacity} positions are more than a buffer of this GPU`);
  const cache = { capacity, owned: [], keys: [], values: [], rope: [], attention: [] };
  const usage = STORAGE | COPY_SRC | COPY_DST;
  const encoder = device.createCommandEncoder();
  for (let l = 0; l < plan.layers; l++) {
    const keys = buffer(m, capacity * row, usage, cache.owned), values = buffer(m, capacity * row, usage, cache.owned);
    if (old) {
      encoder.copyBufferToBuffer(old.keys[l], 0, keys, 0, old.capacity * row);
      encoder.copyBufferToBuffer(old.values[l], 0, values, 0, old.capacity * row);
    }
    cache.keys.push(keys);
    cache.values.push(values);
  }
  device.queue.submit([encoder.finish()]);
  // the old ones go once the copies are done (a buffer destroyed after its submission lives until the GPU is through)
  old?.owned.forEach((buffer) => buffer.destroy());
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(params, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const attentionParams = uniform(m, params, cache.owned);
  for (let l = 0; l < plan.layers; l++) {
    cache.rope.push(bind(m, m.rope, [m.q, m.k, m.v, cache.keys[l], cache.values[l], m.angles, m.ropeShape, m.step]));
    cache.attention.push(bind(m, m.attention.pipeline, [m.q, cache.keys[l], cache.values[l], m.xb, attentionParams, m.step]));
  }
  m.cache = cache;
}

// ---- a block of a prompt
async function prompt({ serial, count, pos }) {
  const { memory, plan } = model;
  const words = new Int32Array(memory.buffer, 0, Math.max(...Object.values(plan.words)) + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  // whether forward.js still waits for this request (T147): it gave up, and the memory may soon be another model's
  const wanted = () => Atomics.load(words, plan.words.wanted) === serial;
  let failed = 1;
  try {
    if (lost) throw new Error(lost);
    await block(model, count, pos, wanted);
    if (lost) throw new Error(lost);
    failed = 0;
  } catch (error) {
    postMessage({ type: "failed", reason: String(error?.message ?? error) });
  } finally {
    clearInterval(beat);
    if (wanted()) {
      Atomics.store(words, plan.words.failed, failed);
      Atomics.store(words, plan.words.done, serial);
      Atomics.notify(words, plan.words.done);
    }
  }
}

async function block(m, count, pos, wanted) {
  const { device, plan } = m, B = plan.batch, half = plan.headSize / 2, kvDim = plan.kvHeads * plan.headSize;
  // the rows forward.js embedded (dense), and the angles of their positions (the tables forward.js has)
  const F = new Float32Array(m.memory.buffer);
  m.rows.set(F.subarray(plan.rows / 4, plan.rows / 4 + count * plan.dim));
  for (let t = 0; t < count; t++) {
    const cos = plan.cos / 4 + (pos + t) * half, sin = plan.sin / 4 + (pos + t) * half;
    m.turns.set(F.subarray(cos, cos + half), t * plan.headSize);
    m.turns.set(F.subarray(sin, sin + half), t * plan.headSize + half);
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.queue.writeBuffer(m.x, 0, m.rows, 0, count * plan.dim);
  device.queue.writeBuffer(m.angles, 0, m.turns, 0, count * plan.headSize);
  device.queue.writeBuffer(m.step, 0, new Uint32Array([count, pos, 0, 0]));
  if (pos + count > m.cache.capacity) grow(m, pos + count);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), form = m.form;
  const multiplied = (product) => multiply(m, pass, form, product.group, product.rows, count);
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantize = (q) => form.packed && dispatch(pass, m.quantize, q.group, q.x, count);
  const flash = m.attention;
  for (let l = 0; l < plan.layers; l++) {
    const layer = m.layers[l];
    dispatch(pass, m.norm, layer.attentionNorm, count);
    quantize(m.quantizeXb);
    multiplied(layer.q);
    multiplied(layer.k);
    multiplied(layer.v);
    dispatch(pass, m.rope, m.cache.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    dispatch(pass, flash.pipeline, m.cache.attention[l], plan.heads * Math.ceil(count / m.wgsl.FLASH_Q_TILE));
    quantize(m.quantizeXb);
    multiplied(layer.o);
    dispatch(pass, m.norm, layer.ffnNorm, count);
    quantize(m.quantizeXb);
    multiplied(layer.gate);
    multiplied(layer.up);
    dispatch(pass, m.swiglu, m.swigluGroup, Math.ceil(plan.hidden / 64), count);
    quantize(m.quantizeGate);
    multiplied(layer.down);
  }
  pass.end();
  // the block's keys of every layer, then its values, [layers][B][kvDim] each in float16, as plan.staging lays them out
  const row = kvDim * 2;
  for (let l = 0; l < plan.layers; l++) {
    encoder.copyBufferToBuffer(m.cache.keys[l], pos * row, m.readback, l * B * row, count * row);
    encoder.copyBufferToBuffer(m.cache.values[l], pos * row, m.readback, (plan.layers + l) * B * row, count * row);
  }
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a block (${(invalid ?? full).message})`);
  await m.readback.mapAsync(MAP_READ);
  try {
    const bytes = 2 * plan.layers * B * row;
    if (wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(m.readback.getMappedRange(), 0, bytes));
  } finally {
    m.readback.unmap();
  }
}
