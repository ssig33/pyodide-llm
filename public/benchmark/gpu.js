// The GPU section of /benchmark/ (T134; T94's stage 0 until then, at /gpu-test/): what WebGPU gives this device,
// measured in a worker (where a GPU forward pass would run). The page (src/pages/benchmark.astro) asks for one step at
// a time and shows what comes back, and ends the worker after the section; nothing here touches the model page.
//
//   { step: "info" }                      the adapter, its limits and features, WGSL's language features
//   { step: "check" }                     the int8 shaders against JavaScript on small matrices (T146: the tiled ones
//                                         too, with their edges)
//   { step: "bandwidth", shape }          GB/s of one int8 matrix times a vector, by every shader of matVecShaders()
//                                         (T134's two and T149's from llama.cpp and ONNX Runtime), and the CPU's
//   { step: "token", model, kind, sample }
//                                         a whole token's work of a model's shapes (every layer's matrices, a few small
//                                         dispatches, the classifier, the logits read back), ms per token. sample: the
//                                         most likely token found on the GPU, and only its id read back instead of every
//                                         logit
//   { step: "layer" }                     T150: one layer of a token of Llama 3.2 1B's width, as its fourteen separate
//                                         steps and fused into five dispatches (shaders.js's fusedMatVec), ms a layer
//   { step: "overhead" }                  what a token costs besides the weights: 240 empty dispatches, a submission
//                                         with and without waiting for it, reading back 4 bytes and all the logits
//   { step: "prompt", counts }            the tokens of a prompt through the matrices all at once (matrix × matrix,
//                                         T135's first candidate), on the made-up model of the CPU section's shape,
//                                         by T135's batched shader and T146's tiled ones, with the GFLOPS of each
//   { step: "ceilings" }                  T168: the device's ceilings: f32 and f16 multiply-adds (GFLOPS), dot4I8Packed
//                                         (GOPS), reading the workgroup's memory and a storage buffer (GB/s), each a
//                                         loop of that alone (shaders.js), for the share of them the prompt's shaders reach
//   { step: "bridge", memory, rounds }    the round trip of a worker that waits with Atomics.wait and this one, which
//                                         answers with Atomics.waitAsync (stage 1's design), in microseconds
//
// The weights are random: only their size and layout matter. int8 in groups of 32 with a float32 scale each, as the
// checkpoints of this project (llama2_numpy's layout), 4 values to a u32.

// the WGSL, shared with the model's GPU worker (public/shaders.js, T135), from the same deployment as this file. Not
// awaited here: a module worker's port opens at its first await, and a message that comes before onmessage is set is
// lost (T109); every step awaits it instead
const shaders = import(new URL(`../shaders.js${new URL(import.meta.url).search}`, import.meta.url));
let WGSL, GROUP, TILE;

// the shapes of the models in the list that the measurement stands for (legacy header: dim, hidden, layers, heads,
// kv heads, vocab), and the int8 matrices of a layer, [rows, n]
const MODELS = {
  "llm-jp-3 150M": { dim: 512, hidden: 2048, layers: 12, heads: 8, kvHeads: 8, vocab: 99584 },
  "Llama 3.2 1B": { dim: 2048, hidden: 8192, layers: 16, heads: 32, kvHeads: 8, vocab: 128256 },
  "Llama 3.2 3B": { dim: 3072, hidden: 8192, layers: 28, heads: 24, kvHeads: 8, vocab: 128256 },
};
const layerMatrices = ({ dim, hidden, heads, kvHeads }) => {
  const kvDim = (dim / heads) * kvHeads;
  return [[dim, dim], [kvDim, dim], [kvDim, dim], [dim, dim], [hidden, dim], [hidden, dim], [dim, hidden]];
};
// the small steps a layer dispatches on its own, as the CPU's forward pass has them (a stand-in each, SMALL; T150's
// layer step runs the real ones)
const SMALL_PER_LAYER = 7;
// the CPU section's made-up model (public/benchmark/sections.js): Llama 3.2 1B's width, two layers
const PROMPT_MODEL = { dim: 2048, hidden: 8192, layers: 2, heads: 32, kvHeads: 8 };
const matrixBytes = ([rows, n]) => rows * n + (rows * n / GROUP) * 4;

let device, adapter, packed = false;
// a fallback adapter (SwiftShader: the CPU pretending to be a GPU, as in CI) says nothing of a GPU's speed, and takes
// 16 s for one token of Llama 3.2 1B: every measurement is taken once there, and only its answers are worth anything
let fallback = false;
async function gpu() {
  if (device) return device;
  if (!self.navigator?.gpu) throw new Error("no navigator.gpu in a worker here");
  adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("navigator.gpu gave no adapter");
  fallback = Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter);
  packed = navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false;
  // as much of a buffer and of a binding as the adapter allows: the weights are the point. T146's tiled shaders use
  // shader-f16 and subgroups where the adapter has them (asked for only then: a device refuses a feature it lacks)
  device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => postMessage({ lost: `${info.reason}: ${info.message}` }));
  return device;
}

async function info() {
  const found = { worker: Boolean(self.navigator?.gpu) };
  if (!found.worker) return found;
  await gpu();
  const about = adapter.info ?? {};
  Object.assign(found, {
    adapter: [about.vendor, about.architecture, about.device, about.description].filter(Boolean).join(" · ") || "(not told)",
    fallback: adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    // what a tiled shader (T146) may take: the device's, though the tiles keep to the defaults (16 KiB, 256)
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    // the DP4A shader's subgroup path runs only where a subgroup is 16 wide (SwiftShader's is not: CI never runs it)
    subgroupSizes: adapter.info?.subgroupMinSize ? [adapter.info.subgroupMinSize, adapter.info.subgroupMaxSize] : null,
    features: [...adapter.features].sort(),
    wgsl: [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort(),
    packed,
  });
  return found;
}

// ---- buffers
// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
function buffer(bytes, usage = STORAGE | COPY_DST) {
  return device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
}
// random bytes, written a few megabytes at a time (writeBuffer copies them, so one block serves them all)
const noise = new Uint8Array(4 << 20).map(() => (Math.random() * 256) | 0);
function fill(target, bytes) {
  for (let at = 0; at < bytes; at += noise.length) device.queue.writeBuffer(target, at, noise, 0, Math.min(noise.length, bytes - at) & ~3);
}
function floats(count, scale = 0.01) {
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) values[i] = (Math.random() - 0.5) * scale;
  return values;
}

let pipelines;
function pipelinesFor() {
  if (pipelines) return pipelines;
  const make = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  pipelines = { widen: make(WGSL.WIDEN), packed: packed ? make(WGSL.PACKED) : null, small: make(WGSL.SMALL),
                batched: make(WGSL.BATCHED), argmax: make(WGSL.ARGMAX), empty: make(WGSL.EMPTY) };
  return pipelines;
}

// T146: the shaders of a prompt: T135's batched one, then the tiled ones of shaders.js: llama.cpp's register tiles
// (f16 in the workgroup's memory where shader-f16 is, else f32) in both shapes of REG_TILES, and ONNX Runtime's DP4A
// (where the packed int8 dot is), with its subgroup path where subgroups are. A shape past the device's workgroup
// memory or threads is not made (none says why). A tiled shader's pipeline is made when first asked for, asynchronously
// and in an error scope: one this device refuses rejects there, and only its own rows say so
const tiledPipelines = new Map();
function promptShaders() {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = device.limits;
  // T147: the tiled ones are the model's GPU worker's candidates (shaders.js's promptForms), the same list
  return [{ name: "batched (T135)", kind: "batched" }, ...WGSL.promptForms({ half: device.features.has("shader-f16"),
    subgroups: device.features.has("subgroups"), packed, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) })];
}
// T149: the shaders of a matrix times one vector (a generated token's): T134's two, then llama.cpp's mul_mat_vec (its
// float form and its MMVQ one, each with the workgroup's reduction and, where subgroups are, with subgroupAdd) and
// ONNX Runtime's MatMulNBits and DP4A for small M (shaders.js). rows: the rows a workgroup takes
function matVecShaders() {
  const noPacked = packed ? undefined : "no packed int8 dot here";
  const subgroups = device.features.has("subgroups");
  const noSubgroupId = navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ? undefined : "no subgroup_id in this WGSL";
  // check: the name of the shader's verdict in check() (T134's two are checked on their own there)
  const shaders = [{ name: "widened (T134)", kind: "widen", check: "widen" },
    { name: "packed int8 (T134)", kind: "packed", check: "packed", none: noPacked }];
  for (const [form, isPacked] of [["mul_mat_vec", false], ["MMVQ", true]]) {
    for (const withSubgroups of subgroups ? [false, true] : [false]) {
      shaders.push({ name: `llama.cpp ${form}, ${WGSL.MUL_MAT_VEC_ROWS} rows${withSubgroups ? ", subgroups" : ""}`,
        code: WGSL.mulMatVec({ packed: isPacked, subgroups: withSubgroups }), rows: WGSL.MUL_MAT_VEC_ROWS, packed: isPacked,
        none: (isPacked && noPacked) || (withSubgroups && noSubgroupId) || undefined });
    }
  }
  shaders.push({ name: `ORT MatMulNBits, ${WGSL.ORT_MATVEC_ROWS} rows`, code: WGSL.ortMatVec, rows: WGSL.ORT_MATVEC_ROWS, packed: false });
  shaders.push({ name: `ORT DP4A small M, ${WGSL.ORT_DP4A_MATVEC_ROWS} rows`, code: WGSL.ortDp4aMatVec,
    rows: WGSL.ORT_DP4A_MATVEC_ROWS, packed: true, none: noPacked });
  return shaders;
}
// what matrix() takes for a shader: "widen", "packed", "batched", or { pipeline, tile or rows, packed } of one with code
// of its own (a tiled one, T146, or a matrix × vector's of T149)
async function kindOf(shader) {
  if (!shader.code) return shader.kind;
  if (!tiledPipelines.has(shader.name)) {
    tiledPipelines.set(shader.name, validated(() => device.createComputePipelineAsync({ layout: "auto",
      compute: { module: device.createShaderModule({ code: shader.code }), entryPoint: "main", constants: shader.constants } })));
  }
  return { pipeline: await tiledPipelines.get(shader.name), tile: shader.tile, rows: shader.rows, packed: shader.packed };
}
// the activations of the packed tiled shaders, quantized on the GPU (shaders.js's QUANTIZE): the first n values of
// each of io's tokens, from io.x into io.xq and io.xs. One thread a group of 32, the tokens along y
let quantizePipeline;
function quantizer(io, n) {
  quantizePipeline ??= device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL.QUANTIZE }), entryPoint: "main" } });
  const shape = buffer(16, UNIFORM | COPY_DST);
  device.queue.writeBuffer(shape, 0, new Uint32Array([n, io.xStride, 0, 0]));
  const group = device.createBindGroup({ layout: quantizePipeline.getBindGroupLayout(0),
    entries: [io.x, io.xq, io.xs, shape, io.step].map((b, binding) => ({ binding, resource: { buffer: b } })) });
  return { dispatch: [quantizePipeline, group, Math.ceil(n / GROUP / 64), io.tokens, 1], owned: [shape] };
}

// A matrix of [rows, n] on the GPU, cut into chunks of rows that each fit a binding (chunk: rows a chunk, for the
// checks; else as many as fit). Returns the dispatches that multiply it by x into y: [pipeline, bind group, workgroups
// x, workgroups y, workgroups z] each. kind "batched" or a tiled one (kindOf): by io.tokens vectors at once (x and y
// hold them io.xStride and io.yStride floats apart); add: the products added to what y holds (the residual stream of
// the model's layers), for the checks
function matrix([rows, n], io, kind = "widen", data, { add = false, chunk } = {}) {
  const held = placed([rows, n], io, data, { add, chunk });
  return { ...bound(held, io, kind), owned: held.owned, bytes: held.bytes };
}
// the weights of a matrix on the GPU, in chunks of rows that each fit a binding, with each chunk's Shape (T149: placed
// once and bound by every shader of a matrix × vector)
function placed([rows, n], io, data, { add = false, chunk } = {}) {
  const words = n / 4, perRow = n / GROUP, rowBytes = n;
  const most = chunk ?? Math.max(1, Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / rowBytes));
  const chunks = [], owned = [];
  for (let first = 0; first < rows; first += most) {
    const count = Math.min(most, rows - first);
    const w = buffer(count * rowBytes), s = buffer(count * perRow * 4), shape = buffer(32, UNIFORM | COPY_DST);
    owned.push(w, s, shape);
    if (data) {
      device.queue.writeBuffer(w, 0, data.w, first * rowBytes, count * rowBytes);
      device.queue.writeBuffer(s, 0, data.s, first * perRow, count * perRow);
    } else {
      fill(w, count * rowBytes);
      device.queue.writeBuffer(s, 0, floats(count * perRow, 0.002));
    }
    // the batched and tiled shaders' shape goes on with the strides of the tokens and "add", the others read four
    device.queue.writeBuffer(shape, 0, new Uint32Array([count, words, perRow, first, io.xStride ?? 0, io.yStride ?? 0, add ? 1 : 0, 0]));
    chunks.push({ w, s, shape, count });
  }
  return { chunks, owned, bytes: matrixBytes([rows, n]) };
}
// the dispatches that multiply placed weights by x into y for a shader: [pipeline, bind group, workgroups x, y, z] each
function bound({ chunks }, io, kind = "widen") {
  const own = typeof kind === "object", tiled = own && Boolean(kind.tile), matVec = own && Boolean(kind.rows);
  const { widen, packed: packedPipeline, batched } = pipelinesFor();
  const pipeline = own ? kind.pipeline : { widen, packed: packedPipeline, batched }[kind];
  const quantized = own ? kind.packed : kind === "packed";
  const dispatches = [];
  for (const { w, s, shape, count } of chunks) {
    const entries = [{ binding: 0, resource: { buffer: w } }, { binding: 1, resource: { buffer: s } },
      { binding: 2, resource: { buffer: quantized ? io.xq : io.x } }, { binding: 3, resource: { buffer: io.y } },
      { binding: 4, resource: { buffer: shape } }];
    if (kind === "packed" || (matVec && quantized)) entries.push({ binding: 5, resource: { buffer: io.xs } });
    if (kind === "batched" || tiled) entries.push({ binding: 5, resource: { buffer: io.step } });
    if (tiled && quantized) entries.push({ binding: 6, resource: { buffer: io.xs } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    if (tiled) {
      // the tiles numbered over x, then y (as both sources number them: the rows' tiles first, then the tokens')
      const tiles = Math.ceil(count / kind.tile.rows) * Math.ceil(io.tokens / kind.tile.tokens);
      const across = Math.min(tiles, device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(tiles / across)]);
    } else if (matVec) {
      // T149: kind.rows rows a workgroup, numbered over x and then y
      const groups = Math.ceil(count / kind.rows), across = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(groups / across)]);
    } else {
      const across = Math.min(count, device.limits.maxComputeWorkgroupsPerDimension);
      dispatches.push([pipeline, group, across, Math.ceil(count / across), kind === "batched" ? Math.ceil(io.tokens / TILE) : 1]);
    }
  }
  return { dispatches };
}
// the vectors every matrix reads and writes: x of the longest row, y of the most rows, tokens of each (the batched and
// tiled shaders take several); xq and xs: x quantized, 8 bits a value and a float32 scale a group of 32, of every token
function vectors(longest, most, tokens = 1) {
  const io = { x: buffer(tokens * longest * 4), xq: buffer(tokens * longest, STORAGE | COPY_DST | COPY_SRC),
               xs: buffer(tokens * (longest / GROUP) * 4, STORAGE | COPY_DST | COPY_SRC),
               y: buffer(tokens * most * 4 + 16, STORAGE | COPY_DST | COPY_SRC), tokens, xStride: longest, yStride: most,
               step: buffer(16, UNIFORM | COPY_DST) };
  device.queue.writeBuffer(io.step, 0, new Uint32Array([tokens, 0, 0, 0]));  // the batched and tiled shaders' tokens
  device.queue.writeBuffer(io.x, 0, floats(tokens * longest, 2));
  fill(io.xq, tokens * longest);
  device.queue.writeBuffer(io.xs, 0, floats(tokens * (longest / GROUP), 0.1));
  return io;
}
const destroyVectors = (io) => [io.x, io.xq, io.xs, io.y, io.step].forEach((b) => b.destroy());
function run(pass, [pipeline, group, x, y, z = 1]) {
  pass.setPipeline(pipeline);
  if (group) pass.setBindGroup(0, group);  // none for the empty dispatch
  pass.dispatchWorkgroups(x, y, z);
}
// the most likely token of the logits in y (the classifier's output), into a buffer of its own
function argmaxOf(io, vocab) {
  const { argmax } = pipelinesFor();
  const chosen = buffer(16, STORAGE | COPY_SRC), count = buffer(16, UNIFORM | COPY_DST);
  device.queue.writeBuffer(count, 0, new Uint32Array([vocab, 0, 0, 0]));
  const group = device.createBindGroup({ layout: argmax.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: io.y } },
    { binding: 1, resource: { buffer: chosen } }, { binding: 2, resource: { buffer: count } }] });
  return { dispatch: [argmax, group, 1, 1], chosen, owned: [chosen, count] };
}
// read size bytes of source back: the copy, the submission and the mapping, as a token's logits come back. target: a
// buffer to read into again and again (a measurement's loop), else one made and destroyed here
async function readBack(encoder, source, size, target) {
  const bytes = Math.ceil(size / 4) * 4, into = target ?? buffer(bytes, MAP_READ | COPY_DST);
  encoder.copyBufferToBuffer(source, 0, into, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await into.mapAsync(MAP_READ);
  const got = into.getMappedRange(0, bytes).slice(0);
  into.unmap();
  if (!target) into.destroy();
  return got;
}
// the median of runs of a measurement, in ms, after warm-up runs (one run and no warm-up on a fallback adapter)
async function median(measure, times = 10, warm = 3) {
  if (fallback) [times, warm] = [1, 0];
  for (let i = 0; i < warm; i++) await measure();
  const ms = [];
  for (let i = 0; i < times; i++) {
    const began = performance.now();
    await measure();
    ms.push(performance.now() - began);
  }
  return ms.sort((a, b) => a - b)[ms.length >> 1];
}

// ---- check: every shader against JavaScript: the two of a matrix × vector on 300 rows of 512 (rows that are not a
// power of two), the batched one on the same with 11 tokens (a tile and a part of one), the tiled ones (checkTiled),
// the argmax on logits of Llama 3's vocabulary and on a tie (the first of the largest, as JavaScript finds it)
async function check() {
  await gpu();
  const rows = 300, n = 512, words = n / 4;
  const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * n / GROUP, 0.01);
  const io = vectors(n, rows), x = floats(n, 2);
  device.queue.writeBuffer(io.x, 0, x);
  const { xq, xs } = WGSL.quantizedLikeCpu(x);
  device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
  device.queue.writeBuffer(io.xs, 0, xs);
  const signed = new Int8Array(w.buffer);
  const verdicts = {};
  for (const kind of packed ? ["widen", "packed"] : ["widen"]) {
    const m = matrix([rows, n], io, kind, { w, s });
    const readback = buffer(rows * 4, MAP_READ | COPY_DST);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    m.dispatches.forEach((d) => run(pass, d));
    pass.end();
    encoder.copyBufferToBuffer(io.y, 0, readback, 0, rows * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(MAP_READ);
    const got = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    let worst = 0;
    for (let r = 0; r < rows; r++) {
      let want = 0;
      for (let i = 0; i < words; i++) {
        let dot = 0;
        for (let k = 0; k < 4; k++) dot += signed[r * n + i * 4 + k] * (kind === "packed" ? xq[i * 4 + k] : x[i * 4 + k]);
        want += dot * s[r * (n / GROUP) + (i >> 3)] * (kind === "packed" ? xs[i >> 3] : 1);
      }
      worst = Math.max(worst, Math.abs(got[r] - want) / (Math.abs(want) + 1e-3));
    }
    // float32 sums in another order: a relative 1e-4 is the rounding, a wrong index is off by a multiple
    verdicts[kind] = { worstRelative: worst, ok: worst < 1e-3 };
    m.owned.forEach((b) => b.destroy());
  }
  destroyVectors(io);
  verdicts.batched = await checkBatched(w, s, rows, n);
  Object.assign(verdicts, await checkMatVec());
  Object.assign(verdicts, await checkTiled());
  verdicts.argmax = await checkArgmax();
  Object.assign(verdicts, await checkLayer());
  return verdicts;
}
// T149: every matrix × vector shader of its own (llama.cpp's and ONNX Runtime's) on 300 rows cut into chunks of 101 (a
// workgroup's 4 or 8 rows and a part of them in each, and shape.first past 0), of widths 544 and 2080: 17 and 65 groups
// of 32, a part of what llama.cpp's 64 groups a pass, ORT's 512 values a step and its DP4A's 32 groups a step take, and
// 2080 more than one of them. x is 64 values longer than the width (a shader that takes the width from the buffer
// reads them), and y holds a sentinel past the rows that must stay. The sums as checkTiled holds them: WGSL.TILED_LINE of
// the sum of the |products| of the row (a wrong index or scale is off by about 1/sqrt(n) of it); the packed ones on the
// vector quantized in JavaScript (as PACKED is checked), whose products are exact integers times the two scales
const SENTINEL = 7.25;
async function checkMatVec() {
  const rows = 300, past = 16, verdicts = {};
  for (const shader of matVecShaders().filter((one) => one.code && !one.none)) {
    try {
      const kind = await kindOf(shader);
      let worst = 0, over = false, touched = false;
      for (const n of [544, 2080]) {
        const perRow = n / GROUP, longest = n + 64;
        const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * perRow, 0.01);
        const signed = new Int8Array(w.buffer);
        const io = vectors(longest, rows + past), x = floats(longest, 2), { xq, xs } = WGSL.quantizedLikeCpu(x);
        device.queue.writeBuffer(io.x, 0, x);
        device.queue.writeBuffer(io.xq, 0, new Uint8Array(xq.buffer));
        device.queue.writeBuffer(io.xs, 0, xs);
        device.queue.writeBuffer(io.y, 0, new Float32Array(rows + past).fill(SENTINEL));
        const owned = [];
        const got = await validated(async () => {
          const m = matrix([rows, n], io, kind, { w, s }, { chunk: 101 });
          owned.push(...m.owned);
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          m.dispatches.forEach((d) => run(pass, d));
          pass.end();
          return new Float32Array(await readBack(encoder, io.y, (rows + past) * 4));
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        for (let r = 0; r < rows; r++) {
          let want = 0, size = 0;
          for (let i = 0; i < n; i++) {
            const g = r * perRow + Math.floor(i / GROUP);
            const product = shader.packed ? signed[r * n + i] * xq[i] * s[g] * xs[Math.floor(i / GROUP)] : signed[r * n + i] * s[g] * x[i];
            want += product;
            size += Math.abs(product);
          }
          const off = Math.abs(got[r] - want);
          worst = Math.max(worst, off / size);
          over ||= !(off < WGSL.TILED_LINE * size);
        }
        for (let r = rows; r < rows + past; r++) touched ||= got[r] !== SENTINEL;
      }
      verdicts[shader.name] = { worstRelative: worst, ok: !over && !touched, ...(touched ? { wrotePastTheRows: true } : {}) };
    } catch (error) {
      verdicts[shader.name] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    }
  }
  return verdicts;
}
// T146: every tiled shader on 300 rows of 544 (17 groups of 32), cut into chunks of 100 rows (tiles of 32 or 64 rows
// and a part of one each, a subtile of 16 and a part, and shape.first past 0), with 11 and 70 tokens (a part of a tile
// of 32 or 64; two or one and a part), and 11 tokens whose x and y are wider than the product (xStride 608 for 544 of
// the width, yStride 320 for 300 rows: the prompt's model reads 2048 of 8192), twice into the same y (the second added
// to the first: shape.add) against JavaScript's product: half of y. The packed ones on what the GPU quantized, and that
// against JavaScript's quantize_x: a scale may differ in its last bits (WGSL's division is not rounded exactly) and a
// value then by 1, a wrong index by far more. The products are held (shaders.js's tiledOff) to 1e-4 of the sum of the |products| of the
// row and token: what a float32 sum in another order may differ by is 544 × 2^-24 = 3.2e-5 of it at most, and a wrong
// index, scale or group is off by about |value| / |sum of |products|| = 1 / sqrt(544) = 4e-2. The f16 tiles hold a
// weight times its scale and an activation as halves, and WGSL leaves the direction of that rounding to the device
// (round to nearest or toward zero, T146's review): each is then within 1 ulp, 2^-10 of it, or 2^-24 where it is
// subnormal, whatever the direction, so a product is within 2^-9 + 2^-20 of it and 2^-24 × (|weight| + |activation|),
// and the float32 sum adds (n + 1) × 2^-24 of the sum of |products|: the f16 tiles are held to that bound, row by row
// (about 2e-3 of the sum; still 1/20 of a wrong index's)
const CASES = [{ tokens: 11, wider: 0 }, { tokens: 70, wider: 0 }, { tokens: 11, wider: 64 }];
async function checkTiled() {
  const rows = 300, n = 544, perRow = n / GROUP;
  const w = new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s = floats(rows * perRow, 0.01);
  const signed = new Int8Array(w.buffer);
  const verdicts = {};
  for (const shader of promptShaders().filter((one) => one.tile && !one.none)) {
    try {
      const kind = await kindOf(shader);
      let worst = 0, over = false, far = false, apart = 0, values = 0;
      for (const { tokens, wider } of CASES) {
        const xStride = n + wider, yStride = rows + (wider ? 20 : 0);
        const io = vectors(xStride, yStride, tokens), x = floats(tokens * xStride, 2);
        device.queue.writeBuffer(io.x, 0, x);
        const owned = [];
        const [got, xq, xs] = await validated(async () => {
          const made = [matrix([rows, n], io, kind, { w, s }, { chunk: 100 }), matrix([rows, n], io, kind, { w, s }, { chunk: 100, add: true })];
          const quantize = shader.packed ? quantizer(io, n) : null;
          owned.push(...made.flatMap((m) => m.owned), ...(quantize?.owned ?? []));
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          if (quantize) run(pass, quantize.dispatch);
          made.forEach((m) => m.dispatches.forEach((d) => run(pass, d)));
          pass.end();
          const y = new Float32Array(await readBack(encoder, io.y, tokens * yStride * 4));
          if (!quantize) return [y];
          return [y, new Int8Array(await readBack(device.createCommandEncoder(), io.xq, tokens * xStride)),
            new Float32Array(await readBack(device.createCommandEncoder(), io.xs, tokens * (xStride / GROUP) * 4))];
        }).finally(() => {
          owned.forEach((b) => b.destroy());
          destroyVectors(io);
        });
        // T147: the comparison is shaders.js's tiledOff, the model's GPU worker's too
        const off = WGSL.tiledOff({ w: signed, s, x, got, xq, xs, rows, n, tokens, xStride, yStride, half: shader.half });
        worst = Math.max(worst, off.worst);
        over ||= Boolean(off.wrong);
        far ||= off.far;
        apart += off.apart;
        values += off.values;
      }
      // the quantized values no more than 1 apart, and apart in no more than 1 of 100
      const quantizing = values ? { apart: apart / values, far } : {};
      verdicts[shader.name] = { worstRelative: worst, ok: !over && !far && apart <= 0.01 * values, ...quantizing };
    } catch (error) {
      verdicts[shader.name] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    }
  }
  return verdicts;
}
// what fn does on the GPU, a validation error of it thrown (a pipeline or a bind group the device refused)
async function validated(fn) {
  device.pushErrorScope("validation");
  try {
    return await fn();
  } finally {
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(invalid.message);
  }
}
async function checkBatched(w, s, rows, n) {
  const tokens = 11, io = vectors(n, rows, tokens), x = floats(tokens * n, 2);
  device.queue.writeBuffer(io.x, 0, x);
  const m = matrix([rows, n], io, "batched", { w, s });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  m.dispatches.forEach((d) => run(pass, d));
  pass.end();
  const got = new Float32Array(await readBack(encoder, io.y, tokens * rows * 4));
  const signed = new Int8Array(w.buffer);
  let worst = 0;
  for (let t = 0; t < tokens; t++) {
    for (let r = 0; r < rows; r++) {
      let want = 0;
      for (let i = 0; i < n; i++) want += signed[r * n + i] * s[r * (n / GROUP) + Math.floor(i / GROUP)] * x[t * n + i];
      worst = Math.max(worst, Math.abs(got[t * rows + r] - want) / (Math.abs(want) + 1e-3));
    }
  }
  m.owned.forEach((b) => b.destroy());
  destroyVectors(io);
  return { worstRelative: worst, ok: worst < 1e-3 };
}
async function checkArgmax() {
  let right = true;
  for (const [vocab, tie] of [[128256, false], [1000, true]]) {
    const io = vectors(4, vocab), logits = floats(vocab, 20);
    if (tie) logits[700] = logits[300] = 50;  // the first of the two
    device.queue.writeBuffer(io.y, 0, logits);
    const picked = argmaxOf(io, vocab);
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    run(pass, picked.dispatch);
    pass.end();
    const got = new Uint32Array(await readBack(encoder, picked.chosen, 4))[0];
    let want = 0;
    for (let i = 1; i < vocab; i++) if (logits[i] > logits[want]) want = i;
    right &&= got === want;
    picked.owned.forEach((b) => b.destroy());
    destroyVectors(io);
  }
  return { worstRelative: 0, ok: right };
}

// ---- bandwidth: one matrix by every shader of matVecShaders() (T149), timed as the ceilings are (paired: a submission
// of 2n of it less one of n, so that what a submission costs besides its work drops out, T168). The weights are
// placed once for every shader (only the bind groups are the shader's), in as many copies as make MATVEC_BYTES, read
// in turn: the same small matrix read again and again stays in the GPU's caches (a phone's system cache of 19 MB and
// more holds Llama 3.2 1B's w1, T149's review), and a token reads each matrix once. The first shader is measured again
// at the end (a device that slows down as it warms shows it there, as the prompt's batched row does). A packed shader
// takes its vector quantized: what QUANTIZE costs for this width, once, goes beside them. On a fallback adapter each
// once, one copy, and no matrix past FALLBACK_BYTES (the check holds the shaders to JavaScript; SwiftShader took 13
// minutes of bench-check's 20 for the GPU section with the classifier's 295 MB 8 times). widen and packed: T134's two;
// cpu: the CPU's kernel on the same bytes
const MATVEC_MOST = 1 << 14, MATVEC_BYTES = 128 << 20, FALLBACK_BYTES = 32 << 20;
async function bandwidth(shape) {
  await gpu();
  const bytes = matrixBytes(shape), shaders = matVecShaders(), rows = [];
  const label = (shader) => ({ shader: shader.name, check: shader.check ?? shader.name });
  if (fallback && bytes > FALLBACK_BYTES) {
    return { rows: shaders.map((shader) => ({ ...label(shader), none: "not on a fallback adapter" })), cpu: await cpuBandwidth(shape) };
  }
  const io = vectors(shape[1], shape[0]), held = [];
  const copies = fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes);
  // what the dispatches do, timed: n of them a submission, each on the next copy (never the one just read)
  let next = 0;
  const timer = (each) => async (n) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (let i = 0; i < n; i++) each[next++ % each.length].forEach((d) => run(pass, d));
    pass.end();
    const began = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - began;
  };
  const time = async (each) => {
    const submission = timer(each);
    if (fallback) return { ms: await submission(1), dispatches: 1 };
    await submission(2);
    return paired(submission, MATVEC_MOST);
  };
  try {
    await scoped(async () => {
      for (let c = 0; c < copies; c++) held.push(placed(shape, io));
      await device.queue.onSubmittedWorkDone();
    });
    const measure = async (shader, again = false) => {
      const row = { ...label(shader), ...(again ? { shader: `${shader.name}, again at the end`, again } : {}) };
      if (shader.none) return { ...row, none: shader.none };
      try {
        const r = await validated(async () => {
          const kind = await kindOf(shader);
          return time(held.map((h) => bound(h, io, kind).dispatches));
        });
        return { ...row, GBps: (r.dispatches * bytes) / (r.ms / 1000) / 1e9, msEach: r.ms / r.dispatches, dispatches: r.dispatches,
          ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) };
      } catch (error) {
        return { ...row, error: String(error?.message ?? error) };
      } finally {
        postMessage({ alive: true });
      }
    };
    for (const shader of shaders) rows.push(await measure(shader));
    rows.push(await measure(shaders[0], true));
  } finally {
    held.forEach((h) => h.owned.forEach((b) => b.destroy()));
  }
  // the vector of a packed shader quantized (QUANTIZE, one dispatch a vector of this width)
  let quantize;
  if (packed) {
    const q = quantizer(io, shape[1]);
    try {
      const r = await validated(() => time([[q.dispatch]]));
      quantize = { msEach: r.ms / r.dispatches, ...(r.unsteady ? { unsteady: true } : {}) };
    } catch (error) {
      quantize = { error: String(error?.message ?? error) };
    } finally {
      q.owned.forEach((b) => b.destroy());
    }
  }
  destroyVectors(io);
  const measured = (kind) => rows.find((row) => row.check === kind && !row.again && row.GBps);
  return { rows, copies, quantize, widen: measured("widen"), packed: measured("packed"), cpu: await cpuBandwidth(shape) };
}

// the CPU on the kernel the model page uses for int8 on one thread (matmul_q8, without relaxed SIMD: every
// browser has it), on four matrices of this shape in turn so that no cache holds them. The page's forward pass does
// more than this with relaxed SIMD and its software threads (2.9 times on the owner's Android, T157), which is why the
// token's table holds the GPU against the CPU section instead.
let cpuKernel;
async function cpuBandwidth([rows, n]) {
  const weights = rows * n, scales = (rows * n / GROUP) * 4, copies = Math.max(1, Math.min(4, Math.floor(128e6 / weights)));
  const bytes = 4096 + n * 8 + rows * 4 + copies * (weights + scales);
  const memory = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) + 1 });
  // the kernels of the same deployment as this file (?v=, GitHub Pages keeps a file for ten minutes)
  cpuKernel ??= await WebAssembly.compile(await (await fetch(new URL(`../simdkernel_plain.wasm${new URL(import.meta.url).search}`, import.meta.url))).arrayBuffer());
  const k = (await WebAssembly.instantiate(cpuKernel, { env: { memory } })).exports;
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const x = 4096, xq = x + n * 4, xs = xq + n, out = xs + (n / GROUP) * 4 + 64, first = Math.ceil((out + rows * 4) / 64) * 64;
  F.set(floats(n, 2), x / 4);
  for (let c = 0; c < copies; c++) {
    const at = first + c * (weights + scales);
    for (let i = 0; i < weights; i += noise.length) U.set(noise.subarray(0, Math.min(noise.length, weights - i)), at + i);
    F.set(floats(rows * n / GROUP, 0.002), (at + weights) / 4);
  }
  k.quantize_x(xq, xs, x, n, 0);
  const pass = (c) => k.matmul_q8(out, xq, xs, first + c * (weights + scales), first + c * (weights + scales) + weights, n, 0, rows);
  for (let c = 0; c < copies; c++) pass(c);
  const rounds = Math.max(1, Math.round(200e6 / weights));
  const began = performance.now();
  for (let r = 0; r < rounds; r++) pass(r % copies);
  const ms = performance.now() - began;
  return { GBps: (rounds * (weights + scales)) / (ms / 1000) / 1e9, msEach: ms / rounds, threads: 1 };
}

// ---- a token's work: every layer's seven matrices and seven small dispatches, the classifier, the logits back.
// sample: the argmax on the GPU and 4 bytes back instead of the logits
async function token(name, kind = "widen", { sample = false } = {}) {
  await gpu();
  const model = MODELS[name];
  const perLayer = layerMatrices(model), small = SMALL_PER_LAYER;
  const shapes = [...Array(model.layers)].flatMap(() => perLayer);
  const classifier = [model.vocab, model.dim];
  const longest = Math.max(model.dim, model.hidden), most = Math.max(2 * model.hidden, model.vocab);
  const io = vectors(longest, most);
  const made = [];
  let bytes = 0;
  // a buffer the device cannot give fails later and quietly, as an error of these scopes
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    for (const shape of [...shapes, classifier]) {
      const m = matrix(shape, io, kind);
      made.push(m);
      bytes += m.bytes;
    }
  } catch (error) {
    await device.popErrorScope();
    await device.popErrorScope();
    made.forEach((m) => m.owned.forEach((b) => b.destroy()));
    return { model: name, error: `could not hold the weights (${(bytes / 1e9).toFixed(2)} GB made): ${error.message}` };
  }
  const smallPipeline = pipelinesFor().small;
  const a = buffer(model.dim * 4), b = buffer(model.dim * 4);
  const smallGroup = device.createBindGroup({ layout: smallPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
  const picked = sample ? argmaxOf(io, model.vocab) : null;
  const back = buffer(picked ? 4 : model.vocab * 4, MAP_READ | COPY_DST);
  // the uploads may still be going: wait for them, and for a device that could not take them
  await device.queue.onSubmittedWorkDone();
  const invalid = await device.popErrorScope(), outOfMemory = await device.popErrorScope();
  const refused = invalid ?? outOfMemory;
  if (refused) {
    made.forEach((m) => m.owned.forEach((x) => x.destroy()));
    return { model: name, error: `the GPU did not take ${(bytes / 1e9).toFixed(2)} GB of weights: ${refused.message}` };
  }
  const once = async () => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    made.forEach((m, i) => {
      m.dispatches.forEach((d) => run(pass, d));
      // after every layer's last matrix, its small steps (and none after the classifier)
      if (i < shapes.length && i % perLayer.length === perLayer.length - 1) {
        for (let j = 0; j < small; j++) run(pass, [smallPipeline, smallGroup, 1, 1]);
      }
    });
    if (picked) run(pass, picked.dispatch);
    pass.end();
    // the token's id, or every logit for the CPU to sample from
    return picked ? readBack(encoder, picked.chosen, 4, back) : readBack(encoder, io.y, model.vocab * 4, back);
  };
  const ms = await median(once);
  made.forEach((m) => m.owned.forEach((x) => x.destroy()));
  picked?.owned.forEach((x) => x.destroy());
  [a, b, back].forEach((x) => x.destroy());
  destroyVectors(io);
  return { model: name, kind, sample, GB: bytes / 1e9, msPerToken: ms, tokPerSecond: 1000 / ms,
    GBps: bytes / (ms / 1000) / 1e9,
    dispatches: made.reduce((n, m) => n + m.dispatches.length, 0) + model.layers * small + (picked ? 1 : 0) };
}

// ---- T150: one layer of a generated token (Llama 3.2 1B's width, at position LAYER_POS), as its fourteen separate
// steps (RMSNorm, q, k, v, RoPE and the cache, the attention, o, the residual add, RMSNorm, gate, up, SwiGLU, down, the
// residual add: the prompt's shaders and T149's matrix × vector) and fused into five (shaders.js's fusedMatVec: q, k
// and v with the norm and RoPE, the attention, o with the add, gate and up with the norm and SwiGLU, down with the
// add), each with the workgroup's reduction and, where subgroups are, with subgroupAdd. Both forms read the same
// weights: a layer's four matrices (q, k and v one after the other; o; gate and up; down), the separate steps a range
// of rows of them each. Timed as the matrix × vector is (T149): n layers a submission, each on the next copy of the
// weights (copies that make MATVEC_BYTES, so that a layer is not read from the GPU's caches), a submission of 2n less
// one of n (paired). The attention is the prompt's (flashTile, f32 in the workgroup's memory and no subgroups, the
// same in both forms: its cost is in both rows alike).
const LAYER_POS = 127, LAYER_MOST = 4096, EPS = 1e-5, THETA = 500000;
const layerShape = ({ dim, hidden, heads, kvHeads }) => {
  const headSize = dim / heads, kvDim = headSize * kvHeads;
  return { dim, hidden, heads, kvHeads, headSize, kvDim,
    matrices: { qkv: [dim + 2 * kvDim, dim], o: [dim, dim], gateUp: [2 * hidden, dim], down: [dim, hidden] } };
};
const layerForms = () => {
  const subgroups = device.features.has("subgroups") && (navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ?? false);
  return (subgroups ? [false, true] : [false]).flatMap((withSubgroups) => [false, true].map((fused) =>
    ({ name: `${fused ? "fused (T150)" : "separate steps"}${withSubgroups ? ", subgroups" : ""}`, fused, subgroups: withSubgroups })));
};
// the check's verdict of a form, by name
const layerCheck = (form) => `a layer, ${form.name}`;
// a pipeline of its WGSL, made once (in a validation scope: a shader this device refuses rejects there)
const layerPipelines = new Map();
async function compiled(code) {
  if (!layerPipelines.has(code)) {
    layerPipelines.set(code, validated(() => device.createComputePipelineAsync({ layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" } })));
  }
  return layerPipelines.get(code);
}
async function layerPipes(shape, subgroups) {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = device.limits;
  const flash = WGSL.flashShape({ headSize: shape.headSize, half: false, subgroups: false, memory,
    threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
  if (flash.none) throw new Error(flash.none);
  // one at a time: each in an error scope of its own
  const pipes = { small: pipelinesFor().small };
  for (const [key, code] of [["norm", WGSL.RMSNORM], ["rope", WGSL.ROPE], ["swiglu", WGSL.SWIGLU], ["flash", WGSL.flashTile(flash)],
    ["product", WGSL.mulMatVec({ packed: false, subgroups })], ["qkv", WGSL.fusedMatVec({ input: "norm", output: "rope", subgroups })],
    ["add", WGSL.fusedMatVec({ input: "plain", output: "add", subgroups })], ["glu", WGSL.fusedMatVec({ input: "norm", output: "swiglu", subgroups })]]) {
    pipes[key] = await compiled(code);
  }
  return pipes;
}
// A layer's buffers: copies of its four matrices (data: the check's weights, else random), the vectors, the norms'
// weights, the angles of the position, the cache (positions up to pos), and the uniforms. owned: where they go to be
// destroyed. dispatches(form, pipes, copy): one layer's, [pipeline, bind group, x, y] each.
function layerParts(shape, pos, copies, owned, data) {
  const { dim, hidden, heads, kvHeads, headSize, kvDim } = shape;
  const make = (bytes, usage = STORAGE | COPY_DST | COPY_SRC) => {
    const b = buffer(bytes, usage);
    owned.push(b);
    return b;
  };
  const uniform = (bytes) => {
    const b = make(bytes.byteLength, UNIFORM | COPY_DST);
    device.queue.writeBuffer(b, 0, bytes);
    return b;
  };
  const copiesOf = [];
  for (let c = 0; c < copies; c++) {
    const one = {};
    for (const [key, [rows, n]] of Object.entries(shape.matrices)) {
      if (rows * n > device.limits.maxStorageBufferBindingSize) throw new Error(`${key} is past a binding of this device`);
      const w = make(rows * n), s = make((rows * n / GROUP) * 4);
      if (data) {
        device.queue.writeBuffer(w, 0, data[key].w);
        device.queue.writeBuffer(s, 0, data[key].s);
      } else {
        fill(w, rows * n);
        device.queue.writeBuffer(s, 0, floats(rows * n / GROUP, 0.002));
      }
      one[key] = { w, s, rows, n };
    }
    copiesOf.push(one);
  }
  const cacheBytes = (pos + 1) * kvDim * 2;
  const v = { h: make(dim * 4), xb: make(dim * 4), q: make(dim * 4), k: make(kvDim * 4), v: make(kvDim * 4), att: make(dim * 4),
    t: make(dim * 4), g: make(hidden * 4), u: make(hidden * 4), norms: make(2 * dim * 4), angles: make(headSize * 4),
    keys: make(cacheBytes), values: make(cacheBytes) };
  const angles = data?.angles ?? layerAngles(headSize, pos), eps = data?.eps ?? EPS;
  device.queue.writeBuffer(v.angles, 0, angles);
  // the state a layer starts from: the residual stream, the norms' weights, the cache of the positions before
  const reset = (state) => {
    device.queue.writeBuffer(v.h, 0, state.h);
    device.queue.writeBuffer(v.norms, 0, state.norms);
    device.queue.writeBuffer(v.keys, 0, state.keys);
    device.queue.writeBuffer(v.values, 0, state.values);
  };
  reset(data ?? layerState(shape, pos));
  const step = uniform(new Uint32Array([1, pos, 0, 0]));
  const normParams = (at) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes, 0, 2).set([dim, at]);
    new Float32Array(bytes, 8, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const flashParams = new ArrayBuffer(16);
  new Uint32Array(flashParams, 0, 2).set([heads, kvHeads]);
  new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(headSize);
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, pos, qRows, kvRows, headSize, turned
  const fusedParams = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / GROUP, second, 0, normAt, pos, dim, kvDim, headSize, headSize, 0]);
    new Float32Array(bytes, 16, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const u = { step, attentionNorm: normParams(0), ffnNorm: normParams(dim), rope: uniform(new Uint32Array([heads, kvHeads, headSize, headSize])),
    flash: uniform(new Uint8Array(flashParams)), swiglu: uniform(new Uint32Array([hidden, 0, 0, 0])),
    qkv: fusedParams(dim + 2 * kvDim, dim), o: fusedParams(dim, dim), gateUp: fusedParams(hidden, dim, hidden, dim), down: fusedParams(dim, hidden) };
  // the matrix × vector's Shape (rows, words, perRow, first) of each range the separate steps read, made once
  const shapes = new Map();
  const shapeOf = (rows, n) => {
    const key = `${rows},${n}`;
    if (!shapes.has(key)) shapes.set(key, uniform(new Uint32Array([rows, n / 4, n / GROUP, 0])));
    return shapes.get(key);
  };
  const group = (pipeline, entries) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(([binding, resource]) => ({ binding, resource: "offset" in resource ? resource : { buffer: resource } })) });
  const dispatches = (form, pipes, copy) => {
    const m = copiesOf[copy];
    const attention = [pipes.flash, group(pipes.flash, [[0, v.q], [1, v.keys], [2, v.values], [3, v.att], [4, u.flash], [5, u.step]]), heads, 1];
    const groups = (rows) => Math.ceil(rows / WGSL.MUL_MAT_VEC_ROWS);
    if (form.fused) {
      return [
        [pipes.qkv, group(pipes.qkv, [[0, m.qkv.w], [1, m.qkv.s], [2, v.h], [3, u.qkv], [4, v.norms], [5, v.q], [6, v.keys], [7, v.values], [8, v.angles]]), groups(m.qkv.rows), 1],
        attention,
        [pipes.add, group(pipes.add, [[0, m.o.w], [1, m.o.s], [2, v.att], [3, u.o], [5, v.h]]), groups(dim), 1],
        [pipes.glu, group(pipes.glu, [[0, m.gateUp.w], [1, m.gateUp.s], [2, v.h], [3, u.gateUp], [4, v.norms], [5, v.g]]), groups(hidden), 1],
        [pipes.add, group(pipes.add, [[0, m.down.w], [1, m.down.s], [2, v.g], [3, u.down], [5, v.h]]), groups(dim), 1]];
    }
    // rows first to first + rows of a matrix, x into y
    const product = ({ w, s, n }, first, rows, x, y) => [pipes.product, group(pipes.product, [
      [0, { buffer: w, offset: first * n, size: rows * n }], [1, { buffer: s, offset: first * n / 8, size: rows * n / 8 }],
      [2, x], [3, y], [4, shapeOf(rows, n)]]), groups(rows), 1];
    const norm = (params) => [pipes.norm, group(pipes.norm, [[0, v.h], [1, v.norms], [2, v.xb], [3, params], [4, u.step]]), 1, 1];
    const add = [pipes.small, group(pipes.small, [[0, v.t], [1, v.h]]), 1, 1];
    return [norm(u.attentionNorm), product(m.qkv, 0, dim, v.xb, v.q), product(m.qkv, dim, kvDim, v.xb, v.k), product(m.qkv, dim + kvDim, kvDim, v.xb, v.v),
      [pipes.rope, group(pipes.rope, [[0, v.q], [1, v.k], [2, v.v], [3, v.keys], [4, v.values], [5, v.angles], [6, u.rope], [7, u.step]]), 1, 1],
      attention, product(m.o, 0, dim, v.att, v.t), add, norm(u.ffnNorm), product(m.gateUp, 0, hidden, v.xb, v.g),
      product(m.gateUp, hidden, hidden, v.xb, v.u), [pipes.swiglu, group(pipes.swiglu, [[0, v.g], [1, v.u], [2, u.swiglu], [3, u.step]]), Math.ceil(hidden / 64), 1],
      product(m.down, 0, dim, v.g, v.t), add];
  };
  return { vectors: v, reset, dispatches };
}
// the cos of a position's headSize / 2 angles, then their sin (Llama 3's theta, unscaled: any angles would do)
function layerAngles(headSize, pos) {
  const angles = new Float32Array(headSize), half = headSize / 2;
  for (let i = 0; i < half; i++) {
    const angle = pos * THETA ** (-2 * i / headSize);
    angles[i] = Math.cos(angle);
    angles[half + i] = Math.sin(angle);
  }
  return angles;
}
// a made-up state: the residual stream, the norms' weights about 1, and the cache before pos of float16 pairs
function layerState({ dim, kvDim }, pos) {
  const h = floats(dim, 4), norms = new Float32Array(2 * dim).map(() => 0.5 + Math.random());
  const cache = () => new Uint16Array((pos + 1) * kvDim).map(() => toHalf((Math.random() - 0.5) * 4));
  return { h, norms, keys: cache(), values: cache() };
}
// float32 to float16's bits, rounded to the nearest (ties to even), as a cache holds its keys and values
const f32 = new Float32Array(1), bits32 = new Uint32Array(f32.buffer);
function toHalf(value) {
  f32[0] = value;
  const b = bits32[0], sign = (b >>> 16) & 0x8000, exponent = ((b >>> 23) & 0xff) - 112;
  let mantissa = b & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - exponent, kept = mantissa >> shift, rest = mantissa & ((1 << shift) - 1), middle = 1 << (shift - 1);
    return sign | (kept + (rest > middle || (rest === middle && kept & 1) ? 1 : 0));
  }
  const kept = (exponent << 10) | (mantissa >> 13), rest = mantissa & 0x1fff;
  return sign | (kept + (rest > 0x1000 || (rest === 0x1000 && kept & 1) ? 1 : 0));
}
function fromHalf(h) {
  const exponent = (h >> 10) & 31, mantissa = h & 1023, sign = h & 0x8000 ? -1 : 1;
  return sign * (exponent ? 2 ** (exponent - 15) * (1 + mantissa / 1024) : 2 ** -14 * (mantissa / 1024));
}
// The layer in JavaScript (float64 sums), as the CPU's forward pass runs it: what both forms are held to. d: the
// check's weights ({w, s} of each matrix), h, norms, keys, values (float16 bits), angles and eps. Returns the residual stream
// after the layer and the float16 bits of the keys and values of the position
function layerReference({ dim, hidden, heads, kvHeads, headSize, kvDim }, pos, d) {
  const product = ({ w, s }, n, first, rows, x) => {
    const signed = new Int8Array(w.buffer, w.byteOffset, w.length), out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += signed[(first + r) * n + i] * s[((first + r) * n + i) / GROUP | 0] * x[i];
      out[r] = sum;
    }
    return out;
  };
  const normed = (x, at) => {
    let squares = 0;
    for (const value of x) squares += value * value;
    const scale = 1 / Math.sqrt(squares / dim + d.eps);
    return x.map((value, i) => d.norms[at + i] * (scale * value));
  };
  const turned = (vector) => {
    for (let j = 0; j < vector.length; j += 2) {
      const i = (j % headSize) / 2, c = d.angles[i], s = d.angles[headSize / 2 + i], [a, b] = [vector[j], vector[j + 1]];
      vector[j] = a * c - b * s;
      vector[j + 1] = a * s + b * c;
    }
    return vector;
  };
  const h = Float64Array.from(d.h), xb = normed(h, 0);
  const q = turned(product(d.qkv, dim, 0, dim, xb)), k = turned(product(d.qkv, dim, dim, kvDim, xb));
  const v = product(d.qkv, dim, dim + kvDim, kvDim, xb);
  const keys = Uint16Array.from(k, toHalf), values = Uint16Array.from(v, toHalf);
  const cached = (all, row) => (p, i) => fromHalf(p === pos ? row[i] : all[p * kvDim + i]);
  const K = cached(d.keys, keys), V = cached(d.values, values), att = new Float64Array(dim);
  for (let head = 0; head < heads; head++) {
    const kv = Math.floor(head / (heads / kvHeads)) * headSize, scores = [];
    for (let p = 0; p <= pos; p++) {
      let score = 0;
      for (let i = 0; i < headSize; i++) score += q[head * headSize + i] * K(p, kv + i);
      scores.push(score / Math.sqrt(headSize));
    }
    const most = Math.max(...scores), weights = scores.map((score) => Math.exp(score - most)), sum = weights.reduce((a, b) => a + b);
    for (let p = 0; p <= pos; p++) for (let i = 0; i < headSize; i++) att[head * headSize + i] += (weights[p] / sum) * V(p, kv + i);
  }
  const o = product(d.o, dim, 0, dim, att), h1 = h.map((value, i) => value + o[i]), xb2 = normed(h1, dim);
  const gate = product(d.gateUp, dim, 0, hidden, xb2), up = product(d.gateUp, dim, hidden, hidden, xb2);
  const down = product(d.down, hidden, 0, dim, gate.map((g, i) => (g / (1 + Math.exp(-g))) * up[i]));
  return { h: h1.map((value, i) => value + down[i]), keys, values };
}
// The check (T150): every form of the layer on a small one (Llama's shape: GQA, 4 heads of 64 and 2 of K and V; a
// hidden width of 544 = 17 groups, a part of mul_mat_vec's 64 a pass), at position 70 (71 positions, two tiles of the
// attention), against layerReference. The residual stream after it is held to LAYER_LINE of what the layer added to
// it (float32 sums in another order are off by about 1e-6 of it; a wrong index, a norm read from the wrong place, a
// residual left out or gate taken for up by a tenth or more), the key and value of the position to 2e-3 of the largest
// (a float16 rounded the other way is 2^-11 of itself), and the cache's other positions must stay as they were.
// The norm folded into the matrices' read (fusedMatVec) is held by the stream and the eps the check starts from: the
// stream is about ±2 with three channels at ±30 (a real stream has such channels: T92's GPT-2 at 1000× the median),
// so the norm's scale is far from 1 (about 0.28: a scale left out shows, and the sum of x² has a few large terms among
// many small), and the check's eps is about a twelfth of the mean of x² (a model's 1e-5 would hide a wrong or missing
// eps under LAYER_LINE; the timing keeps EPS). Larger channels (±60) make the attention's softmax steep enough that a
// key or value of the position rounded the other way in float16 moves the stream by up to 2e-4 (lavapipe, 2026-09-27);
// at ±30 both forms stay within 1e-6 over 45 draws
const LAYER_CHECK = { dim: 256, hidden: 544, heads: 4, kvHeads: 2 }, LAYER_CHECK_POS = 70, LAYER_LINE = 1e-3, CACHE_LINE = 2e-3;
const LAYER_CHECK_OUTLIERS = 3, LAYER_CHECK_OUTLIER = 30, LAYER_CHECK_EPS = 1;
async function checkLayer() {
  const shape = layerShape(LAYER_CHECK), pos = LAYER_CHECK_POS, verdicts = {};
  const data = { ...layerState(shape, pos), angles: layerAngles(shape.headSize, pos), eps: LAYER_CHECK_EPS };
  for (let i = 0; i < LAYER_CHECK_OUTLIERS; i++) {
    data.h[Math.floor((i + 0.5) * shape.dim / LAYER_CHECK_OUTLIERS)] = LAYER_CHECK_OUTLIER * (i % 2 ? -1 : 1);
  }
  for (const [key, [rows, n]] of Object.entries(shape.matrices)) {
    data[key] = { w: new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s: floats(rows * n / GROUP, 0.01) };
  }
  const want = layerReference(shape, pos, data);
  let added = 0, largestKey = 0, largestValue = 0;
  want.h.forEach((value, i) => (added = Math.max(added, Math.abs(value - data.h[i]))));
  want.keys.forEach((bits) => (largestKey = Math.max(largestKey, Math.abs(fromHalf(bits)))));
  want.values.forEach((bits) => (largestValue = Math.max(largestValue, Math.abs(fromHalf(bits)))));
  for (const form of layerForms()) {
    try {
      const got = await scoped(async (owned) => {
        const pipes = await layerPipes(shape, form.subgroups);
        const parts = layerParts(shape, pos, 1, owned, data);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        parts.dispatches(form, pipes, 0).forEach((d) => run(pass, d));
        pass.end();
        const h = new Float32Array(await readBack(encoder, parts.vectors.h, shape.dim * 4));
        const cacheBytes = (pos + 1) * shape.kvDim * 2;
        return { h, keys: new Uint16Array(await readBack(device.createCommandEncoder(), parts.vectors.keys, cacheBytes)),
          values: new Uint16Array(await readBack(device.createCommandEncoder(), parts.vectors.values, cacheBytes)) };
      });
      let off = 0, keyOff = 0, valueOff = 0, touched = false;
      got.h.forEach((value, i) => (off = Math.max(off, Math.abs(value - want.h[i]) / added)));
      for (let p = 0; p <= pos; p++) {
        for (let i = 0; i < shape.kvDim; i++) {
          const at = p * shape.kvDim + i;
          if (p === pos) {
            keyOff = Math.max(keyOff, Math.abs(fromHalf(got.keys[at]) - fromHalf(want.keys[i])) / largestKey);
            valueOff = Math.max(valueOff, Math.abs(fromHalf(got.values[at]) - fromHalf(want.values[i])) / largestValue);
          } else touched ||= got.keys[at] !== data.keys[at] || got.values[at] !== data.values[at];
        }
      }
      const cache = Math.max(keyOff, valueOff);
      verdicts[layerCheck(form)] = { worstRelative: Math.max(off, cache), ok: off < LAYER_LINE && cache < CACHE_LINE && !touched,
        stream: off, cache, ...(touched ? { wroteOtherPositions: true } : {}) };
    } catch (error) {
      verdicts[layerCheck(form)] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles each form's shaders for tens of s
      postMessage({ alive: true });
    }
  }
  return verdicts;
}
async function layer() {
  await gpu();
  const shape = layerShape(MODELS["Llama 3.2 1B"]);
  const bytes = Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0);
  const copies = fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes), rows = [];
  await scoped(async (owned) => {
    const parts = layerParts(shape, LAYER_POS, copies, owned);
    await device.queue.onSubmittedWorkDone();
    for (const form of layerForms()) {
      const row = { form: form.name, check: layerCheck(form), fused: form.fused, subgroups: form.subgroups };
      try {
        const pipes = await layerPipes(shape, form.subgroups);
        const each = [...Array(copies)].map((_, copy) => parts.dispatches(form, pipes, copy));
        // n layers a submission, each on the next copy
        let next = 0;
        const submission = async (n) => {
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          for (let i = 0; i < n; i++) each[next++ % copies].forEach((d) => run(pass, d));
          pass.end();
          const began = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          return performance.now() - began;
        };
        const r = await validated(async () => {
          if (fallback) return { ms: await submission(1), dispatches: 1 };
          await submission(2);
          return paired(submission, LAYER_MOST);
        });
        rows.push({ ...row, dispatches: each[0].length, msPerLayer: r.ms / r.dispatches, layers: r.dispatches,
          ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) });
      } catch (error) {
        rows.push({ ...row, error: String(error?.message ?? error) });
      } finally {
        postMessage({ alive: true });
      }
    }
  });
  return { model: "Llama 3.2 1B", pos: LAYER_POS, layers: MODELS["Llama 3.2 1B"].layers, copies, GB: bytes / 1e9, rows };
}

// ---- what a token costs besides its weights: the dispatches of Llama 3.2 1B's token (seven matrices and seven small
// steps a layer, 16 layers, and the classifier: about 240) doing nothing, a submission with and without waiting for
// the GPU, and reading back the id of a token against Llama 3's 128256 logits
async function overhead() {
  await gpu();
  const { empty } = pipelinesFor();
  const dispatches = 240, vocab = MODELS["Llama 3.2 1B"].vocab;
  const submit = (count, wait) => async () => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    for (let i = 0; i < count; i++) run(pass, [empty, null, 1, 1]);
    pass.end();
    device.queue.submit([encoder.finish()]);
    if (wait) await device.queue.onSubmittedWorkDone();
  };
  const logits = buffer(vocab * 4, STORAGE | COPY_SRC | COPY_DST);
  device.queue.writeBuffer(logits, 0, floats(vocab));
  const targets = { 4: buffer(4, MAP_READ | COPY_DST), [vocab * 4]: buffer(vocab * 4, MAP_READ | COPY_DST) };
  const read = (bytes) => () => readBack(device.createCommandEncoder(), logits, bytes, targets[bytes]);
  const found = {
    dispatches, emptyDispatches: await median(submit(dispatches, true)),
    submitOnly: await median(submit(1, false)), submitAndWait: await median(submit(1, true)),
    readToken: await median(read(4)), readLogits: await median(read(vocab * 4)), vocab,
  };
  [logits, ...Object.values(targets)].forEach((x) => x.destroy());
  return found;
}

// ---- the ceilings (T168): each loop of shaders.js alone. First its loop count doubles until one dispatch takes
// DISPATCH_MS (a dispatch's own cost is then small beside its work), then the dispatches of a submission until it
// takes SUBMISSION_MS. The time of n dispatches is that of a submission of 2n less one of n: what a submission costs
// besides its work (3.7 ms waited for on the owner's Android, T134) is in both and drops out. The two are measured
// in turn, PAIRS times, and the median of the pairs' differences taken: measured one after the other, a device's
// load moved the numbers by several times (T168's review, lavapipe under load: 4 of 55 readings 30% or more off,
// one ten times). A pair's 2n should take about twice its n: past STEADY the pairs are taken again once, and if
// they are still past it the ceiling says "unsteady" and the prompt is not held against it. Everything runs in
// error scopes, validation and out of memory: a pipeline, bind group or buffer the device refused makes an error,
// not a number (a bind group that does not match took no time, and read as 13,915 GB/s). So does a loop that never
// takes DISPATCH_MS in MOST_LOOPS or SUBMISSION_MS in MOST_DISPATCHES (a loop a compiler removed: lavapipe then read
// 60 million GFLOPS). A fallback adapter runs the same (its speed is no GPU's, but a few seconds
// a ceiling)
const DISPATCH_MS = 2, SUBMISSION_MS = 40, PAIRS = 5, STEADY = [1.6, 2.2];
// MOST_LOOPS: a dispatch of 65536 threads that many loops is 4 TFLOP of multiply-adds, 2 s at 2 TFLOPS
const CEILING_GROUPS = 256, FIRST_LOOPS = 4, MOST_LOOPS = 1 << 14, MOST_DISPATCHES = 4096;
// the storage buffer the global read streams through: 128 MiB, WebGPU's default largest binding (or the device's
// largest, if smaller), meant to be larger than the GPU's caches (their sizes on the owner's devices are not measured)
const GLOBAL_BYTES = 128 << 20;
const middle = (values) => [...values].sort((x, y) => x - y)[values.length >> 1];
async function ceilings() {
  await gpu();
  const threads = CEILING_GROUPS * WGSL.CEILING_WORKGROUP;
  const globalBytes = Math.min(GLOBAL_BYTES, device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const globalThreads = Math.floor(globalBytes / WGSL.GLOBAL_PER_THREAD / WGSL.CEILING_WORKGROUP) * WGSL.CEILING_WORKGROUP;
  // what a loop counts (FLOPs, ops or bytes) a second: perLoop a thread and a loop, or perDispatch (the global read)
  const rate = (code, { perLoop, perDispatch, groups = CEILING_GROUPS, bytes = 0 }) => scoped(async (owned) => {
    const out = buffer(Math.max(threads, groups * WGSL.CEILING_WORKGROUP) * 4, STORAGE), plan = buffer(16, UNIFORM | COPY_DST);
    owned.push(out, plan);
    const more = bytes ? [buffer(bytes)] : [];
    owned.push(...more);
    if (bytes) fill(more[0], bytes);
    let loops = FIRST_LOOPS;
    const setLoops = () => device.queue.writeBuffer(plan, 0, new Uint32Array([loops, (Math.random() * 2 ** 32) >>> 0,
      new Uint32Array(new Float32Array([0.999]).buffer)[0], new Uint32Array(new Float32Array([0.001]).buffer)[0]]));
    setLoops();
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [out, plan, ...more].map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const submission = async (n) => {
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let i = 0; i < n; i++) run(pass, [pipeline, group, groups, 1]);
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    await submission(1);
    // a dispatch of DISPATCH_MS (8 of them, so that the submission's own cost is an eighth in each)
    if (perLoop) {
      while ((await submission(8)) / 8 < DISPATCH_MS) {
        if ((loops *= 2) > MOST_LOOPS) throw new Error(`${MOST_LOOPS} loops took less than ${DISPATCH_MS} ms a dispatch: the loop did no work`);
        setLoops();
      }
    }
    const r = await paired(submission, MOST_DISPATCHES, true);
    if (r.short) throw new Error(`${MOST_DISPATCHES} dispatches took less than ${SUBMISSION_MS} ms: the loop did no work`);
    const work = r.dispatches * (perLoop ? threads * loops * perLoop : perDispatch);
    return { rate: work / (r.ms / 1000), loops, dispatches: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) };
  });
  const found = { fallback };
  const measure = async (name, why, how) => {
    if (why) return (found[name] = { none: why });
    try {
      found[name] = await how();
    } catch (error) {
      found[name] = { error: String(error?.message ?? error) };
    } finally {
      postMessage({ alive: true });
    }
  };
  // the multiply-adds in both shapes; the faster is the ceiling (an unsteady one only when both are)
  const fma = async (half) => {
    const shapes = [];
    for (const shape of WGSL.FMA_SHAPES) shapes.push({ shape, ...await rate(WGSL.fmaCeiling(half, shape), { perLoop: WGSL.FMA_PER_LOOP }) });
    const best = [...shapes].sort((x, y) => Boolean(x.unsteady) - Boolean(y.unsteady) || y.rate - x.rate)[0];
    return { GFLOPS: best.rate / 1e9, shape: best.shape, unsteady: best.unsteady, shapes: shapes.map(({ shape, rate: r }) => ({ shape, GFLOPS: r / 1e9 })) };
  };
  const scaled = (key, r) => ({ [key]: r.rate / 1e9, unsteady: r.unsteady, loops: r.loops, dispatches: r.dispatches, ratio: r.ratio });
  await measure("f32", null, () => fma(false));
  await measure("f16", device.features.has("shader-f16") ? null : "no shader-f16 here", () => fma(true));
  await measure("dot4", packed ? null : "no packed int8 dot here",
    async () => scaled("GOPS", await rate(WGSL.DOT4_CEILING, { perLoop: WGSL.DOT4_PER_LOOP })));
  await measure("shared", null, async () => scaled("GBps", await rate(WGSL.SHARED_CEILING, { perLoop: WGSL.SHARED_PER_LOOP })));
  await measure("global", null, async () => ({ MiB: globalThreads * WGSL.GLOBAL_PER_THREAD / 2 ** 20,
    ...scaled("GBps", await rate(WGSL.GLOBAL_CEILING, { perDispatch: globalThreads * WGSL.GLOBAL_PER_THREAD,
      groups: globalThreads / WGSL.CEILING_WORKGROUP, bytes: globalThreads * WGSL.GLOBAL_PER_THREAD })) }));
  return found;
}
// the time of n dispatches (what submission(n) submits and waits for, in ms): n doubles, up to most, until a submission
// takes SUBMISSION_MS (short: it never did), then PAIRS submissions of n and of 2n in turn, and the median of their
// differences; past STEADY the pairs once more, then unsteady, or an error where 2n took no longer than n (lavapipe
// under load once read a negative time, T149). T168's, and T149's matrix × vector. strict: none of the pairs when
// most never took SUBMISSION_MS (a loop that did no work), only { short: true }
async function paired(submission, most, strict = false) {
  let n = 1, ms;
  while ((ms = await submission(n)) < SUBMISSION_MS && n < most) n *= 2;
  if (strict && ms < SUBMISSION_MS) return { short: true };
  for (let tries = 0; ; tries++) {
    const differences = [], ratios = [];
    for (let i = 0; i < PAIRS; i++) {
      const once = await submission(n), twice = await submission(2 * n);
      differences.push(twice - once);
      ratios.push(twice / once);
      postMessage({ alive: true });
    }
    // a median difference of 0 or less (2n no longer than n: the load moved) is no time at all, never steady
    const ratio = middle(ratios), took = middle(differences), steady = took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1];
    if (steady) return { ms: took, dispatches: n, ratio };
    if (tries) {
      if (!(took > 0)) throw new Error(`a submission of ${2 * n} took no longer than one of ${n}: the device's load moved`);
      return { ms: took, dispatches: n, ratio, unsteady: true };
    }
  }
}
// what fn does on the GPU (given an array for the buffers it makes, destroyed after), with an error of the device
// thrown: validation (a pipeline or bind group refused) or out of memory (a buffer it could not give)
async function scoped(fn) {
  const owned = [];
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let result, failure;
  try {
    result = await fn(owned);
  } catch (error) {
    failure = error;
  }
  const invalid = await device.popErrorScope(), outOfMemory = await device.popErrorScope();
  owned.forEach((b) => b.destroy());
  if (invalid ?? outOfMemory) throw new Error((invalid ?? outOfMemory).message);
  if (failure) throw failure;
  return result;
}

// ---- a prompt: its tokens through the matrices of the CPU section's made-up model (two layers of Llama 3.2 1B's
// width, no classifier: a prompt's tokens make no logits) all at once, count tokens at a time, with the small steps
// once a layer as for one token. ms per token, for every shader of promptShaders() (T146: the tiled ones, whose rows
// say none or error on their own where they cannot run), and the GFLOPS of it: a multiply and an add for each weight
// and token. A packed shader's input is quantized first where a matrix reads an input of its own (q: the norm's; o:
// the attention's; gate: the norm's; down: SwiGLU's), as the model's layers would
const NEW_INPUT = new Set([0, 3, 4, 6]);
async function prompt(counts = [1, 16, 64]) {
  await gpu();
  // a fallback adapter measures one count, the block of 16 the CPU also takes: each shader and count takes 10 to 100 s
  // there (SwiftShader on the development machine, T146: 580 s for all three), and its times are no GPU's anyway
  if (fallback) counts = counts.filter((tokens) => tokens === 16).slice(0, 1);
  const model = PROMPT_MODEL, perLayer = layerMatrices(model), shapes = [...Array(model.layers)].flatMap(() => perLayer);
  const weights = shapes.reduce((sum, [rows, n]) => sum + rows * n, 0);
  const longest = Math.max(model.dim, model.hidden), most = model.hidden;
  const smallPipeline = pipelinesFor().small, a = buffer(model.dim * 4), b = buffer(model.dim * 4);
  const smallGroup = device.createBindGroup({ layout: smallPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a } }, { binding: 1, resource: { buffer: b } }] });
  const measure = async (kind, tokens) => {
    const io = vectors(longest, most, tokens), owned = [];
    try {
      return await validated(async () => {
        const made = shapes.map((shape) => matrix(shape, io, kind));
        // one quantizer a width (q, k, v and gate, up read the same width: one each, not one a matrix left unowned)
        const quantize = kind.packed ? new Map([...new Set(perLayer.map(([, n]) => n))].map((n) => [n, quantizer(io, n)])) : null;
        owned.push(...made.flatMap((m) => m.owned), ...[...(quantize?.values() ?? [])].flatMap((q) => q.owned));
        await device.queue.onSubmittedWorkDone();
        const once = async () => {
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          made.forEach((m, i) => {
            if (quantize && NEW_INPUT.has(i % perLayer.length)) run(pass, quantize.get(shapes[i][1]).dispatch);
            m.dispatches.forEach((d) => run(pass, d));
            if (i % perLayer.length === perLayer.length - 1) for (let j = 0; j < SMALL_PER_LAYER; j++) run(pass, [smallPipeline, smallGroup, 1, 1]);
          });
          pass.end();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
        };
        const ms = await median(once, 5, 2);
        return { tokens, ms, msPerToken: ms / tokens, GFLOPS: (2 * weights * tokens) / (ms / 1000) / 1e9 };
      });
    } finally {
      owned.forEach((x) => x.destroy());
      destroyVectors(io);
      // the page stops a section that says nothing for 5 minutes, and a fallback adapter takes minutes for all of these
      postMessage({ alive: true });
    }
  };
  const rows = [];
  for (const shader of promptShaders()) {
    if (shader.none) {
      rows.push({ shader: shader.name, none: shader.none });
      continue;
    }
    try {
      const kind = await kindOf(shader);
      // what the row's GFLOPS are held against (T168): the dot4I8Packed ceiling, the f16 or the f32 one
      for (const tokens of counts) rows.push({ shader: shader.name, packed: shader.packed, half: shader.half, ...await measure(kind, tokens) });
    } catch (error) {
      rows.push({ shader: shader.name, error: String(error?.message ?? error) });
    }
  }
  // the batched shader once more at the most tokens, last: a device that has warmed up and slowed down since shows it
  // here, beside the same shader's row at the start (T146's review)
  const last = counts[counts.length - 1];
  if (last) rows.push({ shader: "batched (T135), again at the end", again: true, ...await measure("batched", last) });
  [a, b].forEach((x) => x.destroy());
  return { rows, layers: model.layers, weights, GB: shapes.reduce((sum, shape) => sum + matrixBytes(shape), 0) / 1e9 };
}

// ---- the bridge: a worker that waits (Atomics.wait, as the model's worker would while Python calls forward())
// and this one, which must not block (it awaits the GPU): Atomics.waitAsync where there is one
async function bridge(memory, rounds) {
  const words = new Int32Array(memory);
  const hasWaitAsync = typeof Atomics.waitAsync === "function";
  if (!hasWaitAsync) return { waitAsync: false };
  // words[0]: the request counter (the waiter adds 1), words[1]: the answer counter (this one sets it)
  let answered = 0;
  while (answered < rounds) {
    const seen = Atomics.load(words, 0);
    if (seen === answered) {
      const result = Atomics.waitAsync(words, 0, seen);
      if (result.async) await result.value;
      continue;
    }
    answered = seen;
    Atomics.store(words, 1, answered);
    Atomics.notify(words, 1);
  }
  return { waitAsync: true };
}

onmessage = async ({ data }) => {
  try {
    WGSL ??= await shaders;
    ({ GROUP, TILE } = WGSL);
    let result;
    if (data.step === "info") result = await info();
    else if (data.step === "check") result = await check();
    else if (data.step === "bandwidth") result = await bandwidth(data.shape);
    else if (data.step === "token") result = await token(data.model, data.kind, data);
    else if (data.step === "layer") result = await layer();
    else if (data.step === "overhead") result = await overhead();
    else if (data.step === "prompt") result = await prompt(data.counts);
    else if (data.step === "ceilings") result = await ceilings();
    else if (data.step === "bridge") result = await bridge(data.memory, data.rounds);
    postMessage({ step: data.step, result });
  } catch (error) {
    postMessage({ step: data.step, error: String(error?.message ?? error) });
  }
};
