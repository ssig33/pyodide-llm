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
//                                         steps and fused into five dispatches (shaders.js's fusedMatVec), ms a layer;
//                                         T175: the same on ORT's DP4A for small M (fusedDp4aMatVec), its vector
//                                         quantized before each matrix
//   { step: "layer steps" }               T202: where the time of a layer goes: each step of the fused forms a token
//                                         would run here alone, the matrices alone, a dispatch that does next to
//                                         nothing, and the whole layer, timed in turn, ms each (T208: the forms the
//                                         layer table's fastest, and the whole layer by timestamps where they are)
//   { step: "generate" }                  T151: tokens generated on the GPU (the sampling too), each read back as it
//                                         comes against 4, 8 and 16 a submission read back once, ms a token; T191:
//                                         each also with the sampling in chunks (many workgroups), and the sampling
//                                         alone both ways
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
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"].filter((name) => adapter.features.has(name)),
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
  verdicts.sampling = await checkSampling();
  verdicts["sampling in chunks"] = await checkSampling("chunks");
  Object.assign(verdicts, await checkGeneration());
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
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles a shader for up to 90 s, and
      // the checks before the layer's took 380 s together under load (T151)
      postMessage({ alive: true });
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
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles a shader for up to 90 s, and
      // the checks before the layer's took 380 s together under load (T151)
      postMessage({ alive: true });
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
// residual add: the prompt's shaders and T149's matrix × vector), fused into five (shaders.js's fusedMatVec: q, k
// and v with the norm and RoPE, the attention, o with the add, gate and up with the norm and SwiGLU, down with the
// add), each with the workgroup's reduction and, where subgroups are, with subgroupAdd; and T175's forms on DP4A
// below (LAYER_KINDS). All read the same weights: a layer's four matrices (q, k and v one after the other; o; gate and up; down), the separate
// steps a range of rows of them each. Timed as the matrix × vector is (T149), n layers a submission, each on the next
// copy of the weights (copies that make MATVEC_BYTES, so that a layer is not read from the GPU's caches), a submission
// of 2n less one of n, but the forms in turn (interleaved: T150's review, as T147's timeForms), so that a device that
// warms up and slows down does so for all of them alike. The attention is the prompt's (flashTile, f32 in the
// workgroup's memory and no subgroups, the same in every form: its cost is in every row alike).
const LAYER_POS = 127, LAYER_MOST = 4096, EPS = 1e-5, THETA = 500000;
const layerShape = ({ dim, hidden, heads, kvHeads }) => {
  const headSize = dim / heads, kvDim = headSize * kvHeads;
  return { dim, hidden, heads, kvHeads, headSize, kvDim,
    matrices: { qkv: [dim + 2 * kvDim, dim], o: [dim, dim], gateUp: [2 * hidden, dim], down: [dim, hidden] } };
};
// T175: the same on ONNX Runtime's DP4A for small M (the matrix × vector that read Llama 3.2 1B's w1 at 96.8% of the
// buffer's reads on the owner's Android, T149, where llama.cpp's read a layer at 18.5%, T150): its vector quantized
// before each matrix (QUANTIZE, four a layer), as separate steps (eighteen dispatches), fused but for the norms (eleven:
// the matrices with T150's writes, RMSNORM and QUANTIZE apart) and fused (nine: the norm with the quantizer,
// NORM_QUANTIZE). Where a device has no packed int8 dot, those rows say so. llama.cpp's rows stay beside them.
// base: whose matrix × vector; fused: the matrices with T150's writes (fusedMatVec, fusedDp4aMatVec); normApart: the
// norms dispatches of their own. T150's llama.cpp form with the norms apart runs only where DP4A cannot (withoutDp4a:
// T175's review): there it is the layer a device without the packed int8 dot chooses from (the owner's Android on
// llama.cpp's matrix: separate 9.27 ms, the norms apart 9.92, fused 10.61, T150); elsewhere the DP4A one stands for what
// folding the norm saves.
const LAYER_KINDS = [{ name: "llama.cpp, separate steps", base: "llama.cpp", fused: false },
  { name: "llama.cpp, fused (T150), the norms apart", base: "llama.cpp", fused: true, normApart: true, withoutDp4a: true },
  { name: "llama.cpp, fused (T150)", base: "llama.cpp", fused: true },
  { name: "DP4A, separate steps", base: "DP4A", dp4a: true, fused: false },
  { name: "DP4A, fused (T175), the norms apart", base: "DP4A", dp4a: true, fused: true, normApart: true },
  { name: "DP4A, fused (T175)", base: "DP4A", dp4a: true, fused: true }];
const layerForms = () => {
  const subgroups = device.features.has("subgroups") && (navigator.gpu.wgslLanguageFeatures?.has("subgroup_id") ?? false);
  const llama = LAYER_KINDS.filter((kind) => !kind.dp4a && !(kind.withoutDp4a && packed)), dp4a = LAYER_KINDS.filter((kind) => kind.dp4a);
  return [...(subgroups ? [false, true] : [false]).flatMap((withSubgroups) => llama.map((kind) =>
    ({ ...kind, name: `${kind.name}${withSubgroups ? ", subgroups" : ""}`, subgroups: withSubgroups }))),
    ...dp4a.map((kind) => ({ ...kind, subgroups: false, ...(packed ? {} : { none: "no packed int8 dot here" }) }))];
};
// the layer a token runs on the GPU (generate(), T151): T175's fused DP4A where the check found it right (checkLayer
// ran in this worker and its verdict is ok: T175's review), else T150's fused one (generate() builds the fused forms
// only; which is fastest on the device is the layer table's, and the engine's choice is T152's)
let layerVerdicts;
// T208: the layer table's rows (layer()), for the steps' table to break down the layer a token would run here
let layerTimes;
const DP4A_TOKEN = "DP4A, fused (T175)";
const tokenForm = () => {
  const dp4a = packed && layerVerdicts?.[`a layer, ${DP4A_TOKEN}`]?.ok === true;
  return { ...LAYER_KINDS.find((kind) => (dp4a ? kind.name === DP4A_TOKEN : kind.name === "llama.cpp, fused (T150)")), subgroups: false };
};
// what a form of the layer dispatches besides the attention, [key, WGSL] each (layer() and generate())
const layerCodes = ({ dp4a, fused, normApart, subgroups }) => {
  if (!dp4a) {
    const input = normApart ? "plain" : "norm";
    return fused ? [...(normApart ? [["norm", WGSL.RMSNORM]] : []), ["qkv", WGSL.fusedMatVec({ input, output: "rope", subgroups })],
      ["add", WGSL.fusedMatVec({ input: "plain", output: "add", subgroups })], ["glu", WGSL.fusedMatVec({ input, output: "swiglu", subgroups })]]
      : [["norm", WGSL.RMSNORM], ["rope", WGSL.ROPE], ["swiglu", WGSL.SWIGLU], ["product", WGSL.mulMatVec({ packed: false, subgroups })]];
  }
  if (!fused) return [["quantize", WGSL.QUANTIZE], ["norm", WGSL.RMSNORM], ["rope", WGSL.ROPE], ["swiglu", WGSL.SWIGLU], ["product", WGSL.ortDp4aMatVec]];
  return [["quantize", WGSL.QUANTIZE], normApart ? ["norm", WGSL.RMSNORM] : ["normQuantize", WGSL.NORM_QUANTIZE],
    ["qkv", WGSL.fusedDp4aMatVec({ output: "rope" })], ["add", WGSL.fusedDp4aMatVec({ output: "add" })], ["glu", WGSL.fusedDp4aMatVec({ output: "swiglu" })]];
};
// T175: a vector of n values quantized (QUANTIZE) into { xq, xs }: a thread a group of 32
const quantizing = (pipes, group, x, into, uniform, n, step) => [pipes.quantize, group(pipes.quantize, [[0, x], [1, into.xq], [2, into.xs], [3, uniform], [4, step]]), Math.ceil(n / GROUP / 64), 1];
// The dispatches of one layer of a token in a fused form (layer() and generate()): T150's five (fusedMatVec), or on
// DP4A (T175) the norm and its quantizing (NORM_QUANTIZE, or with the norms apart RMSNORM and QUANTIZE), q, k and v
// with RoPE and the cache, the attention, its output quantized, o with the add, the norm and its quantizing, gate and up
// with SwiGLU, its output quantized, down with the add. m: the layer's matrices ({w, s, rows} each); cache: its keys and
// values; v: the vectors (h, xb, q, att, g, the norms' weights, the angles, and quantized: the { xq, xs } of each
// quantization, q, k and v's, o's, gate and up's, down's); u: the uniforms (step, flash, the matrices' Params, the norms'
// attentionNorm and ffnNorm, and quantize: QUANTIZE's of dim and of hidden); group(pipeline, entries): a bind group
function fusedLayer(form, pipes, { dim, hidden, heads }, m, cache, v, u, group) {
  const attention = attentionStep(pipes, heads, cache, v, u, group);
  const rope = [[5, v.q], [6, cache.keys], [7, cache.values], [8, v.angles], [9, u.step]];
  // RMSNORM of the stream into xb (the forms with the norms apart)
  const norm = (params) => named("the norm (RMSNORM)", "small", [pipes.norm, group(pipes.norm, [[0, v.h], [1, v.norms], [2, v.xb], [3, params], [4, u.step]]), 1, 1]);
  if (!form.dp4a) {
    const groups = (rows) => Math.ceil(rows / WGSL.MUL_MAT_VEC_ROWS);
    // the matrices read the stream and norm it, or (the norms apart) read xb
    const input = form.normApart ? [[2, v.xb]] : [[2, v.h], [4, v.norms]];
    const withNorm = form.normApart ? "" : " the norm,";
    return [...(form.normApart ? [norm(u.attentionNorm)] : []),
      named(`q, k and v with${withNorm} RoPE and the cache`, "matrix", [pipes.qkv, group(pipes.qkv, [[0, m.qkv.w], [1, m.qkv.s], ...input, [3, u.qkv], ...rope]), groups(m.qkv.rows), 1], "qkv"),
      attention,
      named("o with the residual's add", "matrix", [pipes.add, group(pipes.add, [[0, m.o.w], [1, m.o.s], [2, v.att], [3, u.o], [5, v.h]]), groups(dim), 1], "o"),
      ...(form.normApart ? [norm(u.ffnNorm)] : []),
      named(`gate and up with${withNorm} SwiGLU`, "matrix", [pipes.glu, group(pipes.glu, [[0, m.gateUp.w], [1, m.gateUp.s], ...input, [3, u.gateUp], [5, v.g]]), groups(hidden), 1], "gateUp"),
      named("down with the residual's add", "matrix", [pipes.add, group(pipes.add, [[0, m.down.w], [1, m.down.s], [2, v.g], [3, u.down], [5, v.h]]), groups(dim), 1], "down")];
  }
  const [qkvIn, oIn, gluIn, downIn] = v.quantized;
  const quantized = (x, into, params, n) => named(`a vector of ${n} quantized (QUANTIZE)`, "small", quantizing(pipes, group, x, into, params, n, u.step));
  const normed = (params, into) => (form.normApart
    ? [norm(params), quantized(v.xb, into, u.quantize[0], dim)]
    : [named("the norm with its quantizing (NORM_QUANTIZE)", "small", [pipes.normQuantize, group(pipes.normQuantize, [[0, v.h], [1, v.norms], [2, into.xq], [3, into.xs], [4, params], [5, u.step]]), 1, 1])]);
  // a matrix by its quantized input: rows its workgroups take (gate's, where up's go beside them)
  const product = (step, matrix, pipeline, { w, s }, into, params, rows, output) => named(step, "matrix", [pipeline, group(pipeline, [[0, w], [1, s], [2, into.xq], [3, params], [4, into.xs], ...output]),
    Math.ceil(rows / WGSL.ORT_DP4A_MATVEC_ROWS), 1], matrix);
  return [...normed(u.attentionNorm, qkvIn), product("q, k and v with RoPE and the cache", "qkv", pipes.qkv, m.qkv, qkvIn, u.qkv, m.qkv.rows, rope),
    attention, quantized(v.att, oIn, u.quantize[0], dim), product("o with the residual's add", "o", pipes.add, m.o, oIn, u.o, dim, [[5, v.h]]),
    ...normed(u.ffnNorm, gluIn), product("gate and up with SwiGLU", "gateUp", pipes.glu, m.gateUp, gluIn, u.gateUp, hidden, [[5, v.g]]),
    quantized(v.g, downIn, u.quantize[1], hidden), product("down with the residual's add", "down", pipes.add, m.down, downIn, u.down, dim, [[5, v.h]])];
}
// the attention of a fused layer (flash attention's tile) on cache ({ keys, values }: buffers, or T208's ranges of one)
function attentionStep(pipes, heads, cache, v, u, group) {
  return named("the attention (flash attention's tile)", "attention", [pipes.flash, group(pipes.flash, [[0, v.q], [1, cache.keys], [2, cache.values], [3, v.att], [4, u.flash], [5, u.step]]), heads, 1]);
}
// T202: a dispatch of a fused layer with what it is (run() reads only its first five): step, its name in the steps'
// table (the same name where two dispatches cost the same: the two norms, the quantizing of a vector of dim); kind,
// "matrix" (with what is folded into its write), "attention" or "small"; matrix, the layer's matrix it reads
function named(step, kind, dispatch, matrix) {
  return Object.assign(dispatch, { step, kind, ...(matrix ? { matrix } : {}) });
}
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
// the pipelines a form of the layer runs (what layerCodes() says, the attention, and SMALL for the separate steps' adds)
async function layerPipes(shape, form) {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = device.limits;
  const flash = WGSL.flashShape({ headSize: shape.headSize, half: false, subgroups: false, memory,
    threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
  if (flash.none) throw new Error(flash.none);
  // one at a time: each in an error scope of its own
  const pipes = { small: pipelinesFor().small };
  for (const [key, code] of [["flash", WGSL.flashTile(flash)], ...layerCodes(form)]) pipes[key] = await compiled(code);
  return pipes;
}
// A layer's buffers: copies of its four matrices (data: the check's weights, else random), the vectors, the norms'
// weights, the angles of the position, the cache (positions up to pos), and the uniforms. owned: where they go to be
// destroyed. dispatches(form, pipes, copy): one layer's, [pipeline, bind group, x, y] each. reach (T208, the steps'
// table): the bytes every matrix's ranges (ranges()) make at the least, spare buffers of random weights beside the
// copies where they fall short, and the attention's caches (caches()) as many.
function layerParts(shape, pos, copies, owned, data, reach = 0) {
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
  // a matrix's weights and scales, the check's (given) or random
  const matrixOf = (key, [rows, n], given) => {
    if (rows * n > device.limits.maxStorageBufferBindingSize) throw new Error(`${key} is past a binding of this device`);
    const w = make(rows * n), s = make((rows * n / GROUP) * 4);
    if (given) {
      device.queue.writeBuffer(w, 0, given.w);
      device.queue.writeBuffer(s, 0, given.s);
    } else {
      fill(w, rows * n);
      device.queue.writeBuffer(s, 0, floats(rows * n / GROUP, 0.002));
    }
    return { w, s, rows, n };
  };
  const copiesOf = [];
  for (let c = 0; c < copies; c++) {
    copiesOf.push(Object.fromEntries(Object.entries(shape.matrices).map(([key, size]) => [key, matrixOf(key, size, data?.[key])])));
  }
  const cacheBytes = (pos + 1) * kvDim * 2;
  const v = { h: make(dim * 4), xb: make(dim * 4), q: make(dim * 4), k: make(kvDim * 4), v: make(kvDim * 4), att: make(dim * 4),
    t: make(dim * 4), g: make(hidden * 4), u: make(hidden * 4),
    // T202: where a matrix alone writes (gate and up's rows, the most) and a one-workgroup dispatch adds, read by nothing
    scratch: make(2 * hidden * 4), norms: make(2 * dim * 4), angles: make((pos + 1) * headSize * 4),
    keys: make(cacheBytes), values: make(cacheBytes),
    // T175: the DP4A forms' four quantized vectors, each its own (the check reads every one back)
    quantized: [dim, dim, dim, hidden].map((n) => ({ xq: make(n), xs: make((n / GROUP) * 4) })) };
  const eps = data?.eps ?? EPS;
  // RoPE's table on the GPU, a row a position up to pos (T151: fusedMatVec reads the Step's row; ROPE, the prompt's,
  // takes the rows of its block's positions, here the row at pos bound on its own: a row of headSize 64 is 256 bytes,
  // the alignment of a binding's offset)
  device.queue.writeBuffer(v.angles, 0, ropeTable(headSize, pos + 1));
  const angleRow = { buffer: v.angles, offset: pos * headSize * 4, size: headSize * 4 };
  // the state a layer starts from: the residual stream, the norms' weights, the cache of the positions before
  const reset = (state) => {
    device.queue.writeBuffer(v.h, 0, state.h);
    device.queue.writeBuffer(v.norms, 0, state.norms);
    device.queue.writeBuffer(v.keys, 0, state.keys);
    device.queue.writeBuffer(v.values, 0, state.values);
  };
  const start = data ?? layerState(shape, pos);
  reset(start);
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
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned (T151: the
  // position is the Step's)
  const fusedParams = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / GROUP, second, 0, normAt, dim, kvDim, headSize, headSize, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const u = { step, attentionNorm: normParams(0), ffnNorm: normParams(dim), rope: uniform(new Uint32Array([heads, kvHeads, headSize, headSize])),
    flash: uniform(new Uint8Array(flashParams)), swiglu: uniform(new Uint32Array([hidden, 0, 0, 0])),
    qkv: fusedParams(dim + 2 * kvDim, dim), o: fusedParams(dim, dim), gateUp: fusedParams(hidden, dim, hidden, dim), down: fusedParams(dim, hidden),
    // QUANTIZE's (n, xStride) of dim and of hidden
    quantize: [dim, hidden].map((n) => uniform(new Uint32Array([n, n, 0, 0]))) };
  // the matrix × vector's Shape (rows, words, perRow, first) of each range the separate steps read, made once
  const shapes = new Map();
  const shapeOf = (rows, n) => {
    const key = `${rows},${n}`;
    if (!shapes.has(key)) shapes.set(key, uniform(new Uint32Array([rows, n / 4, n / GROUP, 0])));
    return shapes.get(key);
  };
  const group = (pipeline, entries) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(([binding, resource]) => ({ binding, resource: "offset" in resource ? resource : { buffer: resource } })) });
  // over (T202's review): a fused form's matrices bound elsewhere, { key: a range of ranges(key) }
  const dispatches = (form, pipes, copy, over = {}) => {
    const m = copiesOf[copy];
    if (form.fused) return fusedLayer(form, pipes, shape, { ...m, ...over }, v, v, u, group);
    const attention = [pipes.flash, group(pipes.flash, [[0, v.q], [1, v.keys], [2, v.values], [3, v.att], [4, u.flash], [5, u.step]]), heads, 1];
    const norm = (params) => [pipes.norm, group(pipes.norm, [[0, v.h], [1, v.norms], [2, v.xb], [3, params], [4, u.step]]), 1, 1];
    // the separate steps: rows first to first + rows of a matrix, x into y (on DP4A, x's quantizing into)
    const [qkvIn, oIn, gluIn, downIn] = v.quantized;
    const product = ({ w, s, n }, first, rows, x, y, into) => [pipes.product, group(pipes.product, [
      [0, { buffer: w, offset: first * n, size: rows * n }], [1, { buffer: s, offset: first * n / 8, size: rows * n / 8 }],
      [2, form.dp4a ? into.xq : x], [3, y], [4, shapeOf(rows, n)], ...(form.dp4a ? [[5, into.xs]] : [])]),
      Math.ceil(rows / (form.dp4a ? WGSL.ORT_DP4A_MATVEC_ROWS : WGSL.MUL_MAT_VEC_ROWS)), 1];
    const quantize = (x, into, n) => (form.dp4a ? [quantizing(pipes, group, x, into, u.quantize[n === dim ? 0 : 1], n, u.step)] : []);
    const add = [pipes.small, group(pipes.small, [[0, v.t], [1, v.h]]), 1, 1];
    return [norm(u.attentionNorm), ...quantize(v.xb, qkvIn, dim), product(m.qkv, 0, dim, v.xb, v.q, qkvIn),
      product(m.qkv, dim, kvDim, v.xb, v.k, qkvIn), product(m.qkv, dim + kvDim, kvDim, v.xb, v.v, qkvIn),
      [pipes.rope, group(pipes.rope, [[0, v.q], [1, v.k], [2, v.v], [3, v.keys], [4, v.values], [5, angleRow], [6, u.rope], [7, u.step]]), 1, 1],
      attention, ...quantize(v.att, oIn, dim), product(m.o, 0, dim, v.att, v.t, oIn), add, norm(u.ffnNorm), ...quantize(v.xb, gluIn, dim),
      product(m.gateUp, 0, hidden, v.xb, v.g, gluIn), product(m.gateUp, hidden, hidden, v.xb, v.u, gluIn),
      [pipes.swiglu, group(pipes.swiglu, [[0, v.g], [1, v.u], [2, u.swiglu], [3, u.step]]), Math.ceil(hidden / 64), 1],
      ...quantize(v.g, downIn, hidden), product(m.down, 0, dim, v.g, v.t, downIn), add];
  };
  // T202's review: every range of the copies' matrix buffers that the layer's matrix key can be bound to, the copies
  // one after another. The weights are random bytes, so a matrix read from any range of its size costs the same; a
  // step timed alone on one matrix goes through all of them before it reads one again, as a layer reads each of its
  // matrices once in a round of its copies (T149's review: a matrix read again soon is read from the GPU's caches, and
  // o's two copies alone would be 9.4 MB a round, q, k and v's 14.2, down's 37.7). A range starts on a binding's
  // alignment, 256 bytes, its scales an eighth of the way in (so 2048 bytes of the weights)
  // T208: spare matrices (the largest's size, random) where the copies' ranges make fewer than reach bytes of a matrix:
  // T202 left gate and up two ranges of Llama 3.2 1B's, 75.5 MB a round (the copies' gate and up buffers alone), short
  // of T149's 128 MiB; two spares make it four, 151 MB, and the other matrices' rounds longer
  const spares = [];
  const ranges = (key) => {
    const [rows, n] = shape.matrices[key], bytes = rows * n;
    return [...copiesOf.flatMap((one) => Object.values(one)), ...spares].flatMap((matrix) =>
      [...Array(Math.floor((matrix.rows * matrix.n) / bytes))].map((_, k) => k * bytes).filter((at) => at % 2048 === 0)
        .map((at) => ({ w: { buffer: matrix.w, offset: at, size: bytes }, s: { buffer: matrix.s, offset: at / 8, size: bytes / 8 }, rows, n })));
  };
  const largest = Object.entries(shape.matrices).sort(([, a], [, b]) => b[0] * b[1] - a[0] * a[1])[0];
  while (Object.entries(shape.matrices).some(([key, size]) => ranges(key).length * matrixBytes(size) < reach)) spares.push(matrixOf(...largest));
  // T208: the attention's caches for the step timed alone, each a copy of the cache the layer starts from, in one
  // buffer (a cache's keys, then its values, each on a binding's alignment), as many as make reach bytes: T202 read the
  // one cache of 256 KB again and again, from the GPU's caches, where a token's 16 layers each read their own after
  // the other layers' weights. Made when first asked for (the steps' table only)
  let cacheRanges;
  const caches = () => {
    if (cacheRanges) return cacheRanges;
    const stride = Math.ceil(cacheBytes / 256) * 256, count = Math.max(1, Math.ceil(reach / (2 * stride)));
    const all = make(2 * stride * count, STORAGE | COPY_DST);
    const encoder = device.createCommandEncoder();
    cacheRanges = [...Array(count)].map((_, i) => {
      encoder.copyBufferToBuffer(v.keys, 0, all, 2 * i * stride, cacheBytes);
      encoder.copyBufferToBuffer(v.values, 0, all, (2 * i + 1) * stride, cacheBytes);
      return { keys: { buffer: all, offset: 2 * i * stride, size: cacheBytes }, values: { buffer: all, offset: (2 * i + 1) * stride, size: cacheBytes } };
    });
    device.queue.submit([encoder.finish()]);
    return cacheRanges;
  };
  // T208: the attention alone on a cache of caches()
  const attention = (pipes, cache) => attentionStep(pipes, heads, cache, v, u, group);
  // T202: a matrix of the layer alone on a range of ranges(key), into the scratch buffer (no RoPE, cache, add or
  // SwiGLU after it): its plain matrix × vector (pipeline: mulMatVec, or ortDp4aMatVec reading the quantized vector
  // the fused form reads), one dispatch over all its rows
  const alone = (form, pipeline, key, { w, s, rows, n }) => {
    const input = v.quantized[MATRIX_KEYS.indexOf(key)];
    return named(`${MATRIX_NAMES[key]} alone`, "alone", [pipeline, group(pipeline, [
      [0, w], [1, s], [2, form.dp4a ? input.xq : key === "down" ? v.g : v.xb], [3, v.scratch], [4, shapeOf(rows, n)], ...(form.dp4a ? [[5, input.xs]] : [])]),
      Math.ceil(rows / (form.dp4a ? WGSL.ORT_DP4A_MATVEC_ROWS : WGSL.MUL_MAT_VEC_ROWS)), 1], key);
  };
  // T202: a dispatch of one workgroup that adds a vector of dim into the scratch buffer (SMALL): what a step costs
  // in a chain of dispatches when it does next to nothing (its launch and the barrier before the next)
  const floor = (pipeline) => named("a dispatch of one workgroup (SMALL: a vector of dim added)", "floor", [pipeline, group(pipeline, [[0, v.t], [1, v.scratch]]), 1, 1]);
  // T202: the residual stream written back as it started, before a submission (the adds write over it), so that every
  // submission starts alike. What these steps do does not depend on the values (unlike SAMPLE's work, T191's review),
  // and the stream grows only by a bounded add a layer, so this keeps the submissions alike rather than their times
  const restart = () => device.queue.writeBuffer(v.h, 0, start.h);
  return { vectors: v, reset, dispatches, ranges, caches, attention, alone, floor, restart, spares: spares.length };
}
// T202: the matrices of a layer (layerShape's keys, in the order of the DP4A forms' quantized inputs, v.quantized), and
// as the steps' table names them
const MATRIX_KEYS = ["qkv", "o", "gateUp", "down"];
const MATRIX_NAMES = { qkv: "q, k and v", o: "o", gateUp: "gate and up", down: "down" };
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
// T151: RoPE's table of positions 0 to count - 1, a row of layerAngles() each (as the model's GPU worker would hold the
// CPU's, T152)
function ropeTable(headSize, count) {
  const table = new Float32Array(count * headSize);
  for (let p = 0; p < count; p++) table.set(layerAngles(headSize, p), p * headSize);
  return table;
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
// T175: where a matrix of a layer takes its input (layerReference's inputs), in the order of the DP4A forms' quantized
// vectors: the normed stream before q, k and v; the attention's output before o; the normed stream before gate and up;
// silu(gate) × up before down
const INPUTS = ["qkv", "o", "ffn", "down"];
// a quantized vector's values (int8 × the scale of its group of 32), as a matrix on DP4A takes it
function dequantized(xq, xs) {
  return Float64Array.from(xq, (value, i) => value * xs[(i / GROUP) | 0]);
}
// T175: how a vector the GPU quantized (xq, xs) holds to quantize_x of x, the reference's values where it was made
// (from the GPU's own inputs before it: they differ as float32 and float64 sums in another order, about 1e-6, and a
// key or value of the position rounded the other way in float16 moves the attention's output by up to about 1e-4 of
// itself): each scale within `line` of the reference's (a scale of another group, a norm left out or read from the
// wrong weights is off by far more), each value within 1, and no more than 1% of them off by 1 (a value on a
// rounding's edge goes either way: about 1e-4 of them). The lines (T175's review): where a norm made the vector (q, k
// and v's; gate and up's) NORMED_SCALE_LINE, for the reference's float64 norm and the GPU's float32 differ by a few ulp
// (1.1e-7 to 4.1e-7 on lavapipe, 2026-09-27) and a mean over n - 1 in place of n moves the scales by 2.4e-4 (the
// stream stays within 3e-7: the quantized integers do not depend on the norm's scale, only the group's scale does);
// elsewhere (o's, down's) QUANTIZED_SCALE_LINE, for the attention's output moves by up to about 1e-4 with a key or value
// rounded the other way in float16. Returns { wrong: why or null, scale: the worst relative difference of a scale,
// apart: the values off by 1, of: how many }
const QUANTIZED_SCALE_LINE = 1e-3, NORMED_SCALE_LINE = 3e-5;
const scaleLine = (point) => (point === "qkv" || point === "ffn" ? NORMED_SCALE_LINE : QUANTIZED_SCALE_LINE);
function quantizedOff(x, xq, xs, line) {
  const mine = WGSL.quantizedLikeCpu(Float32Array.from(x));
  let far = false, apart = 0, scale = 0;
  mine.xs.forEach((want, g) => {
    const off = want > 0 ? Math.abs(xs[g] - want) / want : xs[g] === 0 ? 0 : Infinity;
    scale = Math.max(scale, off);
    far ||= !(off <= line);
  });
  mine.xq.forEach((value, i) => {
    far ||= Math.abs(xq[i] - value) > 1;
    apart += xq[i] !== value;
  });
  const wrong = far ? "far from quantize_x's" : apart > 0.01 * x.length ? `${apart} of ${x.length} values not quantize_x's` : null;
  return { wrong, scale, apart, of: x.length };
}
// The layer in JavaScript (float64 sums), as the CPU's forward pass runs it: what every form is held to. d: the
// check's weights ({w, s} of each matrix), h, norms, keys, values (float16 bits), angles and eps. inputs(point, x): the
// vector a matrix takes where x comes in, at the points INPUTS names (T175: the DP4A forms' x quantized; else x itself).
// Returns the residual stream after the layer and the float16 bits of the keys and values of the position
function layerReference({ dim, hidden, heads, kvHeads, headSize, kvDim }, pos, d, inputs = (point, x) => x) {
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
  const h = Float64Array.from(d.h), xb = inputs("qkv", normed(h, 0));
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
  const o = product(d.o, dim, 0, dim, inputs("o", att)), h1 = h.map((value, i) => value + o[i]), xb2 = inputs("ffn", normed(h1, dim));
  const gate = product(d.gateUp, dim, 0, hidden, xb2), up = product(d.gateUp, dim, hidden, hidden, xb2);
  const down = product(d.down, hidden, 0, dim, inputs("down", gate.map((g, i) => (g / (1 + Math.exp(-g))) * up[i])));
  return { h: h1.map((value, i) => value + down[i]), keys, values };
}
// The check (T150; T175: the DP4A forms held to the reference fed their own quantized vectors, each held to
// quantize_x, quantizedOff): every form of the layer (Llama's shape: GQA, 33 heads of 64 and 3 of K and V; a width of 2112 = 66
// groups and a hidden width of 2080 = 65, each past one pass of mul_mat_vec's 64 groups a workgroup, so that the x²
// and the rows' sums of the second pass are seen: T150's review, whose three breakings of them the width of 256 let
// through), at position 70 (71 positions, two tiles of the attention), against layerReference. The weights' scales
// go as 1 / sqrt(width) and the stream's large channels as the width (3 in 256), so that the activations, the
// attention's scores and the norm's scale are as they were at a width of 256. The residual stream after it is held to LAYER_LINE of what the layer added to
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
// T175 (Fable): the stream's first group of 32 is all zeros, so that the first quantized vector (the normed stream
// before q, k and v) has a group whose scale is 0 (NORM_QUANTIZE's and QUANTIZE's select of 1 / scale: a division by
// 0 there makes NaN, which quantizedOff holds to quantize_x's 0). And the two DP4A fused forms, which differ only in
// NORM_QUANTIZE against RMSNORM then QUANTIZE (the same expressions weight × (s × x), the largest / 127 and the
// rounding, in one dispatch or two), must agree within a few ulp (sameAsNormsApart): the scales of their quantized
// vectors within NORMS_APART_ULPS, the values within 1 (no more than 1% of them off by 1), and where every value is the
// same, the stream within 1e-6 of what the layer added. Not to the bit (T175's review): WGSL lets an implementation
// reassociate operations and fuse them where the result is at least as accurate, and a division is within 2.5 ulp
// (§15.7.5), so a device that compiles the two shaders differently (Metal's fast math, gpuweb #2076) may round a scale
// by an ulp or two and a value on an edge the other way, and still be right. Whether they agreed to the bit goes into
// the verdict as a fact. A mean over n - 1 in NORM_QUANTIZE alone moves its scales by 2.4e-4, far past a few ulp
const LAYER_CHECK = { dim: 2112, hidden: 2080, heads: 33, kvHeads: 3 }, LAYER_CHECK_POS = 70, LAYER_LINE = 1e-3, CACHE_LINE = 2e-3;
const NORMS_APART_ULPS = 4, NORMS_APART_STREAM = 1e-6;
const LAYER_CHECK_OUTLIERS = Math.round((3 * LAYER_CHECK.dim) / 256), LAYER_CHECK_OUTLIER = 30, LAYER_CHECK_EPS = 1;
async function checkLayer() {
  const shape = layerShape(LAYER_CHECK), pos = LAYER_CHECK_POS, verdicts = {};
  const data = { ...layerState(shape, pos), angles: layerAngles(shape.headSize, pos), eps: LAYER_CHECK_EPS };
  for (let i = 0; i < LAYER_CHECK_OUTLIERS; i++) {
    data.h[Math.floor((i + 0.5) * shape.dim / LAYER_CHECK_OUTLIERS)] = LAYER_CHECK_OUTLIER * (i % 2 ? -1 : 1);
  }
  data.h.fill(0, 0, GROUP);
  for (const [key, [rows, n]] of Object.entries(shape.matrices)) {
    data[key] = { w: new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s: floats(rows * n / GROUP, 0.01 * Math.sqrt(256 / n)) };
  }
  // what the DP4A fused form with the norms apart left, for the fused form to be held to (within a few ulp)
  let normsApart;
  const sameBytes = (a, b) => {
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    return x.length === y.length && x.every((byte, i) => byte === y[i]);
  };
  // { ok, bitForBit, ulps: the most a scale is apart, apart: the values off by 1, stream: the stream's difference over
  // what the layer added (where every value is the same) }
  const sameAs = (got, other, added) => {
    let ulps = 0, apart = 0, far = false, values = 0;
    got.quantized.forEach((q, i) => {
      const bits = new Int32Array(q.xs.buffer, q.xs.byteOffset, q.xs.length), otherBits = new Int32Array(other.quantized[i].xs.buffer, other.quantized[i].xs.byteOffset, q.xs.length);
      // the scales are 0 or more: their bits are in the order of their values
      bits.forEach((b, g) => (ulps = Math.max(ulps, Math.abs(b - otherBits[g]))));
      q.xq.forEach((value, k) => {
        far ||= Math.abs(value - other.quantized[i].xq[k]) > 1;
        apart += value !== other.quantized[i].xq[k];
      });
      values += q.xq.length;
    });
    let stream = 0;
    got.h.forEach((value, k) => (stream = Math.max(stream, Math.abs(value - other.h[k]) / added)));
    const bitForBit = sameBytes(got.h, other.h) && sameBytes(got.keys, other.keys) && sameBytes(got.values, other.values)
      && got.quantized.every((q, i) => sameBytes(q.xq, other.quantized[i].xq) && sameBytes(q.xs, other.quantized[i].xs));
    const ok = ulps <= NORMS_APART_ULPS && !far && apart <= 0.01 * values && (apart > 0 || stream <= NORMS_APART_STREAM);
    return { ok, bitForBit, ulps, apart, stream };
  };
  for (const form of layerForms()) {
    if (form.none) continue;
    try {
      const got = await scoped(async (owned) => {
        const pipes = await layerPipes(shape, form);
        const parts = layerParts(shape, pos, 1, owned, data);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        parts.dispatches(form, pipes, 0).forEach((d) => run(pass, d));
        pass.end();
        const h = new Float32Array(await readBack(encoder, parts.vectors.h, shape.dim * 4));
        const cacheBytes = (pos + 1) * shape.kvDim * 2;
        const back = async (source, bytes) => readBack(device.createCommandEncoder(), source, bytes);
        // T175: the four vectors the DP4A form quantized, each as the matrix after it took it
        const quantized = [];
        if (form.dp4a) {
          for (const [i, n] of [shape.dim, shape.dim, shape.dim, shape.hidden].entries()) {
            quantized.push({ xq: new Int8Array(await back(parts.vectors.quantized[i].xq, n)),
              xs: new Float32Array(await back(parts.vectors.quantized[i].xs, (n / GROUP) * 4)) });
          }
        }
        return { h, keys: new Uint16Array(await back(parts.vectors.keys, cacheBytes)), values: new Uint16Array(await back(parts.vectors.values, cacheBytes)), quantized };
      });
      // T175: on DP4A the reference takes the GPU's own quantized vectors (a value on a rounding's edge may go either
      // way, and moves a layer's output by more than LAYER_LINE), and each of them is held to quantize_x of the
      // reference's values where it was made (quantizedOff)
      const made = [];
      const want = layerReference(shape, pos, data, form.dp4a ? (point, x) => {
        const i = INPUTS.indexOf(point);
        made[i] = x;
        return dequantized(got.quantized[i].xq, got.quantized[i].xs);
      } : undefined);
      const quantizing = form.dp4a ? INPUTS.map((point, i) => [point, quantizedOff(made[i], got.quantized[i].xq, got.quantized[i].xs, scaleLine(point))]) : [];
      const wrongly = quantizing.filter(([, q]) => q.wrong);
      let added = 0, largestKey = 0, largestValue = 0;
      want.h.forEach((value, i) => (added = Math.max(added, Math.abs(value - data.h[i]))));
      want.keys.forEach((bits) => (largestKey = Math.max(largestKey, Math.abs(fromHalf(bits)))));
      want.values.forEach((bits) => (largestValue = Math.max(largestValue, Math.abs(fromHalf(bits)))));
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
      // the DP4A fused form against the one with the norms apart, within a few ulp (undefined where that one was not run)
      let agreed;
      if (form.dp4a && form.fused) {
        if (form.normApart) normsApart = got;
        else if (normsApart) agreed = sameAs(got, normsApart, added);
      }
      // on DP4A, how each quantized vector held (for CI's logs): the worst scale apart and the values off by 1
      verdicts[layerCheck(form)] = { worstRelative: Math.max(off, cache), ok: off < LAYER_LINE && cache < CACHE_LINE && !touched && !wrongly.length && agreed?.ok !== false,
        stream: off, cache, ...(touched ? { wroteOtherPositions: true } : {}), ...(agreed === undefined ? {} : { sameAsNormsApart: agreed }),
        ...(quantizing.length ? { quantized: quantizing.map(([point, q]) => ({ point, ...q })) } : {}) };
    } catch (error) {
      verdicts[layerCheck(form)] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
    } finally {
      // the page stops a section that says nothing for 5 minutes: SwiftShader compiles each form's shaders for tens of s
      postMessage({ alive: true });
    }
  }
  layerVerdicts = verdicts;
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
    // the forms that can run here, each with what submits n layers of it (each on the next copy) and waits
    const timed = [];
    for (const form of layerForms()) {
      const row = { form: form.name, check: layerCheck(form), base: form.base, fused: form.fused, subgroups: form.subgroups, ...(form.normApart ? { normApart: true } : {}) };
      if (form.none) {
        timed.push({ row: { ...row, none: form.none } });
        continue;
      }
      try {
        const pipes = await layerPipes(shape, form);
        const each = await validated(async () => [...Array(copies)].map((_, copy) => parts.dispatches(form, pipes, copy)));
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
        row.dispatches = each[0].length;
        timed.push({ row, submission });
      } catch (error) {
        timed.push({ row: { ...row, error: String(error?.message ?? error) } });
      } finally {
        postMessage({ alive: true });
      }
    }
    const running = timed.filter((t) => t.submission);
    try {
      const results = await validated(() => interleaved(running.map((t) => t.submission), LAYER_MOST));
      running.forEach((t, i) => {
        const r = results[i];
        Object.assign(t.row, r.error ? { error: r.error } : { msPerLayer: r.ms / r.dispatches, layers: r.dispatches, GBps: bytes / (r.ms / r.dispatches) / 1e6,
          ...(r.ratio ? { ratio: r.ratio } : {}), ...(r.unsteady ? { unsteady: true } : {}) });
      });
    } catch (error) {
      running.forEach((t) => (t.row.error = String(error?.message ?? error)));
    }
    rows.push(...timed.map((t) => t.row));
  });
  layerTimes = rows;
  return { model: "Llama 3.2 1B", pos: LAYER_POS, layers: MODELS["Llama 3.2 1B"].layers, copies, GB: bytes / 1e9, rows };
}

// ---- T202: where the time of a layer goes (T152's review: on the owner's Android the fastest layer took 3.36 ms, its
// matrices read at the matrix × vector's 38.6 GB/s would take 1.77, and what the other 1.59 ms is was not measured).
// The fused forms a token would run here (stepForms(), T208: the layer table's fastest fused form the check found
// right, as the engine chooses, T152, and beside it the same with the norms folded or apart), and for each:
//   - every step of it alone (one of its dispatches, the same name counted where two cost the same),
//   - its four matrices alone (the plain matrix × vector over all of a matrix's rows, nothing folded into the write:
//     the "1.77 ms"),
//   - a dispatch of one workgroup that does next to nothing (what a step costs in a chain for being a dispatch),
//   - the whole layer,
// all timed in turn (interleaved(): n of each a submission, 2n less n, the rounds taken of every item alike). The
// layer on the next copy of the weights each time, as layer() times it; a step or a matrix alone on the next range of
// its size of all the copies' weights and the spares (layerParts' ranges(): T202's review, T149's, T208), the
// attention alone on the next of its caches (caches(), T208), so that none is read again before MATVEC_BYTES of others
// (not from the GPU's caches).
// Why each step alone and not the layer less one step: a step of 20 to 60 µs is 1 or 2% of a layer, about what a
// layer's time moves from one pair to the next, so a difference of two layers could not tell it; alone it is repeated
// until a submission takes SUBMISSION_MS. What alone leaves out, the layer less the sum of its steps says: what the
// chain of different dependent dispatches costs beyond each on its own (a small step on a GPU full of the matrix's
// workgroups waits for their tail). Each dispatch still waits for the one before it (the same buffers written), as
// in the layer. The residual stream is written back before every submission (the adds write over it), so that every
// submission starts alike; the whole layer's attention reads the one cache of positions up to LAYER_POS (256 KB of
// keys and values), as in layer(): between two reads of it, a copy's weights (68 MB).
// T208: then the whole layer once more by the GPU's own clock where the device gives timestamp-query (timestamps()).
async function layerSteps() {
  await gpu();
  const shape = layerShape(MODELS["Llama 3.2 1B"]);
  const bytes = Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0);
  const copies = fallback ? 1 : Math.ceil(MATVEC_BYTES / bytes);
  const { forms, chosen } = stepForms();
  const result = { model: "Llama 3.2 1B", pos: LAYER_POS, layers: MODELS["Llama 3.2 1B"].layers, copies, GB: bytes / 1e9, dp4a: Boolean(forms[0]?.dp4a), chosen };
  if (!forms.length) return { ...result, forms: [], steps: [] };
  return scoped(async (owned) => {
    const parts = layerParts(shape, LAYER_POS, copies, owned, undefined, fallback ? 0 : MATVEC_BYTES);
    const caches = parts.caches();
    // the fewest MB of weights read before the same ones again: the layer's copies or a matrix's ranges (the
    // attention's caches are their own line, result.caches)
    result.cycleMB = Math.min(copies * bytes, ...MATRIX_KEYS.map((key) => parts.ranges(key).length * matrixBytes(shape.matrices[key]))) / 1e6;
    result.spares = parts.spares;
    result.caches = { count: caches.length, MB: (caches.length * 2 * caches[0].keys.size) / 1e6 };
    await device.queue.onSubmittedWorkDone();
    // units: the dispatches of one unit each, on a copy or a range of the weights, taken in turn; a submission of n
    // units, the stream written back first
    const timing = (units) => {
      let next = 0;
      return async (n) => {
        parts.restart();
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        for (let i = 0; i < n; i++) units[next++ % units.length].forEach((d) => run(pass, d));
        pass.end();
        const began = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - began;
      };
    };
    // what is timed: a step { step, kind, matrix } or a form's layer { form }, each with its submission
    const items = [], rows = [], seen = new Set(), layers = [];
    const add = (item, units) => items.push({ ...item, submission: timing(units) });
    for (const form of forms) {
      const row = { form: form.name, check: layerCheck(form), normApart: Boolean(form.normApart) };
      rows.push(row);
      try {
        const pipes = await layerPipes(shape, form);
        const each = await validated(async () => [...Array(copies)].map((_, copy) => parts.dispatches(form, pipes, copy)));
        row.dispatches = each[0].length;
        // how many of each step a layer dispatches, in the layer's order
        row.steps = [];
        each[0].forEach(({ step }) => {
          const known = row.steps.find((one) => one.step === step);
          if (known) known.count++;
          else row.steps.push({ step, count: 1 });
        });
        // each step not timed yet (the first of its name): a matrix's on every range of its size in turn (its other
        // bindings as the layer's), the attention's on every cache in turn, the others' on the copies as the layer
        const fresh = each[0].map((d, k) => ({ d, k })).filter(({ d, k }) => !seen.has(d.step) && each[0].findIndex((e) => e.step === d.step) === k);
        const stepUnits = await validated(async () => fresh.map(({ d: { matrix, kind }, k }) => (matrix
          ? parts.ranges(matrix).map((range) => [parts.dispatches(form, pipes, 0, { [matrix]: range })[k]])
          : kind === "attention" ? caches.map((cache) => [parts.attention(pipes, cache)]) : each.map((layer) => [layer[k]]))));
        add({ form: form.name }, each);
        layers.push({ form: form.name, each });
        fresh.forEach(({ d: { step, kind, matrix } }, i) => {
          seen.add(step);
          add({ step, kind, ...(matrix ? { matrix } : {}) }, stepUnits[i]);
        });
      } catch (error) {
        row.error = String(error?.message ?? error);
      } finally {
        postMessage({ alive: true });
      }
    }
    if (!items.length) return { ...result, forms: rows, steps: [] };
    // the matrices alone (on the plain matrix × vector of the forms' base, with their subgroups or not) and the floor
    // of a dispatch
    const product = await compiled(forms[0].dp4a ? WGSL.ortDp4aMatVec : WGSL.mulMatVec({ packed: false, subgroups: forms[0].subgroups }));
    const alone = await validated(async () => MATRIX_KEYS.map((key) => parts.ranges(key).map((range) => [parts.alone(forms[0], product, key, range)])));
    alone.forEach((units, i) => add({ step: units[0][0].step, kind: "alone", matrix: MATRIX_KEYS[i] }, units));
    const floor = parts.floor(pipelinesFor().small);
    add({ step: floor.step, kind: floor.kind }, [[floor]]);
    postMessage({ alive: true });
    const results = await validated(() => interleaved(items.map((item) => item.submission), MATVEC_MOST));
    const timed = (r) => (r.error ? { error: r.error }
      : { ms: r.ms / r.dispatches, n: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) });
    const steps = [];
    items.forEach((item, i) => {
      if (item.form) Object.assign(rows.find((row) => row.form === item.form), timed(results[i]));
      else steps.push({ step: item.step, kind: item.kind, ...(item.matrix ? { matrix: item.matrix } : {}), ...timed(results[i]) });
    });
    postMessage({ alive: true });
    return { ...result, forms: rows, steps, timestamps: await timestamps(parts, layers) };
  });
}
// T208: the fused forms the steps' table breaks down. The engine runs the fastest fused layer the check found right
// (T152: T150's fusedMatVec, with subgroupAdd where subgroups are, and T175's fusedDp4aMatVec where the packed int8
// dot is), so: the layer table's fastest fused row (layer(), which the page runs just before in this worker) that the
// check found right, and beside it the same form with the norms folded or apart where there is one (the two differ
// only in their norms, and how far their layers less those come apart is how far the run's times move). T202 took
// DP4A's two where the packed int8 dot is and llama.cpp's elsewhere, which on a device whose fastest layer is another
// (Apple, where ONNX Runtime does not use DP4A, or llama.cpp's with subgroups) broke down a layer no token runs. Where
// the layer table gave no time (it failed, or ran in no worker before), as T202.
// The fastest only of the forms the engine has (public/gpu.js's TOKEN_FORMS; T208's review): not llama.cpp's with the
// norms apart (withoutDp4a), which the layer table times where there is no packed int8 dot but no token runs; it can
// still be the partner.
function stepForms() {
  const fused = layerForms().filter((form) => form.fused && !form.none), engine = fused.filter((form) => !form.withoutDp4a);
  const fastest = (layerTimes ?? []).filter((row) => row.fused && row.msPerLayer > 0 && layerVerdicts?.[row.check]?.ok === true)
    .sort((a, b) => a.msPerLayer - b.msPerLayer).map((row) => engine.find((form) => form.name === row.form)).find(Boolean);
  if (!fastest) {
    const forms = LAYER_KINDS.filter((kind) => kind.fused && Boolean(kind.dp4a) === packed).map((kind) => ({ ...kind, subgroups: false }));
    return { forms, chosen: { by: "packed", why: layerTimes ? "the layer table has no fused layer timed and found right" : "no layer table was timed before" } };
  }
  const partner = fused.find((form) => form !== fastest && form.base === fastest.base && form.subgroups === fastest.subgroups && Boolean(form.normApart) !== Boolean(fastest.normApart));
  // in the layer table's order (the norms apart first)
  return { forms: fused.filter((form) => form === fastest || form === partner), chosen: { by: "layer", fastest: fastest.name } };
}
// T208 (T202's review's (d)): the whole layer by the GPU's own clock, each form's: a submission of TIMESTAMP_LAYERS
// layers (on the copies in turn, as layer() times them), each layer a compute pass of its own that writes a timestamp
// as it begins and as it ends (llama.cpp's GGML_WEBGPU_GPU_PROFILE writes them a dispatch a pass). The forms in turn,
// PAIRS rounds, a form's ms a layer the median of its rounds' means. A check of the whole layer only, one line of the
// table: Chrome's Dawn cuts every timestamp down to a multiple of 65.5 µs (2^16 ns; Chrome's own words say 100 µs;
// unless its developer features are on), too coarse for a step of 20 to 60 µs, and each layer's time is off by up to
// that much. T208's review: the mean of the passes is off by 65.5 µs / (2 √32) = 6 µs or less (its spread) where the
// passes start anywhere on that clock's steps (their times are not a multiple of it), and span (the first pass's
// beginning to the last one's end, over the layers) by 65.5 µs / 32 = 2 µs or less whatever they do, the cuts of the
// passes between cancelling. span also holds what comes between two passes; a mean above it says the passes overlapped
// (a pass stamped as begun before the one before it ended), and then the mean is not the layers' work. Against the
// submissions' times: those hold the wait of a submission, taken out as the difference of 2n and n, while these hold
// only what runs on the GPU. { none } where the device gives no timestamp-query, { error } where it failed.
const TIMESTAMP_LAYERS = 32;
async function timestamps(parts, layers) {
  if (!device.features.has("timestamp-query")) return { none: "this device gives no timestamp-query (the GPU's own clock) to this page" };
  if (!layers.length) return { none: "no form's layer ran" };
  const count = fallback ? 1 : TIMESTAMP_LAYERS, rounds = fallback ? 1 : PAIRS;
  try {
    return await scoped(async (owned) => {
      const set = device.createQuerySet({ type: "timestamp", count: 2 * count });
      const resolved = buffer(16 * count, 0x200 | COPY_SRC);  // GPUBufferUsage.QUERY_RESOLVE
      owned.push(resolved, { destroy: () => set.destroy() });
      const means = layers.map(() => []), spans = layers.map(() => []);
      for (let round = 0; round < rounds; round++) {
        for (const [i, { each }] of layers.entries()) {
          parts.restart();
          const encoder = device.createCommandEncoder();
          for (let l = 0; l < count; l++) {
            const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * l, endOfPassWriteIndex: 2 * l + 1 } });
            each[(round * count + l) % each.length].forEach((d) => run(pass, d));
            pass.end();
          }
          encoder.resolveQuerySet(set, 0, 2 * count, resolved, 0);
          const stamps = new BigInt64Array(await readBack(encoder, resolved, 16 * count));
          let ns = 0;
          for (let l = 0; l < count; l++) ns += Number(stamps[2 * l + 1] - stamps[2 * l]);
          means[i].push(ns / count / 1e6);
          spans[i].push(Number(stamps[2 * count - 1] - stamps[0]) / count / 1e6);
        }
        postMessage({ alive: true });
      }
      return { layers: count, rounds, forms: layers.map(({ form }, i) => ({ form, ms: middle(means[i]), span: middle(spans[i]) })) };
    });
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

// ---- T151: generated tokens on the GPU (shaders.js's EMBED, fusedMatVec and SAMPLE): a token a compute pass, the
// sampler's state carried from one to the next on the GPU, and the ids read back once for the run: against reading
// each one back, whose wait (3 to 8.6 ms for 4 bytes on the owner's Android, T134) a token pays alone.
// The layers are tokenForm()'s (T175): the fused layer on DP4A where the packed int8 dot is, else T150's with the
// workgroup's reduction (which layer is faster is the layer table's; every row here runs the same layers, so their
// difference is the reading back alone). On DP4A the classifier takes its normed input quantized too
// (NORM_QUANTIZE, then fusedDp4aMatVec's "write")
const GENERATE_MODEL = { ...PROMPT_MODEL, vocab: 32000 };  // the CPU section's model and its vocabulary
// the settings a token is timed with: the list's sampled models' (src/models.js: temperature 0.7, top-p 0.9, and
// tiny-lm's penalty 1.3)
const GENERATE_SETTINGS = { temperature: 0.7, topp: 0.9, penalty: 1.3 };
// the forms, in turn: tokens a submission (1: each read back as it comes), GENERATE_TOKENS tokens each round
const GENERATE_COUNTS = [1, 4, 8, 16], GENERATE_TOKENS = 16, GENERATE_ROUNDS = 5, GENERATE_MOST = 64, GENERATE_POS = 127;
// a fallback adapter runs the check's small model, 2 tokens a form once: its times are no GPU's
const GENERATE_CHECK = { dim: 256, hidden: 512, heads: 4, kvHeads: 2, layers: 2, vocab: 1003 };
async function generationPipes(headSize, form) {
  const { maxComputeWorkgroupStorageSize: memory, maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX } = device.limits;
  const flash = WGSL.flashShape({ headSize, half: false, subgroups: false, memory, threads: Math.min(maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX) });
  if (flash.none) throw new Error(flash.none);
  const pipes = {};
  const classifier = form.dp4a ? WGSL.fusedDp4aMatVec({ output: "write" }) : WGSL.fusedMatVec({ input: "norm", output: "write", subgroups: false });
  for (const [key, code] of [["embed", WGSL.EMBED], ["flash", WGSL.flashTile(flash)], ...layerCodes(form), ["classifier", classifier]]) {
    pipes[key] = await compiled(code);
    postMessage({ alive: true });
  }
  pipes.sampler = await samplerPipes();
  return pipes;
}
// T191: the two ways of sampling a token on the GPU, measured side by side: SAMPLE's one workgroup (T151) and the
// sampling in chunks of the vocabulary (shaders.js's SAMPLER_STAGES, a workgroup a chunk)
const SAMPLERS = ["one", "chunks"];
async function samplerPipes() {
  const pipes = { one: await compiled(WGSL.SAMPLE), chunks: [] };
  postMessage({ alive: true });
  for (const stage of WGSL.SAMPLER_STAGES) {
    pipes.chunks.push({ ...stage, pipeline: await compiled(stage.code) });
    postMessage({ alive: true });
  }
  return pipes;
}
// a sampler's dispatches, [pipeline, bind group, x, y] each: b holds the buffers by SAMPLE's binding numbers (0 the
// logits, 1 probs, 2 order, 3 the state, 4 chosen, 5 the random numbers, 6 the settings) and 7 the chunks' partial
// results (WGSL.samplePartsBytes of the vocabulary)
function samplerDispatches(kind, pipes, b, vocab) {
  const group = (pipeline, bindings) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((binding) => ({ binding, resource: { buffer: b[binding] } })) });
  if (kind === "one") return [[pipes.one, group(pipes.one, [0, 1, 2, 3, 4, 5, 6]), 1, 1]];
  const chunks = WGSL.sampleChunks(vocab);
  return pipes.chunks.map((stage) => [stage.pipeline, group(stage.pipeline, stage.bindings), stage.chunks ? chunks : 1, 1]);
}
// A model on the GPU for a run of tokens: its layers (T150's four matrices each, the cache of its own), the norms
// (two a layer, then the final one), the classifier (the embedding too: tied, as Llama 3.2 1B's), RoPE's table of
// `positions`, and the sampler's buffers. form: the layer's (tokenForm()). data: the check's ({layers: [{qkv, o,
// gateUp, down: {w, s}}], classifier, norms, keys, values}), else random. encode(encoder): one token's pass and the
// Step copied from the state after it. T175: on DP4A with the check's data, every quantized vector of a pass is its
// own (four a layer, then the classifier's), and encode copies them into `recorded` after the pass, token by token
// from reset() on (recording: their sizes; recorded(bytes) cuts what was read back into [token][vector] {xq, xs})
function generationParts(model, form, pipes, positions, owned, data) {
  const shape = layerShape(model), { dim, hidden, heads, kvDim, headSize } = shape, { vocab, layers } = model;
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
  const weights = ([rows, n], given) => {
    if (rows * n > device.limits.maxStorageBufferBindingSize) throw new Error(`a matrix of ${rows} × ${n} is past a binding of this device`);
    const w = make(rows * n), s = make((rows * n / GROUP) * 4);
    if (given) {
      device.queue.writeBuffer(w, 0, given.w);
      device.queue.writeBuffer(s, 0, given.s);
    } else {
      fill(w, rows * n);
      device.queue.writeBuffer(s, 0, floats(rows * n / GROUP, 0.002));
    }
    return { w, s, rows };
  };
  const cacheBytes = positions * kvDim * 2;
  const stack = [...Array(layers)].map((_, l) => {
    const m = Object.fromEntries(Object.entries(shape.matrices).map(([key, matrix]) => [key, weights(matrix, data?.layers[l][key])]));
    const keys = make(cacheBytes), values = make(cacheBytes);
    if (data) {
      device.queue.writeBuffer(keys, 0, data.keys[l]);
      device.queue.writeBuffer(values, 0, data.values[l]);
    }
    return { ...m, keys, values };
  });
  const classifier = weights([vocab, dim], data?.classifier);
  // T175: one quantized vector serves all of a token's quantizations (each is read before the next is made), but for
  // the check, which reads each one back
  const pair = (n) => ({ n, xq: make(n), xs: make((n / GROUP) * 4) });
  const kept = form.dp4a && data;
  const sizes = [...[...Array(layers)].flatMap(() => [dim, dim, dim, hidden]), dim];
  const quantizedAll = kept ? sizes.map(pair) : Array(sizes.length).fill(pair(Math.max(dim, hidden)));
  const v = { h: make(dim * 4), q: make(dim * 4), att: make(dim * 4), g: make(hidden * 4), logits: make(vocab * 4),
    probs: make(vocab * 4), order: make(vocab * 4), parts: make(WGSL.samplePartsBytes(vocab)), norms: make((2 * layers + 1) * dim * 4), angles: make(positions * headSize * 4),
    state: make(WGSL.STATE_BYTES), chosen: make(Math.max(positions, 4) * 4), randoms: make(positions * 4),
    step: make(16, UNIFORM | COPY_DST), settings: make(WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST) };
  device.queue.writeBuffer(v.norms, 0, data?.norms ?? new Float32Array((2 * layers + 1) * dim).map(() => 0.5 + Math.random()));
  device.queue.writeBuffer(v.angles, 0, ropeTable(headSize, positions));
  const eps = data?.eps ?? EPS;
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / GROUP, second, 0, normAt, dim, kvDim, headSize, headSize, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const flashParams = new ArrayBuffer(16);
  new Uint32Array(flashParams, 0, 2).set([heads, shape.kvHeads]);
  new Float32Array(flashParams, 8, 1)[0] = 1 / Math.sqrt(headSize);
  const step = v.step;
  // RMSNORM's and NORM_QUANTIZE's Norm (size, at, eps): a layer's two norms and the final one, from `at` in the norms
  const normParams = (at) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes, 0, 2).set([dim, at]);
    new Float32Array(bytes, 8, 1)[0] = eps;
    return uniform(new Uint8Array(bytes));
  };
  const common = { step, flash: uniform(new Uint8Array(flashParams)), o: params(dim, dim), down: params(dim, hidden),
    quantize: [dim, hidden].map((n) => uniform(new Uint32Array([n, n, 0, 0]))) };
  const u = { embed: uniform(new Uint32Array([dim, 0, 0, 0])), classifier: params(vocab, dim, 0, 2 * layers * dim), final: normParams(2 * layers * dim),
    layers: stack.map((_, l) => ({ ...common, qkv: params(dim + 2 * kvDim, dim, 0, 2 * l * dim), gateUp: params(hidden, dim, hidden, (2 * l + 1) * dim),
      attentionNorm: normParams(2 * l * dim), ffnNorm: normParams((2 * l + 1) * dim) })) };
  const group = (pipeline, entries) => device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: entries.map(([binding, resource]) => ({ binding, resource: { buffer: resource } })) });
  const dispatches = [[pipes.embed, group(pipes.embed, [[0, classifier.w], [1, classifier.s], [2, v.state], [3, v.h], [4, u.embed]]), 1, 1]];
  stack.forEach((m, l) => dispatches.push(...fusedLayer(form, pipes, shape, m, m, { ...v, quantized: quantizedAll.slice(4 * l, 4 * l + 4) }, u.layers[l], group)));
  const quantized = quantizedAll[4 * layers];
  const rows = Math.ceil(vocab / (form.dp4a ? WGSL.ORT_DP4A_MATVEC_ROWS : WGSL.MUL_MAT_VEC_ROWS));
  const across = Math.min(rows, device.limits.maxComputeWorkgroupsPerDimension);
  if (form.dp4a) {
    dispatches.push([pipes.normQuantize, group(pipes.normQuantize, [[0, v.h], [1, v.norms], [2, quantized.xq], [3, quantized.xs], [4, u.final], [5, step]]), 1, 1],
      [pipes.classifier, group(pipes.classifier, [[0, classifier.w], [1, classifier.s], [2, quantized.xq], [3, u.classifier], [4, quantized.xs], [5, v.logits]]), across, Math.ceil(rows / across)]);
  } else {
    dispatches.push([pipes.classifier, group(pipes.classifier, [[0, classifier.w], [1, classifier.s], [2, v.h], [3, u.classifier], [4, v.norms], [5, v.logits]]), across, Math.ceil(rows / across)]);
  }
  // the token's sampling: SAMPLE's one workgroup, or in chunks (T191), as use() says
  const samplerBuffers = { 0: v.logits, 1: v.probs, 2: v.order, 3: v.state, 4: v.chosen, 5: v.randoms, 6: v.settings, 7: v.parts };
  const samplers = Object.fromEntries(SAMPLERS.map((kind) => [kind, samplerDispatches(kind, pipes.sampler, samplerBuffers, vocab)]));
  let sampling = samplers.one;
  const use = (kind) => (sampling = samplers[kind]);
  // the check's record of the quantized vectors: a token's are perToken bytes, each xq then its xs
  const perToken = kept ? quantizedAll.reduce((sum, { n }) => sum + n + (n / GROUP) * 4, 0) : 0;
  const record = kept ? make(positions * perToken) : undefined;
  let passes = 0;
  const encode = (encoder) => {
    const pass = encoder.beginComputePass();
    dispatches.forEach((d) => run(pass, d));
    sampling.forEach((d) => run(pass, d));
    pass.end();
    // the Step of the next token: the state's first four words (a uniform is not a shader's to write)
    encoder.copyBufferToBuffer(v.state, 0, v.step, 0, 16);
    if (record) {
      let at = passes * perToken;
      for (const { n, xq, xs } of quantizedAll) {
        encoder.copyBufferToBuffer(xq, 0, record, at, n);
        encoder.copyBufferToBuffer(xs, 0, record, at + n, (n / GROUP) * 4);
        at += n + (n / GROUP) * 4;
      }
    }
    passes++;
  };
  // the recorded vectors read back (bytes: count tokens' of them), [token][vector] { xq, xs }
  const recorded = (bytes, count) => [...Array(count)].map((_, k) => {
    let at = k * perToken;
    return quantizedAll.map(({ n }) => {
      const one = { xq: new Int8Array(bytes, at, n), xs: new Float32Array(bytes, at + n, n / GROUP) };
      at += n + (n / GROUP) * 4;
      return one;
    });
  });
  // a run from the start: the state (samplingState), the random numbers and the settings (samplingSettings)
  const reset = (state, randoms, settings) => {
    passes = 0;
    device.queue.writeBuffer(v.state, 0, state);
    device.queue.writeBuffer(v.step, 0, state.subarray(0, 4));
    if (randoms) device.queue.writeBuffer(v.randoms, 0, randoms);
    if (settings) device.queue.writeBuffer(v.settings, 0, settings);
  };
  const bytes = layers * Object.values(shape.matrices).reduce((sum, matrix) => sum + matrixBytes(matrix), 0) + matrixBytes([vocab, dim]);
  return { vectors: v, encode, reset, use, dispatches: dispatches.length + samplers.one.length, chunkDispatches: dispatches.length + samplers.chunks.length,
    bytes, ...(record ? { record, recording: positions * perToken, recorded } : {}) };
}
// what submitting the tokens of a run and reading back their ids and the state costs: count tokens, per of them a
// submission (each read back before the next is submitted, as a token's text is shown), from the state given. Returns
// the ms and what came back: the ids and the state's words
async function generationRun(parts, count, per, target) {
  const v = parts.vectors, bytes = per * 4 + WGSL.STATE_BYTES;
  const into = target ?? buffer(bytes, MAP_READ | COPY_DST);
  const ids = new Uint32Array(count);
  let state;
  const began = performance.now();
  for (let done = 0; done < count; done += per) {
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < per; i++) parts.encode(encoder);
    encoder.copyBufferToBuffer(v.chosen, done * 4, into, 0, per * 4);
    encoder.copyBufferToBuffer(v.state, 0, into, per * 4, WGSL.STATE_BYTES);
    device.queue.submit([encoder.finish()]);
    await into.mapAsync(MAP_READ);
    const words = new Uint32Array(into.getMappedRange(0, bytes).slice(0));
    into.unmap();
    ids.set(words.subarray(0, per), done);
    state = words.subarray(per);
  }
  const ms = performance.now() - began;
  if (!target) into.destroy();
  return { ms, ids, state };
}
// made-up random numbers in [0, 1) (the engine's are NumPy's generator's, drawn by the CPU in the same order)
const randomsOf = (count) => new Float32Array(count).map(() => Math.fround(Math.random()) % 1);
async function generate() {
  await gpu();
  const model = fallback ? GENERATE_CHECK : GENERATE_MODEL, headSize = model.dim / model.heads;
  const counts = fallback ? [1, 2] : GENERATE_COUNTS, tokens = fallback ? 2 : GENERATE_TOKENS;
  const positions = GENERATE_POS + 2 * GENERATE_MOST + GENERATE_TOKENS + 1;
  const history = [...Array(GENERATE_POS + 1)].map(() => (Math.random() * model.vocab) | 0);
  const start = WGSL.samplingState({ token: history[history.length - 1], pos: GENERATE_POS, history });
  const randoms = randomsOf(positions);
  return scoped(async (owned) => {
    const form = tokenForm(), pipes = await generationPipes(headSize, form);
    const parts = generationParts(model, form, pipes, positions, owned);
    parts.reset(start, randoms, WGSL.samplingSettings({ vocab: model.vocab, ...GENERATE_SETTINGS }));
    await device.queue.onSubmittedWorkDone();
    const targets = new Map(counts.map((per) => [per, buffer(per * 4 + WGSL.STATE_BYTES, MAP_READ | COPY_DST)]));
    owned.push(...targets.values());
    postMessage({ alive: true });
    // the ms of a run of `tokens` from the start, per of them a submission, sampled the sampler's way (T191)
    const timed = async (per, kind) => {
      parts.use(kind);
      parts.reset(start);
      return (await generationRun(parts, tokens, per, targets.get(per))).ms;
    };
    // the work of a token without the reading back: n tokens in one submission against 2n (T168's), each sampler's,
    // the samplers in turn (interleaved(): T150's review)
    const works = {};
    if (!fallback) {
      const found = await interleaved(SAMPLERS.map((kind) => async (n) => {
        parts.use(kind);
        parts.reset(start);
        return (await generationRun(parts, n, n)).ms;
      }), GENERATE_MOST);
      SAMPLERS.forEach((kind, k) => {
        const r = found[k];
        if (!r.error) works[kind] = { ms: r.ms / r.dispatches, tokens: r.dispatches, ratio: r.ratio, ...(r.unsteady ? { unsteady: true } : {}) };
      });
    }
    postMessage({ alive: true });
    // the forms and the samplers in turn (T150's review), each a run of `tokens`, after one run each to warm up
    const rounds = fallback ? 1 : GENERATE_ROUNDS, times = SAMPLERS.map(() => counts.map(() => []));
    if (!fallback) for (const per of counts) for (const kind of SAMPLERS) await timed(per, kind);
    for (let round = 0; round < rounds; round++) {
      for (let i = 0; i < counts.length; i++) {
        for (let k = 0; k < SAMPLERS.length; k++) times[k][i].push(await timed(counts[i], SAMPLERS[k]));
      }
      postMessage({ alive: true });
    }
    const work = works.one;
    const rows = counts.map((per, i) => {
      const msPerToken = middle(times[0][i]) / tokens;
      // what a submission costs besides its tokens' work: the submission, the wait and the ids read back
      const fixed = work && !work.unsteady ? msPerToken * per - work.ms * per : undefined;
      return { perSubmission: per, msPerToken, ...(fixed === undefined ? {} : { fixedMs: fixed }), chunks: middle(times[1][i]) / tokens };
    });
    // the sampling alone, on Llama 3's vocabulary: logits as a model's, and flat ones (every token over the floor)
    const sampling = fallback ? undefined : await samplingAlone();
    return { model: fallback ? "the check's small model" : "Llama 3.2 1B's width", layer: form.name, layers: model.layers, vocab: model.vocab,
      GB: parts.bytes / 1e9, dispatches: parts.dispatches, chunkDispatches: parts.chunkDispatches, tokens, settings: GENERATE_SETTINGS, work,
      ...(works.chunks ? { chunkWork: works.chunks } : {}), rows, sampling };
  });
}
// the sampling alone (paired(): T168's n and 2n) on Llama 3's vocabulary, twice: on logits as a model's (madeUpLogits:
// a spread of 2 and 20 tokens far above it, a few percent of the vocabulary over the nucleus's floor) and on flat ones
// (a spread of 1, none above: every token over the floor, so that SAMPLE gathers and reads all of them each round
// of its searches: its worst case). `over` is how many tokens are over the floor.
// Without the repetition penalty (T191's review): the penalty changes the logits in place, sampling after sampling,
// and they are written once for all the submissions, so with it each sampled peak was divided down and the logits as a
// model's went about flat while they were timed: the count over the floor the table named was not what was timed. The
// penalty itself is a thread a token of the window in either sampler.
// T191: SAMPLE's one workgroup, the sampling in chunks, and of that its last stage alone (one workgroup: the nucleus
// and the draw, the same search as SAMPLE's): each submission of the last runs one sampling in chunks, then n of its
// last stage on what that gathered, so that 2n less n is n of the last stage alone
async function samplingAlone() {
  const vocab = MODELS["Llama 3.2 1B"].vocab, positions = 2 * GENERATE_MOST + 2;
  return scoped(async (owned) => {
    const pipes = await samplerPipes();
    const make = (bytes, usage = STORAGE | COPY_DST) => {
      const b = buffer(bytes, usage);
      owned.push(b);
      return b;
    };
    const logits = make(vocab * 4), probs = make(vocab * 4), order = make(vocab * 4), state = make(WGSL.STATE_BYTES),
      chosen = make(positions * 4), randoms = make(positions * 4), settings = make(WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST),
      parts = make(WGSL.samplePartsBytes(vocab));
    device.queue.writeBuffer(randoms, 0, randomsOf(positions));
    device.queue.writeBuffer(settings, 0, WGSL.samplingSettings({ vocab, ...GENERATE_SETTINGS, penalty: 1 }));
    const history = [...Array(64)].map(() => (Math.random() * vocab) | 0), start = WGSL.samplingState({ token: history[63], pos: 0, history });
    const b = { 0: logits, 1: probs, 2: order, 3: state, 4: chosen, 5: randoms, 6: settings, 7: parts };
    // T191: SAMPLE's one workgroup, the sampling in chunks and its last stage alone, in turn (interleaved(): T150's
    // review), each [the dispatches once before the n, the dispatches n times]
    const [one, chunks] = SAMPLERS.map((kind) => samplerDispatches(kind, pipes, b, vocab));
    const last = chunks[WGSL.SAMPLER_STAGES.findIndex((stage) => stage.name === "pick")];
    const lists = [[[], one], [[], chunks], [chunks, [last]]];
    const timed = async (values) => {
      device.queue.writeBuffer(logits, 0, values);
      const found = await interleaved(lists.map(([before, list]) => async (n) => {
        device.queue.writeBuffer(state, 0, start);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        before.forEach((d) => run(pass, d));
        for (let i = 0; i < n; i++) list.forEach((d) => run(pass, d));
        pass.end();
        const began = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - began;
      }), GENERATE_MOST);
      const [once, inChunks, pick] = found.map((r) => (r.error ? { error: r.error } : { msEach: r.ms / r.dispatches, ...(r.unsteady ? { unsteady: true } : {}) }));
      return { ...once, over: overTheFloor(values, GENERATE_SETTINGS.temperature), chunks: { ...inChunks, pick } };
    };
    const peaked = await timed(madeUpLogits(vocab, 2)), flat = await timed(madeUpLogits(vocab, 1, 0));
    return { vocab, ...peaked, flat };
  });
}
// how many of the logits SAMPLE keeps over the nucleus's floor (kernel.ts's: within temperature × ln 1e7 of the largest)
function overTheFloor(logits, temperature) {
  let best = -Infinity;
  for (const value of logits) best = Math.max(best, value);
  const floor = best - temperature * 16.118095;
  let over = 0;
  for (const value of logits) if (value >= floor) over++;
  return over;
}
// logits with equal ones where the draw's order among them shows (the check's ties cases): two at the top (10: the
// first index, 100, is the most likely), a run of 20 at TIED_RUN (a fifth of the top's probability each at temperature
// 0.7, their indices descending as written so the order of the index is not the order written), and a tail of 200
// distinct values below (a fiftieth each, rising by the index) that a nucleus of top-p 0.9 reaches into: no equal
// logits at its border, where SAMPLE takes every equal token and the CPU's sort takes some (a different mass).
// T191: past one chunk of the sampling in chunks (WGSL.SAMPLE_CHUNK), the same spread over the chunks: the second of the
// top at 64000, the run at multiples of 3000 (from 60000 down), the tail at odd indices from 1001 (so that the order
// of equal ones and of the nucleus crosses the chunks)
const TIED_RUN = Math.fround(8.87);
function tiedLogits(vocab) {
  const logits = new Float32Array(vocab), wide = vocab > 64 * WGSL.SAMPLE_CHUNK;
  logits[wide ? 64000 : 700] = 10;
  logits[100] = 10;
  for (let i = 0; i < 20; i++) logits[wide ? 60000 - 3000 * i : 900 - 3 * i] = TIED_RUN;
  for (let i = 0; i < 200; i++) logits[wide ? 1001 + 618 * i : 200 + i] = 7.26 + i * 1e-3;
  return logits;
}
// T191 (Fable's check): a few tokens far over the rest, on the borders of the chunks of the sampling in chunks (the
// first and the last token of a chunk, the very last token of the vocabulary: a thread's fourth), two of them equal
// across a border (1023 and 1024), the rest a narrow normal spread: at temperature 0.4 the floor (best − 6.45) is
// above every other token, so nearly every chunk gathers nothing, the count over the floor is far below a workgroup,
// and the draw without a nucleus walks through chunks whose sums are next to nothing
const SPARSE_PEAKS = [[0, 10], [2047, 10.25], [1023, 10.5], [1024, 10.5], [50000, 10.75], [64511, 11], [128255, 11.25], [64512, 11.5], [128000, 11.75], [1, 12]];
function sparseLogits(vocab) {
  const logits = madeUpLogits(vocab, 0.5, 0);
  for (const [at, value] of SPARSE_PEAKS) logits[at] = value;
  return logits;
}
// logits as a model's look (a few tokens far above the rest), made up: a normal spread and `peaks` tokens 8 to 14 over it
function madeUpLogits(vocab, spread, peaks = 20) {
  const logits = new Float32Array(vocab);
  for (let i = 0; i < vocab; i++) logits[i] = spread * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  for (let i = 0; i < peaks; i++) logits[(Math.random() * vocab) | 0] += 8 + 6 * Math.random();
  return logits;
}

// The check of SAMPLE (T151) against the CPU's sampling in JavaScript (shaders.js's sampleLikeCpu and
// penalizeLikeCpu, which tests/smoke.mjs holds to the kernel): the same logits, history and random number must pick
// the same token. Where a float32 sum in another order moves a border, a token next to it is as right: the token
// passes when it is what the CPU picks, or where the CPU's walk (walkLikeCpu, with top-p as it is and moved by EDGE)
// passes it within EDGE of the mass of the random number's share (a relative 1e-4: about 3 times the worst the GPU's
// float32 sums and exp() can be off by, 501 × 2^-24 ≈ 3.0e-5 of a thread's run and 3.3e-5 with the rest; a wrong
// border moves the draw by a token's probability, 1e-3 of the mass or more), or a token of the same logit (equal probabilities, which the CPU
// takes in no set order); temperature 0 (the most likely token): the first index of the largest logit and, where
// logits come from the GPU's own forward pass, any within `band` of it (and of a token acceptable otherwise). The
// count of tokens that passed by an edge only is in the verdict.
const EDGE = 1e-4;
function acceptable(logits, { temperature, topp }, random, band = 0) {
  const picks = new Set([WGSL.sampleLikeCpu(logits, temperature, topp, random)]), [first] = picks;
  const near = (token) => band > 0 ? logits.forEach((value, i) => Math.abs(value - logits[token]) <= band && picks.add(i)) : picks.add(token);
  // the most likely token: NumPy's first index of the largest logit, exactly where the logits are the same
  if (temperature === 0) {
    near(first);
    return { first, picks };
  }
  const nucleus = topp > 0 && topp < 1;
  for (const p of nucleus ? [topp, topp * (1 - EDGE), topp * (1 + EDGE)] : [topp]) {
    const { tokens, cumulative, mass } = WGSL.walkLikeCpu(logits, temperature, p);
    const low = (random - EDGE) * mass, high = (random + EDGE) * mass;
    tokens.forEach((token, k) => (k ? cumulative[k - 1] : 0) <= high && cumulative[k] >= low && near(token));
  }
  // equal probabilities, which the CPU takes in no set order
  for (const token of [...picks]) logits.forEach((value, i) => value === logits[token] && picks.add(i));
  return { first, picks };
}
// The sampler alone: cases of logits (a vocabulary of 1003, not a multiple of 256, and Llama 3's 128256; normal
// spreads from flat to steep, and tokens far above them), top-p 0.9, 0.5 and none, random numbers at 0, inside and
// just under 1, temperature 0 and 1.3 too, the penalty (the window's tokens among the most likely, a repeated one,
// negative ones, and likely ones older than the window that must not be penalized), and equal logits (tiedLogits: the
// most likely token is the first index of the two at the top; the draw among equals is the exact token in the order of
// the index, as SAMPLE takes them and walkLikeCpu walks them). Then runs of four tokens in one submission (the i-th
// random number for the i-th token, the logits penalized again each time as the CPU's would be), and a run with a stop
// token: it is written, the state stops and nothing after it changes.
async function checkSampling(kind = "one") {
  const pipes = await samplerPipes();
  const cases = [];
  for (const vocab of [1003, 128256]) {
    // (a fallback adapter takes a second or so for each of the big vocabulary's: one spread there)
    for (const spread of vocab === 1003 ? [0.5, 2, 6, 12] : fallback ? [2] : [2, 6]) {
      for (const topp of [0.9, 0.5, 1]) {
        for (const random of [0, Math.random(), 1 - 2 ** -24]) cases.push({ vocab, spread, topp, temperature: 0.7, penalty: 1.3, random });
      }
      cases.push({ vocab, spread, topp: 0.9, temperature: 0, penalty: 1.3, random: 0.5 });
      cases.push({ vocab, spread, topp: 0.9, temperature: 1.3, penalty: 1, random: Math.random() });
    }
  }
  // T191: without a nucleus on flat logits of Llama 3's vocabulary (no peaks), so that the mass is spread over the
  // chunks of the sampling in chunks and the draw passes in a chunk far from the first (a model's logits put nearly
  // all of it on one peak: the mass before that chunk is next to nothing, and forgetting it passed)
  for (const random of [0.3, 0.6, 0.9]) cases.push({ vocab: 128256, spread: 1, peaks: 0, topp: 1, temperature: 0.7, penalty: 1.3, random });
  // T191 (Fable's check): chunks with nothing over the floor, the tokens over it on the chunks' borders (sparseLogits);
  // the most likely token (temperature 0) is the last of the vocabulary (the fourth unpenalized peak: the window holds
  // the three most likely)
  for (const topp of [0.9, 0.5, 1]) {
    for (const random of [0.02, 0.5, 1 - 2 ** -24]) cases.push({ vocab: 128256, sparse: true, topp, temperature: 0.4, penalty: 1.3, random });
  }
  cases.push({ vocab: 128256, sparse: true, topp: 0.9, temperature: 0, penalty: 1.3, random: 0.5 });
  // a history shorter than the window (its empty slots must not count: token 0, among the most likely, is in none)
  for (const spread of [0.5, 2, 6]) for (let i = 0; i < 3; i++) cases.push({ vocab: 1003, spread, topp: 0.9, temperature: 0.7, penalty: 1.3, random: Math.random(), short: true });
  // equal logits (tiedLogits): the draw on the first of the two at the top, on the fifth of the run of 20, and the most
  // likely token (temperature 0)
  // (and over Llama 3's vocabulary, where they cross the chunks of the sampling in chunks: T191)
  for (const vocab of [1003, 128256]) {
    cases.push({ vocab, topp: 0.9, temperature: 0.7, penalty: 1, random: 0.05, ties: true });
    cases.push({ vocab, topp: 0.9, temperature: 0.7, penalty: 1, random: "fifth", ties: true });
    cases.push({ vocab, topp: 0.9, temperature: 0, penalty: 1, random: 0.3, ties: true });
  }
  const most = 128256, owned = [];
  let wrong = 0, edge = 0, checked = 0;
  const problems = [];
  try {
    await validated(async () => {
      const make = (bytes, usage = STORAGE | COPY_DST | COPY_SRC) => {
        const b = buffer(bytes, usage);
        owned.push(b);
        return b;
      };
      const logitsBuffer = make(most * 4), probs = make(most * 4), order = make(most * 4), state = make(WGSL.STATE_BYTES),
        chosen = make(16 * 4), randoms = make(16 * 4), settings = make(WGSL.SAMPLING_BYTES, UNIFORM | COPY_DST),
        parts = make(WGSL.samplePartsBytes(most));
      const b = { 0: logitsBuffer, 1: probs, 2: order, 3: state, 4: chosen, 5: randoms, 6: settings, 7: parts };
      // the sampler's dispatches for a vocabulary (the chunks' count is the vocabulary's)
      const lists = new Map();
      const listOf = (vocab) => lists.get(vocab) ?? lists.set(vocab, samplerDispatches(kind, pipes, b, vocab)).get(vocab);
      const back = make(16 * 4 + WGSL.STATE_BYTES, MAP_READ | COPY_DST);
      // steps SAMPLE dispatches in one submission, from the case's history; the ids and the state back
      const sampled = async (c, logits, history, draws, steps, stops = []) => {
        device.queue.writeBuffer(logitsBuffer, 0, logits);
        device.queue.writeBuffer(state, 0, WGSL.samplingState({ token: history[history.length - 1], pos: 40, history }));
        device.queue.writeBuffer(chosen, 0, new Uint32Array(16).fill(SENTINEL_ID));
        device.queue.writeBuffer(randoms, 0, new Float32Array(draws));
        device.queue.writeBuffer(settings, 0, WGSL.samplingSettings({ vocab: c.vocab, temperature: c.temperature, topp: c.topp, penalty: c.penalty, stops }));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        const list = listOf(c.vocab);
        for (let i = 0; i < steps; i++) list.forEach((d) => run(pass, d));
        pass.end();
        encoder.copyBufferToBuffer(chosen, 0, back, 0, 16 * 4);
        encoder.copyBufferToBuffer(state, 0, back, 16 * 4, WGSL.STATE_BYTES);
        device.queue.submit([encoder.finish()]);
        await back.mapAsync(MAP_READ);
        const words = new Uint32Array(back.getMappedRange().slice(0));
        back.unmap();
        return { ids: words.subarray(0, 16), state: words.subarray(16) };
      };
      // the tokens of a run held to the CPU's, step by step (the CPU's history goes on with the GPU's tokens)
      const judge = (c, logits, history, draws, got, stops = []) => {
        const cpu = Float32Array.from(logits), seen = [...history];
        let k = 0;
        for (; k < draws.length; k++) {
          WGSL.penalizeLikeCpu(cpu, seen, c.penalty);
          const { first, picks } = acceptable(cpu, c, draws[k]);
          checked++;
          // the ties case holds SAMPLE to the token itself: equal probabilities in the order of their index, the
          // random number in the middle of the fifth's share (no border near): a token of the same logit is not enough
          const right = c.ties && c.temperature ? got.ids[k] === first : picks.has(got.ids[k]);
          if (!right) {
            wrong++;
            problems.push(`${c.vocab} spread ${c.spread ?? (c.sparse ? "sparse" : "tied")} top-p ${c.topp} T ${c.temperature} r ${draws[k].toFixed(3)}: ${got.ids[k]}, the CPU ${first}`);
            return;
          }
          if (got.ids[k] !== first) edge++;
          if (stops.includes(got.ids[k])) break;
          seen.push(got.ids[k]);
        }
        // the state: stopped after a stop token (its id written, nothing after it), else at the next position
        const stopped = k < draws.length, taken = stopped ? k + 1 : draws.length;
        const s = got.state, fed = stopped ? k : draws.length;
        const right = s[5] === taken && s[7] === (stopped ? 1 : 0) && s[1] === 40 + fed && s[6] === history.length + fed &&
          s[4] === (fed ? got.ids[fed - 1] : history[history.length - 1]) && got.ids[taken] === SENTINEL_ID &&
          (fed === 0 || s[8 + ((history.length + fed - 1) % WGSL.REPETITION_WINDOW)] === got.ids[fed - 1]);
        if (!right) {
          wrong++;
          problems.push(`the state after ${draws.length} tokens${stops.length ? " and a stop token" : ""}: ${[...s.subarray(0, 8)].join(" ")}`);
        }
      };
      for (const c of cases) {
        const logits = c.ties ? tiedLogits(c.vocab) : c.sparse ? sparseLogits(c.vocab) : madeUpLogits(c.vocab, c.spread, c.peaks);
        const ranked = [...logits.keys()].sort((a, b) => logits[b] - logits[a]);
        // 70 tokens: the 6 before the window two of the most likely (3rd and 4th, which must not be penalized), then the
        // window: the three most likely twice each (a repeat is penalized once), early and late in it (both halves of
        // the ring), with likely ones and negative ones between. Short: 20 tokens, and token 0 made one of the most likely
        const history = c.short ? [ranked[1], ranked[5], ...ranked.slice(10, 28)]
          : [ranked[3], ranked[4], ...ranked.slice(-4), ranked[0], ...ranked.slice(5, 35), ranked[1], ...ranked.slice(-30, -10), ranked[0], ranked[2],
            ranked[1], ...ranked.slice(35, 43), ranked[2]];
        if (c.short) logits[0] = (logits[ranked[0]] + logits[ranked[1]]) / 2;
        let random = c.random;
        if (random === "fifth") {
          // the middle of the fifth tied token's share in the CPU's walk (the ties in the order of their index)
          const walk = WGSL.walkLikeCpu(logits, c.temperature, c.topp);
          const k = walk.tokens.map((token, at) => [token, at]).filter(([token]) => logits[token] === TIED_RUN)[4][1];
          random = (walk.cumulative[k - 1] + walk.cumulative[k]) / 2 / walk.mass;
        }
        judge(c, logits, history, [random], await sampled(c, logits, history, [random], 1));
        postMessage({ alive: true });
      }
      // runs: the i-th random number for the i-th token, and a stop token (the run's third token, taken again), second
      // of two and last of eight: the settings hold the stop tokens four to a vec4, and the list's models have up to
      // five (sarashina2.2, CAT-Translate, llm-jp-4: T151's review)
      for (const vocab of [1003, 128256]) {
        const c = { vocab, spread: 1, topp: 0.9, temperature: 0.7, penalty: 1.3 }, logits = madeUpLogits(vocab, 1, 40);
        const history = [...Array(20)].map(() => (Math.random() * vocab) | 0), draws = [0.05, 0.95, 0.5, 0.25];
        const got = await sampled(c, logits, history, draws, 4);
        judge(c, logits, history, draws, got);
        const stop = got.ids[2];
        for (const stops of [[NOT_A_TOKEN, stop], [...Array(WGSL.STOPS_MOST - 1)].map((_, i) => NOT_A_TOKEN + i).concat(stop)]) {
          judge(c, logits, history, draws, await sampled(c, logits, history, draws, 4, stops), stops);
        }
      }
    });
  } catch (error) {
    return { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
  } finally {
    owned.forEach((b) => b.destroy());
  }
  return { worstRelative: 0, ok: wrong === 0, tokens: checked, edge, ...(problems.length ? { problems: problems.slice(0, 3) } : {}) };
}
const SENTINEL_ID = 0xdeadbeef;
// stop tokens past any vocabulary: the list's before the one that stops the run
const NOT_A_TOKEN = 200000;
// The check of a run of tokens on the GPU (T151): the check's small model (two layers of 256, 4 heads of 64 and 2 of
// K and V, a vocabulary of 1003) generates 6 tokens in one submission from position 5, greedy and sampled (top-p 0.9,
// the penalty), and each token is held to what the CPU's sampling picks from the logits of the same forward pass in
// JavaScript (layerReference's layers, the embedding and the classifier in float64), fed the GPU's tokens before it:
// the first index of the largest logit or one within 1e-4 of the logits' largest magnitude of it (the GPU's float32
// forward pass is off by about 1e-6 of it), or a token acceptable() takes. Then the sampled run again with its fourth
// token as a stop token: the same three tokens, the stop written, and the run stopped there.
// T175: the layers are tokenForm()'s. On DP4A the reference takes the vectors the GPU quantized (each pass records
// them: generationParts), as checkLayer does, and each is held to quantize_x of the reference's values where it was
// made (quantizedOff): quantizing its own values instead, a value on a rounding's edge in float32 goes either way and
// moved this model's logits by up to 6.5% of the largest (a JavaScript emulation, 360 tokens, .tmp/t175/band.mjs)
const GENERATION_CHECK_POS = 5, GENERATION_CHECK_TOKENS = 6;
async function checkGeneration() {
  const model = GENERATE_CHECK, shape = layerShape(model), { dim, hidden, kvDim, headSize } = shape, form = tokenForm();
  const pos = GENERATION_CHECK_POS, count = GENERATION_CHECK_TOKENS, positions = pos + count + 1;
  const matrix = ([rows, n], scale) => ({ w: new Uint8Array(rows * n).map(() => (Math.random() * 256) | 0), s: floats(rows * n / GROUP, scale) });
  const cache = () => new Uint16Array(positions * kvDim).map((_, i) => (i < pos * kvDim ? toHalf((Math.random() - 0.5) * 4) : 0));
  const data = { layers: [...Array(model.layers)].map(() => Object.fromEntries(Object.entries(shape.matrices).map(([key, m]) => [key, matrix(m, 0.01 * Math.sqrt(256 / m[1]))]))),
    classifier: matrix([model.vocab, dim], 0.002), norms: new Float32Array((2 * model.layers + 1) * dim).map(() => 0.5 + Math.random()),
    keys: [...Array(model.layers)].map(cache), values: [...Array(model.layers)].map(cache), eps: EPS };
  const history = [...Array(pos + 1)].map(() => (Math.random() * model.vocab) | 0), draws = randomsOf(count);
  const embedded = (token) => {
    const signed = new Int8Array(data.classifier.w.buffer), h = new Float64Array(dim);
    for (let i = 0; i < dim; i++) h[i] = signed[token * dim + i] * data.classifier.s[(token * dim + i) / GROUP | 0];
    return h;
  };
  // the logits of the GPU's tokens, position by position, in float64 (the caches go on as the GPU's). vectors: on DP4A
  // the GPU's quantized vectors of each pass ([token][layer × 4 + point, then the classifier's] {xq, xs}); quantizing
  // collects where one of them is not quantize_x's of the reference's values
  const referenceLogits = (tokens, vectors, quantizing) => {
    const keys = data.keys.map((k) => Uint16Array.from(k)), values = data.values.map((v) => Uint16Array.from(v)), out = [];
    for (let k = 0; k < tokens.length; k++) {
      const p = pos + k;
      let h = embedded(tokens[k]);
      // the GPU's quantized vector in place of x (T175), held to quantize_x of x
      const taken = (at, x, what) => {
        const { xq, xs } = vectors[k][at], { wrong } = quantizedOff(x, xq, xs, at === 4 * model.layers ? NORMED_SCALE_LINE : scaleLine(INPUTS[at % 4]));
        if (wrong) quantizing.push(`token ${k}, ${what}: ${wrong}`);
        return dequantized(xq, xs);
      };
      data.layers.forEach((m, l) => {
        const r = layerReference(shape, p, { ...m, h, norms: data.norms.subarray(2 * l * dim, (2 * l + 2) * dim), keys: keys[l], values: values[l],
          angles: layerAngles(headSize, p), eps: data.eps }, vectors ? (point, x) => taken(4 * l + INPUTS.indexOf(point), x, `layer ${l} ${point}`) : undefined);
        keys[l].set(r.keys, p * kvDim);
        values[l].set(r.values, p * kvDim);
        h = r.h;
      });
      let squares = 0;
      for (const value of h) squares += value * value;
      const scale = 1 / Math.sqrt(squares / dim + data.eps), final = 2 * model.layers * dim;
      const normed = h.map((value, i) => data.norms[final + i] * (scale * value)), x = vectors ? taken(4 * model.layers, normed, "the classifier") : normed;
      const signed = new Int8Array(data.classifier.w.buffer), logits = new Float32Array(model.vocab);
      for (let r = 0; r < model.vocab; r++) {
        let sum = 0;
        for (let i = 0; i < dim; i++) sum += signed[r * dim + i] * data.classifier.s[(r * dim + i) / GROUP | 0] * x[i];
        logits[r] = sum;
      }
      out.push(logits);
    }
    return out;
  };
  const verdicts = {};
  let tokens = 0, edge = 0;
  const problems = [];
  try {
    await scoped(async (owned) => {
      const pipes = await generationPipes(headSize, form);
      const parts = generationParts(model, form, pipes, positions, owned, data);
      for (const settings of [{ temperature: 0, topp: 0.9, penalty: 1 }, { ...GENERATE_SETTINGS }]) {
        const runOnce = async (stops = []) => {
          // (the caches need no reset: a run writes each position's row before its attention reads it)
          parts.reset(WGSL.samplingState({ token: history[pos], pos, history }), draws, WGSL.samplingSettings({ vocab: model.vocab, ...settings, stops }));
          return generationRun(parts, count, count);
        };
        const got = await runOnce();
        const sampled = Math.min(got.state[5], count), fed = [history[pos], ...got.ids.subarray(0, sampled - 1)];
        const vectors = parts.record ? parts.recorded(await readBack(device.createCommandEncoder(), parts.record, parts.recording), sampled) : undefined;
        const quantizing = [];
        const logits = referenceLogits(fed, vectors, quantizing), seen = [...history];
        problems.push(...quantizing.slice(0, 3).map((why) => `T ${settings.temperature}, ${why}`));
        for (let k = 0; k < sampled; k++) {
          const cpu = logits[k];
          WGSL.penalizeLikeCpu(cpu, seen, settings.penalty);
          let largest = 0;
          for (const value of cpu) largest = Math.max(largest, Math.abs(value));
          const { first, picks } = acceptable(cpu, settings, draws[k], 1e-4 * largest);
          tokens++;
          if (!picks.has(got.ids[k])) {
            problems.push(`T ${settings.temperature}, token ${k}: ${got.ids[k]}, the CPU ${first}`);
            break;
          }
          if (got.ids[k] !== first) edge++;
          seen.push(got.ids[k]);
        }
        if (got.state[5] !== count || got.state[1] !== pos + count || got.state[7] !== 0) problems.push(`T ${settings.temperature}: the state ${[...got.state.subarray(0, 8)].join(" ")}`);
        if (settings.temperature) {
          // the fourth token a stop token (where it is not among the first three), sixth in the list of stop tokens
          const stop = got.ids[3];
          if (!got.ids.subarray(0, 3).includes(stop)) {
            // the stop token sixth of six (in the second vec4 of the settings)
            const again = await runOnce([...Array(5)].map((_, i) => NOT_A_TOKEN + i).concat(stop));
            const same = again.ids.subarray(0, 4).every((id, k) => id === got.ids[k]);
            if (!same || again.state[5] !== 4 || again.state[7] !== 1 || again.state[1] !== pos + 3) {
              problems.push(`a stop token: ${[...again.ids.subarray(0, 4)].join(" ")} against ${[...got.ids.subarray(0, 4)].join(" ")}, the state ${[...again.state.subarray(0, 8)].join(" ")}`);
            }
          }
        }
        postMessage({ alive: true });
      }
    });
    verdicts["tokens on the GPU"] = { worstRelative: 0, ok: problems.length === 0, tokens, edge, layer: form.name, ...(problems.length ? { problems } : {}) };
  } catch (error) {
    verdicts["tokens on the GPU"] = { worstRelative: NaN, ok: false, error: String(error?.message ?? error) };
  }
  return verdicts;
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
// T150's review: paired() for several things at once, in turn (T147's timeForms): n for each as paired() finds it, then
// PAIRS rounds of a submission of n and one of 2n of each, so that a device's warming and load fall on all of them
// alike. Each: { ms, dispatches (n), ratio } as paired() says it, unsteady past STEADY after the rounds are taken
// once more, or { error } where its 2n took no longer than its n. On a fallback adapter one submission of 1 each.
async function interleaved(submissions, most) {
  if (fallback) {
    const results = [];
    for (const submission of submissions) results.push({ ms: await submission(1), dispatches: 1 });
    return results;
  }
  const counts = [];
  for (const submission of submissions) {
    await submission(2);
    let n = 1;
    while ((await submission(n)) < SUBMISSION_MS && n < most) n *= 2;
    counts.push(n);
    postMessage({ alive: true });
  }
  for (let tries = 0; ; tries++) {
    const differences = submissions.map(() => []), ratios = submissions.map(() => []);
    for (let round = 0; round < PAIRS; round++) {
      for (let i = 0; i < submissions.length; i++) {
        const once = await submissions[i](counts[i]), twice = await submissions[i](2 * counts[i]);
        differences[i].push(twice - once);
        ratios[i].push(twice / once);
      }
      postMessage({ alive: true });
    }
    const results = submissions.map((_, i) => {
      const ratio = middle(ratios[i]), took = middle(differences[i]);
      return { ms: took, dispatches: counts[i], ratio, steady: took > 0 && ratio >= STEADY[0] && ratio <= STEADY[1] };
    });
    if (tries || results.every((r) => r.steady)) {
      return results.map(({ steady, ...r }) => (r.ms > 0 ? { ...r, ...(steady ? {} : { unsteady: true }) }
        : { error: `a submission of ${2 * r.dispatches} took no longer than one of ${r.dispatches}: the device's load moved` }));
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
    else if (data.step === "layer steps") result = await layerSteps();
    else if (data.step === "generate") result = await generate();
    else if (data.step === "overhead") result = await overhead();
    else if (data.step === "prompt") result = await prompt(data.counts);
    else if (data.step === "ceilings") result = await ceilings();
    else if (data.step === "bridge") result = await bridge(data.memory, data.rounds);
    postMessage({ step: data.step, result });
  } catch (error) {
    postMessage({ step: data.step, error: String(error?.message ?? error) });
  }
};
