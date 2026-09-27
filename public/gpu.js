// gpu.js (T135, T147, T148): the model page's GPU, in a worker of its own. forward.js (in the model's worker) makes it
// wherever the worker has WebGPU and the model is one it takes (T148: by default, no option), and hands it the blocks
// of a prompt (up to plan.batch tokens) one at a time that it finds faster here than on the CPU, waiting in
// Atomics.wait for the answer: the model's worker cannot wait for a promise while Python calls forwardMany(), and this
// one can (T94's design; T134's bridge measured 12 to 95 µs a round trip).
//
//   { type: "start", memory, plan }  the layers of the model onto the GPU, read from the shared memory of forward.js
//                                    (plan: what forward.js says of them, addresses in that memory). Says
//                                    { type: "progress" } after every step (a layer's weights, a shader compiled,
//                                    checked, timed: STEP_MS each at most, T147), then { type: "ready",
//                                    adapter, key, bytes, seconds, form, attention, forms, remembered, blocks } or
//                                    { type: "unusable", reason }. T148: a fallback adapter (the CPU in the GPU's
//                                    place) is refused before anything is compiled, unless plan.force.fallback
//                                    (tests); plan.remembered ({ key, matrices, attention }: what the page kept of an
//                                    earlier visit) spares the timing of the matrices' shaders where key is this
//                                    adapter's and the shader is still right here; blocks: what a whole block of 16
//                                    and of 64 tokens takes here (forward.js weighs the GPU against the CPU by it)
//   { type: "prompt", serial, count, pos }
//                                    the rows of count tokens at positions pos, pos + 1, ... (embedded by forward.js,
//                                    at plan.rows) through every layer, and their keys and values of every layer into
//                                    plan.staging as float16. The answer is in the control area: words.failed, then
//                                    words.done = serial, while words.beat counts up meanwhile; a failure also says
//                                    { type: "failed", reason }. Nothing is written where words.wanted is no longer
//                                    serial: forward.js gave the request up (T147: a worker that went on late must not
//                                    write into a memory that may hold the next model by then)
//   { type: "tokens", serial, count, pos, from, token, history, length, cache, settings, randoms }
//                                    T152: count steps of a generation from token at pos (the forward pass and the
//                                    sampling of each, SAMPLE's), after the keys and values of positions from to pos - 1
//                                    went up from forward.js's cache; the ids into plan.tokens.ids and the keys and
//                                    values of their positions into plan.staging (see generate() below). The answer is
//                                    in the control area as a prompt's. Where plan.tokens asks for it, "ready" says
//                                    tokens ({ form, ms, forms, remembered }: the layer of a token chosen here and its
//                                    ms a step) or tokensWhy (why they stay on the CPU)
//   { type: "stop" }                 every buffer and the device let go, and the worker ends; T205: it says
//                                    { type: "ended" } as it does (so does a start that ends as "unusable")
//
// A block goes as one submission: per layer the RMSNorm, the matrices of q, k and v (each weight read once for the
// block's tokens, by the tiled shader chosen on this device: see chooseMatrices), T153: their biases (Qwen2) and the
// norms of every head of q and k (Qwen3) where the model has them, RoPE with the keys and values into the
// GPU's own cache (float16), the attention (llama.cpp's flash attention with tiles: every token sees the positions up to
// its own), the output matrix added to the residual, the RMSNorm, the gate and the up matrices, SwiGLU, the down matrix
// added. T154: GPT-2 and GPT-NeoX (plan.layerNorm) have LayerNorm with a bias for either norm, a bias after every
// matrix, no gate (the up matrix is w1, then GELU), GPT-2 no RoPE at all (plan.turned 0: its learned positions are in
// the rows forward.js embeds) and GPT-NeoX RoPE on a part of every head; GPT-NeoX's parallel residual (plan.parallel)
// has the FFN's norm read the layer's input before the output matrix adds to it, into a buffer of its own (xn). The
// last layer stops at its keys and values: nothing of a prompt's token after them is used. Then the block's
// keys and values of every layer are copied out and read back. The activations are float32 (quantized to 8 bits first
// for the packed shaders, as the CPU's matmul_q8 takes them), the weights int8 widened or multiplied as int8. T155:
// int6 weights (T98) are widened to int8 once, as they go onto the GPU; a model in a 64-bit memory (T101) goes as
// one in a 32-bit memory (its addresses are Numbers); a matrix larger than a buffer the device binds, in pieces of rows.
//
// The first message is claimed before anything is awaited (a module worker's port opens at its first await, and a
// message that comes before onmessage is set is lost: T109).

const shaders = import(new URL(`shaders.js${new URL(import.meta.url).search}`, import.meta.url));

// GPUBufferUsage's values (the name itself is missing where there is no WebGPU)
const STORAGE = 0x80, COPY_DST = 0x8, COPY_SRC = 0x4, MAP_READ = 0x1, UNIFORM = 0x40;
// the bytes that go to the GPU through a copy of their own at a time (writeBuffer takes no view of a shared memory
// everywhere, and copies what it is given at once)
const CHUNK = 8 << 20;
// The right tiled shaders are timed together, in turn, on the model's first layer (its seven matrices by a whole block,
// with the quantizations of a packed one): a submission of n passes of the layer and one of 2n, their difference the
// time of n passes (what a submission costs besides its work is in both and drops out: 2.6 to 8.4 ms waited for on
// the owner's Android, T134, where llm-jp-3 150M's gate by 64 tokens is 0.6 ms of work), n doubled until a submission
// of n takes TIMED_MS (at most MOST_PASSES), PAIRS pairs a shader, the shaders in turn within each round (a device that
// warms up or is loaded meanwhile falls on all of them alike: T168's review), the median of each shader's pairs. On a
// fallback adapter (the CPU in the GPU's place: its times are no GPU's) one pair of one pass
const TIMED_MS = 20, MOST_PASSES = 256, PAIRS = 5;

let model = null;  // what is on the GPU for the model: the device, the plan, the buffers, the pipelines, the cache
let starting = false, stopping = false, lost = null;

onmessage = ({ data }) => {
  if (data.type === "open") open(data.plan, data.flow);
  else if (data.type === "weights") take(data);
  else if (data.type === "start") start(data.memory, data.plan);
  else if (data.type === "prompt") prompt(data);
  else if (data.type === "tokens") generate(data);
  else if (data.type === "stop") stop();
};

// The most a step of putting the model on the GPU may take: a shader's compilation, a check, the timing, a layer's
// weights. SwiftShader took up to 90 s to compile llama.cpp's 64×64 tiles on the development machine (T147), a GPU
// compiles in far less; twice that is a GPU that hangs. The worker then gives the GPU up (the prompt stays on the
// CPU), and says { type: "progress" } after every step meanwhile (forward.js gives up a worker that says nothing for
// longer than this: one the browser ended)
const STEP_MS = 180000;
function within(promise, what) {
  let timer;
  const late = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error(`${what} took more than ${STEP_MS / 1000} s`), { late: true })), STEP_MS);
  });
  return Promise.race([promise, late]).finally(() => {
    clearTimeout(timer);
    postMessage({ type: "progress" });
  });
}

// The device, the model's record (model), and the error scopes the layers' buffers go up in; undefined where there is
// none (say(why): unusable() by default) or the worker is stopping
async function openDevice(plan, say = unusable) {
    const wgsl = await shaders;
    if (!self.navigator?.gpu) return say("no WebGPU in a worker here");
    // the browser's own adapter (T148, the review: "high-performance" would keep a laptop's second GPU awake for the
    // whole visit, for prompts that are mostly short; the device measures whichever it gets against its CPU anyway)
    const adapter = await navigator.gpu.requestAdapter();
    if (stopping) return end();
    if (!adapter) return say("no GPU adapter here");
    // T148: a fallback adapter is the CPU doing the GPU's work (SwiftShader, lavapipe): never faster than the CPU's
    // own kernels, and its compilation of the shaders alone took 2 to 4 minutes (T147). Refused before anything is
    // made on it, but where a test asks for it (it is the only WebGPU of CI and of the development machine)
    const info = adapter.info ?? {};
    const fallback = Boolean(info.isFallbackAdapter ?? adapter.isFallbackAdapter);
    if (fallback && !plan.force.fallback) return say("a fallback adapter: the CPU in the GPU's place");
    // the largest buffer the device binds (T155: a matrix larger than it goes in pieces, piecesOf)
    const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    // shader-f16 and subgroups where the adapter has them (a device refuses a feature it lacks), and the adapter's
    // workgroup memory and threads (the tiles and the attention size themselves by them)
    const { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX } = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "subgroups"].filter((name) => adapter.features.has(name)),
      requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX } });
    device.lost.then((info) => { lost = `the GPU was lost (${info.reason}${info.message ? `: ${info.message}` : ""})`; });
    // T148: what the page keeps of an earlier visit counts only for the same adapter and browser
    // (and the shaders of this deployment: a site whose shaders changed chooses anew, the review of T148)
    const key = wgsl.deviceKey(adapter, device);
    const remembered = plan.remembered?.key === key ? plan.remembered : null;
    model = { device, plan, wgsl, owned: [], limit, info, fallback, remembered, adapter, key };
    if (stopping) return end();
    // a buffer the device cannot give fails quietly, as an error of these scopes
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    return model;
}

// ---- T156: a model on the GPU alone. open() makes the device and every buffer of the layers' matrices before a byte
// of them comes (plan: forward.js's gpuOnlyPlan(): each matrix's rows and length, and where a layer's values and scales
// start in the checkpoint; the layers are joined for a token, as a model on the GPU alone always runs its steps here),
// and take() writes each stretch of them to its buffer as the worker posts it (routes: [start, end) in the checkpoint,
// the buffer and the offset in it). flow[0] counts the bytes on the GPU (the worker waits on it: a disk read faster
// than the GPU takes it would pile up in the messages). What comes before the buffers are made waits (backlog). A
// failure (no adapter, a buffer the device refused) takes the rest as if written, and start() says it
let opening = null, opened = false, failure = null, flow = null;
const backlog = [];
function open(plan, shared) {
  starting = true;
  flow = new BigInt64Array(shared);
  opening = (async () => {
    try {
      if (!(await openDevice({ ...plan, tokens: true }, (reason) => { failure = reason; }))) {
        failure ??= "the GPU's worker was stopped";
        return;
      }
      model.direct = { routes: [], partial: new Map(), bytes: 0 };
      model.direct.bytes = await uploadLayers(model);
      if (stopping) failure ??= "the GPU's worker was stopped";
    } catch (error) {
      failure = String(error?.message ?? error);
    } finally {
      opened = true;
      starting = startAsked;  // (a start() waiting on this is still starting)
      for (const data of backlog.splice(0)) take(data);
      if (stopping) end();
    }
  })();
}
// the bytes at offset of the checkpoint (bytes: a Uint8Array of their own) onto the buffers whose routes they fall in.
// writeBuffer takes whole words: the bytes of a word a stretch begins or ends in the middle of wait for the rest of it
// (partial), which another message brings
function take(data) {
  if (!opened) return void backlog.push(data);
  const { offset, bytes } = data;
  if (!failure) {
    try {
      const { routes } = model.direct, end = offset + bytes.length;
      let lo = 0, hi = routes.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (routes[mid][1] <= offset) lo = mid + 1;
        else hi = mid;
      }
      for (let i = lo; i < routes.length && routes[i][0] < end; i++) {
        const [start, stop, target, at] = routes[i];
        let a = Math.max(start, offset) - offset, here = at + (a + offset - start);
        const b = Math.min(stop, end) - offset;
        for (; a < b && here % 4; a++, here++) partialByte(target, here, bytes[a]);
        const whole = (b - a) & ~3;
        if (whole) model.device.queue.writeBuffer(target, here, bytes, a, whole);
        for (let k = a + whole; k < b; k++) partialByte(target, here + (k - a), bytes[k]);
      }
    } catch (error) {
      failure = String(error?.message ?? error);
    }
  }
  const count = BigInt(bytes.length), done = () => {
    Atomics.add(flow, 0, count);
    Atomics.notify(flow, 0);
  };
  if (failure) done();
  else model.device.queue.onSubmittedWorkDone().then(done, done);
}
// a byte of target at the byte offset at, the word it is in written once all four of its bytes are there
function partialByte(target, at, value) {
  const { partial } = model.direct, word = at - (at % 4);
  if (!partial.has(target)) partial.set(target, new Map());
  const words = partial.get(target);
  if (!words.has(word)) words.set(word, { bytes: new Uint8Array(4), count: 0 });
  const held = words.get(word);
  held.bytes[at % 4] = value;
  if (++held.count < 4) return;
  model.device.queue.writeBuffer(target, word, held.bytes);
  words.delete(word);
  if (!words.size) partial.delete(target);
}

let startAsked = false;
async function start(memory, plan) {
  const began = performance.now();
  starting = startAsked = true;
  try {
    let bytes;
    if (plan.direct) {
      // T156: opened before the checkpoint came, its layers written as they came (the worker waited for the last)
      await opening;
      if (failure && !stopping) return unusable(`the layers did not go up to the GPU (${failure})`);
      if (stopping || !model) return end();
      if (model.direct.partial.size) return unusable("the layers' bytes did not all come to the GPU");
      model.memory = memory;
      model.plan = plan;
      bytes = model.direct.bytes + (await uploadRest(model));
    } else {
      if (!(await openDevice(plan))) return;
      model.memory = memory;
      bytes = await upload(model);
    }
    const { device, adapter, key } = model;
    if (stopping) return end();
    await prepare(model);
    await chooseAttention(model);
    if (stopping) return end();
    await chooseMatrices(model);
    if (stopping) return end();
    bindLayers(model);
    grow(model, Math.max(Math.min(plan.kvStart, plan.seqLen), Math.min(plan.batch, plan.seqLen)));
    await within(device.queue.onSubmittedWorkDone(), "the GPU's work");
    const invalid = await device.popErrorScope(), full = await device.popErrorScope();
    if (stopping) return end();
    if (invalid || full) return unusable(`the GPU did not take the layers (${(invalid ?? full).message})`);
    // (not for the page's tests: SwiftShader took more than STEP_MS to time Llama 3.2 1B's shaders, T147 in CI)
    const blocks = plan.force.quick ? [] : await within(timeBlocks(model), "timing a block");
    if (stopping) return end();
    // T152: a token and the ones after it, where the model's layers were put up for them (tokensLayout). What fails
    // here leaves the tokens on the CPU and the prompts on the GPU
    if (plan.tokens && !model.tokensWhy) {
      try {
        await chooseTokens(model);
      } catch (error) {
        model.tokensWhy = String(error?.message ?? error);
        model.gen = null;
      } finally {
        // (the review of T156: the first layer read back for the checks of a model on the GPU alone goes with them: it
        // stayed in this worker for the whole visit, 113 MB of Llama 3.2 3B, 245 MB of Llama 3.1 Swallow 8B)
        model.firstLayer = undefined;
      }
      if (stopping) return end();
    }
    if (lost) return unusable(lost);
    const g = model.gen;
    postMessage({ type: "ready", adapter: describe(adapter), key, bytes, seconds: (performance.now() - began) / 1000,
      form: model.form.name, attention: model.attention.name, forms: model.forms, remembered: Boolean(model.form.remembered), blocks,
      tokens: g?.form ? { form: g.form.name, ms: g.ms, forms: g.forms, remembered: g.forms.some((f) => f.remembered), pieces: model.tables.classifier.length } : null,
      tokensWhy: plan.tokens ? model.tokensWhy ?? null : undefined });
  } catch (error) {
    unusable(String(error?.message ?? error));
  } finally {
    starting = false;
  }
}

function describe(adapter) {
  const info = adapter.info ?? {};
  const name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ") || "a GPU";
  return (info.isFallbackAdapter ?? adapter.isFallbackAdapter) ? `${name} (a fallback adapter: the CPU in the GPU's place)` : name;
}
// T148: the adapter and the browser whose shaders the page remembers: another GPU, driver architecture or browser
// version chooses anew (the user agent carries the browser's version)
// the tiled shaders of T146 this device can make (shaders.js's promptForms)
const candidates = ({ device, wgsl }) => wgsl.devicePromptForms(device);

function unusable(reason) {
  postMessage({ type: "unusable", reason });
  end();
}
function stop() {
  stopping = true;
  if (!starting) end();  // else start() ends at its next step
}
// every buffer and the device let go (T94: a model changed for another leaves nothing on the GPU), and this worker ends.
// T205: it says { type: "ended" } first, for forward.js's release() to read the next model after it. Every way out of
// start() comes here (a stop in the middle of it at its next step), and a stop before or after start() at once
function end() {
  if (model) {
    model.owned.forEach((buffer) => buffer.destroy());
    model.cache?.owned.forEach((buffer) => buffer.destroy());
    model.device.destroy();
    model = null;
  }
  postMessage({ type: "ended" });
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
// bytes of the shared memory at address onto the GPU (at offset of target), a CHUNK at a time through a copy
let scratch;
function copyIn(m, target, address, bytes, offset = 0) {
  scratch ??= new Uint8Array(CHUNK);
  for (let done = 0; done < bytes; done += CHUNK) {
    const n = Math.min(CHUNK, bytes - done);
    scratch.set(new Uint8Array(m.memory.buffer, address + done, n));
    m.device.queue.writeBuffer(target, offset + done, scratch, 0, n);
  }
}
// T152's review (T160): values float32 of the shared memory at address narrowed to float16 onto the GPU (at offset
// of target, in bytes of float16), a CHUNK of them at a time: a float32 cache's keys and values going up
let narrowed;
function narrowIn(m, target, address, values, offset = 0) {
  narrowed ??= new Uint16Array(CHUNK / 2);
  for (let done = 0; done < values; done += narrowed.length) {
    const n = Math.min(narrowed.length, values - done);
    m.wgsl.halvesOf(new Float32Array(m.memory.buffer, address + done * 4, n), narrowed);
    m.device.queue.writeBuffer(target, offset + done * 2, narrowed, 0, n);
  }
}
async function readBack(m, source, bytes, from = 0) {
  const target = m.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: MAP_READ | COPY_DST });
  try {
    const encoder = m.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, from, target, 0, Math.ceil(bytes / 4) * 4);
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
// (a buffer, or { buffer, offset, size }: a part of one, T155's pieces of a matrix writing into their rows of the output)
const bind = (m, pipeline, buffers) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: buffers.map((buffer, binding) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });
function dispatch(pass, pipeline, group, x, y = 1, z = 1) {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(x, y, z);
}

// every layer's matrices (values and scales, as the checkpoint holds them; T154: no w3 where there is no gate) and its
// vectors: the weights of its two norms, and (T153) Qwen2's biases of q, k and v, Qwen3's norms of a head of q and of
// k, (T154) GPT-2's and GPT-NeoX's biases of the two LayerNorms and of every matrix (plan.vectors: each one's address in
// the shared memory and its floats a layer). T155: a matrix in pieces of whole rows (piecesOf), a buffer of values and
// one of scales each; int6 (plan.matrices' six) widened to int8 on the way (widener), its scales as they are (the
// quarter of an int6 group's is already the int8 values' scale, T98). The addresses are Numbers in a 64-bit memory too
// (exact to 2^53; the views and copies take them as they are)
// T152: where a token goes on the GPU too (plan.tokens), q, k and v are one matrix a layer, and gate and up one (the
// layer of a token reads them so: shaders.js's fusedMatVec and fusedDp4aMatVec, T150 and T175), each a range of it for
// the prompt's tiled shaders (tokensLayout), and the classifier, the embedding, the final norm and RoPE's table go up
// as well (uploadTokens)
// T156, a model on the GPU alone (m.direct): the layers' buffers are made as the worker opens, before a byte comes, and
// each stretch of the checkpoint they hold is a route ([start, end) in the checkpoint, the buffer and its offset) that
// take() writes the bytes to as they come; the rest (the tables, the norms) goes up from the shared memory as start()
// is asked (uploadRest)
async function upload(m) {
  try {
    const bytes = await uploadLayers(m);
    if (stopping) return bytes;
    return bytes + (await uploadRest(m));
  } finally {
    m.widen?.done();
  }
}
async function uploadLayers(m) {
  const { plan } = m, group = m.wgsl.GROUP;
  let bytes = 0;
  m.matrices = Object.fromEntries(Object.entries(plan.matrices).map(([name, { rows, n }]) =>
    [name, { rows, n, pieces: piecesOf(m, rows, n).map(([first, count]) => ({ first, rows: count, layers: [] })) }]));
  const together = plan.tokens ? tokensLayout(m) : null;
  m.together = together;
  // (a model on the GPU alone opens before its tables are placed, and is int8: T156)
  const tables = together && !m.direct ? Object.values(tablesOf(plan)) : [];
  const six = [...Object.values(plan.matrices), ...tables].some((matrix) => matrix.six);
  // (kept for the tables, uploadRest; upload() lets it go)
  const widen = m.widen = six ? await widener(m, tables) : null;
  // (T156: a model on the GPU alone reads its first layer back to check a token's layer against: firstLayer)
  const usage = STORAGE | COPY_DST | (m.direct ? COPY_SRC : 0);
  for (let l = 0; l < plan.layers; l++) {
    // a layer's matrices that are one (T152): a buffer of values and one of scales for all of them
    const joined = together && Object.fromEntries(Object.entries(together.sizes).map(([name, [valueBytes, scaleBytes]]) =>
      [name, [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)]]));
    if (joined) (m.joined ??= []).push(joined);
    for (const [name, matrix] of Object.entries(plan.matrices)) {
      const [valuesAt, scalesAt] = matrix.layers[l];
      const home = together?.homes[name];
      for (const piece of m.matrices[name].pieces) {
        const valueBytes = piece.rows * matrix.n, scaleBytes = (valueBytes / group) * 4;
        // its own buffers, or its range of the layer's joined ones
        const [values, scales] = home
          ? joined[home.joined].map((b, i) => ({ buffer: b, offset: home.at[i], size: i ? scaleBytes : valueBytes }))
          : [buffer(m, valueBytes, usage), buffer(m, scaleBytes, usage)];
        const from = [valuesAt + piece.first * matrix.n, scalesAt + (piece.first * matrix.n / group) * 4];
        if (m.direct) {
          // (T156: int8 alone, whose bytes are the buffer's as they come)
          m.direct.routes.push([from[0], from[0] + valueBytes, values.buffer ?? values, values.offset ?? 0],
            [from[1], from[1] + scaleBytes, scales.buffer ?? scales, scales.offset ?? 0]);
        } else {
          // an int6 row is 3/4 of an int8 one (24 bytes a group of 32)
          if (matrix.six) widen.into(values, valuesAt + (piece.first * matrix.n * 3) / 4, valueBytes / group);
          else copyIn(m, values.buffer ?? values, from[0], valueBytes, values.offset ?? 0);
          copyIn(m, scales.buffer ?? scales, from[1], scaleBytes, scales.offset ?? 0);
        }
        piece.layers.push([values, scales]);
        bytes += valueBytes + scaleBytes;
      }
    }
    if (m.direct) continue;
    // what was written waits in memory until the GPU takes it: let it, before more comes
    await within(m.device.queue.onSubmittedWorkDone(), `layer ${l + 1}'s weights`);
    if (stopping) return bytes;
  }
  m.direct?.routes.sort((a, b) => a[0] - b[0]);
  return bytes;
}
// the tables of a token (where the layers are joined for them, T152), and the norms, from the shared memory
async function uploadRest(m) {
  const { plan } = m;
  let bytes = 0;
  if (m.together) bytes += await uploadTokens(m, m.widen);
  m.vectors = {};
  for (const [name, { at, size }] of Object.entries(plan.vectors)) {
    const vectorBytes = plan.layers * size * 4;
    m.vectors[name] = buffer(m, vectorBytes, STORAGE | COPY_DST);
    copyIn(m, m.vectors[name], at, vectorBytes);
    bytes += vectorBytes;
  }
  return bytes;
}

// T155: the pieces of a matrix of rows × n, [first row, rows] each: whole rows, no more bytes than a buffer the device
// binds (m.limit, or plan.force.pieceBytes in the tests), each piece's first row where its part of the output may be
// bound (a multiple of minStorageBufferOffsetAlignment over the 4 bytes of a float32). One piece where the matrix
// fits, as every matrix of the models of the list does at WebGPU's least limit of 128 MiB (the largest, Qwen2.5 7B's
// w1, is 68 MB): llama.cpp's WebGPU keeps each tensor in one buffer within maxStorageBufferBindingSize and refuses a
// larger one (ggml-webgpu.cpp, ggml_backend_webgpu_buffer_type_get_max_size, commit 2145525a, MIT; no line copied),
// where this cuts it by rows, as T94's design had it for a classifier past a buffer
function piecesOf(m, rows, n, most = m.plan.force.pieceBytes) {
  const bytes = Math.min(m.limit, most ?? Infinity), align = m.device.limits.minStorageBufferOffsetAlignment / 4;
  if (rows * n <= bytes) return [[0, rows]];
  const step = Math.floor(bytes / n / align) * align;
  if (!step) throw new Error(`${align} rows of ${n} weights are more than a buffer of this GPU (${bytes} bytes)`);
  return Array.from({ length: Math.ceil(rows / step) }, (_, i) => [i * step, Math.min(step, rows - i * step)]);
}

// T152: the tables a token needs besides the layers (plan.tokens): the classifier, and the embedding where it is
// another table (else the classifier is the embedding too)
const tablesOf = (plan) => ({ classifier: plan.tokens.classifier, ...(plan.tokens.embedding ? { embedding: plan.tokens.embedding } : {}) });
// T152: how a token's layer holds its matrices (shaders.js's fusedMatVec and fusedDp4aMatVec read q, k and v as one
// matrix, gate and up as one, up's rows after gate's): a buffer of values and one of scales a layer for each (sizes),
// and each matrix's range of them (homes: its offsets in bytes), which the prompt's tiled shaders bind as a matrix of
// its own. A range must start where the device binds a buffer (minStorageBufferOffsetAlignment: a matrix of a
// multiple of 2048 weights, the values and the scales alike at 256; stories15M's k starts at 82944 weights, T150's
// (b)), and a matrix a token reads must be one piece (T155). The tables may be in pieces (T209: uploadTokens). null
// where it cannot, and why in m.tokensWhy: the prompts go on the GPU as before, the tokens stay on the CPU
function tokensLayout(m) {
  const { plan } = m, align = m.device.limits.minStorageBufferOffsetAlignment, group = m.wgsl.GROUP;
  const why = (reason) => {
    m.tokensWhy = reason;
    return null;
  };
  for (const name of ["wo", "w2"]) if (m.matrices[name].pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
  const homes = {}, sizes = {};
  for (const [joined, names] of Object.entries({ qkv: ["wq", "wk", "wv"], gateUp: ["w1", "w3"] })) {
    let values = 0, scales = 0;
    for (const name of names) {
      const matrix = m.matrices[name];
      if (matrix.pieces.length > 1) return why(`${name} is past a buffer of this GPU`);
      if (values % align || scales % align) {
        return why(`${name} would not start where this GPU binds a buffer`);
      }
      homes[name] = { joined, at: [values, scales] };
      values += matrix.rows * matrix.n;
      scales += (matrix.rows * matrix.n / group) * 4;
    }
    if (values > m.limit) return why(`${names.join(" and ")} together are past a buffer of this GPU`);
    sizes[joined] = [values, scales];
  }
  return { homes, sizes };
}
// T152: the tables onto the GPU (int6 widened as a layer's matrices are), the final norm's weights, and RoPE's table
// of every position, a row a position: the cos of its headSize / 2 angles, then their sin (shaders.js's fusedMatVec
// reads it so, T151; from the CPU's two tables, Llama 3's scaling in them)
// T209: a table in pieces of rows where it is past what the device binds (piecesOf: Llama 3.2 3B's classifier is
// 394 MB, the owner's Android binds 256 MiB; T152's review (e)), [{ first, rows, values, scales }] each. The
// classifier is then a dispatch a piece into its range of the logits, and EMBED one a piece (tokenPass)
async function uploadTokens(m, widen) {
  const { plan } = m, group = m.wgsl.GROUP, half = plan.headSize / 2;
  let bytes = 0;
  const table = ({ rows, n, six, at: [valuesAt, scalesAt] }) =>
    piecesOf(m, rows, n, plan.force.tablePieceBytes ?? plan.force.pieceBytes).map(([first, count]) => {
      const valueBytes = count * n, scaleBytes = (valueBytes / group) * 4;
      const values = buffer(m, valueBytes, STORAGE | COPY_DST), scales = buffer(m, scaleBytes, STORAGE | COPY_DST);
      if (six) widen.into(values, valuesAt + (first * n * 3) / 4, valueBytes / group);
      else copyIn(m, values, valuesAt + first * n, valueBytes);
      copyIn(m, scales, scalesAt + (first * n / group) * 4, scaleBytes);
      bytes += valueBytes + scaleBytes;
      return { first, rows: count, values, scales };
    });
  const classifier = table(plan.tokens.classifier);
  m.tables = { classifier, embedding: plan.tokens.embedding ? table(plan.tokens.embedding) : classifier };
  m.finalNorm = buffer(m, plan.dim * 4, STORAGE | COPY_DST);
  copyIn(m, m.finalNorm, plan.tokens.final, plan.dim * 4);
  m.angleTable = buffer(m, plan.seqLen * plan.headSize * 4, STORAGE | COPY_DST);
  const rows = Math.max(1, Math.floor(CHUNK / (plan.headSize * 4))), chunk = new Float32Array(rows * plan.headSize);
  for (let p = 0; p < plan.seqLen; p += rows) {
    const count = Math.min(rows, plan.seqLen - p);
    for (let r = 0; r < count; r++) {
      chunk.set(new Float32Array(m.memory.buffer, plan.cos + (p + r) * half * 4, half), r * plan.headSize);
      chunk.set(new Float32Array(m.memory.buffer, plan.sin + (p + r) * half * 4, half), r * plan.headSize + half);
    }
    m.device.queue.writeBuffer(m.angleTable, p * plan.headSize * 4, chunk, 0, count * plan.headSize);
  }
  bytes += plan.dim * 4 + plan.seqLen * plan.headSize * 4;
  await within(m.device.queue.onSubmittedWorkDone(), "the classifier's weights");
  return bytes;
}

// T155: shaders.js's WIDEN_SIX compiled and checked against JavaScript (sixValues), and into(values, address, groups):
// groups of int6 at address in the shared memory widened into values (a buffer of int8, or T152 a range of one) on
// the GPU. The packed bytes go
// through one buffer of their own, written again for every piece (the queue runs a write after the submissions before
// it); done() lets it go.
async function widener(m, tables = []) {
  const { device, wgsl } = m;
  const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.WIDEN_SIX)), "compiling the widening of int6");
  const owned = [];
  // (T152: and the classifier's and the embedding's, one piece each)
  const largest = Math.max(CHECK_GROUPS * 24, ...Object.entries(m.plan.matrices).filter(([, matrix]) => matrix.six)
    .flatMap(([name, { n }]) => m.matrices[name].pieces.map((piece) => (piece.rows * n * 3) / 4)),
  ...tables.filter((table) => table.six).map(({ rows, n }) => (rows * n * 3) / 4));
  const packed = buffer(m, largest, STORAGE | COPY_DST, owned);
  const run = (values, groups, most = device.limits.maxComputeWorkgroupsPerDimension) => {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    dispatch(pass, pipeline, bind(m, pipeline, [packed, values, uniform(m, new Uint32Array([groups, 0, 0, 0]), owned)]),
      ...wgsl.sixDispatch(groups, most));
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  const done = () => owned.splice(0).forEach((b) => b.destroy());
  try {
    const wrong = await within(checkWiden(m, packed, run), "checking the widening of int6");
    if (wrong) throw new Error(`the widening of int6 is wrong on this GPU: ${wrong}`);
  } catch (error) {
    done();
    throw error;
  }
  return {
    into(values, address, groups) {
      copyIn(m, packed, address, groups * 24);
      run(values, groups);
    },
    done,
  };
}
// The check: CHECK_GROUPS groups of random bytes (any 24 bytes are a group), dispatched over rows of 7 workgroups (a
// second dimension, and threads past the last group), into a buffer with 64 groups more of a known byte after them
// (which no thread may write), against JavaScript's to the bit. The reason it is wrong, or null
const CHECK_GROUPS = 1000;
async function checkWiden(m, packed, run) {
  const bytes = new Uint8Array(CHECK_GROUPS * 24).map(() => (Math.random() * 256) | 0), after = 64 * 32, owned = [];
  try {
    const values = buffer(m, CHECK_GROUPS * 32 + after, STORAGE | COPY_DST | COPY_SRC, owned);
    m.device.queue.writeBuffer(values, 0, new Uint8Array(CHECK_GROUPS * 32 + after).fill(0x5a));
    m.device.queue.writeBuffer(packed, 0, bytes);
    run(values, CHECK_GROUPS, 7);
    const got = new Int8Array(await readBack(m, values, CHECK_GROUPS * 32 + after)), want = m.wgsl.sixValues(bytes);
    const at = want.findIndex((v, i) => got[i] !== v);
    if (at >= 0) return `value ${at % 32} of group ${(at / 32) | 0} is ${got[at]}, JavaScript's ${want[at]}`;
    const past = got.subarray(want.length).findIndex((v) => v !== 0x5a);
    return past >= 0 ? `byte ${past} past the last group was written` : null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// the pipelines of a layer's small steps and the buffers of a block's activations
async function prepare(m) {
  const { plan, wgsl } = m, B = plan.batch, gated = Boolean(plan.matrices.w3);
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // T154: LayerNorm and GELU where the model has them (GPT-2, GPT-NeoX), else RMSNorm and SwiGLU
  [m.norm, m.headNorm, m.add, m.rope, m.activation, m.quantize] = await within(Promise.all([plan.layerNorm ? wgsl.LAYER_NORM : wgsl.RMSNORM,
    wgsl.HEAD_NORM, wgsl.ADD, wgsl.ROPE, gated ? wgsl.SWIGLU : wgsl.GELU, wgsl.QUANTIZE].map((code) => pipelineOf(m, code))), "the small steps' shaders");
  // a block's tokens, each array dense: token t's row at t times its width
  m.x = buffer(m, B * plan.dim * 4, STORAGE | COPY_DST);
  m.xb = buffer(m, B * Math.max(plan.dim, qDim) * 4, STORAGE | COPY_DST);
  m.q = buffer(m, B * qDim * 4);
  m.k = buffer(m, B * kvDim * 4);
  m.v = buffer(m, B * kvDim * 4);
  m.gate = buffer(m, B * plan.hidden * 4, STORAGE | COPY_DST);
  m.up = gated ? buffer(m, B * plan.hidden * 4) : null;
  // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, which xb cannot hold (the attention
  // writes there first)
  m.ffnInput = plan.parallel ? buffer(m, B * plan.dim * 4) : m.xb;
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
  const hidden = uniform(m, new Uint32Array([plan.hidden, 0, 0, 0]));
  m.activationGroup = bind(m, m.activation, gated ? [m.gate, m.up, hidden, m.step] : [m.gate, hidden, m.step]);
  // QUANTIZE of the inputs the matrices read: xb (the norm's, dim wide; the attention's, heads × headSize wide, which
  // is not dim where the heads are of another size, T153) and gate (SwiGLU's)
  const quantizing = (from, n) => ({ group: bind(m, m.quantize, [from, m.xq, m.xs, uniform(m, new Uint32Array([n, n, 0, 0])), m.step]),
    x: Math.ceil(n / wgsl.GROUP / 64) });
  m.quantizeXb = quantizing(m.xb, plan.dim);
  m.quantizeAttention = quantizing(m.xb, qDim);
  m.quantizeFfn = plan.parallel ? quantizing(m.ffnInput, plan.dim) : m.quantizeXb;
  m.quantizeGate = quantizing(m.gate, plan.hidden);
}

// ---- the attention: llama.cpp's flash attention with tiles, with f16 in the workgroup's memory where there is
// shader-f16 and its subgroups where there are (and subgroup_id), else the same without them; each checked first
// against JavaScript (checkAttention), the next one tried where it is wrong
async function chooseAttention(m) {
  const { device, plan, wgsl } = m;
  const half = device.features.has("shader-f16");
  const subgroups = device.features.has("subgroups") && Boolean(navigator.gpu.wgslLanguageFeatures?.has("subgroup_id"));
  const limits = { memory: device.limits.maxComputeWorkgroupStorageSize, threads: threadsOf(device),
    subgroupMin: m.info.subgroupMinSize, subgroupMax: m.info.subgroupMaxSize };
  const tried = [];
  const nameOf = (option) => `llama.cpp flash attention tiles${option.half ? ", f16" : ""}${option.subgroups ? ", subgroups" : ""}`;
  // T148: the one the page remembers for this adapter first (it was the first right one then: the other was wrong)
  const options = [{ half, subgroups }, { half: false, subgroups: false }].filter((o, i) => i === 0 || o.half !== half || o.subgroups !== subgroups)
    .sort((a, b) => (nameOf(b) === m.remembered?.attention) - (nameOf(a) === m.remembered?.attention));
  for (const option of options) {
    const shape = wgsl.flashShape({ headSize: plan.headSize, ...option, ...limits });
    const name = nameOf(option);
    if (plan.force.attention && plan.force.attention !== name) continue;
    if (shape.none) {
      tried.push(`${name}: ${shape.none}`);
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, wgsl.flashTile(shape))), `compiling ${name}`);
      const wrong = await within(checkAttention(m, pipeline), `checking ${name}`);
      if (!wrong) {
        m.attention = { name, pipeline, shape };
        return;
      }
      tried.push(`${name}: ${wrong}`);
    } catch (error) {
      if (error?.late) throw error;
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
  const { plan } = m;
  let forms = candidates(m);
  if (plan.force.matrices) forms = forms.filter((form) => form.name === plan.force.matrices);
  m.forms = [];
  // T148: the shader the page remembers for this adapter, alone, where it is one this device still makes and it is
  // still right here (a driver may have changed under the same names): no other is compiled or timed. Else all of them
  const kept = !plan.force.matrices && forms.find((form) => !form.none && form.name === m.remembered?.matrices);
  if (kept) {
    try {
      const tiled = { ...kept, remembered: true, pipeline: await within(validated(m, () => pipelineOf(m, kept.code, kept.constants)), `compiling ${kept.name}`) };
      const wrong = await within(checkForm(m, tiled), `checking ${kept.name}`);
      if (!wrong) {
        m.forms.push({ name: kept.name, remembered: true });
        m.form = tiled;
        return;
      }
      m.forms.push({ name: kept.name, none: `remembered, but wrong now: ${wrong}` });
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: kept.name, none: `remembered, but ${error?.message ?? error}` });
    }
    if (stopping) return;
  }
  const right = [];
  for (const form of forms) {
    if (plan.force.quick && right.length) break;  // T148: the page's tests, the first right one untimed
    if (form === kept) continue;  // wrong just now
    if (form.none) {
      m.forms.push({ name: form.name, none: form.none });
      continue;
    }
    try {
      const pipeline = await within(validated(m, () => pipelineOf(m, form.code, form.constants)), `compiling ${form.name}`);
      const tiled = { ...form, pipeline };
      const wrong = await within(checkForm(m, tiled), `checking ${form.name}`);
      if (wrong) m.forms.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(tiled);
    } catch (error) {
      if (error?.late) throw error;
      m.forms.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) throw new Error(`no tiled shader is right on this GPU (${m.forms.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.matrices}`})`);
  const ms = right.length > 1 ? await within(timeForms(m, right), "timing the tiled shaders") : [0];
  right.forEach((form, i) => m.forms.push({ name: form.name, ms: ms[i] }));
  m.form = right[ms.indexOf(Math.min(...ms))];
}
// the threads of a workgroup of one dimension this device takes
const threadsOf = (device) => Math.min(device.limits.maxComputeInvocationsPerWorkgroup, device.limits.maxComputeWorkgroupSizeX);

// layer l's matrix name by the tokens of from into to (added where add): a bind group of form and its rows for each
// piece (T155), which writes its rows of to (bound from its first row; a token's row of to is the whole matrix's)
const productGroups = (m, form, name, l, from, to, add, owned = m.owned) => {
  const { rows, n, pieces } = m.matrices[name];
  return pieces.map(({ first, rows: count, layers }) => {
    const out = first ? { buffer: to, offset: first * 4, size: to.size - first * 4 } : to;
    return { rows: count, group: bind(m, form.pipeline, [...layers[l], form.packed ? m.xq : from, out,
      uniform(m, new Uint32Array([count, n / 4, n / m.wgsl.GROUP, 0, n, rows, add ? 1 : 0, 0]), owned), m.step, ...(form.packed ? [m.xs] : [])]) };
  });
};
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

// ms of a pass of the model's first layer with each of forms (see TIMED_MS): the seven matrices by plan.batch tokens,
// each packed one's inputs quantized first
async function timeForms(m, forms) {
  const { device, plan } = m, owned = [], layer = [];
  try {
    device.queue.writeBuffer(m.step, 0, new Uint32Array([plan.batch, 0, 0, 0]));
    const products = [["wq", m.xb, m.q, false, m.quantizeXb], ["wk", m.xb, m.k], ["wv", m.xb, m.v], ["wo", m.xb, m.x, true, m.quantizeAttention],
      ["w1", m.ffnInput, m.gate, false, m.quantizeFfn], ["w3", m.ffnInput, m.up], ["w2", m.gate, m.x, true, m.quantizeGate]]
      .filter(([name]) => m.matrices[name]);
    // (a matrix's input quantized once, before its first piece)
    for (const form of forms) {
      layer.push(await validated(m, () => products.flatMap(([name, from, to, add, quantize]) =>
        productGroups(m, form, name, 0, from, to, add, owned).map((piece, i) => ({ ...piece, quantize: i ? null : quantize })))));
    }
    const submission = async (i, passes) => {
      const form = forms[i], encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      for (let p = 0; p < passes; p++) {
        for (const { group, rows, quantize } of layer[i]) {
          if (form.packed && quantize) dispatch(pass, m.quantize, quantize.group, quantize.x, plan.batch);
          multiply(m, pass, form, group, rows, plan.batch);
        }
      }
      pass.end();
      const began = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - began;
    };
    const passes = [], differences = forms.map(() => []);
    for (let i = 0; i < forms.length; i++) {
      await submission(i, 1);  // warm
      let n = 1;
      while (!m.fallback && n < MOST_PASSES && (await submission(i, n)) < TIMED_MS) n *= 2;
      passes.push(n);
    }
    for (let round = 0; round < (m.fallback ? 1 : PAIRS); round++) {
      for (let i = 0; i < forms.length; i++) {
        const once = await submission(i, passes[i]), twice = await submission(i, 2 * passes[i]);
        differences[i].push((twice - once) / passes[i]);
      }
    }
    return differences.map((d) => d.sort((a, b) => a - b)[d.length >> 1]);
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// every layer's bind groups but those of the cache, with the form chosen. T153: a bias (Qwen2's) added to q, k and
// v, or (Qwen3's) the norm of every head of q and of k, where the model has them. T154: LayerNorm with its bias, and a
// bias after o, w1 and w2 (GPT-2's, GPT-NeoX's)
function bindLayers(m) {
  const { plan } = m, form = m.form, V = m.vectors;
  // (a piece each, T155)
  const product = (name, l, from, to, add = false) => productGroups(m, form, name, l, from, to, add);
  m.layers = [];
  for (let l = 0; l < plan.layers; l++) {
    const norm = new ArrayBuffer(16);
    new Uint32Array(norm, 0, 2).set([plan.dim, l * plan.dim]);
    new Float32Array(norm, 8, 1)[0] = plan.eps;
    const normShape = uniform(m, norm);
    const added = (to, name) => V[name] && { group: bind(m, m.add, [to, V[name], uniform(m, new Uint32Array([plan.vectors[name].size, l * plan.vectors[name].size, 0, 0])), m.step]),
      x: Math.ceil(plan.vectors[name].size / 64) };
    // T154: LayerNorm reads its bias from the same place in its own buffer
    const normed = (weight, bias, out) => bind(m, m.norm, plan.layerNorm ? [m.x, V[weight], V[bias], out, normShape, m.step]
      : [m.x, V[weight], out, normShape, m.step]);
    const headNorm = (of, name, heads) => {
      if (!V[name]) return null;
      const shape = new ArrayBuffer(16);
      new Uint32Array(shape, 0, 2).set([plan.headSize, l * plan.headSize]);
      new Float32Array(shape, 8, 1)[0] = plan.eps;
      return { group: bind(m, m.headNorm, [of, V[name], uniform(m, shape), m.step]), heads };
    };
    m.layers.push({
      attentionNorm: normed("attention", "attentionBias", m.xb),
      q: product("wq", l, m.xb, m.q), k: product("wk", l, m.xb, m.k), v: product("wv", l, m.xb, m.v),
      biases: [added(m.q, "bq"), added(m.k, "bk"), added(m.v, "bv")].filter(Boolean),
      headNorms: [headNorm(m.q, "qNorm", plan.heads), headNorm(m.k, "kNorm", plan.kvHeads)].filter(Boolean),
      o: product("wo", l, m.xb, m.x, true), oBias: added(m.x, "bo"),
      ffnNorm: normed("ffn", "ffnBias", m.ffnInput),
      gate: product("w1", l, m.ffnInput, m.gate), gateBias: added(m.gate, "b1"),
      up: m.matrices.w3 ? product("w3", l, m.ffnInput, m.up) : null,
      down: product("w2", l, m.gate, m.x, true), downBias: added(m.x, "b2"),
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

// T148: what a whole block takes here, from the submission to its keys and values read back, for 16 and for 64 tokens
// (plan.batch) at position 0 of the GPU's own cache (which holds nothing of forward.js's yet): forward.js weighs a
// block of a prompt on the GPU against the same tokens on the CPU by it (a line through the two, scaled by the blocks
// it then times itself). The two in turn, BLOCK_ROUNDS rounds after one of each to warm up (the pipelines' first
// dispatches, the weights' first reads), the median of each (a device that warms up or is loaded meanwhile falls on
// both alike); one round on a fallback adapter (tests: its times are no GPU's)
const BLOCK_ROUNDS = 3;
async function timeBlocks(m) {
  const { batch, seqLen } = m.plan, counts = [...new Set([Math.min(16, batch, seqLen), Math.min(batch, seqLen)])], times = counts.map(() => []);
  const timed = async (count) => {
    const began = performance.now();
    await block(m, count, 0, () => false, true);
    return performance.now() - began;
  };
  for (const count of counts) await timed(count);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < counts.length; i++) {
      times[i].push(await timed(counts[i]));
      if (stopping) return [];
    }
  }
  return counts.map((count, i) => ({ count, ms: times[i].sort((a, b) => a - b)[times[i].length >> 1] }));
}

// ---- T152: tokens on the GPU. A step of the engine's generate() (the forward pass of the token fed, the sampling of
// the next) as one compute pass of shaders.js's T151 run: EMBED of the state's token, every layer in a fused form, the
// final norm and the classifier, SAMPLE (the penalty, softmax, top-p and the draw, with the random number the CPU drew
// for the step), after which the state's first four words go to the Step uniform for the next pass (a uniform is not a
// shader's to write). plan.tokens.most steps a submission, their ids read back once (T151: the wait of a submission,
// 3 to 8.6 ms on the owner's Android, T134, is paid once for them), with the keys and values of their positions, which
// go into forward.js's cache as the CPU would have written them: the CPU may take the next step, or the next
// generation, at any position. The keys and values the GPU does not hold (positions the CPU computed since) go up from
// forward.js's cache first: float16 there as here (T110, T147).
//
// The form of a layer, from those this device can make: T150's fusedMatVec (llama.cpp's mul_mat_vec, with the norm on
// its read and RoPE, the residual's add or SwiGLU on its write) with the workgroup's reduction, the same with
// subgroupAdd where there are subgroups, and T175's fusedDp4aMatVec (ORT's DP4A for small M, the vector quantized
// before each matrix, NORM_QUANTIZE where a norm is) where there is the packed int8 dot. Each is checked against
// JavaScript on the model's own weights (checkTokens), and the right ones are timed, a run of plan.tokens.most steps
// each in turn: the fastest is taken (T150's and T175's tables left which is fastest to the device: the owner's
// Android read a layer at 18.5% of its buffer's reads with mul_mat_vec and a matrix at 96.8% with DP4A). The attention
// is the prompt's (llama.cpp's flash attention with tiles, T147, for one token: T150's (a), a KV-split form is left for
// long contexts), on this pass's q. Where it came from: the run of T151 (public/benchmark/gpu.js's generate()), whose
// form is WebLLM's decode loop without its sync of every token (web-llm, src/llm_chat.ts; no line taken) and llama.cpp's
// WebGPU graph of a token, one command encoder for all of it (ggml-webgpu.cpp, commit 2145525a, MIT; no line taken).
// (T175's fused DP4A with the norms apart as well, RMSNORM and QUANTIZE where NORM_QUANTIZE is one: the owner's
// Android ran a layer so in 3.36 ms against 3.67 fused, the fastest of its table, 2026-09-27; T175's condition to
// reverse NORM_QUANTIZE, here chosen on the device)
const TOKEN_FORMS = [{ name: "llama.cpp, fused (T150)", dp4a: false, subgroups: false },
  { name: "llama.cpp, fused (T150), subgroups", dp4a: false, subgroups: true },
  { name: "DP4A, fused (T175)", dp4a: true, subgroups: false },
  { name: "DP4A, fused (T175), the norms apart", dp4a: true, subgroups: false, normApart: true }];
function tokenCandidates(m) {
  const features = navigator.gpu.wgslLanguageFeatures;
  const subgroups = m.device.features.has("subgroups") && Boolean(features?.has("subgroup_id"));
  const packed = Boolean(features?.has("packed_4x8_integer_dot_product"));
  return TOKEN_FORMS.filter((form) => (!form.subgroups || subgroups) && (!form.dp4a || packed));
}
// a form's WGSL, [key, code] each (the pipelines every form shares are compiled apart: tokenBuffers)
const tokenCodes = (wgsl, { dp4a, subgroups, normApart }) => (dp4a
  ? [...(normApart ? [] : [["normQuantize", wgsl.NORM_QUANTIZE]]), ["qkv", wgsl.fusedDp4aMatVec({ output: "rope" })], ["add", wgsl.fusedDp4aMatVec({ output: "add" })],
    ["glu", wgsl.fusedDp4aMatVec({ output: "swiglu" })], ["classifier", wgsl.fusedDp4aMatVec({ output: "write" })]]
  : [["qkv", wgsl.fusedMatVec({ input: "norm", output: "rope", subgroups })], ["add", wgsl.fusedMatVec({ input: "plain", output: "add", subgroups })],
    ["glu", wgsl.fusedMatVec({ input: "norm", output: "swiglu", subgroups })], ["classifier", wgsl.fusedMatVec({ input: "norm", output: "write", subgroups })]]);

// a bind group of the bindings given ([binding, a buffer or a range of one] each: the fused shaders skip binding 4
// where the norm is not on their read)
const bindAt = (m, pipeline, entries) => m.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
  entries: entries.map(([binding, buffer]) => ({ binding, resource: buffer.buffer ? buffer : { buffer } })) });

// The buffers and uniforms of a token, and the pipelines every form shares (EMBED, SAMPLE; QUANTIZE and the attention
// are the prompt's). A token's vectors: the stream h, q, the attention's output, SwiGLU's; the logits and SAMPLE's two
// scratch arrays of the vocabulary; the state, the ids, the random numbers, the Step and the settings; the quantized
// vector (DP4A: one serves every quantization of a token, each read before the next is made: T175's (l)); what a
// submission reads back (the ids, the state, and the keys and values of its positions, [keys, values][layer][step])
async function tokenBuffers(m) {
  const { plan, wgsl } = m, most = plan.tokens.most, vocab = plan.tokens.classifier.rows;
  const qDim = plan.heads * plan.headSize, kvDim = plan.kvHeads * plan.headSize, widest = Math.max(plan.dim, qDim, plan.hidden);
  // (one at a time: each in an error scope of its own)
  const embed = await within(validated(m, () => pipelineOf(m, wgsl.EMBED)), "compiling the embedding's row");
  const sample = await within(validated(m, () => pipelineOf(m, wgsl.SAMPLE)), "compiling the sampling");
  const out = STORAGE | COPY_SRC;
  const g = { embed, sample, most, vocab, h: buffer(m, plan.dim * 4, out | COPY_DST), q: buffer(m, qDim * 4), att: buffer(m, qDim * 4),
    gate: buffer(m, plan.hidden * 4), logits: buffer(m, vocab * 4, out), probs: buffer(m, vocab * 4), order: buffer(m, vocab * 4),
    state: buffer(m, wgsl.STATE_BYTES, out | COPY_DST), chosen: buffer(m, most * 4, out), randoms: buffer(m, most * 4, STORAGE | COPY_DST),
    step: buffer(m, 16, UNIFORM | COPY_DST), settings: buffer(m, wgsl.SAMPLING_BYTES, UNIFORM | COPY_DST),
    xq: buffer(m, widest), xs: buffer(m, (widest / wgsl.GROUP) * 4), xb: buffer(m, plan.dim * 4),
    readback: buffer(m, most * 4 + wgsl.STATE_BYTES + 2 * plan.layers * most * kvDim * 2, MAP_READ | COPY_DST) };
  // fusedMatVec's Params: rows, words, perRow, second, eps, normAt, qRows, kvRows, headSize, turned
  const params = (rows, n, second = 0, normAt = 0) => {
    const bytes = new ArrayBuffer(48);
    new Uint32Array(bytes).set([rows, n / 4, n / wgsl.GROUP, second, 0, normAt, qDim, kvDim, plan.headSize, plan.turned, 0, 0]);
    new Float32Array(bytes, 16, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  // RMSNORM's and NORM_QUANTIZE's Norm: size, at, eps
  const norm = (at) => {
    const bytes = new ArrayBuffer(16);
    new Uint32Array(bytes, 0, 2).set([plan.dim, at]);
    new Float32Array(bytes, 8, 1)[0] = plan.eps;
    return uniform(m, bytes);
  };
  const flash = new ArrayBuffer(16);
  new Uint32Array(flash, 0, 2).set([plan.heads, plan.kvHeads]);
  new Float32Array(flash, 8, 1)[0] = 1 / Math.sqrt(plan.headSize);
  const layers = [...Array(plan.layers)].map((_, l) => l * plan.dim);
  g.u = { embed: m.tables.embedding.map((piece) => uniform(m, new Uint32Array([plan.dim, piece.first, piece.rows, 0]))), flash: uniform(m, flash), o: params(plan.dim, qDim), down: params(plan.dim, plan.hidden),
    qkv: layers.map((at) => params(qDim + 2 * kvDim, plan.dim, 0, at)), gateUp: layers.map((at) => params(plan.hidden, plan.dim, plan.hidden, at)),
    norm: layers.map(norm), final: norm(0), classifier: m.tables.classifier.map((piece) => params(piece.rows, plan.dim)),
    // QUANTIZE's (n, xStride): the attention's output, SwiGLU's
    quantizeAttention: uniform(m, new Uint32Array([qDim, qDim, 0, 0])), quantizeGate: uniform(m, new Uint32Array([plan.hidden, plan.hidden, 0, 0])),
    quantizeNormed: uniform(m, new Uint32Array([plan.dim, plan.dim, 0, 0])) };
  return g;
}

// the workgroups of rows by the fused shaders' rows a workgroup, over x and then y (they number them so)
function spread(m, rows, perGroup) {
  const groups = Math.ceil(rows / perGroup), across = Math.min(groups, m.device.limits.maxComputeWorkgroupsPerDimension);
  return [across, Math.ceil(groups / across)];
}
// The dispatches of one step in a form: EMBED, the layers (from, to: those of a check), the head (the final norm and
// the classifier) and SAMPLE, [pipeline, bind group, x, y] each
function tokenPass(m, form, { from = 0, to = m.plan.layers, head = true, embed = true } = {}) {
  const { plan, wgsl, gen: g } = m, V = m.vectors, P = form.pipes, cache = m.cache, qDim = plan.heads * plan.headSize;
  const rows = form.dp4a ? wgsl.ORT_DP4A_MATVEC_ROWS : wgsl.MUL_MAT_VEC_ROWS;
  const matrix = (pipeline, [w, s], input, params, count, output) =>
    [pipeline, bindAt(m, pipeline, [[0, w], [1, s], ...input, [3, params], ...output]), ...spread(m, count, rows)];
  // what a matrix reads: the stream, normed on the read (T150), or the vector as it is; on DP4A its quantizing (T175)
  const read = form.dp4a ? () => [[2, g.xq], [4, g.xs]] : (x, weights) => [[2, x], ...(weights ? [[4, weights]] : [])];
  // (the norms apart: the prompt's RMSNORM into xb, then its QUANTIZE, as T175's form with the norms apart)
  const normed = (weights, params) => (!form.dp4a ? []
    : form.normApart ? [[m.norm, bind(m, m.norm, [g.h, weights, g.xb, params, g.step]), 1, 1], ...quantized(g.xb, g.u.quantizeNormed, plan.dim)]
      : [[P.normQuantize, bindAt(m, P.normQuantize, [[0, g.h], [1, weights], [2, g.xq], [3, g.xs], [4, params], [5, g.step]]), 1, 1]]);
  const quantized = (x, params, n) => (form.dp4a
    ? [[m.quantize, bindAt(m, m.quantize, [[0, x], [1, g.xq], [2, g.xs], [3, params], [4, g.step]]), Math.ceil(n / wgsl.GROUP / 64), 1]] : []);
  // (T209: a dispatch a piece of the table, each writing the row where the token is in its rows)
  const list = !embed ? [] : m.tables.embedding.map((piece, i) =>
    [g.embed, bindAt(m, g.embed, [[0, piece.values], [1, piece.scales], [2, g.state], [3, g.h], [4, g.u.embed[i]]]), 1, 1]);
  for (let l = from; l < to; l++) {
    const { qkv, gateUp } = m.joined[l], [o] = m.matrices.wo.pieces, [down] = m.matrices.w2.pieces;
    list.push(...normed(V.attention, g.u.norm[l]),
      matrix(P.qkv, qkv, read(g.h, V.attention), g.u.qkv[l], qDim + 2 * plan.kvHeads * plan.headSize,
        [[5, g.q], [6, cache.keys[l]], [7, cache.values[l]], [8, m.angleTable], [9, g.step]]),
      [m.attention.pipeline, bind(m, m.attention.pipeline, [g.q, cache.keys[l], cache.values[l], g.att, g.u.flash, g.step]), plan.heads, 1],
      ...quantized(g.att, g.u.quantizeAttention, qDim), matrix(P.add, o.layers[l], read(g.att), g.u.o, plan.dim, [[5, g.h]]),
      ...normed(V.ffn, g.u.norm[l]), matrix(P.glu, gateUp, read(g.h, V.ffn), g.u.gateUp[l], plan.hidden, [[5, g.gate]]),
      ...quantized(g.gate, g.u.quantizeGate, plan.hidden), matrix(P.add, down.layers[l], read(g.gate), g.u.down, plan.dim, [[5, g.h]]));
  }
  if (head) {
    list.push(...normed(m.finalNorm, g.u.final),
      // (T209: a piece at a time into its range of the logits: its first row is where the device binds, piecesOf)
      ...m.tables.classifier.map((piece, i) => matrix(P.classifier, [piece.values, piece.scales], read(g.h, m.finalNorm), g.u.classifier[i], piece.rows,
        [[5, { buffer: g.logits, offset: piece.first * 4, size: piece.rows * 4 }]])),
      [g.sample, bindAt(m, g.sample, [[0, g.logits], [1, g.probs], [2, g.order], [3, g.state], [4, g.chosen], [5, g.randoms], [6, g.settings]]), 1, 1]);
  }
  return list;
}
// a form's whole step, made again when the cache has grown (its buffers are others then)
function tokenStep(m, form) {
  if (form.step?.cache !== m.cache) form.step = { cache: m.cache, list: tokenPass(m, form) };
  return form.step.list;
}
// count steps of the dispatches from the state given (with the settings and a random number a step), in one
// submission, and read back: the ids and the state's words, and (keep) the keys and values of the positions pos to
// pos + count - 1. extra(encoder): what a check copies out besides. Returns { ids, sampled, stopped, kv: [keys,
// values][layer], the positions' rows of float16 as bytes }
async function runTokens(m, dispatches, { count, pos, state, settings, randoms, keep = false, extra }) {
  const { device, plan, wgsl, gen: g } = m, kvRow = plan.kvHeads * plan.headSize * 2, idsBytes = g.most * 4;
  if (pos + count > m.cache.capacity) throw new Error(`positions to ${pos + count} are past the GPU's cache`);
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.queue.writeBuffer(g.state, 0, state);
  device.queue.writeBuffer(g.step, 0, state, 0, 4);
  if (settings) device.queue.writeBuffer(g.settings, 0, settings);
  device.queue.writeBuffer(g.randoms, 0, randoms ?? new Float32Array(count));
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < count; i++) {
    const pass = encoder.beginComputePass();
    for (const [pipeline, group, x, y] of dispatches) dispatch(pass, pipeline, group, x, y);
    pass.end();
    // the Step of the next pass: the state's first four words
    encoder.copyBufferToBuffer(g.state, 0, g.step, 0, 16);
  }
  encoder.copyBufferToBuffer(g.chosen, 0, g.readback, 0, count * 4);
  encoder.copyBufferToBuffer(g.state, 0, g.readback, idsBytes, wgsl.STATE_BYTES);
  const kvAt = idsBytes + wgsl.STATE_BYTES;
  if (keep) {
    for (let l = 0; l < plan.layers; l++) {
      encoder.copyBufferToBuffer(m.cache.keys[l], pos * kvRow, g.readback, kvAt + l * g.most * kvRow, count * kvRow);
      encoder.copyBufferToBuffer(m.cache.values[l], pos * kvRow, g.readback, kvAt + (plan.layers + l) * g.most * kvRow, count * kvRow);
    }
  }
  extra?.(encoder);
  device.queue.submit([encoder.finish()]);
  const invalid = await device.popErrorScope(), full = await device.popErrorScope();
  if (invalid || full) throw new Error(`the GPU refused a token (${(invalid ?? full).message})`);
  await g.readback.mapAsync(MAP_READ);
  try {
    const words = new Uint32Array(g.readback.getMappedRange(0, idsBytes + wgsl.STATE_BYTES).slice(0));
    const after = words.subarray(g.most);
    const out = { ids: words.slice(0, count), sampled: after[5], stopped: after[7] };
    if (keep) {
      const bytes = new Uint8Array(g.readback.getMappedRange(kvAt, 2 * plan.layers * g.most * kvRow).slice(0));
      out.kv = [0, 1].map((side) => [...Array(plan.layers)].map((_, l) => {
        const at = (side * plan.layers + l) * g.most * kvRow;
        return bytes.subarray(at, at + count * kvRow);
      }));
    }
    return out;
  } finally {
    g.readback.unmap();
  }
}

// The forms of a token this device can make, compiled, checked and (where more than one is right) timed; the fastest
// taken (m.gen.form), with what each came to (m.gen.forms) and its ms a step (m.gen.ms). plan.remembered.tokens: the
// one the page kept for this adapter, alone where it is still right here. plan.force.tokens (tests): that form alone;
// plan.force.quick: the first right one, untimed. Throws where none is right (the tokens then stay on the CPU)
async function chooseTokens(m) {
  const { plan, wgsl } = m;
  m.gen = await tokenBuffers(m);
  // T156: the first layer's matrices of a model on the GPU alone are not in the shared memory: checkTokens reads them
  // back from the GPU
  if (m.direct) m.firstLayer ??= await within(firstLayer(m), "reading the first layer back");
  let forms = tokenCandidates(m);
  if (plan.force.tokens) forms = forms.filter((form) => form.name === plan.force.tokens);
  const kept = !plan.force.tokens && forms.find((form) => form.name === m.remembered?.tokens);
  if (kept) forms = [kept, ...forms.filter((form) => form !== kept)];
  const right = [], tried = [];
  for (const candidate of forms) {
    // the remembered one right: no other is compiled or timed (as the matrices, T148)
    if ((plan.force.quick || (kept && right[0]?.name === kept.name)) && right.length) break;
    const form = { ...candidate, pipes: {} };
    try {
      for (const [key, code] of tokenCodes(wgsl, form)) {
        form.pipes[key] = await within(validated(m, () => pipelineOf(m, code)), `compiling ${form.name}`);
      }
      const wrong = await within(checkTokens(m, form), `checking ${form.name}`);
      if (wrong) tried.push({ name: form.name, none: `wrong: ${wrong}` });
      else right.push(form);
    } catch (error) {
      if (error?.late) throw error;
      tried.push({ name: form.name, none: String(error?.message ?? error) });
    }
    if (stopping) return;
  }
  if (!right.length) {
    throw new Error(`no layer of a token is right on this GPU (${tried.map((f) => `${f.name}: ${f.none}`).join("; ") || `none named ${plan.force.tokens}`})`);
  }
  const ms = plan.force.quick ? right.map(() => undefined) : await within(timeTokens(m, right), "timing a token");
  const best = ms.reduce((b, t, i) => (t !== undefined && (ms[b] === undefined || t < ms[b]) ? i : b), 0);
  m.gen.form = right[best];
  m.gen.ms = ms[best];
  m.gen.forms = [...tried, ...right.map((form, i) => ({ name: form.name, ms: ms[i], ...(kept && form.name === kept.name ? { remembered: true } : {}) }))];
}

// ms a step of each form: runs of plan.tokens.most steps a submission from position 0 (a made-up state, the settings
// of the list's sampled models, T151's), the forms in turn after one run each to warm up, BLOCK_ROUNDS rounds (one on
// a fallback adapter: its times are no GPU's), the median of each over its steps. The runs write the GPU's own keys
// and values of those positions, which hold nothing of forward.js's yet (as timeBlocks)
async function timeTokens(m, forms) {
  const { wgsl, gen: g } = m, most = g.most;
  const state = wgsl.samplingState({ token: 1, pos: 0, history: [1] });
  const settings = wgsl.samplingSettings({ vocab: g.vocab, temperature: 0.7, topp: 0.9, penalty: 1.1 });
  const randoms = new Float32Array(most).map(() => Math.fround(Math.random()) % 1);
  if (most > m.cache.capacity) grow(m, most);
  const timed = async (form) => {
    const began = performance.now();
    await runTokens(m, tokenStep(m, form), { count: most, pos: 0, state, settings, randoms });
    return performance.now() - began;
  };
  for (const form of forms) await timed(form);
  const times = forms.map(() => []);
  for (let round = 0; round < (m.fallback ? 1 : BLOCK_ROUNDS); round++) {
    for (let i = 0; i < forms.length; i++) times[i].push(await timed(forms[i]));
    if (stopping) break;
  }
  return times.map((list) => list.sort((a, b) => a - b)[list.length >> 1] / most);
}

// ---- T152's check of a form on the model's own weights, against JavaScript (the drivers, the subgroups and the
// packed dot differ from device to device, and only the device can say):
//   1. EMBED of a token of the second half of the vocabulary and the first layer at position 1 (the cache's row 0 of random
//      float16), the stream h and the keys and values of position 1 read back: against JavaScript's layer in float64
//      (the weights read from the shared memory as forward.js holds them: int6 widened by shaders.js's sixValues; RoPE
//      from the CPU's tables; the keys and values rounded to float16 where the attention reads them; on DP4A each
//      matrix's input quantized as quantize_x quantizes it, in float32). The stream: no farther than LAYER_LINE of the
//      largest change the layer made; the keys and values no farther than it of their largest. A wrong row, group,
//      head, angle or scale is a tenth and more off; on DP4A a value quantized to the other side of a rounding moves
//      the layer by about 1e-3 (T175), and so its lines are DP4A_LINE.
//   2. the head on a made-up stream (±2), greedy: the logits of every 256th row (and the last) against JavaScript's,
//      no farther than LOGITS_LINE of their largest (DP4A: DP4A_LINE), and the id SAMPLE chose the first largest of the
//      GPU's own logits; then sampled (temperature 0.8, top-p 0.9, penalty 1.3 on a history that holds that id, a
//      random number of 0.7): the logits SAMPLE penalized in place as penalizeLikeCpu penalizes the greedy ones
//      (within 1e-6), and the id one that the CPU's walk (walkLikeCpu) over those logits reaches within 1e-4 of the
//      random number's share of the mass (T151's line: the GPU's exp and its float32 sums).
// The reason it is wrong, or null.
const LAYER_LINE = 2e-3, LOGITS_LINE = 1e-3, DP4A_LINE = 2e-2;
// T156: the first layer's matrices, { name: [values, scales] } as an Int8Array and a Float32Array each, read back from
// their buffers (a piece after another: the rows in order)
async function firstLayer(m) {
  const group = m.wgsl.GROUP, out = {};
  for (const [name, { n, pieces }] of Object.entries(m.matrices)) {
    const rows = pieces.reduce((sum, piece) => sum + piece.rows, 0);
    const values = new Int8Array(rows * n), scales = new Float32Array((rows * n) / group);
    for (const piece of pieces) {
      const [v, s] = piece.layers[0], valueBytes = piece.rows * n, scaleBytes = (valueBytes / group) * 4;
      values.set(new Int8Array(await readBack(m, v.buffer ?? v, valueBytes, v.offset ?? 0)), piece.first * n);
      scales.set(new Float32Array(await readBack(m, s.buffer ?? s, scaleBytes, s.offset ?? 0)), (piece.first * n) / group);
    }
    out[name] = [values, scales];
  }
  return out;
}
async function checkTokens(m, form) {
  const { plan, wgsl, gen: g, device } = m, vocab = g.vocab, dim = plan.dim, headSize = plan.headSize, half = headSize / 2;
  const qDim = plan.heads * headSize, kvDim = plan.kvHeads * headSize, line = form.dp4a ? DP4A_LINE : LAYER_LINE;
  const floats = (address, n) => Float64Array.from(new Float32Array(m.memory.buffer, address, n));
  const round16 = Math.f16round ?? ((x) => x);
  // a matrix's rows ({ n, six }, its values and scales at [valuesAt, scalesAt]) times x, the rows given
  // (T156: or the values and scales themselves, read back from the GPU: firstLayer)
  const product = ({ n, six }, [valuesAt, scalesAt], x, rows) => {
    const q = form.dp4a ? wgsl.quantizedLikeCpu(Float32Array.from(x)) : null, perRow = n / wgsl.GROUP;
    const read = typeof valuesAt !== "number";
    return Float64Array.from(rows, (r) => {
      const w = read ? valuesAt.subarray(r * n, (r + 1) * n)
        : six ? wgsl.sixValues(new Uint8Array(m.memory.buffer, valuesAt + (r * n * 3) / 4, (n * 3) / 4))
          : new Int8Array(m.memory.buffer, valuesAt + r * n, n);
      const s = read ? scalesAt.subarray(r * perRow, (r + 1) * perRow) : new Float32Array(m.memory.buffer, scalesAt + r * perRow * 4, perRow);
      let sum = 0;
      for (let b = 0; b < perRow; b++) {
        let part = 0;
        for (let i = b * wgsl.GROUP; i < (b + 1) * wgsl.GROUP; i++) part += w[i] * (q ? q.xq[i] : x[i]);
        sum += part * s[b] * (q ? q.xs[b] : 1);
      }
      return sum;
    });
  };
  const all = (n) => [...Array(n).keys()];
  const matmul = (name, x) => product(plan.matrices[name], m.firstLayer?.[name] ?? plan.matrices[name].layers[0], x, all(plan.matrices[name].rows));
  const rms = (x, weights) => {
    const s = 1 / Math.sqrt(x.reduce((sum, v) => sum + v * v, 0) / x.length + plan.eps);
    return x.map((v, i) => weights[i] * (s * v));
  };
  const largest = (xs) => xs.reduce((a, v) => Math.max(a, Math.abs(v)), 0);
  const off = (got, want) => largest(want.map((v, i) => got[i] - v));
  const owned = [];
  try {
    // 1. the first layer at position 1
    // a token of the second half of the vocabulary whose row is large: the rows of tokens no text has (llm-jp-3's
    // last) are nearly 0, and their keys and values then float16's subnormals (a check of them was 15% off, CI)
    const embeddingOf = plan.tokens.embedding ?? plan.tokens.classifier, groups = dim / wgsl.GROUP;
    const sizeOf = (t) => new Float32Array(m.memory.buffer, embeddingOf.at[1] + t * groups * 4, groups).reduce((sum, v) => sum + Math.abs(v), 0);
    const token = [...Array(64).keys()].map((i) => vocab - 1 - Math.floor((i * vocab) / 128)).reduce((best, t) => (sizeOf(t) > sizeOf(best) ? t : best));
    const pos = 1;
    const halfBits = () => (Math.random() < 0.5 ? 0x8000 : 0) | ((13 + ((Math.random() * 3) | 0)) << 10) | ((Math.random() * 1024) | 0);
    const row0 = [0, 1].map(() => new Uint16Array(kvDim).map(halfBits));
    if (m.cache.capacity < 2) grow(m, 2);
    device.queue.writeBuffer(m.cache.keys[0], 0, row0[0]);
    device.queue.writeBuffer(m.cache.values[0], 0, row0[1]);
    const readH = buffer(m, dim * 4, MAP_READ | COPY_DST, owned);
    const { kv } = await runTokens(m, tokenPass(m, form, { to: 1, head: false }), { count: 1, pos, keep: true,
      state: wgsl.samplingState({ token, pos, history: [token] }), extra: (encoder) => encoder.copyBufferToBuffer(g.h, 0, readH, 0, dim * 4) });
    await readH.mapAsync(MAP_READ);
    const h = new Float32Array(readH.getMappedRange().slice(0));
    readH.unmap();
    const [gotK, gotV] = kv.map((layers) => Float64Array.from(new Uint16Array(layers[0].buffer, layers[0].byteOffset, kvDim), halfToFloat));
    // JavaScript's layer
    const embedding = plan.tokens.embedding ?? plan.tokens.classifier, perRow = dim / wgsl.GROUP;
    const eRow = embedding.six ? wgsl.sixValues(new Uint8Array(m.memory.buffer, embedding.at[0] + (token * dim * 3) / 4, (dim * 3) / 4))
      : new Int8Array(m.memory.buffer, embedding.at[0] + token * dim, dim);
    const eScales = new Float32Array(m.memory.buffer, embedding.at[1] + token * perRow * 4, perRow);
    const x0 = Float64Array.from(eRow, (v, i) => v * eScales[(i / wgsl.GROUP) | 0]);
    const xn = rms(x0, floats(plan.vectors.attention.at, dim));
    const [q, k, v] = ["wq", "wk", "wv"].map((name) => matmul(name, xn));
    const cos = floats(plan.cos + pos * half * 4, half), sin = floats(plan.sin + pos * half * 4, half);
    const turn = (vector) => {
      for (let at = 0; at < vector.length; at += 2) {
        const i = (at % headSize) / 2;
        if (at % headSize >= plan.turned) continue;
        const [a, b] = [vector[at], vector[at + 1]];
        vector[at] = a * cos[i] - b * sin[i];
        vector[at + 1] = a * sin[i] + b * cos[i];
      }
    };
    turn(q);
    turn(k);
    const keys = [Float64Array.from(row0[0], halfToFloat), k.map(round16)], values = [Float64Array.from(row0[1], halfToFloat), v.map(round16)];
    const att = new Float64Array(qDim), group = plan.heads / plan.kvHeads;
    for (let head = 0; head < plan.heads; head++) {
      const kvAt = Math.floor(head / group) * headSize, at = head * headSize;
      const scores = keys.map((key) => {
        let sum = 0;
        for (let d = 0; d < headSize; d++) sum += q[at + d] * key[kvAt + d];
        return sum / Math.sqrt(headSize);
      });
      const top = Math.max(...scores), weights = scores.map((score) => Math.exp(score - top)), total = weights[0] + weights[1];
      for (let d = 0; d < headSize; d++) att[at + d] = (weights[0] * values[0][kvAt + d] + weights[1] * values[1][kvAt + d]) / total;
    }
    const o = matmul("wo", att), h1 = x0.map((value, i) => value + o[i]);
    const xn2 = rms(h1, floats(plan.vectors.ffn.at, dim));
    const gate = matmul("w1", xn2), up = matmul("w3", xn2);
    const down = matmul("w2", gate.map((value, i) => (value / (1 + Math.exp(-value))) * up[i])), h2 = h1.map((value, i) => value + down[i]);
    const stream = off(h, h2) / largest(h2.map((value, i) => value - x0[i]));
    const keyOff = off(gotK, k) / largest(k), valueOff = off(gotV, v) / largest(v);
    if (!(stream <= line)) return `the first layer's stream is ${stream.toExponential(2)} of its change from JavaScript's (line ${line})`;
    if (!(keyOff <= line && valueOff <= line)) {
      return `the first layer's keys are ${keyOff.toExponential(2)} and its values ${valueOff.toExponential(2)} from JavaScript's (line ${line})`;
    }
    // 2. the head: greedy, then sampled
    const stream2 = new Float32Array(dim).map(() => (Math.random() - 0.5) * 4);
    const rows = [...new Set([...all(Math.ceil(vocab / 256)).map((i) => i * 256), vocab - 1])];
    const readLogits = buffer(m, vocab * 4, MAP_READ | COPY_DST, owned);
    const head = tokenPass(m, form, { to: 0, embed: false });
    const run = async (settings, history, random) => {
      device.queue.writeBuffer(g.h, 0, stream2);
      const { ids } = await runTokens(m, head, { count: 1, pos, state: wgsl.samplingState({ token, pos, history }), settings,
        randoms: new Float32Array([random]), extra: (encoder) => encoder.copyBufferToBuffer(g.logits, 0, readLogits, 0, vocab * 4) });
      await readLogits.mapAsync(MAP_READ);
      const logits = new Float32Array(readLogits.getMappedRange().slice(0));
      readLogits.unmap();
      return { id: ids[0], logits };
    };
    const greedy = await run(wgsl.samplingSettings({ vocab, temperature: 0, topp: 0.9 }), [token], 0);
    const classifier = plan.tokens.classifier;
    const want = product(classifier, classifier.at, rms(Float64Array.from(stream2), floats(plan.tokens.final, dim)), rows);
    const logitsOff = off(rows.map((r) => greedy.logits[r]), want) / largest(want), logitsLine = form.dp4a ? DP4A_LINE : LOGITS_LINE;
    if (!(logitsOff <= logitsLine)) return `the logits are ${logitsOff.toExponential(2)} of their largest from JavaScript's (line ${logitsLine})`;
    if (greedy.id !== wgsl.argmaxLikeCpu(greedy.logits)) return `the greedy token is ${greedy.id}, the largest logit's ${wgsl.argmaxLikeCpu(greedy.logits)}`;
    const sampled = { temperature: 0.8, topp: 0.9, penalty: 1.3 }, history = [token, greedy.id, 0, 1];
    const drawn = await run(wgsl.samplingSettings({ vocab, ...sampled }), history, 0.7);
    const penalized = greedy.logits.slice();
    wgsl.penalizeLikeCpu(penalized, history, sampled.penalty);
    const penaltyOff = off(history.map((t) => drawn.logits[t]), history.map((t) => penalized[t])) / Math.max(largest(history.map((t) => penalized[t])), 1e-30);
    if (!(penaltyOff <= 1e-6)) return `the penalized logits are ${penaltyOff.toExponential(2)} from JavaScript's`;
    const walk = wgsl.walkLikeCpu(drawn.logits, sampled.temperature, sampled.topp), goal = 0.7 * walk.mass, band = 1e-4 * walk.mass;
    const reached = walk.tokens.filter((_, j) => walk.cumulative[j] > goal - band && (j ? walk.cumulative[j - 1] : 0) <= goal + band);
    if (!reached.includes(drawn.id)) return `the sampled token is ${drawn.id}, the CPU's walk reaches ${reached.join(" or ")}`;
    return null;
  } finally {
    owned.forEach((b) => b.destroy());
  }
}

// A request for count steps from token at pos (forward.js's generateMany): the keys and values of positions from to
// pos - 1 up from forward.js's cache first (cache: its addresses of the keys and the values, its capacity and the
// bytes of a position; half: float16 as here, else float32, narrowed on the way up: T160, a grouped-query model's),
// the state (token, pos, the end of the history and its length), the settings and a random number a step; the ids into
// plan.tokens.ids ([sampled, id, id, ...], a stop token last where one came), and the keys and values of the positions
// sampled in float16 into plan.staging as a prompt's block's ([keys, values][layer][plan.batch positions]), which
// forward.js puts into its cache (widened where it is float32), where it still wants the answer
function generate({ serial, count, pos, from, token, history, length, cache, settings, randoms }) {
  serve(serial, async (wanted) => {
    const m = model, { plan, wgsl, gen: g } = m, kvDim = plan.kvHeads * plan.headSize, kvRow = kvDim * 2;
    if (!g?.form) throw new Error("no tokens on this GPU");
    if (pos + count > m.cache.capacity) grow(m, pos + count);
    const at = (block, l, p) => block + l * cache.capacity * cache.row + p * cache.row;
    for (let l = 0; l < plan.layers; l++) {
      for (const [block, target] of [[cache.keys, m.cache.keys[l]], [cache.values, m.cache.values[l]]]) {
        if (cache.half) copyIn(m, target, at(block, l, from), (pos - from) * kvRow, from * kvRow);
        else narrowIn(m, target, at(block, l, from), (pos - from) * kvDim, from * kvRow);
      }
    }
    const out = await runTokens(m, tokenStep(m, g.form), { count, pos, keep: true,
      state: wgsl.samplingState({ token, pos, history, length }), settings: wgsl.samplingSettings({ vocab: g.vocab, ...settings }),
      randoms: Float32Array.from({ length: count }, (_, i) => randoms[i] ?? 0) });
    if (!wanted()) return;
    for (let side = 0; side < 2; side++) {
      for (let l = 0; l < plan.layers; l++) {
        new Uint8Array(m.memory.buffer, plan.staging + (side * plan.layers + l) * plan.batch * kvRow, out.sampled * kvRow)
          .set(out.kv[side][l].subarray(0, out.sampled * kvRow));
      }
    }
    const ids = new Int32Array(m.memory.buffer, plan.tokens.ids, 1 + count);
    ids[0] = out.sampled;
    ids.set(out.ids.subarray(0, out.sampled), 1);
  });
}

// ---- a block of a prompt
function prompt({ serial, count, pos }) {
  serve(serial, (wanted) => block(model, count, pos, wanted));
}
// The answer to a request (a block of a prompt, T152: the steps of a generation), in the control area: work(wanted)
// runs while words.beat counts up, then words.failed and words.done = serial, where forward.js still waits for this
// request (wanted: T147, it gave up, and the memory may soon be another model's)
async function serve(serial, work) {
  const { memory, plan } = model;
  const words = new Int32Array(memory.buffer, 0, Math.max(...Object.values(plan.words)) + 1);
  // the model's worker waits: this says that the work goes on, however long the GPU takes (a software adapter)
  const beat = setInterval(() => Atomics.add(words, plan.words.beat, 1), 250);
  const wanted = () => Atomics.load(words, plan.words.wanted) === serial;
  let failed = 1;
  try {
    if (lost) throw new Error(lost);
    await work(wanted);
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

// A block of count tokens at pos through the layers, its keys and values into plan.staging where wanted() still says
// so. timing (T148, timeBlocks): rows and angles of zeros instead of forward.js's, and nothing written back
async function block(m, count, pos, wanted, timing = false) {
  const { device, plan } = m, B = plan.batch, half = plan.headSize / 2, kvDim = plan.kvHeads * plan.headSize;
  // the rows forward.js embedded (dense), and the angles of their positions (the tables forward.js has); views of just
  // those floats (T155: not a view of the whole of a 64-bit memory of some GB for every block)
  const floats = (address, n) => new Float32Array(m.memory.buffer, address, n);
  if (timing) m.rows.fill(0);
  else m.rows.set(floats(plan.rows, count * plan.dim));
  // (T154: GPT-2 turns nothing and has no tables)
  if (timing || !plan.turned) {
    m.turns.fill(0);
  } else {
    for (let t = 0; t < count; t++) {
      m.turns.set(floats(plan.cos + (pos + t) * half * 4, half), t * plan.headSize);
      m.turns.set(floats(plan.sin + (pos + t) * half * 4, half), t * plan.headSize + half);
    }
  }
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  device.queue.writeBuffer(m.x, 0, m.rows, 0, count * plan.dim);
  device.queue.writeBuffer(m.angles, 0, m.turns, 0, count * plan.headSize);
  device.queue.writeBuffer(m.step, 0, new Uint32Array([count, pos, 0, 0]));
  if (pos + count > m.cache.capacity) grow(m, pos + count);
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(), form = m.form;
  const multiplied = (pieces) => pieces.forEach((piece) => multiply(m, pass, form, piece.group, piece.rows, count));
  // a packed form reads its input quantized: once for the matrices that read the same one
  const quantize = (q) => form.packed && dispatch(pass, m.quantize, q.group, q.x, count);
  const flash = m.attention;
  // a bias added to a matrix's output (T153: Qwen2's q, k and v; T154: GPT-2's and GPT-NeoX's every matrix's)
  const biased = (bias) => bias && dispatch(pass, m.add, bias.group, bias.x, count);
  for (let l = 0; l < plan.layers; l++) {
    const layer = m.layers[l];
    dispatch(pass, m.norm, layer.attentionNorm, 1, count);
    quantize(m.quantizeXb);
    multiplied(layer.q);
    multiplied(layer.k);
    multiplied(layer.v);
    // T153: as the CPU has it (forward.js), the biases, then the norms of the heads, then RoPE
    layer.biases.forEach(biased);
    for (const norm of layer.headNorms) dispatch(pass, m.headNorm, norm.group, norm.heads, count);
    dispatch(pass, m.rope, m.cache.rope[l], count);
    if (l === plan.layers - 1) break;  // the keys and values are all a prompt's token leaves
    // T154: GPT-NeoX's parallel residual: the FFN's norm of the layer's input, before o adds to it
    if (plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    dispatch(pass, flash.pipeline, m.cache.attention[l], plan.heads * Math.ceil(count / m.wgsl.FLASH_Q_TILE));
    quantize(m.quantizeAttention);
    multiplied(layer.o);
    biased(layer.oBias);
    if (!plan.parallel) dispatch(pass, m.norm, layer.ffnNorm, 1, count);
    quantize(m.quantizeFfn);
    multiplied(layer.gate);
    if (layer.up) multiplied(layer.up);
    biased(layer.gateBias);
    dispatch(pass, m.activation, m.activationGroup, Math.ceil(plan.hidden / 64), count);
    quantize(m.quantizeGate);
    multiplied(layer.down);
    biased(layer.downBias);
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
    if (!timing && wanted()) new Uint8Array(m.memory.buffer, plan.staging, bytes).set(new Uint8Array(m.readback.getMappedRange(), 0, bytes));
  } finally {
    m.readback.unmap();
  }
}
