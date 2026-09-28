// T148: how forward.js weighs a block of a prompt on the GPU against the CPU (promptTimes), in Node, without a GPU:
//   node tests/gpu-choice-check.mjs
// Made-up times: the CPU's ms a token of its blocks, gpu.js's two blocks timed as it starts, the blocks the GPU then ran.
import assert from "node:assert/strict";
import { aloneHolds, BOTH_ON_8, aloneVerdict, gpuBytes, gpuLine, gpuOnlyPlan, gpuOnlyUnfit, gpuOnlyWeights, gpuHoles, layerWeightsOf, placer, PROMPTS_CPU,
  PROMPTS_GPU, PROMPTS_UNTIMED, promptTimes, tokenTimes, USAGE_UNKNOWN, weightsPlace } from "../public/forward.js";
import { deviceKey, halvesOf } from "../public/shaders.js";
import { usedAfter } from "../src/bench.js";

// a GPU with a fixed cost of 40 ms a block and 0.5 ms a token (16 tokens 48 ms, 64 tokens 72 ms)
const started = [{ count: 16, ms: 48 }, { count: 64, ms: 72 }];

// nothing is known before the CPU has run TIMED blocks on the threads in use: the prompt stays on the CPU
{
  const times = promptTimes();
  assert.equal(times.of(64, 4), null, "no GPU yet");
  times.started(started);
  assert.equal(times.of(64, 4), null, "no CPU yet");
  times.cpu(4, 2);
  assert.equal(times.of(64, 4), null, "one CPU block is not enough (the first after a pause is the slowest)");
  times.cpu(4, 2);
  assert.ok(times.of(64, 4), "two are");
  assert.equal(times.of(64, 2), null, "another number of threads is timed anew");
}

// a CPU of 2 ms a token: 64 tokens 128 ms against the GPU's 72: the GPU; 16 tokens 32 ms against 48: the CPU. The
// threshold is the fewest tokens the GPU is faster for by more than BETTER (0.95): 40 + 0.5 n < 0.95 × 2 n, n > 28.6
{
  const times = promptTimes();
  times.started(started);
  times.cpu(4, 2);
  times.cpu(4, 2);
  const whole = times.of(64, 4), short = times.of(16, 4);
  assert.equal(whole.cpu, 128);
  assert.equal(whole.gpu, 72);
  assert.equal(whole.faster, true, "a whole block goes to the GPU");
  assert.equal(short.faster, false, "a short one stays on the CPU");
  assert.equal(times.threshold(64, 4), 29);
}

// a GPU that runs its blocks for real three times as slow as it timed itself: the line is scaled, and the CPU wins
{
  const times = promptTimes();
  times.started(started);
  times.cpu(1, 1.5);
  times.cpu(1, 1.5);
  assert.equal(times.of(64, 1).faster, true, "timed at the start: 72 against 96");
  for (let i = 0; i < 3; i++) times.gpu(64, 216);
  assert.equal(times.of(64, 1).gpu, 216);
  assert.equal(times.of(64, 1).faster, false, "216 against 96");
  assert.equal(times.threshold(64, 1), 65, "none: every block on the CPU");
}

// the medians of the last five: one slow block does not move the verdict, five do
{
  const times = promptTimes();
  times.started(started);
  for (const ms of [2, 2, 2, 2, 50]) times.cpu(4, ms);
  assert.equal(times.of(64, 4).cpu, 128, "one slow block among five");
  for (let i = 0; i < 5; i++) times.cpu(4, 50);
  assert.equal(times.of(64, 4).cpu, 3200, "only the last five count");
}

// a GPU whose blocks of 16 took as long as those of 64 (the tiles): a flat line, never below 0 a token
{
  const times = promptTimes();
  times.started([{ count: 16, ms: 80 }, { count: 64, ms: 70 }]);
  times.cpu(4, 1);
  times.cpu(4, 1);
  assert.equal(times.of(16, 4).gpu, 80);
  assert.equal(times.of(64, 4).gpu, 80);
}

// T152: a generation's steps (tokenTimes): nothing before the GPU has a time and the CPU TIMED ones on the threads in
// use; the GPU where a step takes less than 0.95 of the CPU's; the lower medians of the last five
{
  const steps = tokenTimes();
  assert.equal(steps.of(4), null, "nothing timed");
  steps.gpu(10);
  steps.cpu(4, 12);
  assert.equal(steps.of(4), null, "one CPU step is not enough");
  steps.cpu(4, 12);
  assert.deepEqual(steps.of(4), { cpu: 12, gpu: 10, faster: true }, "10 < 0.95 × 12");
  assert.equal(steps.of(2), null, "another number of threads is timed anew");
  for (const ms of [12, 12, 12]) steps.gpu(ms);
  assert.equal(steps.of(4).faster, false, "12 is not below 0.95 × 12");
  for (const ms of [5, 5, 50]) steps.gpu(ms);
  assert.equal(steps.of(4).gpu, 12, "the lower median of the last five: 12, 12, 5, 5, 50");
  steps.gpu(5);
  assert.equal(steps.of(4).gpu, 5, "12, 5, 5, 50, 5");
  assert.equal(steps.of(4).faster, true);
}
// T152's review (T160): halvesOf, a float32 cache's keys and values narrowed on the way up to the GPU. Every float16
// back to itself (NaN to a NaN), the float32 half way between two neighbours to the even one, and a step of float32
// either side of it to the nearer one; past the largest to the infinity, and below half the least subnormal to 0
{
  const toFloat = (h) => {
    const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
    return exponent ? sign * 2 ** (exponent - 15) * (1 + fraction / 1024) : sign * 2 ** -24 * fraction;
  };
  const one = (x) => halvesOf(Float32Array.of(x), new Uint16Array(1))[0];
  const nan = (h) => (h & 0x7c00) === 0x7c00 && (h & 0x3ff) !== 0;
  for (let h = 0; h < 0x10000; h++) {
    if (nan(h)) assert.ok(nan(one(toFloat(h))), `NaN ${h.toString(16)}`);
    else assert.equal(one(toFloat(h)), h, `float16 ${h.toString(16)} back to itself`);
  }
  const f32 = new Float32Array(1), bits = new Uint32Array(f32.buffer);
  const beside = (x, step) => { f32[0] = x; bits[0] += step; return f32[0]; };
  for (const sign of [0, 0x8000]) {
    // (h and h + 1 finite; a float32 one step nearer 0 than the middle goes to h, one step farther to h + 1)
    for (let h = sign; h < sign + 0x7bff; h++) {
      const middle = (toFloat(h) + toFloat(h + 1)) / 2;  // exact in float32
      assert.equal(one(middle), h & 1 ? h + 1 : h, `half way above ${h.toString(16)} to the even one`);
      assert.equal(one(beside(middle, -1)), h, `just short of half way above ${h.toString(16)}`);
      assert.equal(one(beside(middle, 1)), h + 1, `just past half way above ${h.toString(16)}`);
    }
  }
  assert.equal(one(65520), 0x7c00, "past the largest: the infinity");
  assert.equal(one(2 ** -25), 0, "half the least subnormal: 0 (the even one)");
  assert.equal(one(-(2 ** -26)), 0x8000, "below it: -0");
}
// T152: the status line's words (the owner's, 2026-09-27): both on the GPU, before either is timed, the answers alone on
// the CPU (faster here), and a reason for the answers (in the console alone)
{
  assert.equal(gpuLine(PROMPTS_GPU, "gpu"), "prompts and answers on WebGPU");
  assert.equal(gpuLine(PROMPTS_UNTIMED, "untimed"), "WebGPU where it is faster than the CPU");
  assert.equal(gpuLine(PROMPTS_GPU, "cpu"), "prompts on WebGPU, answers on the CPU (faster here)");
  assert.equal(gpuLine(PROMPTS_GPU, "why"), "prompts on WebGPU, answers on the CPU");
  assert.equal(gpuLine(PROMPTS_UNTIMED, "why"), "prompts on WebGPU where it is faster than the CPU, answers on the CPU");
  assert.equal(gpuLine(PROMPTS_CPU, "cpu"), "prompts and answers on the CPU (faster here than WebGPU)");
  assert.equal(gpuLine(PROMPTS_CPU, "gpu"), "prompts on the CPU (faster here than WebGPU), answers on WebGPU");
  assert.equal(gpuLine("prompts of 32 tokens and more on WebGPU", "gpu"), "prompts of 32 tokens and more on WebGPU, answers on WebGPU");
  assert.equal(gpuLine(PROMPTS_GPU, "untimed"), "prompts on WebGPU, answers on WebGPU where it is faster than the CPU");
  assert.equal(gpuLine("prompts on the CPU (no GPU adapter here)", null), "prompts on the CPU (no GPU adapter here)");
  assert.equal(gpuLine(undefined, null), undefined);
}
// T156: where a model goes (the owner's B, 2026-09-27): both where both fit half of the device (8: 8 GB, 4 GB for
// both), else the GPU alone where the model can be and fits (8: no limit, as the CPU alone has none), else the CPU
{
  const GB = 2 ** 30;
  // (the owner, 2026-09-27: both up to 6.5 GiB on a device that says 8, so that the list's models up to 2B stay as they
  // were and 3B and larger go on the GPU alone. The second review of T156, the list's configs with footprint() and
  // gpuBytes(), 4096 positions: sarashina2.2 1B 2.81 + 1.94 GB (4.42 GiB) and llm-jp-3.1 1.8B 3.543 + 2.910 GB (6.010
  // GiB, no GQA: its keys and values are 0.8 GB on either side; 11 MB past the first line of 6 GiB) both as before;
  // Qwen2.5 3B 4.10 + 3.63 GB (7.20 GiB, not eligible: its biases) the CPU with its layers past the room left;
  // Llama 3.2 3B 8.83 GB the GPU alone)
  assert.equal(BOTH_ON_8, 6.5 * GB, "the line of both on a device that says 8");
  assert.deepEqual(weightsPlace({ cpu: 1.5 * GB, gpu: 1.4 * GB, deviceMemory: 8 }), { mode: "both", gpuRoom: 5 * GB }, "Llama 3.2 1B on 8: both");
  assert.equal(weightsPlace({ cpu: 2.812e9, gpuOnly: 0.98e9, gpu: 1.939e9, deviceMemory: 8, eligible: true }).mode, "both", "sarashina2.2 1B on 8: both");
  assert.equal(weightsPlace({ cpu: 3.543e9, gpuOnly: 1.72e9, gpu: 2.910e9, deviceMemory: 8, eligible: true }).mode, "both", "llm-jp-3.1 1.8B: 6.01 GiB, within 6.5");
  assert.equal(weightsPlace({ cpu: 3.25 * GB, gpuOnly: 0.6 * GB, gpu: 3.25 * GB, deviceMemory: 8, eligible: true }).mode, "both", "6.5 GiB: both");
  assert.equal(weightsPlace({ cpu: 3.3 * GB, gpuOnly: 0.6 * GB, gpu: 3.3 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "6.6 GiB: the GPU alone");
  const qwen3B = weightsPlace({ cpu: 4.103e9, gpu: 3.627e9, deviceMemory: 8 });
  assert.ok(qwen3B.mode === "cpu" && qwen3B.gpuRoom < 3.273e9, "Qwen2.5 3B on 8: the CPU, its layers (3.27 GB) past the room left");
  assert.equal(weightsPlace({ cpu: 4.9e9, gpuOnly: 1.3e9, gpu: 3.93e9, deviceMemory: 8, eligible: true }).mode, "gpu", "Llama 3.2 3B on 8: the GPU alone");
  assert.equal(weightsPlace({ cpu: 4.4 * GB, gpuOnly: 1.2 * GB, gpu: 4.1 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "3B on 8: the GPU alone");
  assert.equal(weightsPlace({ cpu: 9.2 * GB, gpuOnly: 1.5 * GB, gpu: 8 * GB, deviceMemory: 8, eligible: true }).mode, "gpu", "7B on 8: no limit");
  const notEligible = weightsPlace({ cpu: 4.4 * GB, gpu: 4.1 * GB, deviceMemory: 8 });
  assert.ok(notEligible.mode === "cpu" && Math.abs(notEligible.gpuRoom - 2.1 * GB) < 1, "not eligible: the CPU, 2.1 GiB for the GPU's layers");
  assert.equal(weightsPlace({ cpu: 1.5 * GB, gpuOnly: 0.7 * GB, gpu: 1.4 * GB, deviceMemory: 4, eligible: true }).mode, "cpu", "1B on 4 GB: 2.1 GB past 2");
  assert.equal(weightsPlace({ cpu: 1.5 * GB, gpuOnly: 0.3 * GB, gpu: 1.4 * GB, deviceMemory: 4, eligible: true }).mode, "gpu", "within 2 GB: the GPU alone");
  assert.equal(weightsPlace({ cpu: 0.1 * GB, gpuOnly: 0.05 * GB, gpu: 0.1 * GB, deviceMemory: 8, eligible: true, forced: true }).mode, "gpu", "?gpuTest=only");
  assert.equal(weightsPlace({ cpu: 0.1 * GB, gpu: 0.1 * GB, deviceMemory: 8, forced: true }).mode, "both", "forced, not eligible");
  // gpuBytes: Llama 3.2 1B (dim 2048, hidden 8192, 16 layers, 32 heads, 8 of keys and values, vocab 128256, 4096)
  const oneB = gpuBytes([2048, 8192, 16, 32, 8, 128256, 4096]);
  const want = 16 * (2 * 2048 * 2048 + 2 * 512 * 2048 + 3 * 8192 * 2048) * 1.125 + 16 * 2 * 2048 * 4 + 2 * 16 * 4096 * 512 * 2 +
    128256 * 2048 * 1.125 + 4096 * 64 * 4 + 3 * 128256 * 4;
  assert.ok(Math.abs(oneB - want) < 1, `gpuBytes ${oneB} against ${want}`);
  assert.ok(gpuBytes([64, 128, 2, 4, 2, 320, 256], { arch: "gpt2" }) < gpuBytes([64, 128, 2, 4, 2, 320, 256]), "no gate on GPT-2");
}
// T156: why a model cannot go on the GPU alone
{
  const adapter = { fallback: false, limits: { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20, minStorageBufferOffsetAlignment: 256 } };
  const oneB = [2048, 8192, 16, 32, 8, 128256, 4096];
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, adapter), null);
  assert.match(gpuOnlyUnfit(oneB, "int8", {}, undefined), /no GPU adapter/);
  assert.match(gpuOnlyUnfit(oneB, "int8", {}, { ...adapter, fallback: true }), /fallback/);
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, { ...adapter, fallback: true }, { fallback: true }), null, "the tests' leave");
  assert.match(gpuOnlyUnfit(oneB, "int8", { bias: true }, adapter), /steps/);
  assert.match(gpuOnlyUnfit(oneB, "int8", { arch: "gpt2" }, adapter), /steps/);
  assert.match(gpuOnlyUnfit(oneB, "int6", {}, adapter), /int6/);
  // stories15M (dim 288, 6 heads): k starts at 288 × 288 weights, no multiple of 2048 (T152)
  assert.match(gpuOnlyUnfit([288, 768, 6, 6, 6, 32000, 256], "int8", {}, adapter), /would not start/);
  // T220: by the device's 256, not by what the adapter could give: stories15M's k starts on 64 (82944 values, 10368
  // bytes of scales), and gpu.js's device, which asks for no alignment, binds on 256 and would refuse it
  const loose = { ...adapter, limits: { ...adapter.limits, minStorageBufferOffsetAlignment: 64 } };
  assert.match(gpuOnlyUnfit([288, 768, 6, 6, 6, 32000, 256], "int8", {}, loose), /would not start/);
  assert.equal(gpuOnlyUnfit(oneB, "int8", {}, loose), null);
  // Qwen2.5 7B's shape (no biases here): gate and up are 135.8 MB (T152's (d)), past 128 MiB
  assert.match(gpuOnlyUnfit([3584, 18944, 28, 28, 4, 152064, 4096], "int8", {}, adapter), /past a buffer/);
}
// T156: the checkpoint's stretches: the layers' matrices to the GPU's worker, the rest packed into memory, whatever the
// stretches it comes in; each byte posted once, and the writer waits on flow
{
  const tensor = (offset, shape, int8) => ({ kind: int8 ? "int8" : "f32", offset, shape, group: int8 ? 32 : 0,
    scales: int8 ? offset + shape.reduce((a, b) => a * b, 1) : 0 });
  // a made-up Llama: 2 layers, dim 64, hidden 96, the tensors in file order
  let end = 28;
  const tensors = {};
  for (const [name, shape, int8] of [["token_embedding_table", [100, 64], true], ["rms_att_weight", [2, 64], false], ["wq", [2, 64, 64], true],
    ["wk", [2, 32, 64], true], ["wv", [2, 32, 64], true], ["wo", [2, 64, 64], true], ["rms_ffn_weight", [2, 64], false], ["w1", [2, 96, 64], true],
    ["w2", [2, 64, 96], true], ["w3", [2, 96, 64], true], ["rms_final_weight", [64], false]]) {
    tensors[name] = tensor(end, shape, int8);
    const count = shape.reduce((a, b) => a * b, 1);
    end += int8 ? count + count / 8 : count * 4;
  }
  const size = end, holes = gpuHoles(tensors), place = placer(holes);
  // (T210: the embedding too, which is the classifier here: one stretch)
  assert.equal(holes.length, 8);
  assert.equal(gpuHoles({ ...tensors, wcls: tensors.token_embedding_table }).length, 8, "a shared classifier is one stretch");
  assert.equal(place(tensors.rms_att_weight.offset), 28, "the first norm goes where the embedding began");
  assert.equal(place(tensors.rms_ffn_weight.offset), 28 + 2 * 64 * 4, "the norm after wo goes after the first norm");
  const checkpoint = Uint8Array.from({ length: size }, (_, i) => (i * 7 + 3) & 255);
  const memory = new WebAssembly.Memory({ initial: 2, maximum: 4, shared: true }), base = 64;
  let flow;
  const posted = new Uint8Array(size), counts = new Uint8Array(size);
  const worker = {
    postMessage({ offset, bytes }) {
      posted.set(bytes, offset);
      for (let i = 0; i < bytes.length; i++) counts[offset + i]++;
      Atomics.add(flow, 0, BigInt(bytes.length));
    },
  };
  const weights = gpuOnlyWeights({ memory, base, size, tensors, worker });
  flow = new BigInt64Array(weights.flow);
  const steps = [1, 3, 4093, 17, 65536];
  for (let at = 0, k = 0; at < size; at += steps[k++ % steps.length]) weights.write(at, checkpoint.subarray(at, Math.min(size, at + steps[k % steps.length])));
  await weights.room();
  await weights.drained();
  const inHole = (i) => holes.some(([a, b]) => i >= a && i < b);
  const stored = new Uint8Array(memory.buffer, base, weights.stored);
  for (let i = 0; i < size; i++) {
    if (inHole(i)) assert.ok(counts[i] === 1 && posted[i] === checkpoint[i], `byte ${i} posted once`);
    else assert.ok(counts[i] === 0 && stored[place(i)] === checkpoint[i], `byte ${i} in memory at ${place(i)}`);
  }
  assert.equal(weights.stored, size - holes.reduce((sum, [a, b]) => sum + b - a, 0));
  // what gpu.js opens with: each layer's values and scales where they start in the checkpoint
  const plan = gpuOnlyPlan([64, 96, 2, 4, 2, 100, 32], tensors);
  assert.deepEqual(plan.matrices.w2.layers[1], [tensors.w2.offset + 64 * 96, tensors.w2.scales + (64 * 96 / 32) * 4]);
  // T210: and the tables, where they start in the checkpoint (the embedding null where the classifier is it)
  const e = tensors.token_embedding_table;
  assert.deepEqual(plan.tables, { classifier: { rows: 100, n: 64, at: [e.offset, e.scales] }, embedding: null });
  const apart = gpuOnlyPlan([64, 96, 2, 4, 2, 100, 32], { ...tensors, wcls: tensor(end, [100, 64], true) });
  assert.deepEqual(apart.tables, { classifier: { rows: 100, n: 64, at: [end, end + 6400] }, embedding: { rows: 100, n: 64, at: [e.offset, e.scales] } });
  assert.equal(gpuHoles({ ...tensors, wcls: tensor(end, [100, 64], true) }).length, 9, "a classifier of its own is a stretch of its own");
}

// T156 (the owner, 2026-09-27): a model on the GPU alone weighed on the prompts too, by the page's use, and the verdict
// kept for the device and /benchmark/'s CPU reading
{
  // Llama 3.2 3B (dim 3072, hidden 8192, 28 layers, 24 heads, 8 of keys and values): 2.8 G multiply-adds a token
  const header = [3072, 8192, 28, 24, 8, 128256, 4096], weights = layerWeightsOf(header);
  assert.equal(weights, 28 * (2 * 3072 * 3072 + 2 * 1024 * 3072 + 3 * 8192 * 3072));
  const size = 3.6e9, cpu = { GBps: 28.7, promptGMACs: 40 };
  // the CPU's step 125 ms (3.6 GB at 28.7 GB/s), its prompt's token 70.6 ms (2.8 G at 40 G MAC/s)
  const slowStep = { stepMs: 160, promptMs: 14 };
  // the answers alone: the CPU faster (125 < 0.95 × 160)
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu: { GBps: 28.7 }, gpu: slowStep }).cpuFaster, true, "the steps alone: the CPU");
  // with the prompts, as many tokens: 125 + 70.6 against 160 + 14: the CPU still (195.6 < 165.3? no): the GPU
  const even = aloneVerdict({ size, layerWeights: weights, cpu, gpu: slowStep, usage: USAGE_UNKNOWN });
  assert.equal(even.cpuFaster, false, `as many prompt tokens as written: the GPU (${even.cpu} against ${even.gpu})`);
  // mostly writing (1 prompt token to 10 written): the CPU
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu, gpu: slowStep, usage: { prompt: 1, written: 10 } }).cpuFaster, true, "writing: the CPU");
  // nothing known of the CPU: the GPU stays
  assert.equal(aloneVerdict({ size, layerWeights: weights, cpu: {}, gpu: slowStep }).cpuFaster, false, "no /benchmark/: the GPU");
  // the use kept: each generation's added to four fifths of what was there
  assert.deepEqual(usedAfter(usedAfter(undefined, 100, 50), 20, 200), { prompt: 100 * 0.8 + 20, written: 50 * 0.8 + 200 });
  // the key: the adapter as the device made of it (the same features and limits), another description another key;
  // the verdict holds for the same key and the same reading of /benchmark/'s CPU, and for nothing else
  const adapter = { info: { vendor: "arm", architecture: "valhall", device: "", description: "Mali-G615" },
    features: new Set(["shader-f16", "subgroups"]), limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 } };
  const device = { features: new Set(adapter.features), limits: { ...adapter.limits } };
  const key = deviceKey(adapter);
  assert.equal(deviceKey(adapter, device), key, "the worker's key is gpu.js's");
  assert.notEqual(deviceKey({ ...adapter, info: { ...adapter.info, description: "Mali-G715" } }), key, "another GPU");
  assert.notEqual(deviceKey({ ...adapter, features: new Set(["subgroups"]) }), key, "other shaders");
  const alone = { key, cpu };
  assert.equal(aloneHolds(alone, key, { ...cpu, threads: 4 }), true);
  assert.equal(aloneHolds(alone, key + "x", cpu), false, "another device, browser or shaders");
  assert.equal(aloneHolds(alone, key, { ...cpu, GBps: 30.1 }), false, "/benchmark/'s CPU measured again");
  assert.equal(aloneHolds(alone, key, undefined), false, "no /benchmark/ now");
  assert.equal(aloneHolds(undefined, key, cpu), false);
}
console.log("ok");
