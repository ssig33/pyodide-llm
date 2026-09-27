// T148: how forward.js weighs a block of a prompt on the GPU against the CPU (promptTimes), in Node, without a GPU:
//   node tests/gpu-choice-check.mjs
// Made-up times: the CPU's ms a token of its blocks, gpu.js's two blocks timed as it starts, the blocks the GPU then ran.
import assert from "node:assert/strict";
import { promptTimes, tokenTimes } from "../public/forward.js";
import { halvesOf } from "../public/shaders.js";

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
console.log("ok");
