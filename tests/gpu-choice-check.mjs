// T148: how forward.js weighs a block of a prompt on the GPU against the CPU (promptTimes), in Node, without a GPU:
//   node tests/gpu-choice-check.mjs
// Made-up times: the CPU's ms a token of its blocks, gpu.js's two blocks timed as it starts, the blocks the GPU then ran.
import assert from "node:assert/strict";
import { promptTimes } from "../public/forward.js";

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
console.log("ok");
