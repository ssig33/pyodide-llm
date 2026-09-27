// jobs.js (T109): what a job of a phase is, for the one that hands them out (forward.js, in the model's worker) and
// for the software threads that take them (helper.js). One place for the layout of the control area at the start of
// a shared memory, the kinds of kernel calls, and how the rows of a job are run, so that the two never drift apart.
//
// A plain ES module. forward.js and helper.js import it with the ?v= of their own URL (GitHub Pages caches a file
// for ten minutes: the three must come from the same deployment).

/** the control area: the first bytes of a shared memory, as 32-bit words (the helpers', then the GPU's: GPU_DONE) */
export const CONTROL_BYTES = 8192;
// GEN: the generation of the phase (odd while the coordinator rewrites the jobs); QUIT: the helpers end; COUNTER:
// the next chunk to take; FINISHED: chunks done; ACTIVE: helpers inside a phase; TOTAL: chunks in all
export const GEN = 0, QUIT = 1, COUNTER = 2, FINISHED = 3, ACTIVE = 4, TOTAL = 5;
// WAKE + share: each helper's own word. The coordinator writes the generation there and wakes exactly the helpers it
// wants, always the same ones. (With one word for all, Atomics.notify wakes whoever has slept longest, so a
// different, cold helper took every phase.)
export const WAKE = 256;
// JOBS: the word that says how many jobs the phase has. The jobs themselves are a table of float64 from byte
// JOB_TABLE, JOB numbers each: a double holds an address exactly up to 2^53, where an int32 cut the addresses of a
// model past 4 GiB of a 64-bit memory (T101: every helper computed on the wrong bytes, only the coordinator right)
export const JOBS = 512, JOB = 16, JOB_TABLE = 2056;
/** T108: the most tokens that go through the layers together, and so the most jobs a phase has */
export const BATCH = 16;
/** scratch for a helper's warm-up, after the last job */
export const SCRATCH = 4608;
// T135: the words of the GPU's worker (gpu.js), which runs the blocks of a prompt that forward.js hands it while
// forward.js waits: GPU_DONE, the number of the last request it finished; GPU_FAILED, whether that one failed;
// GPU_BEAT, counted up while it works (a worker that stopped counting has stopped); GPU_WANTED, the request forward.js
// waits for (T147: 0 once it gave the GPU up, and the GPU's worker writes nothing for a request no longer wanted)
export const GPU_DONE = 1280, GPU_FAILED = 1281, GPU_BEAT = 1282, GPU_WANTED = 1283;
if (JOB_TABLE <= JOBS * 4 || JOB_TABLE % 8 || JOB_TABLE + BATCH * JOB * 8 > SCRATCH || SCRATCH + 448 > GPU_DONE * 4 ||
    (GPU_WANTED + 1) * 4 > CONTROL_BYTES) {
  throw new Error("the control area does not hold its jobs, the scratch and the GPU's words: see jobs.js");
}

// A job: [kind, eight arguments, rows, count, out stride, a stride, b stride], and the control area holds it as it
// is, then the size of its chunks and the number of its first chunk. The kinds and their arguments:
//   0 matmul_q8r     out, xq, xs, w, scales, corrections, n           rows: the matrix's
//   1 matmul_q8      out, xq, xs, w, scales, -, n
//   2 matmul_f32     out, x, -, w, -, -, n
//   3 attention      out, q, keys, values, scores, pos, kv heads, head size   rows: the heads (T109)
//   4 attention_f16  the same over a cache of float16 (T110)
//   5 matmul_q6r     as 0, on int6 weights (T98)
//   6 matmul_q6      as 1, on int6 weights
// count > 1 (T108): the same rows for count tokens, whose out, a and b are that many bytes apart. The rows then go in
// blocks small enough to stay in the cache while every token uses them: each (row, token) is the one kernel call it
// is for a single token, so the numbers are the same as one token at a time. Kind 0 takes the count tokens in one
// call instead (T159: matmul_q8r_tile, four rows by four tokens, each number matmul_q8r's to the bit; a and b are
// then one frame apart, the same stride). Attention is always one token a job.
export const ROWS = 9, COUNT = 10, SIZE = 14, FIRST = 15;
const BLOCK_BYTES = 16384;
export const blockRows = (kind, n) => Math.max(1, Math.floor(BLOCK_BYTES / (kind === 2 ? 4 * n : n)));

/** T101: which arguments of each kernel are addresses (usize in kernels/*.ts; tests/forward-check.mjs holds this
 * table to the source). On a 64-bit memory the kernels take them as BigInt. */
export const ADDRESSES = {
  matmul_f32: [0, 1, 2], quantize_x: [0, 1, 2], quantize6_x: [0, 1, 2], six_sums: [0, 1], int8_sums: [0, 1], widen_bf16: [0, 1], widen_q8_0: [0, 1], matmul_q8: [0, 1, 2, 3, 4],
  matmul_q6: [0, 1, 2, 3, 4], rmsnorm: [0, 1, 2], rope: [0, 1, 2], attention: [0, 1, 2, 3, 4], attention_f16: [0, 1, 2, 3, 4],
  to_f16: [0, 1], from_f16: [0, 1], layernorm: [0, 1, 2, 3], gelu: [0, 1, 2], swiglu: [0, 1, 2], add_columns: [0, 1, 2], add_inplace: [0, 1],
  argmax: [0], penalize: [0, 1], sample: [0, 5, 6], matmul_q8r: [0, 1, 2, 3, 4, 5], matmul_q6r: [0, 1, 2, 3, 4, 5],
  matmul_q8r_tile: [0, 1, 2, 3, 4, 5],
};
/** A kernel module's exports as they are, or on a 64-bit memory (wide) with the addresses made BigInt on the way
 * in: the rest of the code keeps its addresses in Numbers (exact up to 2^53). */
export function addressed(exports, wide) {
  if (!wide) return exports;
  const out = { ...exports };
  for (const [name, at] of Object.entries(ADDRESSES)) {
    const kernel = exports[name];
    if (kernel) {
      out[name] = (...args) => {
        for (const i of at) args[i] = BigInt(args[i]);
        return kernel(...args);
      };
    }
  }
  return out;
}

/** runRows(job, r0, r1) on these kernels: k, the plain module's exports; r, the relaxed module's or null. job is
 * an array or a view of the control area. */
export function runner(k, r) {
  const call = (kind, out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => {
    if (kind === 0) r.matmul_q8r(out, a, b, a4, a5, a6, a7, r0, r1);
    else if (kind === 1) k.matmul_q8(out, a, b, a4, a5, a7, r0, r1);
    else if (kind === 5) r.matmul_q6r(out, a, b, a4, a5, a6, a7, r0, r1);
    else if (kind === 6) k.matmul_q6(out, a, b, a4, a5, a7, r0, r1);
    else if (kind === 2) k.matmul_f32(out, a, a4, a7, r0, r1);
    else if (kind === 3) k.attention(out, a, b, a4, a5, a6, rows, a7, a8, r0, r1);
    else k.attention_f16(out, a, b, a4, a5, a6, rows, a7, a8, r0, r1);
  };
  return (job, r0, r1) => {
    const [kind, out, a, b, a4, a5, a6, a7, a8, rows, count, os, as, bs] = job;
    if (count === 1) return call(kind, out, a, b, a4, a5, a6, a7, a8, rows, r0, r1);
    if (kind === 0) return r.matmul_q8r_tile(out, a, b, a4, a5, a6, a7, r0, r1, count, os, as);
    const step = blockRows(kind, a7);
    for (let r = r0; r < r1; r += step) {
      const end = Math.min(r + step, r1);
      for (let t = 0; t < count; t++) call(kind, out + t * os, a + t * as, b + t * bs, a4, a5, a6, a7, a8, rows, r, end);
    }
  };
}

/** the kernels warmed up: a new thread runs them unoptimized at first, and a measurement would take that for the
 * speed of the count (T93). Tiny calls on the scratch, many times; what they compute is thrown away. */
export function warmUp(k, r) {
  const xq = SCRATCH, xs = SCRATCH + 64, w = SCRATCH + 128, s = SCRATCH + 192, c = SCRATCH + 256, out = SCRATCH + 320;
  for (let i = 0; i < 4000; i++) {
    k.matmul_q8(out, xq, xs, w, s, 32, 0, 1);
    k.matmul_q6(out, xq, xs, w, s, 32, 0, 1);
    k.matmul_f32(out, out + 64, w, 8, 0, 1);
    if (r) {
      r.matmul_q8r(out, xq, xs, w, s, c, 32, 0, 1);
      r.matmul_q6r(out, xq, xs, w, s, c, 32, 0, 1);
      r.matmul_q8r_tile(out, xq, xs, w, s, c, 32, 0, 4, 4, 0, 0);  // one tile: four rows and four tokens, all the same
    }
    k.attention(out, xq, w, w, c, 0, 1, 1, 4, 0, 1);  // one head of 4 at position 0
    k.attention_f16(out, xq, w, w, c, 0, 1, 1, 4, 0, 1);
  }
}
