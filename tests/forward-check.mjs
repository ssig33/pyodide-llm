// The forward pass of public/forward.js (the page's) against the NumPy forward of llama2_numpy.py, on the models of
// this directory (make models kernels) or converted ones. Since T93 there is no other forward to hold it to.
//   float32: the kernels add in another order than NumPy, so the last bits differ; the most likely token must be the
//            same at every position, and no logit may differ by more than 1e-3 (measured 2026-09-25: 1.7e-5 to 3.7e-5).
//   int8: forward.js quantizes the activations too (7 bits with relaxed SIMD), NumPy does not, so the numbers
//         differ by design. Measured 2026-09-25 on NumPy's greedy text: at 128 positions the most likely token the
//         same at 93.8 to 100% and the perplexity -0.23 to +2.00% apart; at 64 positions llm-jp-3 150M was 87.5% and
//         +3.42% (fewer positions, more spread). The line: 85% or more and within 5%, at 128 positions (the
//         default). A real fault (a wrong order, a wrong scale) lands far outside: the agreement near nothing and
//         the perplexity a multiple.
// T108: the same text read by forward.js one token at a time and in blocks (forward_many), from a KV cache that
// starts small so that it grows within the blocks: the last logits must be the same to the bit.
// Then the speeds, both in turn. Runs in the deployment.
//
//   node tests/forward-check.mjs [model id | <out> of tests/perplexity_prepare.py ...] [--rounds 3] [--positions 128]
//        [--without relaxed,int8,sampler] [--plain] [--wide]
//
// The memory is shared, as the page's where it is cross-origin isolated; --plain: not shared, as the page's where it
// is not (the keys and values then stay float32, T110). --wide: a 64-bit memory and its kernels (T101), as the page
// has for a model past 4 GiB.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
import { automaticDtype, footprint, keysInHalf } from "../public/forward.js";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), positions = option("--positions", 128);
const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const without = args.includes("--without") ? args[args.indexOf("--without") + 1].split(",") : [];
const modelOf = (id) => MODELS.find((m) => m.id === id) ?? { name: path.basename(id), checkpoint: path.resolve(`${id}.bin`),
  tokenizer: path.resolve(`${id}.tokenizer.bin`), options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) };
const file = (f) => (path.isAbsolute(f) ? f : root + f);

// T98: six_sums (the corrections of int6 weights for matmul_q6r) against the sums of the int8 values the layout of
// llama2_numpy.pack6 holds, taken apart byte by byte here. T197: a correction is the int32 −64 × the sum (before it
// the float32 scale × the sum)
{
  const memory = new WebAssembly.Memory({ initial: 4 });
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
  const U = new Uint8Array(memory.buffer), N = new Int32Array(memory.buffer);
  const groups = 2000, w = 4096, out = w + groups * 24;
  for (let i = 0; i < groups * 24; i++) U[w + i] = (Math.imul(i, 2654435761) >>> 7) & 255;
  k.six_sums(out, w, groups);
  for (let g = 0; g < groups; g++) {
    let sum = 0;
    for (let j = 0; j < 32; j++) {
      const at = w + g * 24, low = j < 16 ? U[at + j] & 15 : U[at + j - 16] >> 4;
      const top = (U[at + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      sum += (((low | (top << 4)) << 2) << 24) >> 24;
    }
    if (-64 * sum !== N[out / 4 + g]) throw new Error(`six_sums differs at group ${g}: ${N[out / 4 + g]} against ${-64 * sum}`);
  }
  // T123: int8_sums, the same for int8 weights, 32 bytes a group, against the sums taken here; groups at the ends of
  // int8 (all -128: 262144, all 127: -260096)
  const I = new Int8Array(memory.buffer);
  I.fill(-128, w, w + 32);
  I.fill(127, w + 32, w + 64);
  k.int8_sums(out, w, groups * 24 / 32);
  for (let g = 0; g < groups * 24 / 32; g++) {
    let sum = 0;
    for (let j = 0; j < 32; j++) sum += I[w + g * 32 + j];
    if (-64 * sum !== N[out / 4 + g]) throw new Error(`int8_sums differs at group ${g}: ${N[out / 4 + g]} against ${-64 * sum}`);
  }
}

// T230: matmul_t2 (ternary weights, blocks of 128 with a float16 scale) to the bit against its sums taken here: each
// group's 32 products less the activations' bias times the weights' sum, one exact integer, times the block's scale
// times the group's activation scale rounded once, added into lane g % 4 and the lanes added in order at the end. At 1,
// 2, 3 and 7 blocks, with the activations of quantize_x(bias = 0) (-127..127) and of bias = 64 (0..127), the codes all
// three values in every block and every row
{
  const memory = new WebAssembly.Memory({ initial: 8 });
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const rows = 9, x = 4096, xs = 8192, w = 16384, out = 65536;
  let seed = 12345;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const half = (h) => {  // a float16's bits as a number, exactly
    const exp = (h >> 10) & 31, frac = h & 1023, sign = h & 0x8000 ? -1 : 1;
    return sign * (exp ? (1 + frac / 1024) * 2 ** (exp - 15) : frac * 2 ** -24);
  };
  const round = Math.fround;
  for (const blocks of [1, 2, 3, 7]) {
    const n = blocks * 128;
    for (let i = 0; i < rows * blocks; i++) {
      const at = w + i * 34, h = 0x2000 + (next() % 0x3000);  // a normal float16 between about 1e-4 and 8
      U[at] = h & 255; U[at + 1] = h >> 8;
      for (let j = 0; j < 32; j++) U[at + 2 + j] = (next() % 3) | (next() % 3) << 2 | (next() % 3) << 4 | (next() % 3) << 6;
    }
    for (let g = 0; g < n / 32; g++) F[xs / 4 + g] = round(1e-2 * (1 + (next() % 1000)));
    for (const bias of [0, 64]) {
      for (let j = 0; j < n; j++) I[x + j] = bias ? next() % 128 : (next() % 255) - 127;
      F.fill(-7, out / 4, out / 4 + rows);
      k.matmul_t2(out, x, xs, w, n, 0, rows, bias);
      for (let i = 0; i < rows; i++) {
        const lanes = [0, 0, 0, 0];
        for (let b = 0; b < blocks; b++) {
          const at = w + (i * blocks + b) * 34, d = half(U[at] | U[at + 1] << 8);
          for (let g = 0; g < 4; g++) {
            let dot = 0;
            for (let j = 0; j < 32; j++) {
              const v = b * 128 + g * 32 + j, code = (U[at + 2 + ((g * 32 + j) >> 2)] >> (2 * (j & 3))) & 3;
              dot += (code - 1) * (I[x + v] - bias);
            }
            lanes[g] = round(lanes[g] + round(dot * round(d * F[xs / 4 + b * 4 + g])));
          }
        }
        const expected = round(round(round(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
        if (F[out / 4 + i] !== expected) throw new Error(`matmul_t2 differs at row ${i} of ${blocks} blocks, bias ${bias}: ${F[out / 4 + i]} against ${expected}`);
      }
    }
  }
}

// T165: matmul_q8 and matmul_q6 (the path without relaxed SIMD) to the bit against their sums taken here: each group's
// 32 products are exact integers in four int32 lanes (lane k holds products 2k, 2k + 1, 2k + 8, 2k + 9 of each half of
// 16), scaled and added lane by lane in float32, the four lanes added last. The weights take every int8 (-128 too) and
// the activations every value quantize_x gives (-127..127), with groups at the ends: 127 × 127 and -128 × -127 on
// every lane (sums of two products that int16 still holds). T196: at 1 to 8, 11 and 41 groups, as T167's check of
// matmul_q8r below: matmul_q8 takes four groups a turn and the rest one by one, and at 41 alone one group comes after
// the fours (a kernel that scaled the groups after the fours with the scale of the four before them passed there)
{
  const memory = new WebAssembly.Memory({ initial: 8 });
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const rows = 24, n = 32 * 41, ng = n / 32;  // the most groups tried: 41
  const w = 4096, w6 = w + rows * n, ws = w6 + rows * ng * 24, x = ws + rows * ng * 4, xs = x + n, out = xs + ng * 4;
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const pool = Int8Array.from({ length: rows * n }, () => (next() & 255) - 128);  // weight j of row i: pool[i * n + j]
  for (let j = 0; j < n; j++) I[x + j] = (next() % 255) - 127;
  for (let j = 0; j < 32; j++) { I[x + j] = 127; I[x + 32 + j] = -127; }
  for (let i = 0; i < rows * ng; i++) F[ws / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000)));
  for (let g = 0; g < ng; g++) F[xs / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
  for (let i = 0; i < rows * ng * 24; i++) U[w6 + i] = next() & 255;
  const six = (i, j, groups = ng) => {  // the int8 value of int6 weight j of row i of groups groups (six_sums above)
    const at = w6 + (i * groups + (j >> 5)) * 24, m = j & 31;
    const low = m < 16 ? U[at + m] & 15 : U[at + m - 16] >> 4, top = (U[at + 16 + (m % 8)] >> (2 * ((m / 8) | 0))) & 3;
    return (((low | (top << 4)) << 2) << 24) >> 24;
  };
  // rows of groups groups: the weights (row stride groups × 32) and the scales (groups a row) packed for that count
  const reference = (weight, groups) => Array.from({ length: rows }, (_, i) => {
    const lanes = [0, 0, 0, 0];
    for (let g = 0; g < groups; g++) {
      const s = Math.fround(F[ws / 4 + i * groups + g] * F[xs / 4 + g]);
      for (let lane = 0; lane < 4; lane++) {
        let sum = 0;
        for (const j of [2 * lane, 2 * lane + 1, 2 * lane + 8, 2 * lane + 9, 2 * lane + 16, 2 * lane + 17, 2 * lane + 24, 2 * lane + 25]) {
          sum += weight(i, g * 32 + j) * I[x + g * 32 + j];
        }
        lanes[lane] = Math.fround(lanes[lane] + Math.fround(sum * s));
      }
    }
    return Math.fround(Math.fround(Math.fround(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
  });
  for (const groups of [1, 2, 3, 4, 5, 6, 7, 8, 11, ng]) {
    const m = groups * 32;
    for (let i = 0; i < rows; i++) for (let j = 0; j < m; j++) I[w + i * m + j] = pool[i * n + j];
    // the ends: 127 × 127 in row 0's first group, -128 × -127 in row 1's second (-128 × 127 in its first, of one group)
    for (let j = 0; j < 32; j++) { I[w + j] = 127; I[w + m + (groups > 1 ? 32 : 0) + j] = -128; }
    for (const [name, weights, weight] of [["matmul_q8", w, (i, j) => I[w + i * m + j]], ["matmul_q6", w6, (i, j) => six(i, j, groups)]]) {
      F.fill(-7, out / 4, out / 4 + rows);
      k[name](out, x, xs, weights, ws, m, 0, rows);
      const expected = reference(weight, groups);
      for (let i = 0; i < rows; i++) {
        if (F[out / 4 + i] !== expected[i]) throw new Error(`${name} differs at row ${i} of ${groups} groups: ${F[out / 4 + i]} against ${expected[i]}`);
      }
    }
  }
  // T166: matmul_q6r (relaxed SIMD) to the bit against matmul_q8r on the int8 values six() takes apart here (the two
  // add their products in the same order), the activations 0..127 as quantize_x(bias = 64) gives them, and the
  // corrections from six_sums and int8_sums. T167: and matmul_q8r to the bit against its sums taken here: each
  // group's 32 products as one exact integer, times the group's scale (weight scale times activation scale) rounded
  // once, added into lane g % 4 for the groups in fours and the lanes added in order, the groups past the last four
  // added one at a time (T197: the bias of 64 taken out of each group's integer before it is scaled). At 1 to 7 groups (no four at all, and one four with 0 to 3
  // after it) and at 41: T167's review, at 41 alone one group comes after the fours, and a kernel that scaled every
  // group after the fours with the first one's scale passed
  const relaxed = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_relaxed_plain.wasm`)), { env: { memory } }).exports;
  const w8 = out + rows * 4, wc6 = w8 + rows * n, wc8 = wc6 + rows * ng * 4, out8 = wc8 + rows * ng * 4;
  for (let j = 0; j < n; j++) I[x + j] = next() % 128;
  const round = Math.fround;
  // T159's tokens: a frame each (activations, then their scales), and the outputs of the tile and of one at a time
  const frame = 2048, frameScales = 1536, outFrame = 256;
  const frames = 1 << 18, tiles = frames + 9 * frame, singles = tiles + 9 * outFrame;
  if (out8 + rows * 4 > frames || singles + 9 * outFrame > memory.buffer.byteLength) throw new Error("forward-check's memory does not hold T159's frames");
  for (const groups of [1, 2, 3, 4, 5, 6, 7, 8, 11, ng]) {
    const m = groups * 32, fours = groups & ~3;
    for (let i = 0; i < rows; i++) for (let j = 0; j < m; j++) I[w8 + i * m + j] = six(i, j, groups);
    k.six_sums(wc6, w6, rows * groups);
    k.int8_sums(wc8, w8, rows * groups);
    relaxed.matmul_q6r(out, x, xs, w6, ws, wc6, m, 0, rows);
    relaxed.matmul_q8r(out8, x, xs, w8, ws, wc8, m, 0, rows);
    for (let i = 0; i < rows; i++) {
      if (F[out / 4 + i] !== F[out8 / 4 + i]) throw new Error(`matmul_q6r differs at row ${i} of ${groups} groups: ${F[out / 4 + i]} against matmul_q8r's ${F[out8 / 4 + i]}`);
      const part = (g) => {  // T197: the group's exact dot(w, q - 64), scaled once
        let dot = 0;
        for (let j = 0; j < 32; j++) dot += I[w8 + i * m + g * 32 + j] * (I[x + g * 32 + j] - 64);
        return round(dot * round(F[ws / 4 + i * groups + g] * F[xs / 4 + g]));
      };
      const lanes = [0, 0, 0, 0];
      for (let g = 0; g < fours; g++) lanes[g & 3] = round(lanes[g & 3] + part(g));
      let sum = round(round(round(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
      for (let g = fours; g < groups; g++) sum = round(sum + part(g));
      const expected = sum;
      if (F[out8 / 4 + i] !== expected) throw new Error(`matmul_q8r differs at row ${i} of ${groups} groups: ${F[out8 / 4 + i]} against ${expected}`);
    }
    // T159: matmul_q8r_tile (a prompt's count tokens, four rows by four tokens) to the bit against matmul_q8r token by
    // token, at every count of tokens 1 to 9 (no four, one four and 1 to 3 after it, two fours) and row ranges that
    // start and end off the fours (none, one tile, tiles with 1 to 3 rows after them), with every group count above;
    // the rows and tokens outside are left as they were
    for (let t = 0; t < 9; t++) {
      for (let j = 0; j < m; j++) I[frames + t * frame + j] = next() % 128;
      for (let g = 0; g < groups; g++) F[(frames + t * frame + frameScales) / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
    }
    for (const count of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
      for (const [r0, r1] of [[0, rows], [1, rows - 1], [2, 5], [3, 10], [4, 8], [5, 6], [7, 7]]) {
        F.fill(-7, tiles / 4, (tiles + 9 * outFrame) / 4);
        relaxed.matmul_q8r_tile(tiles, frames, frames + frameScales, w8, ws, wc8, m, r0, r1, count, outFrame, frame);
        F.fill(-7, singles / 4, (singles + 9 * outFrame) / 4);
        for (let t = 0; t < count; t++) {
          relaxed.matmul_q8r(singles + t * outFrame, frames + t * frame, frames + t * frame + frameScales, w8, ws, wc8, m, r0, r1);
        }
        for (let t = 0; t < 9; t++) {
          for (let i = 0; i < rows; i++) {
            const got = F[(tiles + t * outFrame) / 4 + i], want = F[(singles + t * outFrame) / 4 + i];
            if (!Object.is(got, want)) {
              throw new Error(`matmul_q8r_tile differs from matmul_q8r at row ${i}, token ${t} of ${count}, rows ${r0}..${r1}, ${groups} groups: ${got} against ${want}`);
            }
          }
        }
      }
    }
  }
}

// T197's review: the ends of the relaxed path's integers, which the check above (weights of int6, from -128 to 124)
// does not reach: rows of -128 and 127 against activations of 127 and 1 (quantize_x's ends with its bias of 64), in
// matmul_q8r and in the tile, with int8_sums' corrections. The scales are powers of 2, so every group's dot(w, q - 64)
// times them and their sum over the 5 groups are exact in float32 (at most 5 × 32 × 128 × 63 < 2^24): the kernels
// must give the float64 sums exactly, whatever the order of their adds.
{
  const memory = new WebAssembly.Memory({ initial: 1 });
  const env = { env: { memory } };
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), env).exports;
  const relaxed = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_relaxed_plain.wasm`)), env).exports;
  const I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const rows = 8, groups = 5, n = groups * 32, frame = 1024, frameScales = 512, outFrame = 64;
  const w = 1024, ws = w + rows * n, wc = ws + rows * groups * 4, frames = 4096, tiles = frames + 4 * frame, singles = tiles + 4 * outFrame;
  let seed = 11;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const weight = [(j) => -128, (j) => 127, (j) => (j & 1 ? 127 : -128), (j) => (j & 31) < 16 ? -128 : 127];
  for (let i = 0; i < rows; i++) for (let j = 0; j < n; j++) I[w + i * n + j] = i < 4 ? weight[i](j) : (next() & 255) - 128;
  const activation = [(j) => 127, (j) => 1, (j) => (j & 1 ? 1 : 127), (j) => 1 + (next() % 127)];
  for (let t = 0; t < 4; t++) {
    for (let j = 0; j < n; j++) I[frames + t * frame + j] = activation[t](j);
    for (let g = 0; g < groups; g++) F[(frames + t * frame + frameScales) / 4 + g] = 2 ** -(2 + t);
  }
  F.fill(2 ** -7, ws / 4, ws / 4 + rows * groups);
  k.int8_sums(wc, w, rows * groups);
  relaxed.matmul_q8r_tile(tiles, frames, frames + frameScales, w, ws, wc, n, 0, rows, 4, outFrame, frame);
  for (let t = 0; t < 4; t++) relaxed.matmul_q8r(singles + t * outFrame, frames + t * frame, frames + t * frame + frameScales, w, ws, wc, n, 0, rows);
  for (let t = 0; t < 4; t++) {
    for (let i = 0; i < rows; i++) {
      let want = 0;
      for (let j = 0; j < n; j++) want += I[w + i * n + j] * (I[frames + t * frame + j] - 64) * 2 ** -7 * 2 ** -(2 + t);
      for (const [name, at] of [["matmul_q8r", singles], ["matmul_q8r_tile", tiles]]) {
        const got = F[(at + t * outFrame) / 4 + i];
        if (got !== want) throw new Error(`${name} at the ends of int8: row ${i}, token ${t}: ${got} against ${want}`);
      }
    }
  }
}

// T101: jobs.js says which arguments of each kernel are addresses (BigInt on a 64-bit memory): the same as the
// usize parameters of the kernels' source, every exported one
{
  const { ADDRESSES } = await import("../public/jobs.js");
  const source = {};
  for (const file of ["kernels/kernel.ts", "kernels/kernel_relaxed.ts"]) {
    for (const [, name, parameters] of fs.readFileSync(root + file, "utf8").matchAll(/export function (\w+)\(([^)]*)\)/g)) {
      source[name] = parameters.split(",").map((p, i) => [i, p.split(":")[1].trim()]).filter(([, type]) => type === "usize").map(([i]) => i);
    }
  }
  if (JSON.stringify(Object.keys(source).sort().map((n) => [n, source[n]])) !== JSON.stringify(Object.keys(ADDRESSES).sort().map((n) => [n, ADDRESSES[n]]))) {
    throw new Error("jobs.js's ADDRESSES is not the kernels' usize parameters");
  }
}

const shared = !args.includes("--plain");
const { pyodide: py, kernels } = await pyodideWithEngine({ shared, wide: args.includes("--wide") });
py.runPython("import time, gc, math, numpy as np, llama2_numpy\nfrom llama2_numpy import Llama");
let failed = false;
// T133: the bits of a model converted with none asked for. Llama-3.2-3B's int8 (3614847004 bytes, its header from
// config.json) does not fit a 32-bit memory with its forward pass (T115: 4.41 GiB shared): int8 on a 64-bit memory
// where the browser has one, six bits where not; a model that fits stays int8 either way
{
  const header = [3072, 8192, 28, 24, 8, 128256, 4096], int8 = 3614847004;
  const after = footprint(header, int8, { dtype: "int8", halfKV: true });
  assert.equal(automaticDtype(int8, after, true), "int8", "a 64-bit memory: int8");
  assert.equal(automaticDtype(int8, after, false), "int6", "no 64-bit memory: six bits");
  const small = [1536, 8960, 28, 12, 2, 151936, 4096], qwen = 1736865820;  // Qwen2.5 1.5B
  assert.equal(automaticDtype(qwen, footprint(small, qwen, { dtype: "int8", halfKV: true }), false), "int8");
}
// T144: heads of another size than dim / heads, which none of the models below has. Qwen3 0.6B's int8 (670744604
// bytes, its header from config.json) with the options the converter gives it: forward.js put 755.3 MiB after it on
// a shared memory and 1427.3 MiB on a plain one (measured in the review of T124). Without head_dim footprint() counts
// its keys and values 45% short (419.1 and 755.1 MiB) and a memory chosen by that runs out near the end of the context.
// T160: a grouped-query model (16 heads, 8 of keys and values) keeps float32 keys and values on a shared memory too:
// the 1427.3 MiB of the plain one either way (the two differ in nothing else footprint() counts)
{
  const header = [1024, 3072, 28, 16, 8, 151936, 4096], int8 = 670744604, MiB = 1 << 20;
  const options = { dtype: "int8", bias: false, arch: "llama", qk_norm: true, head_dim: 128 };
  for (const halfKV of [true, false]) {
    const bound = footprint(header, int8, { ...options, halfKV }) / MiB, placed = 1427.3;
    assert.ok(bound >= placed && bound < placed + 4, `Qwen3 0.6B: ${bound.toFixed(1)} MiB counted, ${placed} placed`);
  }
}
// T160: keys and values in float16 (a shared memory) for a model with a key of every head; for a grouped-query one
// float32, but where that does not fit a 32-bit memory (Qwen2.5 3B: 3.82 GiB, 4.03 in float32; Llama 3.2 3B on a
// 64-bit memory either way, 4.41 GiB in float16: the owner, 2026-09-27). The sizes: llama2_convert.checkpoint_size()
{
  const cases = [["llm-jp-3 150M", [512, 2048, 12, 8, 8, 99584, 4096], 160e6, {}, true],
    ["Qwen2.5 0.5B", [896, 4864, 24, 14, 2, 151936, 4096], 555992604, { bias: true }, false],
    ["Qwen2.5 3B", [2048, 11008, 36, 16, 2, 151936, 4096], 3472375836, { bias: true }, true],
    ["Llama 3.2 3B", [3072, 8192, 28, 24, 8, 128256, 4096], 3614847004, {}, true]];
  for (const [name, header, size, form, half] of cases) {
    const options = { dtype: "int8", ...form };
    assert.equal(keysInHalf(header, size, { ...options, halfKV: true }), half, `${name}: float16 keys and values ${!half}`);
    assert.equal(keysInHalf(header, size, options), false, `${name}: float16 keys and values not on a shared memory`);
    const [f16, f32] = [true, false].map((halfKV) => footprint(header, size, { ...options, halfKV }));
    assert.equal(f16 < f32, half, `${name}: footprint() counts ${half ? "float32" : "float16"} keys and values`);
  }
}
// T160: what forward.js allocates against footprint() (as for the models below), on made-up int8 models of 4096
// positions whose keys and values outweigh the rest: grouped-query (16 heads, 8 of keys and values: float32 on a
// shared memory too) and not (8 and 8: float16 there). None of the models below has grouped-query attention. A
// footprint() that counts the other type is 25 MiB off, past the line's 6 MiB and 5%.
py.globals.set("WITHOUT", py.toPy(without));
py.runPython(`
import struct, numpy as np, llama2_convert

def made_up(dim, heads, kv_heads, hidden=512, layers=4, vocab=320, seq_len=4096):
    """an int8 checkpoint as quantize.py writes one, and its bytes after the checkpoint at the end of the context"""
    rng = np.random.default_rng(0)
    header = (dim, hidden, layers, heads, kv_heads, vocab, seq_len)
    out = [struct.pack("<7i", *header)]
    for shape, is_matrix in llama2_convert.layout(*header):
        if is_matrix is None:
            continue
        values = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        if is_matrix:
            q, scales = llama2_convert.quantize(values.reshape(-1, shape[-1]))
            out += [q.tobytes(), scales.tobytes()]
        else:
            out.append((1.0 + values * 0.1).tobytes())
    pieces = [f"<{i}>".encode() for i in range(vocab)]
    tokenizer = struct.pack("<i", max(map(len, pieces))) + b"".join(struct.pack("<fi", 0.0, len(p)) + p for p in pieces)
    data = b"".join(out)
    llama = kernel_llama(data, tokenizer, dtype="int8", disable=WITHOUT)
    capacity = llama2_numpy.KV_START
    while capacity < seq_len:
        llama.forward(llama.bos, capacity, need_logits=False)
        capacity *= 2
    llama.forward(llama.bos, seq_len - 1, need_logits=False)
    used = int(llama._external[0].memoryBytes())
    llama.release(); del llama; gc.collect()
    return used, len(data), list(header)
`);
for (const [name, dim, heads, kvHeads] of [["made-up, grouped-query", 512, 16, 8], ["made-up, a key for every head", 256, 8, 8]]) {
  const [used, size, header] = py.runPython(`made_up(${dim}, ${heads}, ${kvHeads})`).toJs();
  const after = used - (shared ? 8192 : 64) - size;
  const int8 = !without.includes("int8");
  const options = { dtype: "int8", int8, relaxed: Boolean(kernels.relaxed) && !without.includes("relaxed"),
    halfKV: shared && int8 && !without.includes("kv16") };
  const bound = footprint(header, size, options);
  const close = after <= bound && bound - after <= 0.05 * bound + 6 * 2 ** 20;
  console.log(`${name}: ${(after / 2 ** 20).toFixed(1)} MiB after the checkpoint at the end of the context, footprint ` +
    `${(bound / 2 ** 20).toFixed(1)} MiB, keys and values in ${keysInHalf(header, size, options) ? "float16" : "float32"}` +
    `${close ? "" : " — FAILED"}`);
  failed ||= !close;
}
for (const id of ids.length ? ids : ["stories260K", "stories15M", "tiny-lm", "llm-jp-3-150m"]) {
  const entry = modelOf(id);
  py.FS.writeFile("model.bin", fs.readFileSync(file(entry.checkpoint)));
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(file(entry.tokenizer)));
  py.globals.set("OPTIONS", py.toPy({ ...entry.options, disable: without }));
  // T115: what the forward pass allocates after the checkpoint, at most, against footprint(), which decides a 32-bit
  // or a 64-bit memory and whether a kept one has room: forward() at every position where the KV cache doubles, then
  // at the last one, so that it has grown step by step as a generation grows it, to the whole context
  // (first, before the NumPy engine widens the weights in Pyodide's memory, which never shrinks)
  const [used, header] = py.runPython(`
import struct
data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()
llama = kernel_llama(data, vocabulary, **OPTIONS)
if getattr(llama, "_external", None):
    capacity = llama2_numpy.KV_START
    while capacity < llama.seq_len:
        llama.forward(llama.bos, capacity, need_logits=False)
        capacity *= 2
    llama.forward(llama.bos, llama.seq_len - 1, need_logits=False)
used = int(llama._external[0].memoryBytes()) if getattr(llama, "_external", None) else 0
llama.release(); del llama; gc.collect()
(used, list(struct.unpack_from("<7i", data, 0)))
`).toJs();
  if (used) {
    const size = fs.statSync(file(entry.checkpoint)).size, quantized = ["int8", "int6"].includes(entry.options.dtype);
    const int8 = !without.includes("int8");
    const after = used - (shared ? 8192 : 64) - size;
    const bound = footprint(header, size, { ...entry.options, int8,
      relaxed: Boolean(kernels.relaxed) && !without.includes("relaxed"), halfKV: shared && quantized && int8 && !without.includes("kv16") });
    // above what was used, and by little: a few percent, the megabyte for alignment, and the outlier columns it
    // counts for every quantized model (4 MiB for a vocabulary of 128256; few models have them)
    const close = after <= bound && bound - after <= 0.05 * bound + 6 * 2 ** 20;
    console.log(`${entry.name}: ${(after / 2 ** 20).toFixed(1)} MiB after the checkpoint at the end of the context, ` +
      `footprint ${(bound / 2 ** 20).toFixed(1)} MiB${close ? "" : " — FAILED"}`);
    failed ||= !close;
  }
  const verdict = py.runPython(`
page = kernel_llama(data, vocabulary, **OPTIONS)
numpy = Llama(data, vocabulary, **{k: v for k, v in OPTIONS.items() if k != "disable"})
int8 = any(kind in page.backend for kind in ("int8", "int6", "ternary"))  # these quantize the activations (T98, T230)
sequence, agree, largest, nll = [page.bos], 0, 0.0, [0.0, 0.0]
for pos in range(${positions}):
    a, b = page.forward(sequence[pos], pos).astype(np.float64), numpy.forward(sequence[pos], pos).astype(np.float64)
    largest = max(largest, float(np.abs(a - b).max()))
    agree += int(a.argmax() == b.argmax())
    following = int(b.argmax())  # NumPy's greedy text, which both read
    for i, logits in enumerate((a, b)):
        shifted = logits - logits.max()
        nll[i] -= shifted[following] - math.log(np.exp(shifted).sum())
    sequence.append(following)
agreement, change = agree / ${positions}, math.exp((nll[0] - nll[1]) / ${positions}) - 1
ok = (agreement >= 0.85 and abs(change) <= 0.05) if int8 else (agree == ${positions} and largest <= 1e-3)
kv_start, llama2_numpy.KV_START = llama2_numpy.KV_START, 8
one, many = kernel_llama(data, vocabulary, **OPTIONS), kernel_llama(data, vocabulary, **OPTIONS)
llama2_numpy.KV_START = kv_start
fed = sequence[:${positions}]
for pos, token in enumerate(fed[:-1]):
    one.forward(token, pos, need_logits=False)
blocks = many.forward_many is not None
if blocks:
    for at in range(0, len(fed) - 1, 37):  # blocks that do not line up with forward.js's own BATCH
        many.forward_many(fed[at:min(at + 37, len(fed) - 1)], at)
else:
    for pos, token in enumerate(fed[:-1]):
        many.forward(token, pos, need_logits=False)
same = np.array_equal(one.forward(fed[-1], len(fed) - 1), many.forward(fed[-1], len(fed) - 1))
ok = ok and same
one.release(); many.release(); del one, many
def run(llama, positions):
    token, began = llama.bos, time.perf_counter()
    for pos in range(positions):
        token = int(np.argmax(llama.forward(token, pos)))
    return time.perf_counter() - began
(ok, f"{page.backend}: " + (f"most likely token the same at {agreement * 100:.1f}%, perplexity {change * 100:+.2f}% against NumPy"
     if int8 else f"most likely token the same at {agreement * 100:.1f}%, largest logit difference {largest:.2e} against NumPy")
     + (f"; the prompt in blocks {'the same to the bit' if same else 'DIFFERENT'}" if blocks else "; no blocks (NumPy)"))
`).toJs();
  const [ok, line] = verdict;
  const times = { numpy: [], page: [] };
  for (let r = 0; r < rounds; r++) for (const which of ["numpy", "page"]) times[which].push(positions / py.runPython(`run(${which}, ${positions})`));
  const median = (xs) => [...xs].sort((p, q) => p - q)[xs.length >> 1];
  console.log(`${entry.name}: ${line}${ok ? "" : " — FAILED"}; NumPy ${median(times.numpy).toFixed(1)} against forward.js ${median(times.page).toFixed(1)} tok/s`);
  failed ||= !ok;
  py.runPython("page.release(); del page, numpy; gc.collect()");

}
process.exit(failed ? 1 : 0);
