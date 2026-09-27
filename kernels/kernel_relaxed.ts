// Optional kernel that needs the relaxed-SIMD proposal (Chrome 114+, Firefox 146+, not in shipping Safari).
// Kept in its own module: a browser without relaxed SIMD rejects the whole module at compile time, so the
// Python loader simply falls back to kernel.ts.

import { sixFirst, sixSecond, sixTops } from "./six";

const GS: i32 = 32;

// T167: the four int32 lanes of each of four groups' dot products added up into one lane per group: [sum d0, sum d1,
// sum d2, sum d3] (a transpose by shuffles and two adds, as sums4 in kernel.ts for float32; integers, so the order of
// the adds does not matter). A group's sum is at most 32 × 128 × 127 = 520192, within float32's exact integers.
// @ts-ignore: decorator
@inline function groupSums(d0: v128, d1: v128, d2: v128, d3: v128): v128 {
  const s01 = i32x4.add(v128.shuffle<i32>(d0, d1, 0, 4, 1, 5), v128.shuffle<i32>(d0, d1, 2, 6, 3, 7));  // d0 l0+l2, d1 l0+l2, d0 l1+l3, d1 l1+l3
  const s23 = i32x4.add(v128.shuffle<i32>(d2, d3, 0, 4, 1, 5), v128.shuffle<i32>(d2, d3, 2, 6, 3, 7));
  return i32x4.add(v128.shuffle<i32>(s01, s23, 0, 1, 4, 5), v128.shuffle<i32>(s01, s23, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function laneSum(d: v128): i32 {
  return i32x4.extract_lane(d, 0) + i32x4.extract_lane(d, 1) + i32x4.extract_lane(d, 2) + i32x4.extract_lane(d, 3);
}

// activations from quantize_x(bias = 64). wc = scale * sum(group weights) removes that bias again:
// dot(w, q - 64) = dot(w, q) - 64 * sum(w)
// T167: four groups at a time; their dot products are added up to one integer a group (groupSums), converted and
// multiplied by the four scales (weight scale times activation scale) at once, and added into one accumulator whose
// lane k holds the groups 4j + k: each group's sum times its scale is rounded once (before T167 each of the four
// lanes of a group was scaled and rounded on its own, and four groups cost 4 × (convert, splat, mul, add)). The bias
// correction is a dot product of its own, also by fours. A last group or three take the same steps one at a time.
export function matmul_q8r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  const ng4 = ng & ~3;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    const crow = wc + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0), corrs = f32x4.splat(0);
    let g = 0;
    for (; g < ng4; g += 4) {
      const o = <usize>(g * GS);
      const xsv = v128.load(xs + (<usize>g << 2));
      const scales = f32x4.mul(v128.load(srow + (<usize>g << 2)), xsv);
      corrs = f32x4.add(corrs, f32x4.mul(v128.load(crow + (<usize>g << 2)), xsv));
      let d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o), v128.load(xq + o), i32x4.splat(0));
      let d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 32), v128.load(xq + o + 32), i32x4.splat(0));
      let d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 64), v128.load(xq + o + 64), i32x4.splat(0));
      let d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 96), v128.load(xq + o + 96), i32x4.splat(0));
      d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 16), v128.load(xq + o + 16), d0);
      d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 48), v128.load(xq + o + 48), d1);
      d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 80), v128.load(xq + o + 80), d2);
      d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 112), v128.load(xq + o + 112), d3);
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupSums(d0, d1, d2, d3)), scales));
    }
    let sum: f32 = f32x4.extract_lane(facc, 0) + f32x4.extract_lane(facc, 1) + f32x4.extract_lane(facc, 2) + f32x4.extract_lane(facc, 3);
    let corr: f32 = f32x4.extract_lane(corrs, 0) + f32x4.extract_lane(corrs, 1) + f32x4.extract_lane(corrs, 2) + f32x4.extract_lane(corrs, 3);
    for (; g < ng; g++) {
      const o = <usize>(g * GS);
      let acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o), v128.load(xq + o), i32x4.splat(0));
      acc = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(row + o + 16), v128.load(xq + o + 16), acc);
      const xsg = load<f32>(xs + (<usize>g << 2));
      sum += <f32>laneSum(acc) * (load<f32>(srow + (<usize>g << 2)) * xsg);
      corr += load<f32>(crow + (<usize>g << 2)) * xsg;
    }
    store<f32>(xout + (<usize>i << 2), sum - <f32>64.0 * corr);
  }
}

// T159: matmul_q8r for count tokens of a prompt (T108) at once: token t's activations and their scales at xq + t * frame
// and xs + t * frame (one frame a token), its outputs at xout + t * os. Four rows by four tokens a tile, where a group's
// 32 bytes of a weight row are loaded once for the four tokens and a token's 32 bytes once for the four rows
// (matmul_q8r loads both for every row and token). The form is that of the 4 x 4 GEMM of llama.cpp's CPU backend for
// Q8_0 (ggml-cpu, MIT: int8 dot products of 4 rows by 4 tokens, each group's sums scaled once), not its lines: the
// weights stay in the checkpoint's order (no repacking), and a row's four integer sums of a group, one a token, come
// out of groupSums with the tokens in the lanes.
//
// Every (row, token) is matmul_q8r's number to the bit. A lane k of matmul_q8r adds the groups 4j + k in order and
// the four lanes are added in order, then the groups past the last four one at a time, the corrections likewise; so
// the tile takes the groups 4j + k in one pass for each k (its lanes are the tokens), adds the passes in that order,
// then the groups past the fours, each with the same float32 operations (the weight scale times the activation scale,
// times the group's exact integer sum). The rows past the last four and the tokens past the last four go through
// matmul_q8r itself.
// @ts-ignore: decorator
@inline function scalesOf(xs: usize, frame: usize, g: i32): v128 {  // the four tokens' scales of group g
  const at = xs + (<usize>g << 2);
  let v = v128.load32_splat(at);
  v = v128.load32_lane(at + frame, v, 1);
  v = v128.load32_lane(at + 2 * frame, v, 2);
  return v128.load32_lane(at + 3 * frame, v, 3);
}
// @ts-ignore: decorator
@inline function sumsOf(w: usize, x0: v128, x1: v128, x2: v128, x3: v128, x4: v128, x5: v128, x6: v128, x7: v128): v128 {
  // a row's group (32 bytes at w) with the four tokens' (x0 and x1 the first token's two halves, ...): its four
  // exact integer sums, as float32 (at most 520192, exact)
  const lo = v128.load(w), hi = v128.load(w, 16);
  const d0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(hi, x1, i32x4.relaxed_dot_i8x16_i7x16_add_s(lo, x0, i32x4.splat(0)));
  const d1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(hi, x3, i32x4.relaxed_dot_i8x16_i7x16_add_s(lo, x2, i32x4.splat(0)));
  const d2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(hi, x5, i32x4.relaxed_dot_i8x16_i7x16_add_s(lo, x4, i32x4.splat(0)));
  const d3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(hi, x7, i32x4.relaxed_dot_i8x16_i7x16_add_s(lo, x6, i32x4.splat(0)));
  return f32x4.convert_i32x4_s(groupSums(d0, d1, d2, d3));
}

// the tile of rows i..i+3 and the four tokens at xq, xs (frame apart), writing xout (os apart)
function tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, i: i32, os: usize, frame: usize): void {
  const ng = n / GS;
  const ng4 = ng & ~3;
  const w0 = wq + <usize>i * <usize>n, w1 = w0 + <usize>n, w2 = w1 + <usize>n, w3 = w2 + <usize>n;
  const sb = <usize>ng << 2;
  const s0 = ws + <usize>i * sb, s1 = s0 + sb, s2 = s1 + sb, s3 = s2 + sb;
  const c0 = wc + <usize>i * sb, c1 = c0 + sb, c2 = c1 + sb, c3 = c2 + sb;
  const x1 = xq + frame, x2 = x1 + frame, x3 = x2 + frame;
  // f: the rows' sums so far, k: their corrections (a lane a token)
  let f0 = f32x4.splat(0), f1 = f0, f2 = f0, f3 = f0, k0 = f0, k1 = f0, k2 = f0, k3 = f0;
  for (let lane = 0; lane < 4; lane++) {
    let a0 = f32x4.splat(0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (let g = lane; g < ng4; g += 4) {
      const o = <usize>(g * GS), gs = <usize>g << 2;
      const t0 = v128.load(xq + o), t1 = v128.load(xq + o, 16), t2 = v128.load(x1 + o), t3 = v128.load(x1 + o, 16);
      const t4 = v128.load(x2 + o), t5 = v128.load(x2 + o, 16), t6 = v128.load(x3 + o), t7 = v128.load(x3 + o, 16);
      const xsv = scalesOf(xs, frame, g);
      a0 = f32x4.add(a0, f32x4.mul(sumsOf(w0 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s0 + gs), xsv)));
      a1 = f32x4.add(a1, f32x4.mul(sumsOf(w1 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s1 + gs), xsv)));
      a2 = f32x4.add(a2, f32x4.mul(sumsOf(w2 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s2 + gs), xsv)));
      a3 = f32x4.add(a3, f32x4.mul(sumsOf(w3 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s3 + gs), xsv)));
      b0 = f32x4.add(b0, f32x4.mul(v128.load32_splat(c0 + gs), xsv));
      b1 = f32x4.add(b1, f32x4.mul(v128.load32_splat(c1 + gs), xsv));
      b2 = f32x4.add(b2, f32x4.mul(v128.load32_splat(c2 + gs), xsv));
      b3 = f32x4.add(b3, f32x4.mul(v128.load32_splat(c3 + gs), xsv));
    }
    if (lane == 0) {
      f0 = a0; f1 = a1; f2 = a2; f3 = a3; k0 = b0; k1 = b1; k2 = b2; k3 = b3;
    } else {
      f0 = f32x4.add(f0, a0); f1 = f32x4.add(f1, a1); f2 = f32x4.add(f2, a2); f3 = f32x4.add(f3, a3);
      k0 = f32x4.add(k0, b0); k1 = f32x4.add(k1, b1); k2 = f32x4.add(k2, b2); k3 = f32x4.add(k3, b3);
    }
  }
  for (let g = ng4; g < ng; g++) {  // the groups past the fours, one at a time
    const o = <usize>(g * GS), gs = <usize>g << 2;
    const t0 = v128.load(xq + o), t1 = v128.load(xq + o, 16), t2 = v128.load(x1 + o), t3 = v128.load(x1 + o, 16);
    const t4 = v128.load(x2 + o), t5 = v128.load(x2 + o, 16), t6 = v128.load(x3 + o), t7 = v128.load(x3 + o, 16);
    const xsv = scalesOf(xs, frame, g);
    f0 = f32x4.add(f0, f32x4.mul(sumsOf(w0 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s0 + gs), xsv)));
    f1 = f32x4.add(f1, f32x4.mul(sumsOf(w1 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s1 + gs), xsv)));
    f2 = f32x4.add(f2, f32x4.mul(sumsOf(w2 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s2 + gs), xsv)));
    f3 = f32x4.add(f3, f32x4.mul(sumsOf(w3 + o, t0, t1, t2, t3, t4, t5, t6, t7), f32x4.mul(v128.load32_splat(s3 + gs), xsv)));
    k0 = f32x4.add(k0, f32x4.mul(v128.load32_splat(c0 + gs), xsv));
    k1 = f32x4.add(k1, f32x4.mul(v128.load32_splat(c1 + gs), xsv));
    k2 = f32x4.add(k2, f32x4.mul(v128.load32_splat(c2 + gs), xsv));
    k3 = f32x4.add(k3, f32x4.mul(v128.load32_splat(c3 + gs), xsv));
  }
  const bias = f32x4.splat(64);
  const r0 = f32x4.sub(f0, f32x4.mul(bias, k0)), r1 = f32x4.sub(f1, f32x4.mul(bias, k1));
  const r2 = f32x4.sub(f2, f32x4.mul(bias, k2)), r3 = f32x4.sub(f3, f32x4.mul(bias, k3));
  // token t's four rows are lane t of r0..r3: a transpose
  const p01 = v128.shuffle<f32>(r0, r1, 0, 4, 1, 5), p23 = v128.shuffle<f32>(r2, r3, 0, 4, 1, 5);
  const q01 = v128.shuffle<f32>(r0, r1, 2, 6, 3, 7), q23 = v128.shuffle<f32>(r2, r3, 2, 6, 3, 7);
  const at = xout + (<usize>i << 2);
  v128.store(at, v128.shuffle<f32>(p01, p23, 0, 1, 4, 5));
  v128.store(at + os, v128.shuffle<f32>(p01, p23, 2, 3, 6, 7));
  v128.store(at + 2 * os, v128.shuffle<f32>(q01, q23, 0, 1, 4, 5));
  v128.store(at + 3 * os, v128.shuffle<f32>(q01, q23, 2, 3, 6, 7));
}

// rows r0..r1 for count tokens: the tiles row by row of four, so that a tile's four weight rows stay in the first
// cache for every token (T108's blocks of 16 KB are not needed here).
// T159's review: a quad of rows up to 10 KB is read once in order before its tiles. A tile reads its rows in four
// lane passes, each every fourth group: every other 64-byte line, 128 bytes apart, and a pass over a row of 2 KB is
// 16 steps long. From memory (a model's weights, read once a block) the x86 prefetchers did not keep up with passes
// that short: on the EPYC 9V74 and 9V45 (Zen 4 and 5) the tile took up to 1.6 times as long as the blocks before it,
// at 4 to 8 tokens (the numbers are in TODO.md's T159). Past rows of 2560 bytes the passes are long enough, and a read
// ahead of 12 KB and more fell out of the first cache before the tile got to it (slower than none on the 9V45). The
// sum is there only so that the reads stay (a load whose value is not used goes away): it is stored where the tile
// writes next.
export function matmul_q8r_tile(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32, count: i32, os: i32, frame: i32): void {
  const rows4 = r0 + ((r1 - r0) & ~3), count4 = count & ~3;
  const outStride = <usize>os, frameStride = <usize>frame, quad = <usize>n << 2;
  for (let i = r0; i < rows4; i += 4) {
    if (count4 > 0 && quad <= 10240) {
      const w = wq + <usize>i * <usize>n;
      let sum = i32x4.splat(0);
      for (let p: usize = 0; p < quad; p += 64) sum = i32x4.add(sum, v128.load(w + p));
      v128.store(xout + (<usize>i << 2), sum);  // rows i..i+3 of the first token, which its tile writes next
    }
    let t = 0;
    for (; t < count4; t += 4) {
      tile(xout + <usize>t * outStride, xq + <usize>t * frameStride, xs + <usize>t * frameStride, wq, ws, wc, n, i, outStride, frameStride);
    }
    for (; t < count; t++) matmul_q8r(xout + <usize>t * outStride, xq + <usize>t * frameStride, xs + <usize>t * frameStride, wq, ws, wc, n, i, i + 4);
  }
  if (rows4 < r1) {
    for (let t = 0; t < count; t++) matmul_q8r(xout + <usize>t * outStride, xq + <usize>t * frameStride, xs + <usize>t * frameStride, wq, ws, wc, n, rows4, r1);
  }
}

// T98: the same on int6 weights (six.ts: 24 bytes a group, widened straight into int8), in the same order as
// matmul_q8r: four groups at a time, one sum a group (T167), their scales and corrections as vectors. wc as
// for matmul_q8r (scale times the sum of the group's int8 values).
// @ts-ignore: decorator
@inline function dot6(p: usize, x: usize): v128 {
  const low = v128.load(p), t = sixTops(p);
  const d = i32x4.relaxed_dot_i8x16_i7x16_add_s(sixFirst(low, t), v128.load(x), i32x4.splat(0));
  return i32x4.relaxed_dot_i8x16_i7x16_add_s(sixSecond(low, t), v128.load(x, 16), d);
}

export function matmul_q6r(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, wc: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  const ng4 = ng & ~3;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>ng * 24;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    const crow = wc + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0), corrs = f32x4.splat(0);
    let g = 0;
    for (; g < ng4; g += 4) {
      const p = row + <usize>g * 24, o = xq + <usize>(g * GS);
      const xsv = v128.load(xs + (<usize>g << 2));
      const scales = f32x4.mul(v128.load(srow + (<usize>g << 2)), xsv);
      corrs = f32x4.add(corrs, f32x4.mul(v128.load(crow + (<usize>g << 2)), xsv));
      const d0 = dot6(p, o), d1 = dot6(p + 24, o + 32), d2 = dot6(p + 48, o + 64), d3 = dot6(p + 72, o + 96);
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupSums(d0, d1, d2, d3)), scales));
    }
    let sum: f32 = f32x4.extract_lane(facc, 0) + f32x4.extract_lane(facc, 1) + f32x4.extract_lane(facc, 2) + f32x4.extract_lane(facc, 3);
    let corr: f32 = f32x4.extract_lane(corrs, 0) + f32x4.extract_lane(corrs, 1) + f32x4.extract_lane(corrs, 2) + f32x4.extract_lane(corrs, 3);
    for (; g < ng; g++) {
      const xsg = load<f32>(xs + (<usize>g << 2));
      sum += <f32>laneSum(dot6(row + <usize>g * 24, xq + <usize>(g * GS))) * (load<f32>(srow + (<usize>g << 2)) * xsg);
      corr += load<f32>(crow + (<usize>g << 2)) * xsg;
    }
    store<f32>(xout + (<usize>i << 2), sum - <f32>64.0 * corr);
  }
}
