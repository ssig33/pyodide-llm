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
