// WASM SIMD128 kernels for a llama2.c-style transformer (AssemblyScript).
// Loaded into Pyodide with ctypes as an Emscripten side module, they work in place on NumPy-owned memory.
// No static data and no std math on purpose: nothing relocates a data segment in this hand-made side module.

import { sixFirst, sixSecond, sixTops } from "./six";

const GS: i32 = 32; // int8 quantization group size, as in quantize.py

// @ts-ignore: decorator
@inline function hsum(a: v128): f32 {
  return f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 2) + f32x4.extract_lane(a, 3);
}

// @ts-ignore: decorator
@inline function fexp(x: f32): f32 { // table-free Cephes expf
  if (x < -87.0) return 0;
  if (x > 88.0) x = 88.0;
  const k = nearest<f32>(x * <f32>1.44269504088896341);
  const r = x - k * <f32>0.693359375 - k * <f32>-2.12194440e-4;
  let p: f32 = 1.9875691500e-4;
  p = p * r + <f32>1.3981999507e-3;
  p = p * r + <f32>8.3334519073e-3;
  p = p * r + <f32>4.1665795894e-2;
  p = p * r + <f32>1.6666665459e-1;
  p = p * r + <f32>5.0000001201e-1;
  const e = p * (r * r) + r + <f32>1.0;
  return e * reinterpret<f32>(<u32>(<i32>k + 127) << 23);
}

// fexp() on four numbers at a time, bit for bit, except that below -87 it gives exp(-87) (1.6e-38) where fexp gives 0
// @ts-ignore: decorator
@inline function vexp(x: v128): v128 {
  x = f32x4.min(f32x4.max(x, f32x4.splat(-87.0)), f32x4.splat(88.0));
  const k = f32x4.nearest(f32x4.mul(x, f32x4.splat(1.44269504088896341)));
  const r = f32x4.sub(f32x4.sub(x, f32x4.mul(k, f32x4.splat(0.693359375))), f32x4.mul(k, f32x4.splat(-2.12194440e-4)));
  let p = f32x4.splat(1.9875691500e-4);
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(1.3981999507e-3));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(8.3334519073e-3));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(4.1665795894e-2));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(1.6666665459e-1));
  p = f32x4.add(f32x4.mul(p, r), f32x4.splat(5.0000001201e-1));
  const e = f32x4.add(f32x4.add(f32x4.mul(p, f32x4.mul(r, r)), r), f32x4.splat(1.0));
  const scale = i32x4.shl(i32x4.add(i32x4.trunc_sat_f32x4_s(k), i32x4.splat(127)), 23);
  return f32x4.mul(e, scale);
}

// W (d,n) @ x (n,) -> xout, rows [r0, r1)
export function matmul_f32(xout: usize, x: usize, w: usize, n: i32, r0: i32, r1: i32): void {
  const n16 = n & ~15;
  for (let i = r0; i < r1; i++) {
    const row = w + ((<usize>i * <usize>n) << 2);
    let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
    let j = 0;
    for (; j < n16; j += 16) {
      const o = <usize>j << 2;
      a0 = f32x4.add(a0, f32x4.mul(v128.load(row + o), v128.load(x + o)));
      a1 = f32x4.add(a1, f32x4.mul(v128.load(row + o + 16), v128.load(x + o + 16)));
      a2 = f32x4.add(a2, f32x4.mul(v128.load(row + o + 32), v128.load(x + o + 32)));
      a3 = f32x4.add(a3, f32x4.mul(v128.load(row + o + 48), v128.load(x + o + 48)));
    }
    let val: f32 = hsum(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)));
    for (; j < n; j++) {
      const o = <usize>j << 2;
      val += load<f32>(row + o) * load<f32>(x + o);
    }
    store<f32>(xout + (<usize>i << 2), val);
  }
}

// the largest |value| of a group of 32, lane by lane as the scalar comparisons were (see quantize_x)
// @ts-ignore: decorator
@inline function groupMax(v0: v128, v1: v128, v2: v128, v3: v128, v4: v128, v5: v128, v6: v128, v7: v128): f32 {
  const m = f32x4.max(f32x4.max(f32x4.max(f32x4.abs(v0), f32x4.abs(v1)), f32x4.max(f32x4.abs(v2), f32x4.abs(v3))),
                      f32x4.max(f32x4.max(f32x4.abs(v4), f32x4.abs(v5)), f32x4.max(f32x4.abs(v6), f32x4.abs(v7))));
  let amax = f32x4.extract_lane(m, 0);
  const m1 = f32x4.extract_lane(m, 1), m2 = f32x4.extract_lane(m, 2), m3 = f32x4.extract_lane(m, 3);
  if (m1 > amax) amax = m1;
  if (m2 > amax) amax = m2;
  if (m3 > amax) amax = m3;
  return amax;
}

// T123: n bfloat16 values (raw) to float32 (out): a bfloat16 is the upper half of a float32, so each is its 16 bits
// shifted up, exactly as the converter's NumPy bfloat16() does. 8 at a time, the last few one by one.
export function widen_bf16(out: usize, raw: usize, n: i32): void {
  let i = 0;
  for (; i + 8 <= n; i += 8) {
    const h = v128.load(raw + (<usize>i << 1));
    v128.store(out + (<usize>i << 2), i32x4.shl(i32x4.extend_low_i16x8_u(h), 16));
    v128.store(out + (<usize>i << 2), i32x4.shl(i32x4.extend_high_i16x8_u(h), 16), 16);
  }
  for (; i < n; i++) store<u32>(out + (<usize>i << 2), <u32>load<u16>(raw + (<usize>i << 1)) << 16);
}

// a float16 (its 16 bits) as float32, exactly: every float16 is a float32. The bits as NumPy's software conversion
// makes them: infinities and NaNs keep their payload (shifted up), subnormals become normal float32.
// @ts-ignore: decorator
@inline function halfToFloat(h: u32): f32 {
  const sign = (h & 0x8000) << 16, exp = h & 0x7c00, sig = h & 0x03ff;
  if (exp == 0x7c00) return reinterpret<f32>(sign | 0x7f800000 | (sig << 13));
  if (exp == 0) {
    const tiny = <f32>sig * <f32>5.9604644775390625e-8; // sig * 2^-24: exact, a power of two times an integer < 1024
    return sign ? -tiny : tiny;
  }
  return reinterpret<f32>(sign | (((h & 0x7fff) << 13) + (112 << 23))); // the exponent bias 15 -> 127
}

// T136: n blocks of GGUF's Q8_0 (raw, 34 bytes each: a float16 scale and 32 int8) to 32 n float32 (out), each int8
// times its block's scale in float32: exactly the converter's NumPy q8_0() (one rounding, the product).
export function widen_q8_0(out: usize, raw: usize, n: i32): void {
  for (let b = 0; b < n; b++) {
    const block = raw + <usize>b * 34, to = out + (<usize>b << 7);
    const scale = f32x4.splat(halfToFloat(<u32>load<u16>(block)));
    const lo = v128.load(block, 2), hi = v128.load(block, 18);
    const l0 = i16x8.extend_low_i8x16_s(lo), l1 = i16x8.extend_high_i8x16_s(lo);
    const h0 = i16x8.extend_low_i8x16_s(hi), h1 = i16x8.extend_high_i8x16_s(hi);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(l0)), scale));
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(l0)), scale), 16);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(l1)), scale), 32);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(l1)), scale), 48);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(h0)), scale), 64);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(h0)), scale), 80);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_low_i16x8_s(h1)), scale), 96);
    v128.store(to, f32x4.mul(f32x4.convert_i32x4_s(i32x4.extend_high_i16x8_s(h1)), scale), 112);
  }
}

// bias = 0: signed int8 in [-127,127];  bias = 64: 7-bit unsigned, real value = (q - 64) * scale (for kernel_relaxed.ts)
export function quantize_x(xq: usize, xs: usize, x: usize, n: i32, bias: i32): void {
  // SIMD, 32 values (one group) at a time. Every step is the scalar one lane by lane (abs, max, the division, the
  // product, round half to even), so the int8 and the scales are the same to the bit as before, and as NumPy's
  // llama2_convert.quantize(), which the converter now hands to this (T89). Only a NaN in the input tells them
  // apart (SIMD max keeps it, the scalar compare skipped it), and a NaN in a checkpoint is broken on every path.
  const qmax: f32 = bias == 0 ? 127.0 : 63.0;
  const offset = i32x4.splat(bias);
  for (let g = 0; g < n; g += GS) {
    const p = x + (<usize>g << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    const v4 = v128.load(p, 64), v5 = v128.load(p, 80), v6 = v128.load(p, 96), v7 = v128.load(p, 112);
    const amax = groupMax(v0, v1, v2, v3, v4, v5, v6, v7);
    const scale: f32 = amax / qmax;
    store<f32>(xs + (<usize>(g / GS) << 2), scale);
    const inv = f32x4.splat(scale > 0 ? <f32>1.0 / scale : 0);
    const q0 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v0, inv))), offset);
    const q1 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v1, inv))), offset);
    const q2 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v2, inv))), offset);
    const q3 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v3, inv))), offset);
    const q4 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v4, inv))), offset);
    const q5 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v5, inv))), offset);
    const q6 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v6, inv))), offset);
    const q7 = i32x4.add(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v7, inv))), offset);
    const out = xq + <usize>g;
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q0, q1), i16x8.narrow_i32x4_s(q2, q3)));
    v128.store(out, i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(q4, q5), i16x8.narrow_i32x4_s(q6, q7)), 16);
  }
}

// T98: the converter's six bits (llama2_numpy.quantize6 and pack6 in one pass, the same bytes): per group of 32,
// scale = the largest |value| / 31, v = round(value / scale) in -32..31, stored as the int8 4 v packed into 24 bytes
// (six.ts reads them back) and the scale as scale / 4 (xs).
// round(value / scale), clipped to six bits (-32..31), as NumPy's rint and clip
// @ts-ignore: decorator
@inline function six(v: v128, inv: v128): v128 {
  return i32x4.max_s(i32x4.min_s(i32x4.trunc_sat_f32x4_s(f32x4.nearest(f32x4.mul(v, inv))), i32x4.splat(31)), i32x4.splat(-32));
}
export function quantize6_x(out: usize, xs: usize, x: usize, n: i32): void {
  const low6 = i8x16.splat(63);
  for (let g = 0; g < n; g += GS) {
    const p = x + (<usize>g << 2);
    const v0 = v128.load(p), v1 = v128.load(p, 16), v2 = v128.load(p, 32), v3 = v128.load(p, 48);
    const v4 = v128.load(p, 64), v5 = v128.load(p, 80), v6 = v128.load(p, 96), v7 = v128.load(p, 112);
    const scale: f32 = groupMax(v0, v1, v2, v3, v4, v5, v6, v7) / <f32>31.0;
    store<f32>(xs + (<usize>(g / GS) << 2), scale * <f32>0.25);
    const inv = f32x4.splat(scale > 0 ? <f32>1.0 / scale : 0);
    // the six bits of values 0..15 and 16..31
    const first = v128.and(i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(six(v0, inv), six(v1, inv)),
                                                i16x8.narrow_i32x4_s(six(v2, inv), six(v3, inv))), low6);
    const second = v128.and(i8x16.narrow_i16x8_s(i16x8.narrow_i32x4_s(six(v4, inv), six(v5, inv)),
                                                 i16x8.narrow_i32x4_s(six(v6, inv), six(v7, inv))), low6);
    const at = out + <usize>(g / GS) * 24;
    v128.store(at, v128.or(v128.and(first, i8x16.splat(15)), i8x16.shl(second, 4)));
    // top two bits: lane k of a holds those of values k and k + 16; byte k takes lanes k and k + 8 of a
    const a = v128.or(i8x16.shr_u(first, 4), i8x16.shl(i8x16.shr_u(second, 4), 4));
    const b = i8x16.shuffle(a, a, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15);
    v128.store64_lane(at + 16, v128.or(a, i8x16.shl(b, 2)), 0);
  }
}

// T98: the corrections of matmul_q6r for int6 weights (w, groups of 24 bytes; ws their scales): out[g] = the scale
// times the sum of the group's int8 values, as forward.js computes them for int8 (an integer sum is exact in float32)
export function six_sums(out: usize, w: usize, ws: usize, groups: i32): void {
  for (let g = 0; g < groups; g++) {
    const p = w + <usize>g * 24, low = v128.load(p), t = sixTops(p);
    const pairs = i16x8.add(i16x8.extadd_pairwise_i8x16_s(sixFirst(low, t)), i16x8.extadd_pairwise_i8x16_s(sixSecond(low, t)));
    const quads = i32x4.extadd_pairwise_i16x8_s(pairs);
    const sum = i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
    store<f32>(out + (<usize>g << 2), load<f32>(ws + (<usize>g << 2)) * <f32>sum);
  }
}

// T123: the same for int8 weights: the sums of groups of 32 int8 values (w), times the group's scale (ws), the
// corrections of matmul_q8r. forward.js made them one value at a time in JavaScript (7B: 266 s of its construct)
export function int8_sums(out: usize, w: usize, ws: usize, groups: i32): void {
  for (let g = 0; g < groups; g++) {
    const p = w + (<usize>g << 5);
    const pairs = i16x8.add(i16x8.extadd_pairwise_i8x16_s(v128.load(p)), i16x8.extadd_pairwise_i8x16_s(v128.load(p + 16)));
    const quads = i32x4.extadd_pairwise_i16x8_s(pairs);
    const sum = i32x4.extract_lane(quads, 0) + i32x4.extract_lane(quads, 1) + i32x4.extract_lane(quads, 2) + i32x4.extract_lane(quads, 3);
    store<f32>(out + (<usize>g << 2), load<f32>(ws + (<usize>g << 2)) * <f32>sum);
  }
}

// T165: the 32 products of one group (a0 b0, a1 b1: 16 int8 each), summed into four int32 lanes. Two products are
// added in int16 before the widening: 9 instructions, not 11 (V8 on arm64 folds the add into smlal: 6, not 9).
// int8 times int8 is 16384 at most (-128 times -128), and two such sums are within int16 unless both factors are
// -128 twice over. The weights and the activations reach -128 only in a group whose scale is below 2.9e-39
// (quantize_x where 1/scale overflows, T136), so the scale of that product (the two scales multiplied) is 0 in
// float32 and the lane is multiplied by 0 either way (at most the sign of a zero changes). The lanes hold the same
// products as before (lane k: products 2k, 2k + 1, 2k + 8, 2k + 9 of each half), so the sums are the same integers.
// @ts-ignore: decorator
@inline function dot32(a0: v128, b0: v128, a1: v128, b1: v128): v128 {
  return i32x4.add(
    i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extmul_low_i8x16_s(a0, b0), i16x8.extmul_high_i8x16_s(a0, b0))),
    i32x4.extadd_pairwise_i16x8_s(i16x8.add(i16x8.extmul_low_i8x16_s(a1, b1), i16x8.extmul_high_i8x16_s(a1, b1))));
}

// int8 weights (wq) with one float32 scale per group (ws), int8 activations from quantize_x(bias = 0)
// T165: four groups a turn share one load of their scales (ws times xs, four at a time: the same products as one by
// one) and one step of the loop, and add into facc in the same order, so the result is the same to the bit
export function matmul_q8(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>n;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    let g = 0;
    for (; g + 4 <= ng; g += 4) {
      const sv = f32x4.mul(v128.load(srow + (<usize>g << 2)), v128.load(xs + (<usize>g << 2)));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g)), f32x4.splat(f32x4.extract_lane(sv, 0))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 1)), f32x4.splat(f32x4.extract_lane(sv, 1))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 2)), f32x4.splat(f32x4.extract_lane(sv, 2))));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g + 3)), f32x4.splat(f32x4.extract_lane(sv, 3))));
    }
    for (; g < ng; g++) {
      const s = load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(groupQ8(row, xq, g)), f32x4.splat(s)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// the four int32 lanes of group g of an int8 row (row) against the int8 activations (xq)
// @ts-ignore: decorator
@inline function groupQ8(row: usize, xq: usize, g: i32): v128 {
  const o = <usize>(g * GS);
  return dot32(v128.load(row + o), v128.load(xq + o), v128.load(row + o + 16), v128.load(xq + o + 16));
}

// T98: int6 weights (24 bytes a group, six.ts) with one float32 scale per group: matmul_q8 on the widened groups
export function matmul_q6(xout: usize, xq: usize, xs: usize, wq: usize, ws: usize, n: i32, r0: i32, r1: i32): void {
  const ng = n / GS;
  for (let i = r0; i < r1; i++) {
    const row = wq + <usize>i * <usize>ng * 24;
    const srow = ws + ((<usize>i * <usize>ng) << 2);
    let facc = f32x4.splat(0);
    for (let g = 0; g < ng; g++) {
      const p = row + <usize>g * 24, o = <usize>(g * GS);
      const low = v128.load(p), t = sixTops(p);
      const a0 = sixFirst(low, t), b0 = v128.load(xq + o);
      const a1 = sixSecond(low, t), b1 = v128.load(xq + o + 16);
      const acc = dot32(a0, b0, a1, b1);
      const s = load<f32>(srow + (<usize>g << 2)) * load<f32>(xs + (<usize>g << 2));
      facc = f32x4.add(facc, f32x4.mul(f32x4.convert_i32x4_s(acc), f32x4.splat(s)));
    }
    store<f32>(xout + (<usize>i << 2), hsum(facc));
  }
}

// eps: the model's (config.json's rms_norm_eps): Qwen3's 1e-6 against 1e-5 moved its perplexity by 0.12% (T124)
export function rmsnorm(out: usize, x: usize, w: usize, n: i32, eps: f32): void {
  let acc = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { const v = v128.load(x + (<usize>j << 2)); acc = f32x4.add(acc, f32x4.mul(v, v)); }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); ss += v * v; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + eps);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * load<f32>(x + o)));
  }
}

export function rope(v: usize, fcr: usize, fci: usize, nh: i32, hs: i32, rot: i32): void {
  // rot: how many values of each head to turn (GPT-NeoX turns only the first ones, everyone else all of them)
  const turned = rot > 0 && rot < hs ? rot : hs;
  for (let h = 0; h < nh; h++) {
    const base = v + (<usize>(h * hs) << 2);
    for (let i = 0; i < turned; i += 2) {
      const c = load<f32>(fcr + (<usize>(i >> 1) << 2)), s = load<f32>(fci + (<usize>(i >> 1) << 2));
      const p0 = base + (<usize>i << 2);
      const v0 = load<f32>(p0), v1 = load<f32>(p0 + 4);
      store<f32>(p0, v0 * c - v1 * s);
      store<f32>(p0 + 4, v0 * s + v1 * c);
    }
  }
}

// kc / vc: this layer's cache laid out [seq][nkv * hs] (NOT the NumPy forward's [kv heads][seq][hs]);
// att: scratch of at least nh * (pos + 1) floats. Grouped-query attention: nh / nkv query heads share one kv head.
// The cache is walked position by position, all heads of a row at once: a head at a time meant nh strided passes
// over a cache that does not fit the caches of the CPU (12 layers of it), which cost more than the arithmetic.
// heads h0..h1 of nh (T109: the software threads take heads as they take rows of a matmul). Every head is computed
// alone, the same whatever the range, so the numbers do not depend on how the heads are shared out.
export function attention(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  attentionOf<f32>(out, q, kc, vc, att, pos, nh, nkv, hs, h0, h1);
}
// T110: the same over a cache of float16 keys and values (half the bytes to read, the longer the context the more
// that is). Each is widened to the float32 it stands for, exactly, and then everything is as above: the numbers are
// those of attention() on the widened cache.
export function attention_f16(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  attentionOf<u16>(out, q, kc, vc, att, pos, nh, nkv, hs, h0, h1);
}

// float16 to float32 without the FP16 proposal (T104 is on hold): the bits of the magnitude, moved to where float32
// keeps them, are the number times 2^-112, which one multiplication by 2^112 puts right (exactly, subnormals and
// zero included). An infinity (what to_f16 writes past 65520) reads back as 65536, and NaN is not kept: neither is
// expected of a key or value.
// T160: in four instructions, not seven. The half is loaded sign-extended, so that the shift by 13 puts its sign in
// bit 31 as well as in bits 30..28 (the copies the extension made); one AND clears those three, and the sign rides
// through the multiplication (-a times 2^112 is -(a times 2^112) to the bit, -0 included). The same bits as the
// seven-instruction form (magnitude, multiply, then OR the sign back in) for every one of the 65536 halves.
const HALF_BITS: i32 = <i32>0x8fffffff;  // the sign, and the exponent and mantissa of a half shifted by 13
const HALF_SCALE: f32 = 5.192296858534828e33;  // 2^112 (0x77800000)
// @ts-ignore: decorator
@inline function halves4(p: usize): v128 {
  return f32x4.mul(v128.and(i32x4.shl(v128.load16x4_s(p), 13), i32x4.splat(HALF_BITS)), f32x4.splat(HALF_SCALE));
}
// @ts-ignore: decorator
@inline function half(p: usize): f32 {
  return reinterpret<f32>((<i32>load<i16>(p) << 13) & HALF_BITS) * HALF_SCALE;
}
// four keys or values from the cache as float32, and one
// @ts-ignore: decorator
@inline function kv4<T>(p: usize): v128 {
  return sizeof<T>() == 2 ? halves4(p) : v128.load(p);
}
// @ts-ignore: decorator
@inline function kv<T>(p: usize): f32 {
  return sizeof<T>() == 2 ? half(p) : load<f32>(p);
}

// T161: the sums of four accumulators, lane by lane: [sum a0, sum a1, sum a2, sum a3], each as (l0 + l2) + (l1 + l3),
// the order of sum4 below (a transpose by shuffles, then two adds: six instructions for four sums)
// @ts-ignore: decorator
@inline function sums4(a0: v128, a1: v128, a2: v128, a3: v128): v128 {
  const s01 = f32x4.add(v128.shuffle<f32>(a0, a1, 0, 4, 1, 5), v128.shuffle<f32>(a0, a1, 2, 6, 3, 7));  // a0 l0+l2, a1 l0+l2, a0 l1+l3, a1 l1+l3
  const s23 = f32x4.add(v128.shuffle<f32>(a2, a3, 0, 4, 1, 5), v128.shuffle<f32>(a2, a3, 2, 6, 3, 7));
  return f32x4.add(v128.shuffle<f32>(s01, s23, 0, 1, 4, 5), v128.shuffle<f32>(s01, s23, 2, 3, 6, 7));
}
// @ts-ignore: decorator
@inline function sum4(a: v128): f32 {
  return (f32x4.extract_lane(a, 0) + f32x4.extract_lane(a, 2)) + (f32x4.extract_lane(a, 1) + f32x4.extract_lane(a, 3));
}

// @ts-ignore: decorator
@inline function attentionOf<T>(out: usize, q: usize, kc: usize, vc: usize, att: usize, pos: i32, nh: i32, nkv: i32, hs: i32, h0: i32, h1: i32): void {
  const E: usize = sizeof<T>();  // bytes per key or value
  const kvDim = nkv * hs;
  const kvMul = nh / nkv;
  const hs4 = hs & ~3;
  const count = pos + 1, count4 = count & ~3;
  const isq: f32 = <f32>1.0 / sqrt<f32>(<f32>hs);
  const stride = <usize>kvDim * E;
  // 1. the scores of every head against every position. T161: four positions of a head at a time, each with its own
  // accumulator, so that the query's four floats are loaded once for four keys and one transposing reduction turns
  // the four accumulators into the four scores, stored at once (the scores of a head are side by side in att). The
  // form of several rows against one shared vector is llama.cpp's ggml_vec_dot_f16_unroll (ggml/src/ggml-cpu/vec.h,
  // MIT; no line of it is copied). A last position left over takes the same sums in the same order, one accumulator,
  // so a score does not depend on where its position falls.
  let t = 0;
  for (; t < count4; t += 4) {
    const row = kc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const qh = q + (<usize>(h * hs) << 2);
      const k0 = row + <usize>((h / kvMul) * hs) * E;
      const k1 = k0 + stride, k2 = k1 + stride, k3 = k2 + stride;
      let a0 = f32x4.splat(0), a1 = f32x4.splat(0), a2 = f32x4.splat(0), a3 = f32x4.splat(0);
      let j = 0;
      for (; j < hs4; j += 4) {
        const x = v128.load(qh + (<usize>j << 2)), e = <usize>j * E;
        a0 = f32x4.add(a0, f32x4.mul(x, kv4<T>(k0 + e)));
        a1 = f32x4.add(a1, f32x4.mul(x, kv4<T>(k1 + e)));
        a2 = f32x4.add(a2, f32x4.mul(x, kv4<T>(k2 + e)));
        a3 = f32x4.add(a3, f32x4.mul(x, kv4<T>(k3 + e)));
      }
      let sc = sums4(a0, a1, a2, a3);
      for (; j < hs; j++) {
        const x = load<f32>(qh + (<usize>j << 2)), e = <usize>j * E;
        sc = f32x4.add(sc, f32x4(x * kv<T>(k0 + e), x * kv<T>(k1 + e), x * kv<T>(k2 + e), x * kv<T>(k3 + e)));
      }
      v128.store(att + (<usize>(h * count + t) << 2), f32x4.mul(sc, f32x4.splat(isq)));
    }
  }
  for (; t < count; t++) {
    const row = kc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const qh = q + (<usize>(h * hs) << 2);
      const kt = row + <usize>((h / kvMul) * hs) * E;
      let a = f32x4.splat(0);
      let j = 0;
      for (; j < hs4; j += 4) a = f32x4.add(a, f32x4.mul(v128.load(qh + (<usize>j << 2)), kv4<T>(kt + <usize>j * E)));
      let sc: f32 = sum4(a);
      for (; j < hs; j++) sc += load<f32>(qh + (<usize>j << 2)) * kv<T>(kt + <usize>j * E);
      store<f32>(att + (<usize>(h * count + t) << 2), sc * isq);
    }
  }
  // 2. softmax, head by head, four exponentials at a time
  for (let h = h0; h < h1; h++) {
    const scores = att + (<usize>(h * count) << 2);
    let mx: f32 = -f32.MAX_VALUE;
    for (let t = 0; t < count; t++) mx = max<f32>(mx, load<f32>(scores + (<usize>t << 2)));
    const mxs = f32x4.splat(mx);
    let sums = f32x4.splat(0);
    let t = 0;
    for (; t < count4; t += 4) {
      const e = vexp(f32x4.sub(v128.load(scores + (<usize>t << 2)), mxs));
      v128.store(scores + (<usize>t << 2), e);
      sums = f32x4.add(sums, e);
    }
    let sum: f32 = hsum(sums);
    for (; t < count; t++) {
      const e = fexp(load<f32>(scores + (<usize>t << 2)) - mx);
      store<f32>(scores + (<usize>t << 2), e);
      sum += e;
    }
    const inv: f32 = <f32>1.0 / sum;
    const invs = f32x4.splat(inv);
    for (t = 0; t < count4; t += 4) v128.store(scores + (<usize>t << 2), f32x4.mul(v128.load(scores + (<usize>t << 2)), invs));
    for (; t < count; t++) store<f32>(scores + (<usize>t << 2), load<f32>(scores + (<usize>t << 2)) * inv);
  }
  // 3. the weighted sum of the values, again row by row, four positions at a time: out is loaded and stored once
  // for the four of them
  for (let j = h0 * hs; j < h1 * hs; j++) store<f32>(out + (<usize>j << 2), 0);
  for (t = 0; t < count4; t += 4) {
    const row = vc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const weights = att + (<usize>(h * count + t) << 2);
      const w0 = f32x4.splat(load<f32>(weights)), w1 = f32x4.splat(load<f32>(weights + 4));
      const w2 = f32x4.splat(load<f32>(weights + 8)), w3 = f32x4.splat(load<f32>(weights + 12));
      const oh = out + (<usize>(h * hs) << 2);
      const v0 = row + <usize>((h / kvMul) * hs) * E;
      const v1 = v0 + stride, v2 = v1 + stride, v3 = v2 + stride;
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2, e = <usize>j * E;
        const pair0 = f32x4.add(f32x4.mul(w0, kv4<T>(v0 + e)), f32x4.mul(w1, kv4<T>(v1 + e)));
        const pair1 = f32x4.add(f32x4.mul(w2, kv4<T>(v2 + e)), f32x4.mul(w3, kv4<T>(v3 + e)));
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.add(pair0, pair1)));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2, e = <usize>j * E;
        store<f32>(oh + o, load<f32>(oh + o) + load<f32>(weights) * kv<T>(v0 + e) + load<f32>(weights + 4) * kv<T>(v1 + e)
          + load<f32>(weights + 8) * kv<T>(v2 + e) + load<f32>(weights + 12) * kv<T>(v3 + e));
      }
    }
  }
  for (; t < count; t++) {
    const row = vc + <usize>(t * kvDim) * E;
    for (let h = h0; h < h1; h++) {
      const weight: f32 = load<f32>(att + (<usize>(h * count + t) << 2));
      const a = f32x4.splat(weight);
      const oh = out + (<usize>(h * hs) << 2);
      const vt = row + <usize>((h / kvMul) * hs) * E;
      let j = 0;
      for (; j < hs4; j += 4) {
        const o = <usize>j << 2;
        v128.store(oh + o, f32x4.add(v128.load(oh + o), f32x4.mul(a, kv4<T>(vt + <usize>j * E))));
      }
      for (; j < hs; j++) {
        const o = <usize>j << 2;
        store<f32>(oh + o, load<f32>(oh + o) + weight * kv<T>(vt + <usize>j * E));
      }
    }
  }
}

// T110: float32 to float16, rounded to the nearest (ties to even), as NumPy's astype(float16) does: the keys and
// values of a token, into the cache. Too large a number becomes infinity (a key never is one).
export function to_f16(out: usize, x: usize, n: i32): void {
  for (let i = 0; i < n; i++) {
    const bits = reinterpret<u32>(load<f32>(x + (<usize>i << 2)));
    const sign = (bits >> 16) & 0x8000;
    const exponent = <i32>((bits >> 23) & 0xff) - 112;  // rebased for float16 (127 - 15)
    let mantissa = bits & 0x7fffff;
    let result: u32;
    if (exponent >= 31) {
      result = sign | 0x7c00;
    } else if (exponent <= 0) {
      // a subnormal float16 (or zero): the implicit bit comes in, and the rest is shifted out with rounding
      if (exponent < -10) {
        result = sign;
      } else {
        mantissa |= 0x800000;
        const shift = <u32>(14 - exponent);
        const rest = mantissa & ((1 << shift) - 1), middle = <u32>1 << (shift - 1);
        let value = mantissa >> shift;
        if (rest > middle || (rest == middle && (value & 1))) value++;
        result = sign | value;
      }
    } else {
      const rest = mantissa & 0x1fff;
      let value = (<u32>exponent << 10) | (mantissa >> 13);
      if (rest > 0x1000 || (rest == 0x1000 && (value & 1))) value++;  // a carry moves into the exponent, as it should
      result = sign | value;
    }
    store<u16>(out + (<usize>i << 1), <u16>result);
  }
}

export function layernorm(out: usize, x: usize, w: usize, b: usize, n: i32): void {
  // GPT-2 normalizes by the mean and the variance, and adds a bias after the scale
  let sum = f32x4.splat(0);
  let j = 0;
  const n4 = n & ~3;
  for (; j < n4; j += 4) { sum = f32x4.add(sum, v128.load(x + (<usize>j << 2))); }
  let total: f32 = hsum(sum);
  for (; j < n; j++) { total += load<f32>(x + (<usize>j << 2)); }
  const mean: f32 = total / <f32>n;
  let acc = f32x4.splat(0);
  const meanv = f32x4.splat(mean);
  for (j = 0; j < n4; j += 4) {
    const d = f32x4.sub(v128.load(x + (<usize>j << 2)), meanv);
    acc = f32x4.add(acc, f32x4.mul(d, d));
  }
  let ss: f32 = hsum(acc);
  for (; j < n; j++) { const d = load<f32>(x + (<usize>j << 2)) - mean; ss += d * d; }
  const s: f32 = <f32>1.0 / sqrt<f32>(ss / <f32>n + <f32>1e-5);
  for (j = 0; j < n; j++) {
    const o = <usize>j << 2;
    store<f32>(out + o, load<f32>(w + o) * (s * (load<f32>(x + o) - mean)) + load<f32>(b + o));
  }
}

// T162: gelu and swiglu four at a time by vexp. Their results are fexp's bit for bit: where vexp and fexp differ (below
// -87), 1 + exp(x) is 1 either way.
export function gelu(out: usize, x: usize, b: usize, n: i32): void {
  // GPT-2's gelu_new, with the bias of the projection added first. 0.5 * (1 + tanh(z)) is 1 / (1 + exp(-2z)).
  const one = f32x4.splat(1.0), c = f32x4.splat(0.7978845608028654), cube = f32x4.splat(0.044715), two = f32x4.splat(-2.0);
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    const v = f32x4.add(v128.load(x + o), v128.load(b + o));
    const inner = f32x4.mul(c, f32x4.add(v, f32x4.mul(f32x4.mul(f32x4.mul(cube, v), v), v)));
    v128.store(out + o, f32x4.div(v, f32x4.add(one, vexp(f32x4.mul(two, inner)))));
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(x + o) + load<f32>(b + o);
    const inner = <f32>0.7978845608028654 * (v + <f32>0.044715 * v * v * v);
    store<f32>(out + o, v / (<f32>1.0 + fexp(<f32>-2.0 * inner)));
  }
}

export function swiglu(out: usize, h1: usize, h3: usize, n: i32): void {
  const one = f32x4.splat(1.0);
  let j = 0;
  for (; j + 4 <= n; j += 4) {
    const o = <usize>j << 2;
    const v = v128.load(h1 + o);
    v128.store(out + o, f32x4.mul(f32x4.div(v, f32x4.add(one, vexp(f32x4.neg(v)))), v128.load(h3 + o)));
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    const v = load<f32>(h1 + o);
    store<f32>(out + o, v / (<f32>1.0 + fexp(-v)) * load<f32>(h3 + o));
  }
}

// y[j] += sum over c of a[c] * rows[c][j]: the outlier channels of the classifier's input, multiplied by their own
// columns of the classifier in float32 (OUTLIER_CHANNELS in llama2_numpy.py). rows is (count, n) row-major.
export function add_columns(y: usize, rows: usize, a: usize, count: i32, n: i32): void {
  const n4 = n & ~3;
  let j = 0;
  for (; j < n4; j += 4) {
    const o = <usize>j << 2;
    let acc = v128.load(y + o);
    for (let c = 0; c < count; c++) {
      const row = rows + ((<usize>c * <usize>n) << 2);
      acc = f32x4.add(acc, f32x4.mul(f32x4.splat(load<f32>(a + (<usize>c << 2))), v128.load(row + o)));
    }
    v128.store(y + o, acc);
  }
  for (; j < n; j++) {
    const o = <usize>j << 2;
    let v = load<f32>(y + o);
    for (let c = 0; c < count; c++) v += load<f32>(a + (<usize>c << 2)) * load<f32>(rows + ((<usize>c * <usize>n) << 2) + o);
    store<f32>(y + o, v);
  }
}

export function add_inplace(x: usize, y: usize, n: i32): void {
  for (let j = 0; j < n; j++) { const o = <usize>j << 2; store<f32>(x + o, load<f32>(x + o) + load<f32>(y + o)); }
}

export function argmax(x: usize, n: i32): i32 {
  let mi = 0; let mv = load<f32>(x);
  for (let j = 1; j < n; j++) { const v = load<f32>(x + (<usize>j << 2)); if (v > mv) { mv = v; mi = j; } }
  return mi;
}

// ---------------------------------------------------------------------------------------------- sampling
// What Llama.sample() does in NumPy, without walking a vocabulary of 50000 or 100000 tokens several times.

// tokens: the recently generated ones; each is made less likely once, however often it occurs
export function penalize(logits: usize, tokens: usize, count: i32, penalty: f32): void {
  for (let i = 0; i < count; i++) {
    const token = load<i32>(tokens + (<usize>i << 2));
    let seen = false;
    for (let j = 0; j < i; j++) {
      if (load<i32>(tokens + (<usize>j << 2)) == token) { seen = true; break; }
    }
    if (seen) continue;
    const address = logits + (<usize>token << 2);
    const value = load<f32>(address);
    store<f32>(address, value > 0 ? value / penalty : value * penalty);
  }
}

// The state of sortNucleus(): globals are no static data
let nucleusMass: f64 = 0;
let nucleusLimit: f64 = 0;
let nucleusLast: i32 = -1;

// adds probs[lo..hi], already in their final order, to the nucleus: true when it is complete
// @ts-ignore: decorator
@inline function intoNucleus(probs: usize, lo: i32, hi: i32): bool {
  for (let k = lo; k <= hi; k++) {
    nucleusMass += load<f32>(probs + (<usize>k << 2));
    if (nucleusMass >= nucleusLimit) { nucleusLast = k; return true; }
  }
  return false;
}

// Sorts probs[lo..hi] in descending order, and index[] along with it, from the left and only as far as the nucleus
// reaches: usually a few dozen tokens out of thousands. Quicksort (median of three), insertion sort for short ranges.
function sortNucleus(probs: usize, index: usize, lo: i32, hi: i32): void {
  while (hi - lo > 12) {
    const mid = lo + ((hi - lo) >> 1);
    let a = load<f32>(probs + (<usize>lo << 2)), b = load<f32>(probs + (<usize>mid << 2));
    const c = load<f32>(probs + (<usize>hi << 2));
    if (a < b) { const t = a; a = b; b = t; }
    if (b < c) { b = c; if (a < b) b = a; }
    const pivot = b;
    let i = lo, j = hi;
    while (i <= j) {
      while (load<f32>(probs + (<usize>i << 2)) > pivot) i++;
      while (load<f32>(probs + (<usize>j << 2)) < pivot) j--;
      if (i <= j) {
        const pi = probs + (<usize>i << 2), pj = probs + (<usize>j << 2), xi = index + (<usize>i << 2), xj = index + (<usize>j << 2);
        const p = load<f32>(pi); store<f32>(pi, load<f32>(pj)); store<f32>(pj, p);
        const x = load<i32>(xi); store<i32>(xi, load<i32>(xj)); store<i32>(xj, x);
        i++; j--;
      }
    }
    sortNucleus(probs, index, lo, j);
    if (nucleusLast >= 0) return;
    if (intoNucleus(probs, j + 1, i - 1)) return; // equal to the pivot
    lo = i;
  }
  for (let i = lo + 1; i <= hi; i++) {
    const p = load<f32>(probs + (<usize>i << 2));
    const x = load<i32>(index + (<usize>i << 2));
    let j = i - 1;
    while (j >= lo && load<f32>(probs + (<usize>j << 2)) < p) {
      store<f32>(probs + (<usize>(j + 1) << 2), load<f32>(probs + (<usize>j << 2)));
      store<i32>(index + (<usize>(j + 1) << 2), load<i32>(index + (<usize>j << 2)));
      j--;
    }
    store<f32>(probs + (<usize>(j + 1) << 2), p);
    store<i32>(index + (<usize>(j + 1) << 2), x);
  }
  intoNucleus(probs, lo, hi);
}

// Writes value and token at probs[count] and index[count], and counts them when past is 1: a list that keeps some of
// a sequence in its order without a branch on each (T189: whether a token stays is not predictable). count must be
// at most the position read from, so that nothing not yet read is written over
// @ts-ignore: decorator
@inline function kept(probs: usize, index: usize, count: i32, value: f32, token: i32, past: i32): i32 {
  store<f32>(probs + (<usize>count << 2), value);
  store<i32>(index + (<usize>count << 2), token);
  return count + past;
}

// kept() for the four of d, the bits of past telling which of them are past the floor (T189)
// @ts-ignore: decorator
@inline function keptFour(probs: usize, index: usize, count: i32, d: v128, i: i32, past: i32): i32 {
  count = kept(probs, index, count, f32x4.extract_lane(d, 0), i, past & 1);
  count = kept(probs, index, count, f32x4.extract_lane(d, 1), i + 1, (past >> 1) & 1);
  count = kept(probs, index, count, f32x4.extract_lane(d, 2), i + 2, (past >> 2) & 1);
  return kept(probs, index, count, f32x4.extract_lane(d, 3), i + 3, past >> 3);
}

// Draws a token from softmax(logits / temperature), restricted to the nucleus when 0 < topp < 1.
// random: one number in [0, 1) from Python's generator, so that a seed reproduces. probs and index: scratch of n each.
export function sample(logits: usize, n: i32, temperature: f32, topp: f32, random: f64, probs: usize, index: usize): i32 {
  let best = load<f32>(logits);
  let i = 0;
  if (n >= 4) {
    // T189: four maxima side by side. x86 has no instruction for f32x4.max (NaN, -0): V8 writes several, and one
    // maximum waited on itself (on the x86 runners 1.3 ns a logit, three quarters of the walk; the arm64 runner 0.15)
    let b0 = v128.load(logits), b1 = b0, b2 = b0, b3 = b0;
    for (i = 4; i + 16 <= n; i += 16) {
      const at = logits + (<usize>i << 2);
      b0 = f32x4.max(b0, v128.load(at));
      b1 = f32x4.max(b1, v128.load(at, 16));
      b2 = f32x4.max(b2, v128.load(at, 32));
      b3 = f32x4.max(b3, v128.load(at, 48));
    }
    for (; i + 4 <= n; i += 4) b0 = f32x4.max(b0, v128.load(logits + (<usize>i << 2)));
    const bests = f32x4.max(f32x4.max(b0, b1), f32x4.max(b2, b3));
    best = max(max(f32x4.extract_lane(bests, 0), f32x4.extract_lane(bests, 1)), max(f32x4.extract_lane(bests, 2), f32x4.extract_lane(bests, 3)));
  }
  for (; i < n; i++) best = max(best, load<f32>(logits + (<usize>i << 2)));
  const nucleus = topp > 0 && topp < 1;
  // With a nucleus, tokens less than a ten millionth as probable as the best one cannot matter (ln 1e-7 = -16.118):
  // they are left out before exp(), which is the expensive part
  const floor: f32 = nucleus ? best - temperature * <f32>16.118095 : -f32.MAX_VALUE;
  // T189: eight at a time. Eight that all stay below the floor (most of the vocabulary) cost two comparisons and one
  // branch; of the others each is written and only those past the floor are counted, in their order, without a
  // branch each (whether a token passes is not predictable). A branch for each four was slower (on the arm64 runner
  // the walk 1.23 times main's where this is 2.1), and none at all slower still (0.45 times)
  const floors = f32x4.splat(floor), shift = f32x4.splat(best);
  let count = 0;
  for (i = 0; i + 8 <= n; i += 8) {
    const at = logits + (<usize>i << 2);
    const a = v128.load(at), b = v128.load(at, 16);
    const pa = f32x4.ge(a, floors), pb = f32x4.ge(b, floors);
    if (!v128.any_true(v128.or(pa, pb))) continue;
    count = keptFour(probs, index, count, f32x4.sub(a, shift), i, i32x4.bitmask(pa));
    count = keptFour(probs, index, count, f32x4.sub(b, shift), i + 4, i32x4.bitmask(pb));
  }
  for (; i < n; i++) {
    const v = load<f32>(logits + (<usize>i << 2));
    count = kept(probs, index, count, v - best, i, <i32>(v >= floor));
  }
  const inverse = f32x4.splat(<f32>1.0 / temperature);
  for (i = 0; i + 4 <= count; i += 4) {
    const address = probs + (<usize>i << 2);
    v128.store(address, vexp(f32x4.mul(v128.load(address), inverse)));
  }
  for (; i < count; i++) {
    const address = probs + (<usize>i << 2);
    store<f32>(address, fexp(load<f32>(address) / temperature));
  }
  let total: f64 = 0;
  let top: f32 = 0;
  for (i = 0; i < count; i++) {
    const p = load<f32>(probs + (<usize>i << 2));
    total += p;
    top = max(top, p);
  }
  let last = count - 1;
  let mass = total;
  if (nucleus) {
    // Tokens below (1 - topp) / (n - 1) cannot be part of the nucleus (llama2.c), so they need not be sorted. That
    // holds while one token at least stays: when all are below it (n * topp < 1, which the floor above makes possible,
    // T178), the others add up to less than (1 - topp), so the nucleus is the most probable token alone. It always stays
    const cutoff: f64 = min((1.0 - <f64>topp) / <f64>(count > 1 ? count - 1 : 1) * total, <f64>top);
    let likely = 0;
    for (let k = 0; k < count; k++) {
      const p = load<f32>(probs + (<usize>k << 2));
      likely = kept(probs, index, likely, p, load<i32>(index + (<usize>k << 2)), <i32>(<f64>p >= cutoff));
    }
    // the most probable tokens whose probabilities add up to topp
    nucleusMass = 0;
    nucleusLimit = <f64>topp * total;
    nucleusLast = -1;
    sortNucleus(probs, index, 0, likely - 1);
    last = nucleusLast >= 0 ? nucleusLast : likely - 1;
    mass = nucleusMass;
  }
  const target: f64 = random * mass;
  let cumulative: f64 = 0;
  for (let k = 0; k <= last; k++) {
    cumulative += load<f32>(probs + (<usize>k << 2));
    if (cumulative > target) return load<i32>(index + (<usize>k << 2));
  }
  return load<i32>(index + (<usize>last << 2));
}
