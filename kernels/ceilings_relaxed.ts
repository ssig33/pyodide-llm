// T163: the relaxed SIMD ceiling of the CPU section of /benchmark/ (ceilings.ts has the others and says why these
// loops are here). Its own module: a browser without relaxed SIMD cannot compile it, and says "not in this browser".

// relaxed_dot_i8x16_i7x16_add_s with the two loads every dot has in matmul_q8r (weights, activations), from 8 KB at p
// that stay in L1, 8 accumulators: what the int8 matrix products can reach at most (a prompt's, T108, and a token's
// when its weights are in the cache). 4096 multiply-adds a pass. The second 4 KB (x) must hold bytes of 0 to 127, as
// the activations of matmul_q8r are: then every browser's relaxed_dot gives the exact sums, and the checksum (the
// xor of the lanes of the accumulators' sum) is one the page computes too.
export function dot(p: usize, passes: i32): i32 {
  let a0 = i32x4.splat(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
  const x = p + 4096;
  for (let r = 0; r < passes; r++) {
    for (let o: usize = 0; o < 4096; o += 128) {
      a0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o), v128.load(x + o), a0);
      a1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 16), v128.load(x + o + 16), a1);
      a2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 32), v128.load(x + o + 32), a2);
      a3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 48), v128.load(x + o + 48), a3);
      a4 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 64), v128.load(x + o + 64), a4);
      a5 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 80), v128.load(x + o + 80), a5);
      a6 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 96), v128.load(x + o + 96), a6);
      a7 = i32x4.relaxed_dot_i8x16_i7x16_add_s(v128.load(p + o + 112), v128.load(x + o + 112), a7);
    }
  }
  const s = i32x4.add(i32x4.add(i32x4.add(a0, a1), i32x4.add(a2, a3)), i32x4.add(i32x4.add(a4, a5), i32x4.add(a6, a7)));
  return i32x4.extract_lane(s, 0) ^ i32x4.extract_lane(s, 1) ^ i32x4.extract_lane(s, 2) ^ i32x4.extract_lane(s, 3);
}

// relaxed_dot on registers only, 8 accumulators: the instruction's own rate (what a form that keeps weights and
// activations in registers, a tile of rows and tokens, could reach; T158's dot_regs). 4096 multiply-adds is 32 dots:
// this is 8 dots, 128 multiply-adds, a pass. a moves by step each pass (so that nothing is hoisted); b stays (0 to
// 127: exact in every browser). a, b and the step come from the caller: V8 made a constant splat again before each dot
// (movi). The accumulators start apart (lanes 1000k + j) and the checksum is the sum of the lanes of their sum, which
// the page computes too.
export function dotRegisters(passes: i32, first: i32, times: i32, step: i32): i32 {
  let a = i8x16.splat(<i8>first);
  const b = i8x16.splat(<i8>times), one = i8x16.splat(<i8>step);
  let a0 = i32x4(0, 1, 2, 3), a1 = i32x4(1000, 1001, 1002, 1003), a2 = i32x4(2000, 2001, 2002, 2003), a3 = i32x4(3000, 3001, 3002, 3003);
  let a4 = i32x4(4000, 4001, 4002, 4003), a5 = i32x4(5000, 5001, 5002, 5003), a6 = i32x4(6000, 6001, 6002, 6003), a7 = i32x4(7000, 7001, 7002, 7003);
  for (let i = 0; i < passes; i++) {
    a0 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a0);
    a1 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a1);
    a2 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a2);
    a3 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a3);
    a4 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a4);
    a5 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a5);
    a6 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a6);
    a7 = i32x4.relaxed_dot_i8x16_i7x16_add_s(a, b, a7);
    a = i8x16.add(a, one);
  }
  const s = i32x4.add(i32x4.add(i32x4.add(a0, a1), i32x4.add(a2, a3)), i32x4.add(i32x4.add(a4, a5), i32x4.add(a6, a7)));
  return i32x4.extract_lane(s, 0) + i32x4.extract_lane(s, 1) + i32x4.extract_lane(s, 2) + i32x4.extract_lane(s, 3);
}
