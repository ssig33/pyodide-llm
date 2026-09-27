// T163: the ceilings of this CPU as the browser gives them to WebAssembly, for the CPU section of /benchmark/: loops of
// one kind of instruction only, the forms T158 measured on the development machine (docs/notes/t158-cpu-audit-2026-09-27.md).
// Not kernels: nothing of the model page calls them. The relaxed SIMD loop is in ceilings_relaxed.ts (a browser
// without relaxed SIMD, Safari, cannot compile a module that holds it).
//
// Each loop returns a checksum of everything it computed, which the page computes too and compares: a loop the
// compiler cut short (its work unused) returns another number, or is too fast to time (T163's watch).

// A read-only loop: v128.load and i32x4.add into 4 accumulators, over bytes at p (a multiple of 64), passes times.
// What a token of the generation does with its weights, without the multiply-adds: the memory's ceiling. The checksum
// is the xor of the lanes of the sum, the lane j of which is passes times the sum of the int32s at p + 16k + 4j.
export function read(p: usize, bytes: usize, passes: i32): i32 {
  let a0 = i32x4.splat(0), a1 = a0, a2 = a0, a3 = a0;
  const end = p + bytes;
  for (let r = 0; r < passes; r++) {
    for (let q = p; q < end; q += 64) {
      a0 = i32x4.add(a0, v128.load(q));
      a1 = i32x4.add(a1, v128.load(q, 16));
      a2 = i32x4.add(a2, v128.load(q, 32));
      a3 = i32x4.add(a3, v128.load(q, 48));
    }
  }
  const s = i32x4.add(i32x4.add(a0, a1), i32x4.add(a2, a3));
  return i32x4.extract_lane(s, 0) ^ i32x4.extract_lane(s, 1) ^ i32x4.extract_lane(s, 2) ^ i32x4.extract_lane(s, 3);
}

// f32x4 mul + add on registers only, 8 accumulators (the form of matmul_f32 and of attention's scores; T158 found
// relaxed_madd no faster under V8, so the page's kernels do not use it). 32 multiply-adds a pass. w and x move a
// little each pass so that nothing is hoisted out of the loop. The checksum is the sum of the lanes of the sum of the
// accumulators: the page repeats the passes in float32 (Math.fround) to the same bits.
export function fma(passes: i32): f32 {
  let w = f32x4.splat(1.0001), x = f32x4.splat(0.9999);
  const d = f32x4.splat(1e-7);
  // accumulators that start apart, so that the checksum sees each (one dropped would change it)
  let a0 = f32x4.splat(0), a1 = f32x4.splat(1), a2 = f32x4.splat(2), a3 = f32x4.splat(3);
  let a4 = f32x4.splat(4), a5 = f32x4.splat(5), a6 = f32x4.splat(6), a7 = f32x4.splat(7);
  for (let i = 0; i < passes; i++) {
    a0 = f32x4.add(a0, f32x4.mul(w, x)); a1 = f32x4.add(a1, f32x4.mul(w, x));
    a2 = f32x4.add(a2, f32x4.mul(w, x)); a3 = f32x4.add(a3, f32x4.mul(w, x));
    a4 = f32x4.add(a4, f32x4.mul(w, x)); a5 = f32x4.add(a5, f32x4.mul(w, x));
    a6 = f32x4.add(a6, f32x4.mul(w, x)); a7 = f32x4.add(a7, f32x4.mul(w, x));
    w = f32x4.add(w, d);
    x = f32x4.sub(x, d);
  }
  const s = f32x4.add(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)), f32x4.add(f32x4.add(a4, a5), f32x4.add(a6, a7)));
  return f32x4.extract_lane(s, 0) + f32x4.extract_lane(s, 1) + f32x4.extract_lane(s, 2) + f32x4.extract_lane(s, 3);
}
