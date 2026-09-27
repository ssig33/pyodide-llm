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

// f32x4 mul + add on registers only, 8 accumulators (the form of matmul_f32 and of attention's scores): 32 multiplies
// and 32 adds a pass, and two subtracts that move x and y (so that nothing is hoisted out of the loop). Each
// accumulator has a product of its own (w0 to w3 times x or y): with one w and one x for all 8, Binaryen merged the
// equal f32x4.mul into one and the loop measured 1 multiply and 8 adds (T163's review; T158's 12.1 G MAC/s was that
// loop). w, x and the step come from the caller: V8 made every constant splat again inside the loop (movz, movk, dup
// before each multiply). 15 registers, so that x86's 16 hold them all. The accumulators start apart (0 to 7) so that the checksum
// (the sum of the lanes of their sum) sees each: the page repeats the passes in float32 (Math.fround) to the same bits.
export function fma(passes: i32, w: f32, x0: f32, step: f32): f32 {
  const w0 = f32x4.splat(w), w1 = f32x4.splat(w + 0.0009765625), w2 = f32x4.splat(w + 0.001953125), w3 = f32x4.splat(w + 0.0029296875);
  let x = f32x4.splat(x0), y = f32x4.splat(x0 + 0.25);
  const d = f32x4.splat(step);
  let a0 = f32x4.splat(0), a1 = f32x4.splat(1), a2 = f32x4.splat(2), a3 = f32x4.splat(3);
  let a4 = f32x4.splat(4), a5 = f32x4.splat(5), a6 = f32x4.splat(6), a7 = f32x4.splat(7);
  for (let i = 0; i < passes; i++) {
    a0 = f32x4.add(a0, f32x4.mul(w0, x)); a1 = f32x4.add(a1, f32x4.mul(w1, x));
    a2 = f32x4.add(a2, f32x4.mul(w2, x)); a3 = f32x4.add(a3, f32x4.mul(w3, x));
    a4 = f32x4.add(a4, f32x4.mul(w0, y)); a5 = f32x4.add(a5, f32x4.mul(w1, y));
    a6 = f32x4.add(a6, f32x4.mul(w2, y)); a7 = f32x4.add(a7, f32x4.mul(w3, y));
    x = f32x4.sub(x, d);
    y = f32x4.sub(y, d);
  }
  const s = f32x4.add(f32x4.add(f32x4.add(a0, a1), f32x4.add(a2, a3)), f32x4.add(f32x4.add(a4, a5), f32x4.add(a6, a7)));
  return f32x4.extract_lane(s, 0) + f32x4.extract_lane(s, 1) + f32x4.extract_lane(s, 2) + f32x4.extract_lane(s, 3);
}
