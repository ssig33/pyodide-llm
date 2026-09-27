// six.ts (T98): the 32 values of one int6 group, widened to the int8 the dot products take. kernel.ts (matmul_q6,
// six_sums) and kernel_relaxed.ts (matmul_q6r) import it. The layout is llama2_numpy.pack6's: an int6 value is an
// int8 with its two low bits zero, so a group widens straight into int8 with no offset. 24 bytes a group: the four
// low bits of the six of value j and of value j + 16 in byte j (0..15), the top two bits of values k, k + 8, k + 16,
// k + 24 in byte 16 + k (0..7) at bits 0, 2, 4, 6. Masks and shifts by constants only: no table.
//
// T166: 8 wasm instructions a group, not 15 (the loads of the tops included). The eight top bytes are read twice over (lanes 0..7 and 8..15 both
// hold byte k) and lanes 0..7 multiplied by 4 as int16 (one mul): then in every lane bits 2-3 hold the top of the
// first half's value (k or k + 8) and bits 6-7 that of the second half's (k + 16 or k + 24), so that each half takes
// its top with one select and no shift per lane. Shifts by 16-bit lanes are used wherever the bits a byte takes from
// its neighbour are masked away (x86 has no 8-bit shifts: V8 makes each of them two instructions).

/** the top bits of the group at p, read once for both halves: in every lane, bits 2-3 the first half's, bits 6-7 the
 * second half's (bits 0-1 and 4-5 are left over and never used) */
// @ts-ignore: decorator
@inline export function sixTops(p: usize): v128 {
  return i16x8.mul(v128.load64_splat(p + 16), i16x8(4, 4, 4, 4, 1, 1, 1, 1));
}

/** values 0..15 of a group, from its first 16 bytes and sixTops(): the low nibble to bits 2-5 (bits 0-1 zero), the
 * top to bits 6-7 */
// @ts-ignore: decorator
@inline export function sixFirst(low: v128, t: v128): v128 {
  return v128.bitselect(i16x8.shl(t, 4), i8x16.shl(low, 2), i8x16.splat(-64));
}

/** values 16..31: the high nibble to bits 2-5, the top already at bits 6-7 */
// @ts-ignore: decorator
@inline export function sixSecond(low: v128, t: v128): v128 {
  return v128.bitselect(t, v128.and(i16x8.shr_u(low, 2), i8x16.splat(60)), i8x16.splat(-64));
}
