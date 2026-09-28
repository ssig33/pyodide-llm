// tests/attention-check.mjs (T217, the review of T201): attention's softmax against its meaning where the largest score
// decides something. The kernel takes exp(score - largest) with its exponent held to at most 88 (fexp and vexp), and
// divides by the sum, so a largest that leaves out one position (the largest one) moves nothing past rounding: that
// position's weight is exp(its lead) over the sum either way, and past a lead of 88 every other weight is below
// float32's reach in both. It does matter where two or more positions lead the largest taken by more than 88: they
// all get exp(88) and so the same weight (and where the largest taken is too high, every weight is exp(-87) or 0).
// forward-check's models never lead by that much, so its line (1e-3) does not see a largest that leaves positions out
// (T201's record: only attention-compare --exact did, to the bit). Here the scores of a head are made so: two
// positions far above the rest, 3 apart, at every pair 1, 4 and 32 apart (a lane, a vector of four, a turn of the
// eight maxima), in every count of positions from 2 to 72; one head far above zero and one far below it (a largest
// begun at 0 instead of -f32.MAX_VALUE is wrong only there). Their weights are then 1 / (1 + e^-3) and the rest,
// whatever the kernel's exp() does below -87: the output is those two values' mix to 1e-3, float32 and float16 caches.
// The expected output is float64 from the very numbers the kernel reads (the float16 cache's halves widened here).
//   node tests/attention-check.mjs            (after make kernels; exit 1 if a head is off)
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const memory = new WebAssembly.Memory({ initial: 32 });
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
const F = new Float32Array(memory.buffer), H = new Uint16Array(memory.buffer);

const halfToFloat = (h) => {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};
let seed = 12345;
const random = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8) / 2 ** 24;  // [0, 1)

const nh = 2, hs = 8, kvDim = nh * hs, most = 72, LINE = 1e-3, LEAD = 3;
const q = 1 << 20, att = q + 4096, out = att + 4096, kc = out + 4096, vc = kc + most * kvDim * 4;
const kh = vc + most * kvDim * 4, vh = kh + most * kvDim * 2;
// one query for both heads (a kv head of its own each): the direction u
const u = Array.from({ length: hs }, () => random() * 2 - 1), uu = u.reduce((sum, x) => sum + x * x, 0);
for (let h = 0; h < nh; h++) for (let i = 0; i < hs; i++) F[q / 4 + h * hs + i] = u[i];
// head 0: the rest from -2 to 2 and the two at 110 and 107; head 1: the rest from -260 to -250 and the two at -100 and -103
const levels = [{ rest: [-2, 2], top: 110 }, { rest: [-260, -250], top: -100 }];

let cases = 0, worst = 0, failed = 0;
for (let count = 2; count <= most; count++) {
  for (const apart of [1, 4, 32]) {
    for (let a = 0; a + apart < count; a++) {
      const b = a + apart;
      for (let h = 0; h < nh; h++) {
        const { rest: [low, high], top } = levels[h];
        for (let t = 0; t < count; t++) {
          // a key along u gives the score it is scaled to (score × √hs / |u|²); the values: +1 at a, -1 at b
          const score = t === a ? top : t === b ? top - LEAD : low + (high - low) * random();
          for (let i = 0; i < hs; i++) {
            F[kc / 4 + t * kvDim + h * hs + i] = (score * Math.sqrt(hs) / uu) * u[i];
            F[vc / 4 + t * kvDim + h * hs + i] = t === a ? 1 : t === b ? -1 : random() * 2 - 1;
          }
        }
      }
      k.to_f16(kh, kc, count * kvDim);
      k.to_f16(vh, vc, count * kvDim);
      for (const [name, keys, values, read] of [["attention", kc, vc, (at, i) => F[at / 4 + i]], ["attention_f16", kh, vh, (at, i) => halfToFloat(H[at / 2 + i])]]) {
        F.fill(NaN, out / 4, out / 4 + nh * hs);
        k[name](out, q, keys, values, att, count - 1, nh, nh, hs, 0, nh);
        for (let h = 0; h < nh; h++) {
          const scores = Array.from({ length: count }, (_, t) => {
            let sum = 0;
            for (let i = 0; i < hs; i++) sum += F[q / 4 + h * hs + i] * read(keys, t * kvDim + h * hs + i);
            return sum / Math.sqrt(hs);
          });
          const largest = Math.max(...scores), weights = scores.map((s) => Math.exp(s - largest));
          const total = weights.reduce((sum, w) => sum + w, 0);
          for (let i = 0; i < hs; i++) {
            let want = 0;
            for (let t = 0; t < count; t++) want += (weights[t] / total) * read(values, t * kvDim + h * hs + i);
            const off = Math.abs(F[out / 4 + h * hs + i] - want);
            if (!(off <= worst)) worst = Number.isNaN(off) ? NaN : Math.max(worst, off);
            if (!(off <= LINE)) {
              if (failed < 5) console.log(`${name}: ${count} positions, the two far largest at ${a} and ${b}, head ${h} (${top > 0 ? "above" : "below"} 0): ${F[out / 4 + h * hs + i]} where ${want.toFixed(6)} is right`);
              failed++;
              break;
            }
          }
        }
        cases++;
      }
    }
  }
}
console.log(`attention-check: ${cases} calls (${nh} heads each: two positions far above the rest, 1, 4 and 32 apart, 2 to ${most} positions), ` +
  `the worst ${Number.isNaN(worst) ? "NaN" : worst.toExponential(1)} from the right mix (line ${LINE})${failed ? `, ${failed} heads off — FAILED` : ""}`);
process.exit(failed ? 1 : 0);
