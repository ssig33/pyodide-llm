// T161: the attention kernels of another commit (the old) against this tree's (the new), in one process, taking
// turns: what each writes (the same to a few float32 roundings; the heads shared in two ranges the same to the bit
// as all at once) and how long each takes, float32 and float16 caches, at short and long contexts.
//   git fetch --depth=1 origin main && node tests/attention-compare.mjs FETCH_HEAD [rounds]
// The old kernels are compiled here from that commit's kernels/*.ts (as kernels/build.py's plain module), the new
// ones are public/simdkernel_plain.wasm (make kernels). Exit 1 if the outputs differ.
// T160: at the end, the float16 cache against the float32 one, old and new (above 1: the float16 cache is slower).
// T201: --exact, for a change that must write the old's bits (the softmax's maximum): any bit that differs fails, and
// before the timing each position of short heads (1 to 72 positions, around the 4 and 32 of the maximum's steps)
// takes a turn as the one far largest score (its key ten times the query), so that a maximum that skips a position
// (a lane, an accumulator, the tail) moves every weight of that head (exp(0) is 1 only against the true maximum).
//   node tests/attention-compare.mjs FETCH_HEAD [rounds] --exact
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const exact = process.argv.includes("--exact");
const [ref = "origin/main", rounds = "3"] = process.argv.slice(2).filter((a) => a !== "--exact");
const dir = `${root}.tmp/attention-compare/`;
fs.mkdirSync(dir, { recursive: true });
for (const file of ["kernel.ts", "six.ts"]) {
  fs.writeFileSync(dir + file, execFileSync("git", ["show", `${ref}:kernels/${file}`], { cwd: root }));
}
execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
  `${dir}kernel.ts`, "-o", `${dir}old.wasm`, "--enable", "simd"], { cwd: root, stdio: "inherit" });

const memory = new WebAssembly.Memory({ initial: 4200 });
const load = (file) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(file)), { env: { memory } }).exports;
const kernels = { old: load(`${dir}old.wasm`), new: load(`${root}public/simdkernel_plain.wasm`) };
const F = new Float32Array(memory.buffer);

let failed = false;
if (exact) {
  // T201: each position the one largest score in turn, both caches, old and new to the bit
  const nh = 2, hs = 8, kvDim = nh * hs, seq = 72;
  const q = 1 << 20, att = q + 4096, out = att + 4096, other = out + 4096, kc = other + 4096;
  const vc = kc + seq * kvDim * 4, kh = vc + seq * kvDim * 4, vh = kh + seq * kvDim * 2;
  for (let i = q / 4; i < q / 4 + nh * hs; i++) F[i] = Math.random() * 2 - 1;
  for (let i = kc / 4; i < kh / 4; i++) F[i] = Math.random() * 2 - 1;
  kernels.new.to_f16(vh, vc, seq * kvDim);
  let checked = 0, differ = 0;
  for (let pos = 0; pos < seq; pos++) {
    for (let p = 0; p <= pos; p++) {
      const row = kc / 4 + p * kvDim, saved = F.slice(row, row + kvDim);
      for (let i = 0; i < kvDim; i++) F[row + i] = 10 * F[q / 4 + i];
      kernels.new.to_f16(kh, kc, seq * kvDim);
      for (const [name, k, v] of [["attention", kc, vc], ["attention_f16", kh, vh]]) {
        kernels.old[name](out, q, k, v, att, pos, nh, nh, hs, 0, nh);
        kernels.new[name](other, q, k, v, att, pos, nh, nh, hs, 0, nh);
        checked++;
        for (let i = 0; i < nh * hs; i++) {
          if (!Object.is(F[out / 4 + i], F[other / 4 + i])) {
            if (differ < 5) console.log(`${name} ${nh}/${nh}/${hs} pos ${pos}, the largest at ${p}: old ${F[out / 4 + i]}, new ${F[other / 4 + i]}`);
            differ++;
            break;
          }
        }
      }
      F.set(saved, row);
    }
  }
  if (differ) failed = true;
  console.log(`attention-compare --exact: the largest score at each position, ${checked} calls, ${differ} differ from the old`);
}
const times = {};  // T160: "shape, position" -> {attention: {old, new}, attention_f16: {old, new}}
console.log(`| kernel | heads / kv heads / head size, position | old µs | new µs | old ÷ new | G MAC/s old → new |\n|---|---|---:|---:|---:|---|`);
for (const [nh, nkv, hs, positions] of [[8, 8, 64, [16, 256, 1000, 2000, 4000]], [32, 8, 64, [256, 4000]], [4, 2, 6, [0, 1, 2, 3, 4, 5, 6, 7]], [3, 3, 10, [9]]]) {
  const seq = 4096, kvDim = nkv * hs;
  let top = 1 << 20;
  const alloc = (bytes) => { const at = top; top = Math.ceil((at + bytes) / 64) * 64; return at; };
  const q = alloc(nh * hs * 4), att = alloc(nh * seq * 4), out = alloc(nh * hs * 4), other = alloc(nh * hs * 4), halves = alloc(nh * hs * 4);
  const kc = alloc(seq * kvDim * 4), vc = alloc(seq * kvDim * 4), kh = alloc(seq * kvDim * 2), vh = alloc(seq * kvDim * 2);
  for (let i = q / 4; i < vh / 4; i++) F[i] = Math.random() * 2 - 1;
  kernels.new.to_f16(kh, kc, seq * kvDim);
  kernels.new.to_f16(vh, vc, seq * kvDim);
  for (const pos of positions) {
    for (const [name, k, v] of [["attention", kc, vc], ["attention_f16", kh, vh]]) {
      const call = (K, o, h0 = 0, h1 = nh) => K[name](o, q, k, v, att, pos, nh, nkv, hs, h0, h1);
      call(kernels.old, out);
      call(kernels.new, other);
      call(kernels.new, halves, 0, nh >> 1);
      call(kernels.new, halves, nh >> 1, nh);
      let largest = 0, difference = 0;
      for (let i = 0; i < nh * hs; i++) {
        largest = Math.max(largest, Math.abs(F[out / 4 + i]));
        difference = Math.max(difference, Math.abs(F[out / 4 + i] - F[other / 4 + i]));
        if (F[halves / 4 + i] !== F[other / 4 + i]) { failed = true; console.log(`${name} ${nh}/${nkv}/${hs} pos ${pos}: the heads in two ranges differ from all at once`); break; }
      }
      if (exact && difference > 0) { failed = true; console.log(`${name} ${nh}/${nkv}/${hs} pos ${pos}: old and new differ by ${difference} (--exact)`); }
      if (difference > 1e-5 * largest) { failed = true; console.log(`${name} ${nh}/${nkv}/${hs} pos ${pos}: old and new differ by ${difference} of ${largest}`); }
      if (pos < 16) continue;  // the small shapes check the edges only
      const n = Math.max(10, Math.floor(20000 / (pos + 1)));
      const best = { old: Infinity, new: Infinity };
      for (let round = 0; round < +rounds; round++) {
        for (const which of ["old", "new"]) {
          for (let r = 0; r < 5; r++) {
            const t = performance.now();
            for (let i = 0; i < n; i++) call(kernels[which], out);
            best[which] = Math.min(best[which], (performance.now() - t) / n * 1e3);
          }
        }
      }
      (times[`${nh}/${nkv}/${hs}, ${pos}`] ??= {})[name] = best;
      const macs = 2 * nh * (pos + 1) * hs;
      console.log(`| ${name} | ${nh}/${nkv}/${hs}, ${pos} | ${best.old.toFixed(1)} | ${best.new.toFixed(1)} | ${(best.old / best.new).toFixed(2)} | ${(macs / best.old / 1e3).toFixed(2)} → ${(macs / best.new / 1e3).toFixed(2)} |`);
    }
  }
}
console.log(`\n| heads / kv heads / head size, position | float16 ÷ float32 old | new |\n|---|---:|---:|`);
for (const [shape, { attention: f32, attention_f16: f16 }] of Object.entries(times)) {
  console.log(`| ${shape} | ${(f16.old / f32.old).toFixed(2)} | ${(f16.new / f32.new).toFixed(2)} |`);
}
console.log(failed ? "attention-compare: FAILED" : "attention-compare: the outputs agree");
process.exit(failed ? 1 : 0);
