// T161: the attention kernels of another commit (the old) against this tree's (the new), in one process, taking
// turns: what each writes (the same to a few float32 roundings; the heads shared in two ranges the same to the bit
// as all at once) and how long each takes, float32 and float16 caches, at short and long contexts.
//   git fetch --depth=1 origin main && node tests/attention-compare.mjs FETCH_HEAD [rounds]
// The old kernels are compiled here from that commit's kernels/*.ts (as kernels/build.py's plain module), the new
// ones are public/simdkernel_plain.wasm (make kernels). Exit 1 if the outputs differ.
// T160: at the end, the float16 cache against the float32 one, old and new (above 1: the float16 cache is slower).
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const [ref = "origin/main", rounds = "3"] = process.argv.slice(2);
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
