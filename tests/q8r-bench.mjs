// T167: matmul_q8r and matmul_q6r (relaxed SIMD, the one-token path of Chromium and Firefox) with main's kernels and
// this tree's, in one process, taking turns (AGENTS.md: the old and the new side by side). The two forms round each
// group differently (T167: one rounding a group, before it one a lane), so the outputs are compared with a float64
// sum of the same integers and scales: the error of each against it, and how far the two are apart. Compiles each
// form with AssemblyScript into .tmp/q8r-bench/ (needs `npm ci`; `make kernels` not).
//
//   node tests/q8r-bench.mjs [--rounds 3] [--turns 7]
//
// Three sizes: 8192 x 8192 (64 MiB of int8, far past the caches), 2048 x 2048 (4 MiB: T167's size "in the caches"
// of a server's last level) and 256 x 1024 (in the first levels, called over and over). GB/s counts the bytes a row
// reads: 32 a group of int8 (24 of int6), its float32 scale and its correction (4 bytes: float32 before T197, int32 from it).
// T167's review: and the runner's ceilings, T163's loops (kernels/ceilings*.ts): at each size the read-only loop over
// as many bytes as matmul_q8r reads (40 a group), taking turns with the kernels, and once relaxed_dot with its two
// loads in L1. A form at the read's GB/s is held by the memory there, not by its own instructions.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), turns = option("--turns", 7);
const work = root + ".tmp/q8r-bench/";

// main's kernels (CI checks out one commit: fetch main's)
try { execFileSync("git", ["fetch", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
const forms = { main: {}, tree: {} };
for (const file of ["kernel.ts", "kernel_relaxed.ts", "six.ts"]) {
  forms.main[file] = execFileSync("git", ["show", `origin/main:kernels/${file}`], { cwd: root, encoding: "utf8" });
  forms.tree[file] = fs.readFileSync(`${root}kernels/${file}`, "utf8");
}

const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1"];
const memory = new WebAssembly.Memory({ initial: 1, maximum: 8192 });
const instance = (file) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(file)), { env: { memory } }).exports;
const kernels = {};
for (const [name, files] of Object.entries(forms)) {
  const dir = `${work}${name}/`;
  fs.mkdirSync(dir, { recursive: true });
  for (const [file, text] of Object.entries(files)) fs.writeFileSync(dir + file, text);
  execFileSync("npx", [...asc, dir + "kernel.ts", "-o", dir + "plain.wasm", "--enable", "simd"], { cwd: root, stdio: "inherit" });
  execFileSync("npx", [...asc, dir + "kernel_relaxed.ts", "-o", dir + "relaxed.wasm", "--enable", "simd,relaxed-simd"], { cwd: root, stdio: "inherit" });
  kernels[name] = { plain: instance(dir + "plain.wasm"), relaxed: instance(dir + "relaxed.wasm") };
}
execFileSync("npx", [...asc, root + "kernels/ceilings.ts", "-o", work + "ceilings.wasm", "--enable", "simd"], { cwd: root, stdio: "inherit" });
execFileSync("npx", [...asc, root + "kernels/ceilings_relaxed.ts", "-o", work + "ceilings_relaxed.wasm", "--enable", "simd,relaxed-simd"], { cwd: root, stdio: "inherit" });
const ceilings = { ...instance(work + "ceilings.wasm"), ...instance(work + "ceilings_relaxed.wasm") };

// the runner's CPU by name: GitHub's ubuntu-latest hands out different ones (T167's review: EPYC 7763, 9V74 and 9V45,
// Xeon 6973P-C, 8573C and 8370C, where the same change was 0.88 to 1.18 times main)
console.log(`cpu: ${os.cpus()[0]?.model ?? "unknown"}, ${os.cpus().length} logical cores, ${process.arch}`);
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
// the corrections of matmul_q8r and matmul_q6r as a form makes them: before T197 int8_sums and six_sums took the
// scales (out, w, scales, groups: float32 scale × sum), from T197 not (out, w, groups: int32 −64 × sum)
const sums = (plain, kernel, out, w, ws, groups) => (plain[kernel].length === 4 ? plain[kernel](out, w, ws, groups) : plain[kernel](out, w, groups));
{  // relaxed_dot with its two loads, on 8 KB at 4096 (below the matrices), the second 4 KB 0..127 as the loop wants
  const I = new Int8Array(memory.buffer);
  for (let j = 0; j < 8192; j++) I[4096 + j] = j < 4096 ? (Math.imul(j, 2654435761) >>> 24) - 128 : (j * 37) % 128;
  const passes = 20000, ms = [];
  for (let t = 0; t < turns * 3; t++) { const t0 = performance.now(); ceilings.dot(4096, passes); ms.push(performance.now() - t0); }
  console.log(`ceiling: relaxed_dot with two loads in L1 ${(passes * 4096 / median(ms) / 1e6).toFixed(2)} G MAC/s`);
}
for (const [rows, n, calls] of [[8192, 8192, 1], [2048, 2048, 12], [256, 1024, 200]]) {
  const ng = n / 32;
  let top = 65536;
  const take = (bytes) => { const at = top; top += Math.ceil(bytes / 64) * 64; return at; };
  // each form's corrections in its own place: T197 changed them from the float32 scale × sum to the int32 −64 × sum
  const w8 = take(rows * n), w6 = take(rows * ng * 24), ws = take(rows * ng * 4);
  const wc8 = { main: take(rows * ng * 4), tree: take(rows * ng * 4) }, wc6 = { main: take(rows * ng * 4), tree: take(rows * ng * 4) };
  const x = take(n), xs = take(ng * 4), out = take(rows * 4);
  if (top > memory.buffer.byteLength) memory.grow(Math.ceil((top - memory.buffer.byteLength) / 65536));
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
  let seed = 11;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  for (let i = 0; i < rows * n; i++) I[w8 + i] = (next() & 255) - 128;
  for (let i = 0; i < rows * ng * 24; i++) U[w6 + i] = next() & 255;
  for (let i = 0; i < rows * ng; i++) F[ws / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000)));
  for (let g = 0; g < ng; g++) F[xs / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
  for (let j = 0; j < n; j++) I[x + j] = next() % 128;  // quantize_x(bias = 64): 0..127
  for (const [name, k] of Object.entries(kernels)) {
    sums(k.plain, "int8_sums", wc8[name], w8, ws, rows * ng);
    sums(k.plain, "six_sums", wc6[name], w6, ws, rows * ng);
  }
  const six = (i, j) => {  // the int8 value of int6 weight j of row i (forward-check's six())
    const at = w6 + (i * ng + (j >> 5)) * 24, m = j & 31;
    const low = m < 16 ? U[at + m] & 15 : U[at + m - 16] >> 4, t = (U[at + 16 + (m % 8)] >> (2 * ((m / 8) | 0))) & 3;
    return (((low | (t << 4)) << 2) << 24) >> 24;
  };
  // float64, the same integers and scales: each group's dot(w, q − 64) (the activations carry quantize_x's bias of 64)
  const exact = (weight) => Array.from({ length: rows }, (_, i) => {
    let sum = 0;
    for (let g = 0; g < ng; g++) {
      let dot = 0;
      for (let j = g * 32; j < g * 32 + 32; j++) dot += weight(i, j) * (I[x + j] - 64);
      sum += dot * F[ws / 4 + i * ng + g] * F[xs / 4 + g];
    }
    return sum;
  });
  const runs = {
    q8r: { run: (k, name) => k.relaxed.matmul_q8r(out, x, xs, w8, ws, wc8[name], n, 0, rows), bytes: rows * ng * 40, reference: exact((i, j) => I[w8 + i * n + j]) },
    q6r: { run: (k, name) => k.relaxed.matmul_q6r(out, x, xs, w6, ws, wc6[name], n, 0, rows), bytes: rows * ng * 32, reference: exact(six) },
  };
  // each form against the float64 sums, and the two against each other: the largest error over the rows, relative
  // to the largest |output| of the matrix (a row's own output can be near 0 while its terms are not)
  for (const [kernel, { run, reference }] of Object.entries(runs)) {
    const scale = Math.max(...reference.map(Math.abs));
    const outputs = {};
    for (const [name, k] of Object.entries(kernels)) { F.fill(0, out / 4, out / 4 + rows); run(k, name); outputs[name] = F.slice(out / 4, out / 4 + rows); }
    const error = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i]))) / scale;
    const same = outputs.main.filter((v, i) => v === outputs.tree[i]).length;
    console.log(`${rows} x ${n} ${kernel}: against float64 main ${error(outputs.main, reference).toExponential(2)}, tree ${error(outputs.tree, reference).toExponential(2)}; `
      + `tree against main ${error(outputs.tree, outputs.main).toExponential(2)}, ${same} of ${rows} rows the same to the bit`);
  }
  const time = (run, k, name) => { const t0 = performance.now(); for (let c = 0; c < calls; c++) run(k, name); return (performance.now() - t0) / calls; };
  // the read: as many bytes as q8r's, from w8 on, after the turns of the kernels and not between them (between them
  // it read w8 just before main's q8r and not before the tree's)
  const readBytes = rows * ng * 40, read = () => ceilings.read(w8, readBytes, 1);
  for (let r = 0; r < rounds; r++) {
    const ms = { read: [] };
    for (let t = 0; t < turns; t++) {
      for (const [name, k] of Object.entries(kernels)) for (const [kernel, { run }] of Object.entries(runs)) (ms[`${kernel} ${name}`] ??= []).push(time(run, k, name));
    }
    for (let t = 0; t < turns; t++) ms.read.push(time(read));
    const readSpeed = readBytes / median(ms.read) / 1e6;
    const line = [`read ${readSpeed.toFixed(2)} GB/s`];
    for (const [kernel, { bytes }] of Object.entries(runs)) {
      const base = median(ms[`${kernel} main`]);
      for (const name of Object.keys(kernels)) {
        const m = median(ms[`${kernel} ${name}`]);
        line.push(`${kernel} ${name} ${(bytes / m / 1e6).toFixed(2)} GB/s (${Math.round(100 * bytes / m / 1e6 / readSpeed)}% of the read) ${(rows * n / m / 1e6).toFixed(2)} G MAC/s ${(base / m).toFixed(2)}x`);
      }
    }
    console.log(`${rows} x ${n}, round ${r + 1}: ${line.join(" | ")}`);
  }
}
