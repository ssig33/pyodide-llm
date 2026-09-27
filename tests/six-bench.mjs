// T166: the int6 matrix products (matmul_q6r with relaxed SIMD, matmul_q6 without) with main's kernels and this
// tree's, in one process, taking turns (AGENTS.md: the old and the new side by side), with matmul_q8r (int8) for the
// ratio T98 wrote down (0.45). The tree's output must be main's to the bit (matmul_q6r's only while the two make
// the same corrections: T197 changed them). (The forms T166 measured and dropped are
// in TODO.md's T166: put another kernels/six.ts in the tree to compare one.) Compiles each form with AssemblyScript into .tmp/six-bench/ (needs `npm ci`; `make kernels` not).
//
//   node tests/six-bench.mjs [--rounds 3] [--turns 7]
//
// Two sizes: 8192 x 8192 (the weights far past the caches: 48 MiB of int6) and 256 x 1024 (in the caches, called
// over and over). GB/s counts the bytes a row reads: 24 a group of int6 (32 of int8), and its float32 scale (and the
// correction, relaxed).
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), turns = option("--turns", 7);
const work = root + ".tmp/six-bench/";

// main's kernels (CI checks out one commit: fetch main's)
try { execFileSync("git", ["fetch", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
const forms = { main: {}, tree: {} };
for (const file of ["kernel.ts", "kernel_relaxed.ts", "six.ts"]) {
  forms.main[file] = execFileSync("git", ["show", `origin/main:kernels/${file}`], { cwd: root, encoding: "utf8" });
  forms.tree[file] = fs.readFileSync(`${root}kernels/${file}`, "utf8");
}

const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1"];
const memory = new WebAssembly.Memory({ initial: 1, maximum: 4096 });
const kernels = {};
for (const [name, files] of Object.entries(forms)) {
  const dir = `${work}${name}/`;
  fs.mkdirSync(dir, { recursive: true });
  for (const [file, text] of Object.entries(files)) fs.writeFileSync(dir + file, text);
  execFileSync("npx", [...asc, dir + "kernel.ts", "-o", dir + "plain.wasm", "--enable", "simd"], { cwd: root, stdio: "inherit" });
  execFileSync("npx", [...asc, dir + "kernel_relaxed.ts", "-o", dir + "relaxed.wasm", "--enable", "simd,relaxed-simd"], { cwd: root, stdio: "inherit" });
  const instance = (file) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(dir + file)), { env: { memory } }).exports;
  kernels[name] = { plain: instance("plain.wasm"), relaxed: instance("relaxed.wasm") };
}

const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
// the corrections as a form makes them: before T197 int8_sums and six_sums took the scales (out, w, scales, groups),
// from T197 not (out, w, groups)
const sums = (plain, kernel, out, w, ws, groups) => (plain[kernel].length === 4 ? plain[kernel](out, w, ws, groups) : plain[kernel](out, w, groups));
for (const [rows, n, calls] of [[8192, 8192, 1], [256, 1024, 200]]) {
  const ng = n / 32;
  let top = 65536;
  const take = (bytes) => { const at = top; top += Math.ceil(bytes / 64) * 64; return at; };
  const w6 = take(rows * ng * 24), w8 = take(rows * n), ws = take(rows * ng * 4);
  // each form's corrections in its own place (T197 changed them from float32 scale × sum to int32 −64 × sum)
  const wc6 = new Map(), wc8 = new Map();
  for (const k of Object.values(kernels)) { wc6.set(k, take(rows * ng * 4)); wc8.set(k, take(rows * ng * 4)); }
  const x = take(n), xs = take(ng * 4), out = take(rows * 4), expected = take(rows * 4);
  if (top > memory.buffer.byteLength) memory.grow(Math.ceil((top - memory.buffer.byteLength) / 65536));
  const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
  let seed = 11;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  for (let i = 0; i < rows * ng * 24; i++) U[w6 + i] = next() & 255;
  for (let i = 0; i < rows * ng; i++) F[ws / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000)));
  for (let g = 0; g < ng; g++) F[xs / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
  for (let j = 0; j < n; j++) I[x + j] = next() % 128;
  for (let g = 0; g < rows * ng; g++) {  // the int8 values of main's widening, for matmul_q8r
    for (let j = 0; j < 32; j++) {
      const at = w6 + g * 24, low = j < 16 ? U[at + j] & 15 : U[at + j - 16] >> 4, t = (U[at + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      I[w8 + g * 32 + j] = (((low | (t << 4)) << 2) << 24) >> 24;
    }
  }
  for (const k of Object.values(kernels)) { sums(k.plain, "six_sums", wc6.get(k), w6, ws, rows * ng); sums(k.plain, "int8_sums", wc8.get(k), w8, ws, rows * ng); }
  const runs = {
    q6r: (k) => k.relaxed.matmul_q6r(out, x, xs, w6, ws, wc6.get(k), n, 0, rows),
    q6: (k) => k.plain.matmul_q6(out, x, xs, w6, ws, n, 0, rows),
  };
  const bytes = { q6r: rows * ng * 32, q6: rows * ng * 28, q8r: rows * ng * 40 };
  // each form's output against main's, to the bit (matmul_q6r only where the two forms make the same corrections:
  // T197 changed matmul_q6r's numbers, and each form's matmul_q6r is held to its own matmul_q8r below)
  const sameSums = kernels.main.plain.six_sums.length === kernels.tree.plain.six_sums.length;
  for (const [kernel, run] of Object.entries(runs)) {
    if (kernel === "q6r" && !sameSums) continue;
    run(kernels.main);
    F.copyWithin(expected / 4, out / 4, out / 4 + rows);
    for (const [name, k] of Object.entries(kernels)) {
      F.fill(0, out / 4, out / 4 + rows);
      run(k);
      for (let i = 0; i < rows; i++) if (F[out / 4 + i] !== F[expected / 4 + i]) throw new Error(`${name}'s ${kernel} differs from main's at row ${i}`);
    }
  }
  for (const [name, k] of Object.entries(kernels)) {
    k.relaxed.matmul_q8r(expected, x, xs, w8, ws, wc8.get(k), n, 0, rows);
    k.relaxed.matmul_q6r(out, x, xs, w6, ws, wc6.get(k), n, 0, rows);
    for (let i = 0; i < rows; i++) if (F[out / 4 + i] !== F[expected / 4 + i]) throw new Error(`${name}'s matmul_q6r differs from its matmul_q8r at row ${i}`);
  }
  console.log(`${rows} x ${n}: every form's output is main's to the bit${sameSums ? "" : " (matmul_q6 only: the corrections of matmul_q6r differ, T197)"}, and each matmul_q6r is its matmul_q8r's on the widened values`);
  runs.q8r = (k) => k.relaxed.matmul_q8r(out, x, xs, w8, ws, wc8.get(k), n, 0, rows);
  const time = (run, k) => { const t0 = performance.now(); for (let c = 0; c < calls; c++) run(k); return (performance.now() - t0) / calls; };
  for (let r = 0; r < rounds; r++) {
    const ms = {};
    for (let t = 0; t < turns; t++) {
      for (const [name, k] of Object.entries(kernels)) {
        for (const kernel of ["q6r", "q6"]) (ms[`${kernel} ${name}`] ??= []).push(time(runs[kernel], k));
      }
      (ms["q8r tree"] ??= []).push(time(runs.q8r, kernels.tree));
    }
    const line = [];
    for (const kernel of ["q6r", "q6"]) {
      const base = median(ms[`${kernel} main`]);
      for (const name of Object.keys(kernels)) {
        const m = median(ms[`${kernel} ${name}`]);
        line.push(`${kernel} ${name} ${(bytes[kernel] / m / 1e6).toFixed(2)} GB/s ${(base / m).toFixed(2)}x`);
      }
    }
    const q8r = median(ms["q8r tree"]);
    line.push(`q8r ${(bytes.q8r / q8r / 1e6).toFixed(2)} GB/s; q6r against q8r (the time of a row): main ${(q8r / median(ms["q6r main"])).toFixed(2)}, tree ${(q8r / median(ms["q6r tree"])).toFixed(2)}`);
    console.log(`${rows} x ${n}, round ${r + 1}: ${line.join(" | ")}`);
  }
}
