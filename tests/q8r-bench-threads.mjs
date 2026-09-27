// T167's review: matmul_q8r of main and of this tree with 1, 2 and 4 threads on one shared memory, taking turns,
// at 8192 x 8192 (past the caches) and 2048 x 2048. The page runs the token with software threads where it is
// cross-origin isolated, so a form that is slower with one thread at the memory's wall (the EPYC 9V45) may not be
// slower with four (it was not: 0.99 to 1.01). Rows split evenly, as the static share of the threads. Two threads
// swing too much on the 4-vCPU runners to read (0.65 to 1.88).
//
//   node tests/q8r-bench-threads.mjs [--rounds 3] [--turns 7]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, workerData } from "node:worker_threads";

const CONTROL = 16;  // int32s: 0 go, 1 done, 2 form, 3 n, 4..7 addresses (out, x, xs, w8), 8 ws, 9 wc8, 10 rows, 11 threads

if (!isMainThread) {
  const { memory, forms, control, index } = workerData;
  const c = new Int32Array(control);
  const kernels = forms.map((file) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(file)), { env: { memory } }).exports);
  let seen = 0;
  for (;;) {
    Atomics.wait(c, 0, seen);
    seen = Atomics.load(c, 0);
    if (seen < 0) break;
    const rows = c[10], threads = c[11], share = Math.ceil(rows / threads);
    if (index >= threads) continue;  // not asked this time
    const r0 = Math.min(rows, index * share), r1 = Math.min(rows, r0 + share);
    kernels[c[2]].matmul_q8r(c[4], c[5], c[6], c[7], c[8], c[9], c[3], r0, r1);
    Atomics.add(c, 1, 1);
    Atomics.notify(c, 1);
  }
} else {
  const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
  const args = process.argv.slice(2);
  const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
  const rounds = option("--rounds", 3), turns = option("--turns", 7);
  const work = root + ".tmp/q8r-bench-threads/";
  try { execFileSync("git", ["fetch", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
  const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
    "--sharedMemory", "--maximumMemory", "65536"];
  const files = [];
  for (const name of ["main", "tree"]) {
    const dir = `${work}${name}/`;
    fs.mkdirSync(dir, { recursive: true });
    for (const file of ["kernel.ts", "kernel_relaxed.ts", "six.ts"]) {
      fs.writeFileSync(dir + file, name === "main" ? execFileSync("git", ["show", `origin/main:kernels/${file}`], { cwd: root, encoding: "utf8" })
        : fs.readFileSync(`${root}kernels/${file}`, "utf8"));
    }
    execFileSync("npx", [...asc, dir + "kernel_relaxed.ts", "-o", dir + "relaxed.wasm", "--enable", "simd,relaxed-simd,threads"], { cwd: root, stdio: "inherit" });
    files.push(dir + "relaxed.wasm");
  }
  execFileSync("npx", [...asc, `${work}tree/kernel.ts`, "-o", work + "plain.wasm", "--enable", "simd,threads"], { cwd: root, stdio: "inherit" });
  console.log(`cpu: ${os.cpus()[0]?.model ?? "unknown"}, ${os.cpus().length} logical cores, ${process.arch}`);

  const memory = new WebAssembly.Memory({ initial: 1, maximum: 65536, shared: true });
  const plain = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(work + "plain.wasm")), { env: { memory } }).exports;
  const control = new SharedArrayBuffer(CONTROL * 4), c = new Int32Array(control);
  const workers = [0, 1, 2, 3].map((index) => new Worker(new URL(import.meta.url), { workerData: { memory, forms: files, control, index } }));
  const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];

  for (const [rows, n, calls] of [[8192, 8192, 1], [2048, 2048, 12]]) {
    const ng = n / 32;
    let top = 65536;
    const take = (bytes) => { const at = top; top += Math.ceil(bytes / 64) * 64; return at; };
    const w8 = take(rows * n), ws = take(rows * ng * 4), wc8 = take(rows * ng * 4), x = take(n), xs = take(ng * 4), out = take(rows * 4);
    const pages = Math.ceil(top / 65536) - memory.buffer.byteLength / 65536;
    if (pages > 0) memory.grow(pages);
    const I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
    let seed = 11;
    const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
    for (let i = 0; i < rows * n; i++) I[w8 + i] = (next() & 255) - 128;
    for (let i = 0; i < rows * ng; i++) F[ws / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000)));
    for (let g = 0; g < ng; g++) F[xs / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
    for (let j = 0; j < n; j++) I[x + j] = next() % 128;
    plain.int8_sums(wc8, w8, ws, rows * ng);
    Object.assign(c, { 3: n, 4: out, 5: x, 6: xs, 7: w8, 8: ws, 9: wc8, 10: rows });
    const run = (form, threads) => {
      c[2] = form; c[11] = threads;
      Atomics.store(c, 1, 0);
      Atomics.add(c, 0, 1);
      Atomics.notify(c, 0);
      while (Atomics.load(c, 1) < threads) Atomics.wait(c, 1, Atomics.load(c, 1), 100);
    };
    const time = (form, threads) => {
      const t0 = performance.now();
      for (let k = 0; k < calls; k++) run(form, threads);
      return (performance.now() - t0) / calls;
    };
    for (const threads of [1, 2, 4]) {
      for (let r = 0; r < rounds; r++) {
        const ms = [[], []];
        for (let t = 0; t < turns; t++) for (const form of [0, 1]) ms[form].push(time(form, threads));
        const [a, b] = ms.map(median), bytes = rows * ng * 40;
        console.log(`${rows} x ${n}, ${threads} thread(s), round ${r + 1}: q8r main ${(bytes / a / 1e6).toFixed(2)} GB/s | tree ${(bytes / b / 1e6).toFixed(2)} GB/s ${(a / b).toFixed(2)}x`);
      }
    }
  }
  Atomics.store(c, 0, -1);
  Atomics.notify(c, 0);
  for (const w of workers) await w.terminate();
}
