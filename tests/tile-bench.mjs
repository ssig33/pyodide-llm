// T159: a prompt's matrix product, 16 tokens, as jobs.js ran it before T159 (matmul_q8r for each token over blocks of
// rows of 16 KB, T108) and as matmul_q8r_tile (four rows by four tokens), in one process, taking turns (AGENTS.md: the
// old and the new side by side). One thread. The two must give the same numbers to the bit. Compiles this tree's
// kernel_relaxed.ts with AssemblyScript into .tmp/tile-bench/ (needs `npm ci`; `make kernels` not).
//
//   node tests/tile-bench.mjs [--rounds 3] [--turns 7] [--tokens 16]
//
// The shapes of Llama 3.2 1B's layers (2048 x 2048, 8192 x 2048, 2048 x 8192) and llm-jp-3 150M's (512 x 512,
// 2048 x 512). G MAC/s is rows x n x tokens a second.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), turns = option("--turns", 7), tokens = option("--tokens", 16);
const work = root + ".tmp/tile-bench/";
fs.mkdirSync(work, { recursive: true });
execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
  root + "kernels/kernel_relaxed.ts", "-o", work + "relaxed.wasm", "--enable", "simd,relaxed-simd"], { cwd: root, stdio: "inherit" });
const memory = new WebAssembly.Memory({ initial: 1, maximum: 8192 });
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(work + "relaxed.wasm")), { env: { memory } }).exports;
console.log(`${os.cpus()[0]?.model ?? "unknown"}, ${os.cpus().length} logical cores, ${tokens} tokens, one thread`);

const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
for (const [rows, n] of [[2048, 2048], [8192, 2048], [2048, 8192], [512, 512], [2048, 512]]) {
  const ng = n / 32, frame = Math.ceil((n + ng * 4) / 64) * 64, outFrame = Math.ceil(rows * 4 / 64) * 64;
  let top = 65536;
  const take = (bytes) => { const at = top; top += Math.ceil(bytes / 64) * 64; return at; };
  const w = take(rows * n), ws = take(rows * ng * 4), wc = take(rows * ng * 4), frames = take(tokens * frame);
  const outA = take(tokens * outFrame), outB = take(tokens * outFrame);
  if (top > memory.buffer.byteLength) memory.grow(Math.ceil((top - memory.buffer.byteLength) / 65536));
  const I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer);
  let seed = 5;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  for (let i = 0; i < rows * n; i++) I[w + i] = (next() & 255) - 128;
  for (let i = 0; i < rows * ng; i++) { F[ws / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000))); F[wc / 4 + i] = Math.fround(1e-2 * ((next() % 2001) - 1000)); }
  for (let t = 0; t < tokens; t++) {
    for (let j = 0; j < n; j++) I[frames + t * frame + j] = next() % 128;
    for (let g = 0; g < ng; g++) F[(frames + t * frame + n) / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
  }
  const step = Math.max(1, Math.floor(16384 / n));  // jobs.js's blockRows before T159
  const blocks = () => {
    for (let r = 0; r < rows; r += step) {
      const end = Math.min(r + step, rows);
      for (let t = 0; t < tokens; t++) k.matmul_q8r(outA + t * outFrame, frames + t * frame, frames + t * frame + n, w, ws, wc, n, r, end);
    }
  };
  const tile = () => k.matmul_q8r_tile(outB, frames, frames + n, w, ws, wc, n, 0, rows, tokens, outFrame, frame);
  blocks();
  tile();
  for (let t = 0; t < tokens; t++) {
    for (let i = 0; i < rows; i++) {
      if (!Object.is(F[(outA + t * outFrame) / 4 + i], F[(outB + t * outFrame) / 4 + i])) throw new Error(`${rows} x ${n}: the tile differs at row ${i}, token ${t}`);
    }
  }
  const calls = Math.max(1, Math.round(2e8 / (rows * n * tokens)));
  const time = (run) => { const t0 = performance.now(); for (let c = 0; c < calls; c++) run(); return (performance.now() - t0) / calls; };
  const macs = rows * n * tokens;
  for (let r = 0; r < rounds; r++) {
    const a = [], b = [];
    for (let t = 0; t < turns; t++) { a.push(time(blocks)); b.push(time(tile)); }
    const ma = median(a), mb = median(b);
    console.log(`${rows} x ${n}, round ${r + 1}: blocks of 16 KB ${(macs / ma / 1e6).toFixed(1)} G MAC/s, tile ${(macs / mb / 1e6).toFixed(1)} G MAC/s, ${(ma / mb).toFixed(2)}x (the same to the bit)`);
  }
}
