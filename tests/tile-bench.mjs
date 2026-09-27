// T159: a prompt's int8 matrix product, matmul_q8r_tile (four rows by four tokens), of main and of this tree, in one
// process, taking turns (AGENTS.md: the old and the new side by side), at 2 to 16 tokens, on 1 and 4 threads of one
// shared memory, the rows in chunks taken in turn as forward.js's phase() cuts them. (Before T197 it set the tile
// against jobs.js's form before T159, blocks of 16 KB with matmul_q8r for each token: that is in TODO.md's T159.)
//
// T197 changed the corrections (float32 scale × sum → int32 −64 × sum, added to each group's integer sum), so the two
// forms' numbers differ: each is held to the float64 sums of the same integers and scales (the largest error over the
// rows of 4 tokens, relative to the largest |output|), and each to itself on 1 and 4 threads, to the bit. Each form
// reads its own corrections (the copies hold both).
//
// T159's review: the matrices are read from memory, as a model's are (each read once a block): copies of each past
// every cache (--megabytes, default 512), a call on each in turn. One matrix timed again and again stays in the caches
// and hid that the tile was up to 1.6 times as slow on the EPYC 9V74 and 9V45 at 4 to 8 tokens (--megabytes 0 times
// that way). Compiles main's and this tree's kernel_relaxed.ts with AssemblyScript into .tmp/tile-bench/ (needs `npm ci`).
//
//   node tests/tile-bench.mjs [--rounds 2] [--turns 5] [--megabytes 512] [--shapes 2048x2048,4096x4096]
//
// The shapes of llm-jp-3 150M's layers (512, hidden 2048), Qwen2.5 0.5B's (896, 4864), Llama 3.2 1B's (2048, its k
// and v 512, hidden 8192) and the widths of the 3B to 8B models (2560 to 4096). G MAC/s is rows x n x tokens a second.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, workerData } from "node:worker_threads";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const work = root + ".tmp/tile-bench/";
const FORMS = ["main", "tree"];
// the control words: 0 go (a generation), 1 threads done, 2 the next chunk, 3 form (0 main, 1 tree), 4 the first
// copy's weights, 5 n, 6 rows, 7 count, 8 threads, 9 out, 10 frames, 11 frame, 12 out frame, 13 calls, 14 copy bytes,
// 15 copies, 16 the copy to start at
const GO = 0, DONE = 1, NEXT = 2, FORM = 3, FIRST = 4, N = 5, ROWS = 6, COUNT = 7, THREADS = 8, OUT = 9, FRAMES = 10,
  FRAME = 11, OUT_FRAME = 12, CALLS = 13, STRIDE = 14, COPIES = 15, START = 16;
// a copy: the int8 weights, their scales, main's corrections, the tree's corrections
const tiles = (k, c, w, r0, r1) => {
  const n = c[N], rows = c[ROWS], ng = n / 32;
  k.matmul_q8r_tile(c[OUT], c[FRAMES], c[FRAMES] + n, w, w + rows * n, w + rows * n + (1 + c[FORM]) * rows * ng * 4, n, r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME]);
};
// forward.js's phase(): chunks of a quarter of a thread's share (T93), in fours of rows for a prompt (T159), taken in
// turn; one thread runs the whole matrix in one call
const chunkOf = (rows, threads) => (threads === 1 ? rows : 4 * Math.ceil(rows / (threads * 16)));
function runCalls(k, c, index) {
  const threads = c[THREADS], rows = c[ROWS];
  const size = chunkOf(rows, threads), chunks = Math.ceil(rows / size), calls = c[CALLS];
  const weights = (call) => c[FIRST] + ((c[START] + call) % c[COPIES]) * c[STRIDE];
  if (threads === 1) {
    for (let call = 0; call < calls; call++) if (index === 0) tiles(k, c, weights(call), 0, rows);
    return;
  }
  // chunk j of call m is number m * chunks + j, taken in turn (no wait between the calls: the threads may be on two
  // calls at once near the end of one, on different copies)
  for (let at = Atomics.add(c, NEXT, 1); at < calls * chunks; at = Atomics.add(c, NEXT, 1)) {
    const r0 = (at % chunks) * size;
    tiles(k, c, weights(Math.floor(at / chunks)), r0, Math.min(r0 + size, rows));
  }
}
const instances = (memory) => FORMS.map((form) => new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${work}${form}/relaxed-shared.wasm`)), { env: { memory } }).exports);

if (!isMainThread) {
  const { memory, control, index } = workerData;
  const c = new Int32Array(control);
  const forms = instances(memory);
  let seen = 0;
  for (;;) {
    Atomics.wait(c, GO, seen);
    seen = Atomics.load(c, GO);
    if (seen < 0) break;
    if (index < c[THREADS]) runCalls(forms[c[FORM]], c, index);
    Atomics.add(c, DONE, 1);
    Atomics.notify(c, DONE);
  }
  process.exit(0);
}

const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 2), turns = option("--turns", 5), megabytes = option("--megabytes", 512);
// main's kernels (CI checks out one commit: fetch main's)
try { execFileSync("git", ["fetch", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
for (const form of FORMS) {
  const dir = `${work}${form}/`;
  fs.mkdirSync(dir, { recursive: true });
  for (const file of ["kernel_relaxed.ts", "six.ts"]) {
    fs.writeFileSync(dir + file, form === "main" ? execFileSync("git", ["show", `origin/main:kernels/${file}`], { cwd: root, encoding: "utf8" })
      : fs.readFileSync(`${root}kernels/${file}`, "utf8"));
  }
  execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
    "--sharedMemory", "--maximumMemory", "32768", dir + "kernel_relaxed.ts", "-o", dir + "relaxed-shared.wasm",
    "--enable", "simd,relaxed-simd,threads"], { cwd: root, stdio: "inherit" });
}
const listed = args.includes("--shapes") ? args[args.indexOf("--shapes") + 1].split(",").map((shape) => shape.split("x").map(Number)) : null;
const shapes = listed ?? [[512, 512], [2048, 512], [512, 2048], [896, 896], [4864, 896], [896, 4864], [2048, 2048], [8192, 2048],
  [2048, 8192], [2560, 2560], [3072, 3072], [3584, 3584], [4096, 4096]];
const most = Math.max(...shapes.map(([rows, n]) => rows * n * 1.4));
const pages = Math.ceil((Math.max(megabytes * 2 ** 20, 2 * most) + 8 * 2 ** 20) / 65536);
const memory = new WebAssembly.Memory({ initial: pages, maximum: 32768, shared: true });
const control = new SharedArrayBuffer(32 * 4), c = new Int32Array(control);
instances(memory);
const workers = Array.from({ length: 4 }, (_, index) => new Worker(new URL(import.meta.url), { workerData: { memory, control, index } }));
const cpu = () => {
  const model = os.cpus()[0]?.model;
  if (model && model !== "unknown") return model;
  const part = (fs.existsSync("/proc/cpuinfo") ? fs.readFileSync("/proc/cpuinfo", "utf8") : "").match(/CPU part\s*:\s*(0x[0-9a-f]+)/)?.[1];
  return part ? `arm64 ${{ "0xd0c": "Neoverse-N1", "0xd49": "Neoverse-N2", "0xd40": "Neoverse-V1", "0xd4f": "Neoverse-V2" }[part] ?? part}` : "unknown";
};
console.log(`${cpu()}, ${os.cpus().length} logical cores, ${megabytes ? `matrices read from ${megabytes} MB of copies` : "one matrix again and again (in the caches)"}`);

let gen = 0;
function go() {
  Atomics.store(c, DONE, 0);
  Atomics.store(c, NEXT, 0);
  Atomics.store(c, GO, ++gen);
  Atomics.notify(c, GO);
  for (let done = Atomics.load(c, DONE); done < workers.length; done = Atomics.load(c, DONE)) Atomics.wait(c, DONE, done, 1000);
}
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
const summary = {}, errors = { main: [], tree: [] };
for (const [rows, n] of shapes) {
  const ng = n / 32, frame = Math.ceil((n + ng * 4) / 64) * 64, outFrame = Math.ceil(rows * 4 / 64) * 64;
  const frames = 65536, outs = [frames + 16 * frame, frames + 16 * frame + 16 * outFrame, frames + 16 * frame + 32 * outFrame];
  const first = Math.ceil((outs[2] + 16 * outFrame) / 65536) * 65536;
  const stride = Math.ceil((rows * n + 3 * rows * ng * 4) / 4096) * 4096;
  const copies = megabytes ? Math.max(2, Math.floor((memory.buffer.byteLength - first) / stride)) : 1;
  const I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N32 = new Int32Array(memory.buffer), U = new Uint8Array(memory.buffer);
  let seed = 5;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  const scales = first + rows * n, mainSums = scales + rows * ng * 4, treeSums = mainSums + rows * ng * 4;
  for (let i = 0; i < rows * n; i++) I[first + i] = (next() & 255) - 128;
  for (let i = 0; i < rows * ng; i++) {
    F[scales / 4 + i] = Math.fround(1e-3 * (1 + (next() % 1000)));
    let sum = 0;
    for (let j = 0; j < 32; j++) sum += I[first + i * 32 + j];
    F[mainSums / 4 + i] = Math.fround(F[scales / 4 + i] * sum);  // before T197: float32 scale × sum
    N32[treeSums / 4 + i] = -64 * sum;  // from T197: int32 −64 × sum
  }
  for (let copy = 1; copy < copies && first + (copy + 1) * stride <= memory.buffer.byteLength; copy++) U.copyWithin(first + copy * stride, first, first + stride);
  for (let t = 0; t < 16; t++) {
    for (let j = 0; j < n; j++) I[frames + t * frame + j] = next() % 128;
    for (let g = 0; g < ng; g++) F[(frames + t * frame + n) / 4 + g] = Math.fround(1e-2 * (1 + (next() % 1000)));
  }
  Object.assign(c, { [FIRST]: first, [N]: n, [ROWS]: rows, [FRAMES]: frames, [FRAME]: frame, [OUT_FRAME]: outFrame, [STRIDE]: stride, [COPIES]: copies });
  // the float64 sums of the first 4 tokens: dot(w, q − 64) of each group times the two scales
  const reference = [];
  for (let t = 0; t < 4; t++) {
    const at = frames + t * frame;
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let g = 0; g < ng; g++) {
        let dot = 0;
        for (let j = g * 32; j < g * 32 + 32; j++) dot += I[first + i * n + j] * (I[at + j] - 64);
        sum += dot * F[scales / 4 + i * ng + g] * F[(at + n) / 4 + g];
      }
      reference.push(sum);
    }
  }
  const largest = Math.max(...reference.map(Math.abs));
  const lines = [];
  for (const count of [2, 4, 6, 8, 16]) {
    c[COUNT] = count;
    // each form the same to the bit on one thread and on four; at 4 tokens, each against the float64 sums
    for (const form of [0, 1]) {
      c[FORM] = form;
      for (const [slot, threads] of [[0, 1], [1, 4]]) {
        c[THREADS] = threads; c[CALLS] = 1; c[START] = 0;
        F.fill(-7, outs[slot] / 4, (outs[slot] + 16 * outFrame) / 4);
        c[OUT] = outs[slot];
        go();
      }
      for (let t = 0; t < 16; t++) {
        for (let i = 0; i < rows; i++) {
          if (!Object.is(F[(outs[0] + t * outFrame) / 4 + i], F[(outs[1] + t * outFrame) / 4 + i])) {
            throw new Error(`${rows} x ${n}, ${count} tokens, ${FORMS[form]}: 4 threads differ from 1 at row ${i}, token ${t}`);
          }
        }
      }
      if (count === 4) {
        let worst = 0;
        for (let t = 0; t < 4; t++) {
          for (let i = 0; i < rows; i++) worst = Math.max(worst, Math.abs(F[(outs[0] + t * outFrame) / 4 + i] - reference[t * rows + i]));
        }
        errors[FORMS[form]].push(worst / largest);
        lines.push(`${rows} x ${n}, ${FORMS[form]}: against float64 ${(worst / largest).toExponential(2)} (of the largest |output| ${largest.toExponential(2)})`);
      }
    }
    for (const threads of [1, 4]) {
      c[THREADS] = threads;
      const macs = rows * n * count, calls = Math.max(megabytes ? 4 : 1, Math.round((threads === 1 ? 2e8 : 6e8) / macs));
      c[CALLS] = calls;
      let start = 0;
      const time = (form) => {
        c[FORM] = form; c[OUT] = outs[2]; c[START] = start; start = (start + calls) % copies;
        const t0 = performance.now();
        go();
        return (performance.now() - t0) / calls;
      };
      time(0); time(1);
      const speeds = [[], []];
      for (let r = 0; r < rounds; r++) {
        const ts = [[], []];
        for (let t = 0; t < turns; t++) { ts[0].push(time(0)); ts[1].push(time(1)); }
        ts.forEach((list, i) => speeds[i].push(macs / median(list) / 1e6));
      }
      const ratios = speeds[1].map((s, r) => s / speeds[0][r]), lo = Math.min(...ratios).toFixed(2), hi = Math.max(...ratios).toFixed(2);
      (summary[`${count} tokens, ${threads} thread${threads > 1 ? "s" : ""}`] ??= []).push(median(ratios));
      lines.push(`${rows} x ${n}, ${count} tokens, ${threads} thread${threads > 1 ? "s" : ""}: main ${median(speeds[0]).toFixed(1)} G MAC/s, ` +
        `tree ${median(speeds[1]).toFixed(1)} G MAC/s, ${lo === hi ? lo : `${lo}–${hi}`}x`);
    }
  }
  console.log(lines.join("\n"));
}
const geo = (list) => Math.exp(list.reduce((a, b) => a + Math.log(b), 0) / list.length);
for (const [key, list] of Object.entries(summary)) console.log(`${key}: the tree's tile against main's, geometric mean over the shapes ${geo(list).toFixed(3)}x, least ${Math.min(...list).toFixed(3)}x`);
for (const [form, list] of Object.entries(errors)) console.log(`${form}: against float64 (4 tokens), largest over the shapes ${Math.max(...list).toExponential(2)}, geometric mean ${geo(list).toExponential(2)}`);
Atomics.store(c, GO, -1);
Atomics.notify(c, GO);
process.exit(0);
