// T189: the sampling kernel (kernels/kernel.ts's sample) of main and of this tree, and of any commits given, in one
// process, taking turns (AGENTS.md: the old and the new side by side), on the real logits of the models' generation.
// Every form must draw main's token for the same random number (the same logits, the same penalty): it is checked on
// every position with several random numbers before anything is timed. Compiles each form's kernel.ts with
// AssemblyScript into .tmp/sample-bench/ (needs `npm ci` and `make models kernels`).
//
//   node tests/sample-bench.mjs [model id ...] [--rounds 5] [--tokens 128] [--commits <sha>,<sha>]
//
// The logits: generate() as tests/overhead.mjs runs it, one position after another from BOS with the model's
// temperature, top-p and repetition penalty (penalized, as sample() gets them), the tokens drawn by the page's kernel.
// Per model, µs a call, the median of the rounds (the first round warms up and is left out):
//   sample     each position's logits copied into the kernel's array (which warms them, as T164's "again") and one
//              call; the forms take turns at each position, in an order that turns round
//   walk       the logits of which the best alone passes the floor (T164's "1 past"), 200 calls in a loop: what
//              walking the whole vocabulary costs
// and what the floor and llama2.c's cutoff leave, the median of the positions (computed here in float64, as the
// kernel does to within a rounding): C past the floor, L past the cutoff (what sortNucleus() sorts), K the nucleus.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const rounds = Number(option("--rounds", 5)), tokens = Number(option("--tokens", 128));
const commits = option("--commits", "").split(",").filter(Boolean);
const staged = args.includes("--stages");
const ids = args.filter((a, i) => !a.startsWith("--") && !["--rounds", "--tokens", "--commits"].includes(args[i - 1]));
const work = root + ".tmp/sample-bench/";

// main's kernels (CI checks out one commit: fetch main's, and the commits asked for)
const git = (...a) => execFileSync("git", a, { cwd: root, encoding: "utf8" });
try { execFileSync("git", ["fetch", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
for (const sha of commits) { try { execFileSync("git", ["fetch", "--depth=1", "origin", sha], { cwd: root, stdio: "inherit" }); } catch {} }
const forms = { main: (file) => git("show", `origin/main:kernels/${file}`) };
for (const sha of commits) forms[sha.slice(0, 7)] = (file) => git("show", `${sha}:kernels/${file}`);
forms.tree = (file) => fs.readFileSync(`${root}kernels/${file}`, "utf8");
// --stages: the tree's sample() made to return after each of its steps (stop_at(s)), to see what each step costs
const STAGES = ["the best", "the floor", "exp()", "the total", "the cutoff", "the sort"];
const ANCHORS = ["  const nucleus = topp > 0 && topp < 1;", "  const inverse = f32x4.splat(<f32>1.0 / temperature);", "  let total: f64 = 0;",
  "  let last = count - 1;", "    // the most probable tokens whose probabilities add up to topp", "  const target: f64 = random * mass;"];
if (staged) {
  forms.stages = (file) => {
    let text = forms.tree(file);
    if (file !== "kernel.ts") return text;
    ANCHORS.forEach((anchor, s) => {
      if (text.split(anchor).length !== 2) throw new Error(`--stages: kernel.ts's sample() has no single ${JSON.stringify(anchor)}`);
      text = text.replace(anchor, `  if (stopAt == ${s}) return ${s};\n${anchor}`);
    });
    return text + "\nlet stopAt: i32 = -1;\nexport function stop_at(s: i32): void { stopAt = s; }\n";
  };
}

const memory = new WebAssembly.Memory({ initial: 1, maximum: 16384 });
const kernels = {};
for (const [name, read] of Object.entries(forms)) {
  const dir = `${work}${name}/`;
  fs.mkdirSync(dir, { recursive: true });
  for (const file of ["kernel.ts", "six.ts"]) fs.writeFileSync(dir + file, read(file));
  execFileSync("npx", ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
    dir + "kernel.ts", "-o", dir + "plain.wasm", "--enable", "simd"], { cwd: root, stdio: "inherit" });
  kernels[name] = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(dir + "plain.wasm")), { env: { memory } }).exports;
}
const names = Object.keys(kernels).filter((n) => n !== "stages");

const { pyodide: py } = await pyodideWithEngine();
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
console.log(`${os.cpus()[0]?.model ?? "?"} (${process.arch}) × ${os.cpus().length}, Node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}; ` +
  `${rounds} rounds, ${tokens} tokens; forms ${names.join(", ")}`);
const rows = [], stageRows = [];
for (const id of ids.length ? ids : ["llm-jp-3-150m", "tiny-lm"]) {
  const entry = MODELS.find((m) => m.id === id);
  if (!entry) throw new Error(`${id} is not a model of src/models.js`);
  py.globals.set("CHECKPOINT", root + entry.checkpoint);
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
  py.globals.set("OPTIONS", py.toPy(entry.options));
  py.globals.set("GEN", py.toPy(entry.generation));
  const got = py.runPython(`
import numpy as np
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
temperature, topp, penalty = GEN["temperature"], GEN.get("topp", 0.9), GEN.get("repetition_penalty", 1.0)
rng, token, history, saved = np.random.default_rng(1), llama.bos, [llama.bos], []
for pos in range(${tokens}):
    out = llama.forward(token, pos)
    if penalty != 1.0:
        llama.penalize(out, history, penalty)
    saved.append(out.copy())
    token = llama.sample(out, temperature, topp, rng)
    history.append(token)
llama.release(); del llama
(np.stack(saved).tobytes(), saved[0].size, temperature, topp)`);
  const [bytes, V, temperature, topp] = got.toJs();
  got.destroy();
  const all = new Float32Array(new Uint8Array(bytes).buffer);  // a copy of its own, at a multiple of 4

  // the kernel's arrays: the logits, and probs and index (scratch of V each)
  const need = 65536 + 3 * 4 * V + 3 * 64;
  if (need > memory.buffer.byteLength) memory.grow(Math.ceil((need - memory.buffer.byteLength) / 65536));
  const logits = 65536, probs = logits + 4 * V + 64, index = probs + 4 * V + 64;
  const F = () => new Float32Array(memory.buffer, logits, V);
  const at = (pos) => all.subarray(pos * V, (pos + 1) * V);

  // what the floor and the cutoff leave
  const C = [], L = [], K = [];
  for (let pos = 0; pos < tokens; pos++) {
    const x = at(pos);
    let best = -Infinity;
    for (const v of x) best = Math.max(best, v);
    const floor = Math.fround(best - Math.fround(temperature * Math.fround(16.118095)));
    const p = [];
    for (const v of x) if (v >= floor) p.push(Math.fround(Math.exp(Math.fround(Math.fround(v - best) / temperature))));
    const total = p.reduce((a, b) => a + b, 0), top = p.reduce((a, b) => Math.max(a, b), 0);
    const cutoff = Math.min((1 - topp) / Math.max(p.length - 1, 1) * total, top);
    const likely = p.filter((q) => q >= cutoff).sort((a, b) => b - a);
    let mass = 0, k = 0;
    while (k < likely.length && (mass += likely[k]) < topp * total) k++;
    C.push(p.length); L.push(likely.length); K.push(Math.min(k + 1, likely.length));
  }

  // every form draws main's token
  let checked = 0;
  for (let pos = 0; pos < tokens; pos++) {
    for (const r of [0, 0.25, 0.5, 0.75, 0.999999, (pos * 0.6180339887) % 1]) {
      F().set(at(pos));
      const wanted = kernels.main.sample(logits, V, temperature, topp, r, probs, index);
      for (const name of names) {
        F().set(at(pos));
        const token = kernels[name].sample(logits, V, temperature, topp, r, probs, index);
        if (token !== wanted) throw new Error(`${entry.name}: ${name} drew ${token} where main drew ${wanted} (position ${pos}, r ${r})`);
      }
      checked++;
    }
  }

  const alone = new Float32Array(V).fill(-1000);
  alone[V >> 1] = 0;
  const cells = {};
  for (let r = 0; r <= rounds; r++) {
    const spent = Object.fromEntries(names.map((n) => [n, 0]));
    for (let pos = 0; pos < tokens; pos++) {
      for (let t = 0; t < names.length; t++) {
        const name = names[(pos + t) % names.length];
        F().set(at(pos));
        const began = performance.now();
        kernels[name].sample(logits, V, temperature, topp, (pos * 0.6180339887) % 1, probs, index);
        spent[name] += performance.now() - began;
      }
    }
    const walk = {};
    for (let t = 0; t < names.length; t++) {
      const name = names[(r + t) % names.length];
      F().set(alone);
      const began = performance.now();
      for (let c = 0; c < 200; c++) kernels[name].sample(logits, V, temperature, topp, 0.5, probs, index);
      walk[name] = (performance.now() - began) / 200 * 1000;
    }
    if (!r) continue;
    for (const name of names) {
      (cells[`sample ${name}`] ??= []).push(spent[name] / tokens * 1000);
      (cells[`walk ${name}`] ??= []).push(walk[name]);
    }
  }
  const m = Object.fromEntries(Object.entries(cells).map(([k, v]) => [k, median(v)]));
  const cell = (kind, name) => `${m[`${kind} ${name}`].toFixed(1)}${name === "main" ? "" : ` (${(m[`${kind} main`] / m[`${kind} ${name}`]).toFixed(2)}×)`}`;
  rows.push(`| ${entry.name} | ${V} | ${median(C)} | ${median(L)} | ${median(K)} | ${names.map((n) => cell("sample", n)).join(" | ")} | ${names.map((n) => cell("walk", n)).join(" | ")} |`);
  if (staged) {
    const stages = kernels.stages, spent = {};
    for (let r = 0; r <= rounds; r++) {
      const got = Array(STAGES.length + 1).fill(0);
      for (let pos = 0; pos < tokens; pos++) {
        for (let t = 0; t <= STAGES.length; t++) {
          const s = (pos + t) % (STAGES.length + 1);
          stages.stop_at(s < STAGES.length ? s : -1);
          F().set(at(pos));
          const began = performance.now();
          stages.sample(logits, V, temperature, topp, (pos * 0.6180339887) % 1, probs, index);
          got[s] += performance.now() - began;
        }
      }
      if (r) got.forEach((v, s) => (spent[s] ??= []).push(v / tokens * 1000));
    }
    const until = Object.values(spent).map(median);
    stageRows.push(`| ${entry.name} | ${until.map((v, s) => `${(s ? v - until[s - 1] : v).toFixed(1)}`).join(" | ")} | ${until[STAGES.length].toFixed(1)} |`);
  }
  console.log(`${entry.name}: every form drew main's token in ${checked} draws (${tokens} positions)`);
}
console.log("The sampling kernel: µs a call, the median of the rounds (× against main's)");
console.log(`| model | vocabulary | C | L | K | ${names.map((n) => `sample ${n}`).join(" | ")} | ${names.map((n) => `walk ${n}`).join(" | ")} |`);
console.log(`|---|${"---:|".repeat(4 + 2 * names.length)}`);
for (const row of rows) console.log(row);
if (staged) {
  console.log("Each step of the tree's sample() (--stages: returning after it), µs a call, the median of the rounds");
  console.log(`| model | ${STAGES.join(" | ")} | the draw | all |`);
  console.log(`|---|${"---:|".repeat(STAGES.length + 2)}`);
  for (const row of stageRows) console.log(row);
}
console.log(`load after: ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
process.exit(0);
