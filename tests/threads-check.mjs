// T93 stage 2: the forward pass of forward.js with helper threads on a shared memory. The forward pass runs in a
// worker, as on the page (the model's worker coordinates); run from the main thread next to Pyodide it waited on
// its helpers several times longer. Python in the main thread only makes the plan.
//   1. Every row is computed whole by one thread with the same kernel, so the logits of a greedy run must be the
//      same to the bit with any number of threads.
//   2. The speeds, the counts in turn, several rounds, the median.
//   3. The search (stage 2b): which count it chooses from navigator.hardwareConcurrency, within a few generations,
//      against the fastest of 2.
//   4. T108: the same text as a prompt in blocks (forwardMany): the last logits the same to the bit with every
//      count, and what a token of a prompt costs in blocks of 1, 2, 4, 8 and 16, with each count.
//
//   node tests/threads-check.mjs [model id ...] [--threads 1,2,4,8] [--rounds 5] [--positions 64] [--kv-start 16]
//        [--from 0] [--without kv16]
//
// --from: the speeds of step 2 are measured from this position on, the KV cache filled up to it first (T109: the
// attention of a long context).
//
// --without: the optimizations to leave out (the engine's disable, T52), e.g. kv16 for a float32 cache (T160).
//
// --kv-start: the KV cache starts this small (the page's KV_START is 256), so that it has to grow, and move, under
// the helper threads within the positions of a run (Fable's review of T93).
import fs from "node:fs";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { compileKernels, createForward, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
// a helper thread: resolves once it has its kernels on the shared memory
const spawn = (data) => new Promise((resolve) => {
  const worker = new Worker(new URL("../public/helper.js", import.meta.url));
  worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
  worker.postMessage(data);
});

if (isMainThread) {
  const { pyodideWithEngine } = await import("./engine.mjs");
  const { MODELS } = await import("../src/models.js");
  const args = process.argv.slice(2);
  const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
  const counts = option("--threads", "1,2,4,8").split(",").map(Number);
  const rounds = Number(option("--rounds", 5)), positions = Number(option("--positions", 64)), kvStart = Number(option("--kv-start", 16));
  const from = Number(option("--from", 0));
  const without = option("--without", "").split(",").filter(Boolean);
  const wide = args.includes("--wide");
  // T101: --high (with --wide) puts the checkpoint 4 GiB up a 64-bit memory, so that every address the forward pass
  // uses is past 2^32 (the pages below are never touched, so they cost no memory)
  const high = wide && args.includes("--high") ? 2 ** 32 : 0;
  const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
  const { pyodide: py } = await pyodideWithEngine();
  let failed = false;
  for (const id of ids.length ? ids : ["tiny-lm", "llm-jp-3-150m"]) {
    // a model of the list, or <out> of tests/perplexity_prepare.py (or quantize.py: <out>.bin, .tokenizer.bin, .json)
    const entry = MODELS.find((m) => m.id === id) ?? { name: path.basename(id), checkpoint: path.resolve(`${id}.bin`),
      tokenizer: path.resolve(`${id}.tokenizer.bin`), options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) };
    const file = (f) => (path.isAbsolute(f) ? f : root + f);
    const checkpoint = fs.readFileSync(file(entry.checkpoint));
    const { memory, base: low } = weightsMemory(checkpoint.length + high, { shared: true, wide });
    const base = low + high;
    new Uint8Array(memory.buffer).set(checkpoint, base);
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(file(entry.tokenizer)));
    let plan;
    // Python says where every tensor is; the forward pass itself is made in the worker
    const outside = { size: checkpoint.length, read: (o, l) => new Uint8Array(memory.buffer, base + o, l).slice(),
      start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } };
    py.globals.set("OUTSIDE", outside);
    py.globals.set("OPTIONS", py.toPy(entry.options));
    py.runPython(`import llama2_numpy\nfrom llama2_numpy import Llama\nllama2_numpy.KV_START = ${kvStart}\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, disable=${JSON.stringify(without)}, **OPTIONS)`);
    const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size: checkpoint.length, plan, counts, rounds, positions, from, wide } });
    const result = await new Promise((resolve, reject) => { worker.once("message", resolve); worker.once("error", reject); });
    console.log(`${entry.name}: ${result}`);
    failed ||= result.includes("DIFFER");
    await worker.terminate();
  }
  process.exit(failed ? 1 : 0);
} else {
  const { memory, base, size, plan, counts, rounds, positions, from, wide } = workerData;
  const suffix = wide ? "64" : "";  // T101: a 64-bit memory and its kernels
  const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_shared${suffix}.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_shared${suffix}.wasm`), wide);
  const engine = createForward({ memory, base, size, kernels, plan, spawn });
  const greedy = () => {
    const seen = [];
    let token = plan.bos ?? 1;
    for (let pos = 0; pos < positions; pos++) {
      engine.forward(token, pos, true);
      const logits = engine.logits();
      seen.push(logits.slice());
      token = logits.indexOf(Math.max(...logits));
    }
    return seen;
  };
  let reference, differ = [];
  for (const n of counts) {
    await engine.setThreads(n);
    const seen = greedy();
    if (!reference) reference = seen;
    else if (!seen.every((row, i) => row.every((v, j) => Object.is(v, reference[i][j])))) differ.push(n);
  }
  // T108: the greedy text fed back as a prompt, in blocks
  const text = [plan.bos ?? 1, ...reference.slice(0, -1).map((logits) => logits.indexOf(Math.max(...logits)))];
  const blocksDiffer = [];
  for (const n of counts) {
    await engine.setThreads(n);
    for (let at = 0; at < positions - 1; at += 5) engine.forwardMany(text.slice(at, Math.min(at + 5, positions - 1)), at);
    engine.forward(text[positions - 1], positions - 1, true);
    if (!engine.logits().every((v, j) => Object.is(v, reference[positions - 1][j]))) blocksDiffer.push(n);
  }
  const blockSizes = [1, 2, 4, 8, 16], prompt = Object.fromEntries(counts.map((n) => [n, Object.fromEntries(blockSizes.map((k) => [k, []]))]));
  for (let r = 0; r < rounds; r++) {
    for (const n of counts) {
      await engine.setThreads(n);
      for (const k of blockSizes) {
        const began = performance.now();
        for (let at = 0; at < positions; at += k) engine.forwardMany(text.slice(at, Math.min(at + k, positions)).map((t) => t ?? 1), at);
        prompt[n][k].push((positions * 1000) / (performance.now() - began));
      }
    }
  }
  const times = Object.fromEntries(counts.map((n) => [n, []]));
  for (let r = 0; r < rounds; r++) {
    for (const n of counts) {
      await engine.setThreads(n);
      if (from) engine.forwardMany(new Array(from).fill(1), 0);  // the KV cache up to where the timing starts
      for (let pos = from; pos < from + 8; pos++) engine.forward(1, pos, true);  // the helpers wake up
      const began = performance.now();
      for (let pos = from; pos < from + positions; pos++) engine.forward(1, pos, true);
      times[n].push((positions * 1000) / (performance.now() - began));
    }
  }
  const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
  const one = median(times[counts[0]]);
  // the search, as the page runs it: from the hint, over generations of `positions` tokens
  const hint = globalThis.navigator?.hardwareConcurrency ?? 4;
  let found = 0, generationsUsed = 0;
  const verdicts = [];  // T114: what the page writes to the console, one per comparison
  await engine.findThreads({ from: hint, chose: (n) => { found = n; }, compared: (verdict) => verdicts.push(verdict) });
  for (; !found && generationsUsed < 8; generationsUsed++) {
    engine.newGeneration();
    for (let pos = 0; pos < positions; pos++) engine.forward(1, pos, true);
    await new Promise((resolve) => setTimeout(resolve, 0));  // let a helper that is being started come up
  }
  if (verdicts.length !== engine.searchLog.length || verdicts.some((v, i) => v.best !== engine.searchLog[i].best
      || v.candidate !== engine.searchLog[i].candidate || v.faster !== engine.searchLog[i].faster || !(v.bestMs > 0) || !(v.candidateMs > 0))) {
    throw new Error(`the verdicts told (${JSON.stringify(verdicts)}) are not the search's log`);
  }
  const fastest = counts.reduce((a, b) => (median(times[b]) > median(times[a]) ? b : a));
  const comparisons = engine.searchLog.map(({ best, candidate, times, faster }) =>
    `${best} [${times[best].map((t) => t.toFixed(2)).join(" ")}] vs ${candidate} [${times[candidate].map((t) => t.toFixed(2)).join(" ")}] ${faster ? "->" : "stay"}`).join("; ");
  const searchLine = `the search from ${hint} chose ${found || "nothing"} in ${generationsUsed} generation(s) (${comparisons}); the fastest measured was ${fastest}` +
    (found && times[found] ? ` (${found} runs at ${(median(times[found]) / median(times[fastest]) * 100).toFixed(0)}% of it)` : "");
  const promptLine = "a prompt in blocks of " + blockSizes.join(", ") + ": " + counts.map((n) =>
    `${n} thread(s) ${blockSizes.map((k) => median(prompt[n][k]).toFixed(0)).join(" / ")} tok/s (${(median(prompt[n][16]) / median(prompt[n][1])).toFixed(2)}×)`).join(", ");
  // T96: the page keeps one memory for model after model. A second engine on this memory, after the first has
  // ended its threads, must run as the first did (the control area holds what the first left there).
  engine.stopThreads();
  const again = createForward({ memory, base, size, kernels, plan, spawn });
  await again.setThreads(counts[counts.length - 1]);
  let token = plan.bos ?? 1, reused = true;
  for (let pos = 0; pos < positions && reused; pos++) {
    again.forward(token, pos, true);
    const logits = again.logits();
    reused = logits.every((v, j) => Object.is(v, reference[pos][j]));
    token = logits.indexOf(Math.max(...logits));
  }
  again.stopThreads();
  // The review of T120: helpers that come up after their engine let its helpers go (a visitor choosing another model
  // while the search starts more) are ended at once, not left for the next engine on this memory to wake
  let alive = 0;
  const counted = (data) => spawn(data).then((helper) => {
    alive += 1;
    return { terminate: () => { alive -= 1; return helper.terminate(); } };
  });
  const released = createForward({ memory, base, size, kernels, plan, spawn: counted });
  const starting = released.setThreads(4);  // three helpers, one after another
  released.release();
  const late = (await starting) === 1 && alive === 0;
  // T120: a software thread that the browser stops in the middle of its chunk (iOS may end a worker for memory):
  // this one takes a chunk and ends without counting it. The coordinator must give its helpers up, run the phase
  // again on its own, and go on with one thread, to the same logits
  const { WAKE, COUNTER, ACTIVE, CONTROL_BYTES } = await import("../public/jobs.js");
  const dying = ({ memory: shared, share }) => new Promise((resolve) => {
    const worker = new Worker(`
      const { parentPort, workerData: { memory, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } } = require("node:worker_threads");
      const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
      parentPort.postMessage("ready");
      for (let gen = Atomics.load(ctl, WAKE + share); ; gen = Atomics.load(ctl, WAKE + share)) {
        Atomics.wait(ctl, WAKE + share, gen);
        if (Atomics.load(ctl, WAKE + share) & 1) continue;
        Atomics.add(ctl, ACTIVE, 1);
        Atomics.add(ctl, COUNTER, 1);  // a chunk taken, never done
        process.exit(0);
      }`, { eval: true, workerData: { memory: shared, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } });
    worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
  });
  const stopped = createForward({ memory, base, size, kernels, plan, spawn: dying, stalledMs: 500 });
  await stopped.setThreads(2);
  let went = true;
  const warned = console.warn;
  console.warn = () => {};  // the one line forward.js writes about it
  token = plan.bos ?? 1;
  for (let pos = 0; pos < positions && went; pos++) {
    stopped.forward(token, pos, true);
    const logits = stopped.logits();
    went = logits.every((v, j) => Object.is(v, reference[pos][j]));
    token = logits.indexOf(Math.max(...logits));
  }
  console.warn = warned;
  went &&= stopped.lostThreads && stopped.threads === 1 && (await stopped.setThreads(4)) === 1;
  stopped.stopThreads();
  parentPort.postMessage(`${went ? "a software thread that stops mid-chunk is given up and the text is the same; " : "a software thread that stops mid-chunk DIFFERS or hangs; "}` +
    `${late ? "helpers that come up after a release are ended; " : "helpers that come up after a release are left alive (DIFFERS); "}` +
    `${reused ? "a second engine on the same memory runs the same; " : "a second engine on the same memory DIFFERS or hangs; "}${engine.backend}: ${differ.length ? `logits DIFFER with ${differ.join(", ")} threads` : `logits the same to the bit with ${counts.join(", ")} threads`}; ` +
    `${blocksDiffer.length ? `the prompt in blocks DIFFERS with ${blocksDiffer.join(", ")} threads` : "the prompt in blocks the same to the bit"}; ` +
    counts.map((n) => `${n}: ${median(times[n]).toFixed(1)} tok/s (${(median(times[n]) / one).toFixed(2)}×)`).join(", ") + `; ${promptLine}; ${searchLine}`);
  engine.stopThreads();
}
