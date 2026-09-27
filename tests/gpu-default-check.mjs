// T148 (the review's must-fix 2): the page's default choice of the GPU or the CPU for the blocks of a prompt (not the
// tests' gpuForce.always), on forward.js's real CPU forward pass of llm-jp-3 150M, with a made-up GPU's worker that only
// sleeps for as long as its block would take and answers in the control area (no WebGPU: Node alone). Its times are
// set as multiples of the CPU's own ms a prompt token, measured first, so that the verdicts do not hang on this
// machine's speed. Each generation feeds a prompt as Python does (promptBlock at a time). Checked:
//   - a GPU far faster: the first prompt on the CPU (it is timed first), then every block on the GPU; the eighth
//     generation after the first verdict times the CPU again on the prompt's last 32 tokens only; a short prompt
//     stays on the CPU; another number of threads is timed on the CPU again;
//   - a GPU far slower: every prompt on the CPU, but the eighth after the first verdict, whose first block of 64 goes to
//     the GPU once;
//   - a GPU faster from 36 tokens on (the band of 17 to 64 where the first version of T148 put the prompt on blocks of
//     16 on the GPU in its check, 1.47 times as long): the generation that checks again takes no longer than the others.
//   node tests/gpu-default-check.mjs [--forward <another forward.js, to see a broken one fail>]
import fs from "node:fs";
import path from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : path.join(root, "public", "forward.js");

// the made-up GPU: ready with the times of a block of 16 and of 64 on its line (fixed + a token), then each block a
// sleep of that long and the answer
const FAKE = `
const { parentPort, workerData: line } = require("node:worker_threads");
let ctl, words;
const nap = new Int32Array(new SharedArrayBuffer(4)), ms = (n) => line.fixed + line.perToken * n;
parentPort.on("message", (data) => {
  if (data.type === "start") {
    ctl = new Int32Array(data.memory.buffer, 0, 2048);
    words = data.plan.words;
    parentPort.postMessage({ type: "ready", adapter: "made up", key: "k", bytes: 1, seconds: 0, form: "made up", attention: "made up",
      forms: [], remembered: false, blocks: [{ count: 16, ms: ms(16) }, { count: 64, ms: ms(64) }] });
  } else if (data.type === "prompt") {
    Atomics.wait(nap, 0, 0, ms(data.count));
    if (Atomics.load(ctl, words.wanted) !== data.serial) return;
    Atomics.store(ctl, words.failed, 0);
    Atomics.store(ctl, words.done, data.serial);
    Atomics.notify(ctl, words.done);
  } else if (data.type === "stop") process.exit(0);
});`;

if (isMainThread) {
  const { pyodideWithEngine } = await import("./engine.mjs");
  const { MODELS } = await import("../src/models.js");
  const { weightsMemory, footprint } = await import(forwardFile);
  const entry = MODELS.find((m) => m.id === "llm-jp-3-150m");
  const { pyodide: py } = await pyodideWithEngine();
  const checkpoint = fs.readFileSync(path.join(root, entry.checkpoint));
  // the plan forward.js gets from Python (where every tensor is), recorded
  let plan;
  py.globals.set("OUTSIDE", { size: checkpoint.length, read: (o, l) => new Uint8Array(checkpoint.buffer, checkpoint.byteOffset + o, l).slice(),
    start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } });
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(path.join(root, entry.tokenizer)));
  py.globals.set("OPTIONS", py.toPy(entry.options ?? {}));
  py.runPython(`from llama2_numpy import Llama\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS).release()`);
  const header = [plan.dim, plan.hidden_dim, plan.n_layers, plan.n_heads, plan.n_kv_heads, plan.vocab_size, plan.seq_len];
  const { memory, base } = weightsMemory(checkpoint.length, { shared: true, after: footprint(header, checkpoint.length, { dtype: "int8", halfKV: true, gpu: true, kvStart: plan.kv_start }) });
  new Uint8Array(memory.buffer).set(checkpoint, base);
  // forward.js waits in Atomics.wait: in a worker, as in the page
  const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size: checkpoint.length, plan, forwardFile } });
  const code = await new Promise((resolve) => {
    worker.on("message", (line) => console.log(line));
    worker.once("exit", resolve);
    worker.once("error", (error) => { console.error(`FAILED\n- ${error.stack ?? error}`); resolve(1); });
  });
  process.exit(code);
} else {
  const { memory, base, size, plan, forwardFile } = workerData;
  const { compileKernels, createForward } = await import(forwardFile);
  const kernels = compileKernels(fs.readFileSync(path.join(root, "public/simdkernel_shared.wasm")), fs.readFileSync(path.join(root, "public/simdkernel_relaxed_shared.wasm")));
  const spawn = (data) => new Promise((resolve) => {
    const helper = new Worker(path.join(root, "public/helper.js"));
    helper.once("message", () => resolve({ terminate: () => helper.terminate() }));
    helper.postMessage(data);
  });
  const say = (line) => parentPort.postMessage(line);
  const failures = [];
  console.info = () => {};  // forward.js's verdicts: the harness reads what ran where instead
  const PROMPT = 230;  // fed: blocks of 64, 64, 64 and 38
  const prompt = (count) => [plan.bos ?? 1, ...Array.from({ length: count - 1 }, (_, i) => 100 + (i % 500))];
  // one generation's prompt as Python hands it over: [ms, tokens on the GPU]
  const feed = (engine, count = PROMPT) => {
    engine.newGeneration();
    const fed = prompt(count), began = performance.now();
    for (let at = 0; at < fed.length;) {
      const block = fed.slice(at, at + engine.promptBlock);
      engine.forwardMany(block, at);
      at += block.length;
    }
    return [performance.now() - began, engine.gpuTokens];
  };
  // the CPU's ms a prompt token here, on one thread (the median of five prompts of a GPU-less engine)
  const alone = createForward({ memory, base, size, kernels, plan });
  const perToken = [0, 1, 2, 3, 4].map(() => feed(alone)[0] / PROMPT).sort((a, b) => a - b)[2];
  alone.release();
  say(`the CPU: ${perToken.toFixed(2)} ms a prompt token`);

  const run = async (name, fixed, tokenCost, generations, after) => {
    const line = { fixed: fixed * perToken, perToken: tokenCost * perToken };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: line });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); } };
    };
    const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu });
    await engine.setThreads(1);
    await engine.gpu;  // ready before the first prompt
    const seen = [];
    for (let g = 1; g <= generations; g++) seen.push(feed(engine));
    const more = after ? await after(engine) : {};
    engine.release();
    say(`${name}: tokens on the GPU ${seen.map(([, t]) => t).join(" ")}; ms ${seen.map(([ms]) => ms.toFixed(0)).join(" ")}` +
      `${Object.keys(more).length ? `; ${JSON.stringify(more)}` : ""}; status ${engine.gpuStatus}`);
    return { seen, more };
  };
  const expect = (what, got, want) => {
    if (JSON.stringify(got) !== JSON.stringify(want)) failures.push(`${what}: ${JSON.stringify(got)}, not ${JSON.stringify(want)}`);
  };
  const all = (from, to, value) => Array.from({ length: to - from + 1 }, () => value);

  // a GPU far faster: 10 ms-a-token of fixed cost, a twentieth of the CPU's a token (faster from 12 tokens on)
  {
    const { seen, more } = await run("a GPU far faster", 10, 0.05, 12, async (engine) => {
      const short = feed(engine, 6)[1];
      await engine.setThreads(2);
      return { short, twoThreads: [feed(engine)[1], feed(engine)[1]] };
    });
    const tokens = seen.map(([, t]) => t);
    expect("the first prompt on the CPU", tokens[0], 0);
    expect("then every block on the GPU", tokens.slice(1, 8), all(2, 8, PROMPT));
    expect("the eighth generation after the first verdict: the last 32 tokens on the CPU", tokens[8], PROMPT - 38);
    expect("and the GPU again", tokens.slice(9), all(10, 12, PROMPT));
    expect("a short prompt on the CPU", more.short, 0);
    expect("two threads: the CPU timed again, then the GPU", more.twoThreads, [0, PROMPT]);
  }
  // a GPU far slower: 60 fixed and twice the CPU's a token
  {
    const tokens = (await run("a GPU far slower", 60, 2, 12)).seen.map(([, t]) => t);
    expect("every prompt on the CPU but the GPU's check, its first block", tokens, [0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0]);
  }
  // a GPU faster from 36 tokens on (30 fixed, a tenth of the CPU's a token): whole blocks on the GPU, the last block of
  // 38 either way; the check must cost little (the first version of T148: about three times here)
  {
    const { seen } = await run("a GPU faster from 36 tokens", 30, 0.1, 12);
    const normal = seen.slice(1, 8).map(([ms]) => ms).sort((a, b) => a - b)[3], check = seen[8][0];
    expect("the check: the last 32 tokens on the CPU", seen[8][1], PROMPT - 38);
    if (!(check <= 1.5 * normal)) failures.push(`the check took ${check.toFixed(0)} ms, the others ${normal.toFixed(0)} (more than 1.5 times)`);
  }
  if (failures.length) say(`FAILED\n- ${failures.join("\n- ")}`);
  else say("ok");
  process.exit(failures.length ? 1 : 0);
}
