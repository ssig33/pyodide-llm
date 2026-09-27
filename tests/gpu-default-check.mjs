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
// T184 (the review's must-fix): /benchmark/'s page path (forward.js's timePrompts, src/bench.js's pathTable) on the same
// made-up GPU: far faster, every block of the GPU's side on it and a ratio; failing on its sixth block, the GPU's cells
// empty with why and no ratio anywhere (its time was the CPU's: 0.90× before); no GPU, the CPU's side alone.
// T190: the page path's number of threads: a search run to its end (forward.js's endSearch) and timed on the count it
// chose; a count the model page remembers taken with no search.
// T152: the steps of a generation (forward.js's tokenBlock and generateMany, tokenTimes) on the same made-up GPU, whose
// step sleeps a multiple of the CPU's own ms of a token and writes made-up ids: generations of STEPS steps after a short
// prompt, as Python takes them (tokenBlock at a time on the GPU, else one on the CPU). Far faster: the first steps on the
// CPU (timed first), then every one on the GPU, and the eighth generation after the first verdict its first 4 on the
// CPU; far slower: every step on the CPU but that generation's first run; failing on its second request: the CPU from
// there on, the step it gave back taken by the CPU. And Python's generate() through external(), as the page's worker
// runs it, on the far faster one: sampled and greedy, the steps on the GPU and the counts right (the made-up GPU fails a
// request whose random numbers, history or settings are not what Python should hand over); and through Python on one
// that fails on its second request: the generation goes on whole on the CPU (T152's review: JavaScript's null is jsnull).
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
let ctl, words, ids, blocks = 0, requests = 0;
const nap = new Int32Array(new SharedArrayBuffer(4)), ms = (n) => line.fixed + line.perToken * n;
parentPort.on("message", (data) => {
  if (data.type === "start") {
    ctl = new Int32Array(data.memory.buffer, 0, 2048);
    words = data.plan.words;
    // T152: where forward.js asked for the steps and the line has a cost of a step
    ids = data.plan.tokens && line.step !== undefined ? new Int32Array(data.memory.buffer, data.plan.tokens.ids, 1 + data.plan.tokens.most) : null;
    parentPort.postMessage({ type: "ready", adapter: "made up", key: "k", bytes: 1, seconds: 0, form: "made up", attention: "made up",
      forms: [], remembered: false, blocks: [{ count: 16, ms: ms(16) }, { count: 64, ms: ms(64) }],
      ...(ids ? { tokens: { form: "made up", ms: line.step, forms: [] } } : {}) });
  } else if (data.type === "prompt") {
    Atomics.wait(nap, 0, 0, ms(data.count));
    if (Atomics.load(ctl, words.wanted) !== data.serial) return;
    const fail = Boolean(line.failAt) && ++blocks >= line.failAt;  // T184: a GPU that fails on its failAt-th block
    Atomics.store(ctl, words.failed, fail ? 1 : 0);
    Atomics.store(ctl, words.done, data.serial);
    Atomics.notify(ctl, words.done);
  } else if (data.type === "tokens") {
    // T152: count steps of a made-up generation: each id the one after the token fed
    Atomics.wait(nap, 0, 0, line.step * data.count);
    if (Atomics.load(ctl, words.wanted) !== data.serial) return;
    // (T152: what Python hands over, as a step on the GPU takes it: a random number a step where sampled, the end of
    // the history and its length, the settings)
    const odd = data.randoms.length !== (data.settings.temperature ? data.count : 0) || data.history.length > 64 ||
      data.length < data.history.length || data.history.at(-1) !== data.token || !Array.isArray(data.settings.stops);
    const fail = odd || (Boolean(line.failTokensAt) && ++requests >= line.failTokensAt);
    ids[0] = data.count;
    for (let i = 0; i < data.count; i++) ids[1 + i] = data.token + 1 + i;
    Atomics.store(ctl, words.failed, fail ? 1 : 0);
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
  const worker = new Worker(new URL(import.meta.url), { workerData: { memory, base, size: checkpoint.length, plan, forwardFile,
    tokenizer: path.join(root, entry.tokenizer), options: entry.options ?? {} } });
  const code = await new Promise((resolve) => {
    worker.on("message", (line) => console.log(line));
    worker.once("exit", resolve);
    worker.once("error", (error) => { console.error(`FAILED\n- ${error.stack ?? error}`); resolve(1); });
  });
  process.exit(code);
} else {
  const { memory, base, size, plan, forwardFile, tokenizer, options } = workerData;
  const { compileKernels, createForward, endSearch, timePrompts, external } = await import(forwardFile);
  const { pathTable } = await import(path.join(root, "src/bench.js"));
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

  const run = async (name, fixed, tokenCost, generations, after, failAt = 0) => {
    const line = { fixed: fixed * perToken, perToken: tokenCost * perToken, failAt };
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
  // T184: the page path of /benchmark/, as worker.js's timedPaths times it (2 rounds here), and its table
  const paths = (engine) => {
    const ready = engine.gpuReady;  // as timedPaths: what the GPU was before the sides were timed
    const rows = timePrompts(engine, { words: [100, 101, 102, 103, 104], counts: [64, 256], rounds: 2 });
    const gpu = ready ? { seconds: 0, matrices: "made up", attention: "made up", ...(engine.gpuWhyNot ? { lost: engine.gpuWhyNot } : {}) }
      : { why: engine.gpuWhyNot };
    return { rows, table: pathTable({ threads: engine.threads, gpu, status: engine.gpuStatus, rows }) };
  };
  {
    const { more } = await run("the page path, a GPU far faster", 10, 0.05, 0, async (engine) => paths(engine));
    say(more.table);
    expect("the GPU's side all on the GPU", more.rows.map((row) => row.gpu.gpuTokens), [64, 256]);
    expect("as chosen: the GPU", more.rows.map((row) => row.chosen.gpuTokens), [64, 256]);
    expect("the CPU's side on the CPU", more.rows.map((row) => row.cpu.gpuTokens), [0, 0]);
    if ((more.table.match(/×/g) ?? []).length !== 2) failures.push("the page path: a ratio a prompt where the GPU ran them");
  }
  {
    const { more } = await run("the page path, a GPU that fails on its sixth block", 10, 0.05, 0, async (engine) => paths(engine), 6);
    say(more.table);
    // the sixth block is the first of 256's warm-up: 64's GPU side ran whole before it, 256's on the CPU after it
    expect("the GPU's cell of 256 empty, with why", more.rows.map((row) => row.gpu.skip), [undefined, "the GPU failed on a block of the prompt"]);
    if (/×/.test(more.table) || !more.table.includes("WebGPU stopped while timed")) failures.push("the page path: a ratio, or no word of the GPU that stopped");
  }
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    await engine.setThreads(1);
    const { rows, table } = paths(engine);
    engine.release();
    say(table);
    expect("no GPU: the CPU's side alone", rows.map((row) => [row.chosen.same, row.gpu.skip]), [["cpu", "no WebGPU in a worker here"], ["cpu", "no WebGPU in a worker here"]]);
  }
  // T190: the page path is timed on the model page's number of threads. A search from the logical cores runs to its end
  // (endSearch), on generations of few logits each (the owner's Android timed it on 1 thread where the page runs 4:
  // T184 stopped the search after 8 generations); a count the model page remembers is taken as is, with no search
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    let chose = 0, tokens = 0;
    await engine.findThreads({ from: 2, chose: (count) => (chose = count) });
    // a generation of 6 tokens with logits: a comparison takes 20 of them, so more than 3 generations
    const write = () => { for (let pos = 0; pos < 6; pos++, tokens++) engine.forward(100 + pos, pos, true); };
    const ended = await endSearch(engine, write);
    const log = engine.searchLog.map(({ best, candidate, faster }) => `${best} or ${candidate}: ${faster ? candidate : best}`);
    engine.release();
    say(`the search to its end: ${ended.threads} threads after ${ended.generations} generations of 6 tokens (${log.join(", ")}), the search chose ${chose}`);
    expect("the search ended", [ended.ended, engine.searching], [true, false]);
    expect("timed on the count the search chose", ended.threads, chose);
    if (!log.length) failures.push("the search to its end: no comparison made");
  }
  {
    const engine = createForward({ memory, base, size, kernels, plan, spawn });
    let tokens = 0;
    await engine.findThreads({ from: 2, remembered: 3 });
    const ended = await endSearch(engine, () => tokens++);
    engine.release();
    expect("a count the model page remembers: no search, that count", [ended.threads, ended.generations, tokens], [3, 0, 0]);
  }
  // T190's review: a software thread that stops in the search (T120: the engine gives its helpers up and goes on with
  // one) leaves found at 1 as well, so "fewer than found" never says it: the page path's head reads lostThreads. This
  // helper takes a chunk and ends without counting it (threads-check's)
  {
    const { WAKE, COUNTER, ACTIVE, CONTROL_BYTES } = await import(path.join(root, "public/jobs.js"));
    const dying = ({ memory: shared, share }) => new Promise((resolve) => {
      const helper = new Worker(`
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
      helper.once("message", () => resolve({ terminate: () => helper.terminate() }));
    });
    const engine = createForward({ memory, base, size, kernels, plan, spawn: dying, stalledMs: 300 });
    await engine.findThreads({ from: 2 });
    const warned = console.warn;
    console.warn = () => {};  // the one line forward.js writes about it
    const ended = await endSearch(engine, () => { for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true); });
    console.warn = warned;
    const lost = engine.lostThreads;
    engine.release();
    expect("a software thread that stops in the search: one thread, found 1 too, lostThreads", [ended.threads, ended.found, lost], [1, 1, true]);
  }
  // T152: the steps of a generation. The CPU's ms of a token here (the median of five of a GPU-less engine); a prompt of
  // 16 (on the CPU: the made-up GPU's prompt is slow here), then STEPS steps as Python takes them: [on the GPU, on the CPU]
  const STEPS = 20;
  const cpuStep = (() => {
    const engine = createForward({ memory, base, size, kernels, plan });
    engine.forwardMany(prompt(16), 0);
    const ms = [16, 17, 18, 19, 20].map((pos) => {
      const began = performance.now();
      engine.forward(100, pos);
      return performance.now() - began;
    }).sort((a, b) => a - b)[2];
    engine.release();
    return ms;
  })();
  say(`the CPU: ${cpuStep.toFixed(2)} ms a token with its logits`);
  const write = (engine) => {
    engine.newGeneration();
    const fed = prompt(16);
    engine.forwardMany(fed, 0);
    let token = 100, pos = fed.length, onGpu = 0, onCpu = 0;
    const history = [...fed, token];
    while (pos < fed.length + STEPS) {
      const many = Math.min(engine.tokenBlock, fed.length + STEPS - pos);
      const ids = many > 0 ? engine.generateMany(token, pos, history.slice(-64), history.length, many, 0, 0.9, 1, [], []) : null;
      const chosen = ids ?? (engine.forward(token, pos), [token + 1]);
      if (ids) onGpu += ids.length;
      else onCpu += 1;
      for (const id of chosen) {
        history.push(id);
        token = id;
        pos += 1;
      }
    }
    return [onGpu, onCpu];
  };
  const steps = async (name, stepCost, generations, failTokensAt = 0) => {
    const line = { fixed: 60 * perToken, perToken: 2 * perToken, step: stepCost * cpuStep, failTokensAt };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: line });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); } };
    };
    const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu });
    await engine.setThreads(1);
    await engine.gpu;
    const seen = [];
    for (let g = 1; g <= generations; g++) seen.push(write(engine));
    say(`${name}: [on the GPU, on the CPU] ${seen.map((pair) => pair.join("/")).join(" ")}; status ${engine.gpuStatus}`);
    engine.release();
    return seen;
  };
  {
    const seen = await steps("the steps, a GPU far faster", 0.2, 12);
    expect("the first generation: 2 steps on the CPU (timed), the rest on the GPU", seen[0], [STEPS - 2, 2]);
    expect("then every step on the GPU", seen.slice(1, 8), all(2, 8, [STEPS, 0]));
    expect("the eighth after the first verdict: its first 4 on the CPU", seen[8], [STEPS - 4, 4]);
    expect("and the GPU again", seen.slice(9), all(10, 12, [STEPS, 0]));
  }
  {
    const seen = await steps("the steps, a GPU far slower", 5, 12);
    expect("every step on the CPU but the check's first run", seen, [...all(1, 8, [0, STEPS]), [4, STEPS - 4], ...all(10, 12, [0, STEPS])]);
  }
  {
    const seen = await steps("the steps, a GPU that fails on its second request", 0.2, 2, 2);
    expect("the CPU from the failure on", seen, [[4, STEPS - 4], [0, STEPS]]);
  }
  // T152: Python's generate() through forward.js's external() (as the page's worker has it) on the made-up GPU far
  // faster: the steps go to it (Python draws the random numbers and hands the history over as the made-up GPU expects),
  // and the text and the counts are those of the steps
  {
    const { loadPyodide } = await import("pyodide");
    const py = await loadPyodide();
    await py.loadPackage("numpy", { messageCallback: () => {} });
    for (const name of ["llama2_numpy.py", "llama2_convert.py", "simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
      py.FS.writeFile(name, fs.readFileSync(path.join(root, "public", name)));
    }
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(tokenizer));
    const line = { fixed: 60 * perToken, perToken: 2 * perToken, step: 0.2 * cpuStep };
    const gpuOf = (workerData) => () => {
      const fake = new Worker(FAKE, { eval: true, workerData });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); } };
    };
    const outside = external({ memory, base, size, kernels, gpu: gpuOf(line) });
    py.globals.set("OUTSIDE", outside);
    py.globals.set("OPTIONS", py.toPy(options));
    py.runPython(`from llama2_numpy import Llama\nllama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    await outside.engine.gpu;
    const written = [];
    for (const settings of ["temperature=0.8, topp=0.9, repetition_penalty=1.1, seed=3", "temperature=0.0"]) {
      outside.engine.newGeneration();
      py.runPython(`text = "".join(llama.generate("こんにちは、今日は", steps=48, ${settings}))`);
      const stats = py.runPython("llama.stats").toJs({ dict_converter: Object.fromEntries });
      written.push({ sampled: stats.sampled, gpu: outside.engine.gpuSampled, prompt: stats.prompt_tokens, text: py.globals.get("text").length });
    }
    py.runPython("llama.release()");
    say(`Python's generate() on the made-up GPU: ${JSON.stringify(written)}; status ${outside.engine.gpuStatus}`);
    written.forEach(({ sampled, gpu, prompt, text }, i) => {
      if (!(sampled === 48 - prompt && gpu >= sampled - 2 && text > 0)) {
        failures.push(`Python's generate() ${i ? "greedy" : "sampled"}: ${sampled} sampled, ${gpu} on the GPU, ${text} characters`);
      }
    });
    // T152's review: the same through Python on a GPU that fails on its second request. generateMany gives the steps
    // back (null), and Python takes them on the CPU: the generation goes on whole (Pyodide makes JavaScript's null
    // jsnull, not None, unless asked: generateMany says undefined, which is None)
    const failing = external({ memory, base, size, kernels, gpu: gpuOf({ ...line, failTokensAt: 2 }) });
    py.globals.set("OUTSIDE", failing);
    py.runPython(`llama = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
    await failing.engine.gpu;
    failing.engine.newGeneration();
    let broke = null;
    try {
      py.runPython(`text = "".join(llama.generate("こんにちは、今日は", steps=48, temperature=0.8, topp=0.9, repetition_penalty=1.1, seed=3))`);
    } catch (error) {
      broke = String(error?.message ?? error).split("\n").filter(Boolean).at(-1);
    }
    const after = py.runPython("llama.stats").toJs({ dict_converter: Object.fromEntries });
    py.runPython("llama.release()");
    say(`Python's generate() on a GPU that fails on its second request: ${broke ?? `${after.sampled} sampled, ${failing.engine.gpuSampled} on the GPU`}; status ${failing.engine.gpuStatus}`);
    if (broke || !(after.sampled === 48 - after.prompt_tokens && failing.engine.gpuSampled === 4 && /failed on a token/.test(failing.engine.gpuStatus ?? ""))) {
      failures.push(`Python's generate() on a GPU that fails: ${broke ?? `${after.sampled} sampled, ${failing.engine.gpuSampled} on the GPU, ${failing.engine.gpuStatus}`}`);
    }
  }
  if (failures.length) say(`FAILED\n- ${failures.join("\n- ")}`);
  else say("ok");
  process.exit(failures.length ? 1 : 0);
}
