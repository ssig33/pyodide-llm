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
// T199: the search's verdict on a made-up clock (forward.js's clock), with one block of every comparison slowed 3 times:
// the count that is fastest still chosen (4 of the owner's Android's 8 logical cores, 2 of CI's runner's 4).
// T223: on the same clock, the owner's Android's llm-jp-3 150M (2 threads fastest, 4 a third of 1's speed) chooses 2;
// a search timed while the GPU gets ready is not remembered, and is searched again once the GPU is ready (the times
// beside the GPU say 4, those after it 2: 2 is remembered); a count remembered from an earlier visit is searched again
// on the first generation (not only on the eighth); and the page's key of the count is one a model.
// T152: the steps of a generation (forward.js's tokenBlock and generateMany, tokenTimes) on the same made-up GPU, whose
// step sleeps a multiple of the CPU's own ms of a token and writes made-up ids: generations of STEPS steps after a short
// prompt, as Python takes them (tokenBlock at a time on the GPU, else one on the CPU). Far faster: the first steps on the
// CPU (timed first), then every one on the GPU, and the eighth generation after the first verdict its first 4 on the
// CPU; far slower: every step on the CPU but that generation's first run; failing on its second request: the CPU from
// there on, the step it gave back taken by the CPU. And Python's generate() through external(), as the page's worker
// runs it, on the far faster one: sampled and greedy, the steps on the GPU and the counts right (the made-up GPU fails a
// request whose random numbers, history or settings are not what Python should hand over); and through Python on one
// that fails on its second request: the generation goes on whole on the CPU (T152's review: JavaScript's null is jsnull).
// T219: a GPU that samples -1 (SAMPLE's NONE) or an id past the vocabulary: that request refused whole, the CPU from
// there on, and the status line says why; a model on the GPU alone stops with words and is not loaded again.
// T205: release() waits for the GPU's worker to say "ended" (a slow one; one that never does, terminated after 5 s; one
// stopped while it got ready), and so does Llama.release() called from JavaScript through Python, as the page's worker
// calls it (the review); a browser that does not say the device's memory keeps the steps on the CPU.
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
let ctl, words, ids, blocks = 0, requests = 0, asked = 0, held = null;
const nap = new Int32Array(new SharedArrayBuffer(4)), ms = (n) => line.fixed + line.perToken * n;
parentPort.on("message", (data) => {
  if (data.type === "start") {
    ctl = new Int32Array(data.memory.buffer, 0, 2048);
    words = data.plan.words;
    // T152: where forward.js asked for the steps and the line has a cost of a step
    ids = data.plan.tokens && line.step !== undefined ? new Int32Array(data.memory.buffer, data.plan.tokens.ids, 1 + data.plan.tokens.most) : null;
    const ready = { type: "ready", adapter: "made up", key: "k", bytes: 1, seconds: 0, form: "made up", attention: "made up",
      forms: [], remembered: false, blocks: [{ count: 16, ms: ms(16) }, { count: 64, ms: ms(64) }],
      ...(ids ? { tokens: { form: "made up", ms: line.step, forms: [] } } : {}) };
    // T223: a GPU that gets ready only when the test says so ({ type: "go" }, sent to this worker alone)
    if (line.readyOnGo) held = ready;
    else parentPort.postMessage(ready);
  } else if (data.type === "go") {
    parentPort.postMessage(held);
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
    // T219: a GPU whose outsideAt-th request samples outsideId (-1, SAMPLE's NONE, or one past the vocabulary) last
    const outside = Boolean(line.outsideAt) && ++asked === line.outsideAt;
    ids[0] = data.count;
    for (let i = 0; i < data.count; i++) ids[1 + i] = outside && i === data.count - 1 ? line.outsideId : data.token + 1 + i;
    Atomics.store(ctl, words.failed, fail ? 1 : 0);
    Atomics.store(ctl, words.done, data.serial);
    Atomics.notify(ctl, words.done);
  } else if (data.type === "stop") {
    // T205: as gpu.js's end(): { type: "ended" } once its device is let go (endAfter ms: a slow let-go; never: a worker
    // that hangs in a compilation), then the worker ends
    if (line.endAfter === Infinity) return;
    setTimeout(() => {
      parentPort.postMessage({ type: "ended" });
      parentPort.close();
    }, line.endAfter ?? 0);
  }
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
  const { compileKernels, createForward, endSearch, timePrompts, external, OUTSIDE_VOCABULARY } = await import(forwardFile);
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
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
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
  // T199: the search's verdict stands where one block of 4 timed tokens is slowed whole (a collection of the garbage,
  // another tab). On a made-up clock (forward.js's clock) a token of n threads takes ms[n], and in every comparison the
  // tokens of one block (0 and 3 are the best count's, 1 and 2 the candidate's) take 3 times as long: the owner's
  // Android (4 threads fastest, from its 8 logical cores) and CI's runner where 2 threads were 1.3 times as fast as 1
  // (the upper median chose 1 twice, T190's review)
  for (const [name, ms, from, want] of [["the owner's Android", { 1: 16.4, 2: 10.5, 4: 8.8, 8: 12 }, 8, 4],
                                        ["CI's runner", { 1: 13, 2: 10, 4: 11 }, 4, 2]]) {
    const chosen = [];
    for (const slowed of [null, 0, 1, 2, 3]) {
      let engine = null, starting = false, time = 0, comparison = -1, token = 0;
      const clock = () => {
        starting = !starting;  // the search reads the clock as a token starts and as it ends
        if (starting) return time;
        // the end of a token: its count is the one the engine runs it on. No helper is started on the way to the
        // count wanted (the search goes down from counts whose helpers findThreads started), so a comparison's 20 tokens
        // are its 4 blocks of 5. (A count past ms: a search gone the wrong way)
        if (engine.searchLog.length !== comparison) [comparison, token] = [engine.searchLog.length, 0];
        const block = Math.floor(token++ / 5);
        time += (ms[engine.threads] ?? 30) * (block === slowed ? 3 : 1);
        return time;
      };
      engine = createForward({ memory, base, size, kernels, plan, spawn, clock });
      await engine.findThreads({ from });
      const ended = await endSearch(engine, () => { for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true); });
      const log = engine.searchLog.map(({ best, candidate, faster }) => `${best} or ${candidate}: ${faster ? candidate : best}`);
      engine.release();
      chosen.push(`${slowed === null ? "none slowed" : `block ${slowed} slowed`}: ${ended.threads} (${log.join(", ")})`);
      expect(`${name}, ${slowed === null ? "no block" : `block ${slowed}`} slowed 3 times: the count chosen`, [ended.ended, ended.threads], [true, want]);
    }
    say(`the search on a made-up clock, ${name}: ${chosen.join("; ")}`);
  }
  // T223: the owner's Android's writing of llm-jp-3 150M on the CPU alone (/benchmark/, site d531cd5): 1 thread 110,
  // 2 123, 4 38.6, 8 32.3 tok/s; as ms a token
  const LLM_JP = { 1: 1000 / 110, 2: 1000 / 123, 4: 1000 / 38.6, 8: 1000 / 32.3 };
  const madeUpClock = (msNow) => {
    let starting = false, time = 0;
    return () => {
      starting = !starting;  // the search reads the clock as a token starts and as it ends
      if (!starting) time += msNow();
      return time;
    };
  };
  const generation = (engine) => {
    engine.newGeneration();
    for (let pos = 0; pos < 6; pos++) engine.forward(100 + pos, pos, true);
  };
  const verdicts = (engine) => engine.searchLog.map(({ best, candidate, faster, whileGpu }) => `${best} or ${candidate}: ${faster ? candidate : best}${whileGpu ? " (GPU getting ready)" : ""}`);
  {
    let engine = null;
    engine = createForward({ memory, base, size, kernels, plan, spawn, clock: madeUpClock(() => LLM_JP[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, chose: (count) => told.push(count) });
    const ended = await endSearch(engine, () => generation(engine));
    say(`T223, llm-jp-3 150M of the owner's Android on a made-up clock: ${ended.threads} (${verdicts(engine).join(", ")})`);
    engine.release();
    expect("T223: llm-jp-3 150M of the owner's Android from its 8 logical cores: 2 threads, remembered", [ended.threads, told], [2, [2]]);
  }
  {
    // the GPU gets ready when told; until then a token's times are those beside it (4 fastest), then llm-jp's
    let engine = null, gpuReady = false, fake = null;
    const beside = { 1: 16, 2: 12, 4: 8, 8: 10 };
    const gpu = () => {
      fake = new Worker(FAKE, { eval: true, workerData: { fixed: 1e3, perToken: 1e3, readyOnGo: true } });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    engine = createForward({ memory, base, size, kernels, plan, spawn, gpu, clock: madeUpClock(() => (gpuReady ? LLM_JP : beside)[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, chose: (count) => told.push(count) });
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && engine.searching; g++) {
      generation(engine);
      await turn();
    }
    const beforeReady = { threads: engine.threads, told: [...told], searched: engine.searchLog.length };
    for (let g = 0; g < 3; g++) generation(engine);  // no search begins while the GPU gets ready
    const idle = engine.searching || engine.searchLog.length !== beforeReady.searched;
    fake.postMessage({ type: "go" });
    await engine.gpu;
    gpuReady = true;
    for (let g = 0; g < 200 && (g === 0 || engine.searching); g++) {
      generation(engine);
      await turn();
    }
    say(`T223, a search while the GPU gets ready: ${beforeReady.threads} then ${engine.threads}, remembered ${JSON.stringify(told)} (${verdicts(engine).join(", ")})`);
    const log = engine.searchLog;
    engine.release();
    expect("T223: while the GPU gets ready, the search's 4 used and not remembered", [beforeReady.threads, beforeReady.told], [4, []]);
    expect("T223: no search begins while the GPU gets ready", idle, false);
    expect("T223: the verdicts beside the GPU marked, those after it not", log.map((v) => v.whileGpu),
      log.map((_, i) => i < beforeReady.searched));
    expect("T223: searched again once the GPU is ready: 2, remembered", [engine.threads, told], [2, [2]]);
  }
  {
    // a count remembered from an earlier visit (4, the owner's Android's) is searched again on the first generation
    let engine = null;
    engine = createForward({ memory, base, size, kernels, plan, spawn, clock: madeUpClock(() => LLM_JP[engine.threads] ?? 40) });
    const told = [];
    await engine.findThreads({ from: 8, remembered: 4, chose: (count) => told.push(count) });
    const idle = engine.searching;
    generation(engine);
    const began = engine.searching;
    const turn = () => new Promise((resolve) => setTimeout(resolve, 0));
    for (let g = 0; g < 200 && engine.searching; g++) {
      generation(engine);
      await turn();
    }
    say(`T223, a remembered 4 on its first generation: ${engine.threads}, remembered ${JSON.stringify(told)} (${verdicts(engine).join(", ")})`);
    engine.release();
    expect("T223: a remembered count: no search as it loads, one on the first generation, 2 remembered", [idle, began, engine.threads, told], [false, true, 2, [2]]);
  }
  {
    const { threadsKey } = await import(path.join(root, "src/bench.js"));
    const device = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "made up" };
    expect("T223: the page remembers the count one a model", threadsKey("llm-jp-3-150m", device) !== threadsKey("tiny-lm", device), true);
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
  const steps = async (name, stepCost, generations, more = {}) => {
    const line = { fixed: 60 * perToken, perToken: 2 * perToken, step: stepCost * cpuStep, ...more };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: line });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu });
    await engine.setThreads(1);
    await engine.gpu;
    const seen = [];
    for (let g = 1; g <= generations; g++) seen.push(write(engine));
    say(`${name}: [on the GPU, on the CPU] ${seen.map((pair) => pair.join("/")).join(" ")}; status ${engine.gpuStatus}`);
    const status = engine.gpuStatus;
    engine.release();
    return Object.assign(seen, { status });
  };
  {
    const seen = await steps("the steps, a GPU far faster", 0.2, 12);
    expect("the first generation: 2 steps on the CPU (timed), the rest on the GPU", seen[0], [STEPS - 2, 2]);
    expect("then every step on the GPU", seen.slice(1, 8), all(2, 8, [STEPS, 0]));
    expect("the eighth after the first verdict: its first 4 on the CPU", seen[8], [STEPS - 4, 4]);
    expect("and the GPU again", seen.slice(9), all(10, 12, [STEPS, 0]));
    // (the owner's words, 2026-09-27: "prompts and answers on WebGPU", or the prompts' own verdict and "answers on WebGPU")
    expect(`the status line says the answers are on WebGPU (${seen.status})`, /answers on WebGPU$/.test(seen.status), true);
  }
  {
    const seen = await steps("the steps, a GPU far slower", 5, 12);
    expect("every step on the CPU but the check's first run", [...seen], [...all(1, 8, [0, STEPS]), [4, STEPS - 4], ...all(10, 12, [0, STEPS])]);
    // ("prompts on WebGPU, answers on the CPU (faster here)", or both on the CPU where the prompts are too)
    expect(`the status line says the answers are on the CPU, faster (${seen.status})`,
      /(, answers on the CPU \(faster here\)|^prompts and answers on the CPU \(faster here than WebGPU\))$/.test(seen.status), true);
  }
  {
    const seen = await steps("the steps, a GPU that fails on its second request", 0.2, 2, { failTokensAt: 2 });
    expect("the CPU from the failure on", [...seen], [[4, STEPS - 4], [0, STEPS]]);
    expect(`the status line says why the prompts are on the CPU, and nothing of the answers (${seen.status})`,
      /^prompts on the CPU \(.*failed on a token/.test(seen.status) && !/answers/.test(seen.status), true);
  }
  // T219: a GPU whose second request samples an id outside the vocabulary last (what SAMPLE gives for logits that are
  // not finite: -1, or one past the end): the whole request refused, the CPU takes the steps from there (and would stop
  // with T195's NOT_FINITE where its own logits were not finite either; here they are), the GPU stopped
  for (const outsideId of [-1, plan.vocab_size]) {
    const seen = await steps(`the steps, a GPU that samples ${outsideId} on its second request`, 0.2, 2, { outsideAt: 2, outsideId });
    expect(`${outsideId}: the CPU from that request on, none of its ids taken`, [...seen], [[4, STEPS - 4], [0, STEPS]]);
    expect(`${outsideId}: the status line says the GPU sampled outside the vocabulary (${seen.status})`,
      new RegExp(`^prompts on the CPU \\(the GPU sampled ${outsideId}, outside the vocabulary`).test(seen.status), true);
  }
  // T219: a model on the GPU alone (T156's direct) cannot take the step on the CPU: generateMany stops with words (the
  // page says them), and does not load the model again on the CPU (onLost: the CPU's logits would be those of the same
  // weights); a request of good ids after it is taken on the GPU as before
  for (const outsideId of [-1, plan.vocab_size]) {
    const line = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep, outsideAt: 1, outsideId };
    const fake = new Worker(FAKE, { eval: true, workerData: line });
    const gpu = () => ({ postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
      set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() });
    let lost = null;
    const engine = createForward({ memory, base, size, kernels, plan, gpu, direct: { onLost: (why) => { lost = why; } } });
    await engine.gpu;
    engine.newGeneration();
    const fed = prompt(16);
    engine.forwardMany(fed, 0);
    let thrown = null, first;
    try {
      first = engine.generateMany(100, fed.length, [...fed, 100].slice(-64), fed.length + 1, 4, 0, 0.9, 1, [], []);
    } catch (error) {
      thrown = error.message;
    }
    const second = thrown ? engine.generateMany(100, fed.length, [...fed, 100].slice(-64), fed.length + 1, 4, 0, 0.9, 1, [], []) : undefined;
    say(`T219: on the GPU alone, ${outsideId} sampled: ${thrown ?? `no error (${JSON.stringify(first)})`}; lost ${lost}; then ${JSON.stringify(second)}`);
    expect(`${outsideId} on the GPU alone: stopped with words, not loaded again on the CPU, the GPU still taking the steps`,
      [thrown === OUTSIDE_VOCABULARY, lost, second], [true, null, [101, 102, 103, 104]]);
    await engine.release();
  }
  // T205: the next model is read after the GPU's worker let go of its device (an iPhone's tab went down in
  // /benchmark/'s rounds where the one before still held it). release() waits for its "ended": a slow let-go of 400 ms
  // is waited for; one that never comes is given up after 5 s and the worker terminated; a model let go while its GPU
  // is getting ready waits the same (the late "ready" after the stop changes nothing)
  {
    const fast = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep };
    const released = async (more, ready = true) => {
      let exited = false;
      const gpu = () => {
        const fake = new Worker(FAKE, { eval: true, workerData: { ...fast, ...more } });
        fake.once("exit", () => { exited = true; });
        return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
          set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
      };
      const warned = console.warn;
      console.warn = () => {};  // the line forward.js writes of a worker that did not end
      const engine = createForward({ memory, base, size, kernels, plan, gpu });
      if (ready) await engine.gpu;
      const began = performance.now(), ended = await engine.release(), ms = performance.now() - began;
      console.warn = warned;
      await new Promise((resolve) => setTimeout(resolve, 300));
      say(`T205: a GPU's worker ${more.endAfter === Infinity ? "that never ends" : `ending ${more.endAfter} ms after the stop`}` +
        `${ready ? "" : ", stopped while it got ready"}: release() ${ended ? "saw it end" : "gave it up"} in ${ms.toFixed(0)} ms` +
        `, the worker ${exited ? "gone" : "still there"}; status ${engine.gpuStatus}`);
      return { ended, ms, exited, status: engine.gpuStatus };
    };
    const slow = await released({ endAfter: 400 });
    expect("release() waits for a slow GPU's worker to end", [slow.ended, slow.ms >= 380 && slow.ms < 2500, slow.exited], [true, true, true]);
    const hung = await released({ endAfter: Infinity });
    expect("release() gives up a GPU's worker that never ends after 5 s, and terminates it",
      [hung.ended, hung.ms >= 4900 && hung.ms < 7000, hung.exited], [false, true, true]);
    const early = await released({ endAfter: 200 }, false);
    expect("a model let go while its GPU gets ready: waited for, and the late ready ignored",
      [early.ended, early.ms >= 180 && early.ms < 2500, early.exited, /WebGPU/.test(early.status ?? "") && !/CPU/.test(early.status ?? "")], [true, true, true, false]);
  }
  // T205: a browser that does not say the device's memory (Safari, Firefox: no navigator.deviceMemory) keeps a
  // generation's steps on the CPU (no classifier and embedding on the GPU), and the prompts' blocks on the GPU
  {
    const fast = { fixed: 10 * perToken, perToken: 0.05 * perToken, step: 0.2 * cpuStep };
    const gpu = () => {
      const fake = new Worker(FAKE, { eval: true, workerData: fast });
      return { postMessage: (data) => fake.postMessage(data), set onmessage(f) { fake.on("message", (data) => f({ data })); },
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    const outcome = async (memoryUnsaid) => {
      const engine = createForward({ memory, base, size, kernels, plan, spawn, gpu, memoryUnsaid });
      await engine.setThreads(1);
      await engine.gpu;
      const prompts = [1, 2, 3].map(() => feed(engine)[1]);
      const written = [1, 2, 3].map(() => write(engine));
      const got = { planned: engine.gpuTokensPlanned, why: engine.gpuTokensWhyNot, prompts, written, status: engine.gpuStatus };
      await engine.release();
      say(`T205: memoryUnsaid ${memoryUnsaid}: ${JSON.stringify(got)}`);
      return got;
    };
    const unsaid = await outcome(true), said = await outcome(false);
    expect("no deviceMemory: the steps not asked of the GPU, and why", [unsaid.planned, /does not say how much memory/.test(unsaid.why ?? "")], [false, true]);
    expect("no deviceMemory: the prompts' blocks still on the GPU (after the first, timed on the CPU)", unsaid.prompts.slice(1), [PROMPT, PROMPT]);
    expect("no deviceMemory: every step on the CPU", unsaid.written, all(1, 3, [0, STEPS]));
    expect("no deviceMemory: the status line says no reason (the console does)", /memory/.test(unsaid.status ?? ""), false);
    expect("with deviceMemory: the steps asked of the GPU, and taken there", [said.planned, said.written.at(-1)[0] > 0], [true, true]);
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
        set onerror(f) { fake.on("error", (error) => f({ message: error.message })); }, terminate: () => fake.terminate() };
    };
    // (T205's review: its worker says ended 300 ms after the stop, for the release through Python below)
    const outside = external({ memory, base, size, kernels, gpu: gpuOf({ ...line, endAfter: 300 }) });
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
    // T205's review: the release as the page's worker makes it (worker.js's load()): Llama.release() called from
    // JavaScript hands forward.js's promise back through Python, and the next model is read once it settles. An await
    // of anything else (None, a PyProxy) waits for nothing, and every other test here would still pass
    const pythonLlama = py.globals.get("llama");
    const releasedAt = performance.now(), released = pythonLlama.release?.();
    const promised = typeof released?.then === "function";
    const endedThrough = promised ? await released : undefined, releaseMs = performance.now() - releasedAt;
    pythonLlama.destroy();
    say(`T205: Llama.release() through Python: ${promised ? `a promise, ${endedThrough ? "its worker ended" : "not ended"} after ${releaseMs.toFixed(0)} ms` : `${released} (no promise)`}`);
    if (!(promised && endedThrough === true && releaseMs >= 280)) {
      failures.push(`Llama.release() through Python: ${promised ? `${endedThrough} after ${releaseMs.toFixed(0)} ms (its worker ends 300 ms after the stop)` : "no promise to wait for"}`);
    }
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
