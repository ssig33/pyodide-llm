// T164: what a token costs outside forward.js, in Node's Pyodide, as the page runs it: the forward pass in
// public/forward.js, the sampling kernels of simdkernel.so through ctypes, and generate() in Python, driven from
// JavaScript as public/worker.js drives it. One thread: tests/engine.mjs starts no software threads (the page's
// threads shorten the forward pass and leave the outside as it is, see T164 in TODO.md).
//
// A generated token, at the positions 0..N-1, with the model's temperature, top-p and repetition penalty:
//   JS forward     engine.forward() in a loop of JavaScript: the forward pass, and the copy of the logits into
//                  Python's array that forward() ends with ("copy": that alone, made the same way from JavaScript)
//   generate()     N tokens through generate(), the stop tokens off, every piece taken by next() from JavaScript and
//                  posted on a MessageChannel as worker.js does (not its breath every 50 ms, a turn of the event loop,
//                  which runPython does not give back): 1000 / tok/s, the number of the status line
//   outside        generate() less the JS forward of the same round; the median of the rounds and their range
// and what makes the outside up, each timed alone with Python's garbage collector off, µs a call. The kernels are
// timed right after a forward pass, as generate() runs them (the pass has just streamed the weights through the
// caches: "again" shows what that costs):
//   penalize, sample   the page's kernels on the real logits of each position; C is how many tokens pass the floor
//                  (the best's 1e-7) that sample() keeps, the median of the positions
//   again          sample() once more on the same logits at once: the caches warm again
//   1 past         sample() on logits of which the best alone passes the floor: its walks over the whole vocabulary
//   T158's         sample() on T158's artificial logits (uniform in ±10: most of the vocabulary passes the floor)
//   argmax         sample() at temperature 0: the greedy models' choice (not in the sampled models' outside)
//   crossing       a call from Python of a JavaScript function that does nothing, with forward()'s arguments
//   ctypes         a call of a kernel that has nothing to do (add_inplace of 0 numbers)
//   GC             Python's garbage collector during generate() (gc.callbacks), µs a token
//   the rest       the outside less crossing, penalize, sample and GC: forward()'s closure, the decoding of the
//                  text, the generator, next() and the message
// A prompt of P tokens (the model's own prompt, repeated up to 64 tokens at least): the tokenizer's encode(), which
// generate() runs before its clock starts, forwardMany() from JavaScript, and generate() with P + 1 steps.
//
// The measures take turns in each round (AGENTS.md: the same process, alternating); the first round warms up and is
// left out, and each cell is the median of the rounds.
//
//   node tests/overhead.mjs [model id ...] [--rounds 5] [--tokens 128]
import fs from "node:fs";
import os from "node:os";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 5), tokens = option("--tokens", 128);
const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));

const { pyodide: py } = await pyodideWithEngine();
// JavaScript's side: ms a call of step(i), the mean of n
const timed = (n, step) => {
  const began = performance.now();
  for (let i = 0; i < n; i++) step(i);
  return (performance.now() - began) / n;
};
py.globals.set("js_forward", (engine, n) => timed(n, (pos) => engine.forward(1, pos, true)));
py.globals.set("js_prompt", (engine, n) => {
  const block = Number(engine.promptBlock ?? 16), fed = new Array(n).fill(1);
  const began = performance.now();
  for (let at = 0; at < n; at += block) engine.forwardMany(fed.slice(at, at + block), at);
  return (performance.now() - began) / n;
});
py.globals.set("js_nothing", () => {});
// what forward() ends with (public/forward.js): a view of the logits made anew, into Python's array
py.globals.set("js_copy", (array, n) => {
  const source = new ArrayBuffer(4 * Number(array.size));
  return timed(n, () => {
    const view = array.getBuffer("f32");
    view.data.set(new Float32Array(source, 0, view.data.length));
    view.release();
  });
});
// generate() as public/worker.js's generate() drives it
py.globals.set("js_generate", (generate, prompt, settings) => {
  const channel = new MessageChannel();
  const pieces = generate.callKwargs(prompt, settings.toJs({ dict_converter: Object.fromEntries }));
  try {
    for (let piece = pieces.next(); !piece.done; piece = pieces.next()) channel.port2.postMessage({ type: "token", text: piece.value });
  } finally {
    pieces.destroy();
    channel.port1.close();
  }
});

const clock = py.runPython(`
import time
clock = time.perf_counter
began = clock()
for _ in range(10000):
    clock()
(clock() - began) / 10000 * 1e6`);
console.log(`${os.cpus()[0]?.model ?? "?"} (${process.arch}) × ${os.cpus().length}, Node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}; ` +
  `${rounds} rounds, ${tokens} tokens; time.perf_counter() costs ${clock.toFixed(2)} µs`);
const tokenRows = [], promptRows = [];
for (const id of ids.length ? ids : ["llm-jp-3-150m", "tiny-lm"]) {
  const entry = MODELS.find((m) => m.id === id);
  if (!entry) throw new Error(`${id} is not a model of src/models.js`);
  py.globals.set("CHECKPOINT", root + entry.checkpoint);
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
  py.globals.set("OPTIONS", py.toPy(entry.options));
  py.globals.set("GEN", py.toPy(entry.generation));
  py.globals.set("PROMPT", entry.prompt ?? "Once upon a time");
  const rows = py.runPython(`
import time, math, gc, statistics
import numpy as np
import llama2_numpy
N, ROUNDS = ${tokens}, ${rounds}
clock = time.perf_counter
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
engine = llama._external[0]
logits = llama.forward(llama.bos, 0)  # the array forward.js writes into, every time
V = logits.size
settings = dict(temperature=GEN["temperature"], topp=GEN.get("topp", 0.9), repetition_penalty=GEN.get("repetition_penalty", 1.0))
temperature, topp, penalty = settings["temperature"], settings["topp"], settings["repetition_penalty"]
past_floor = lambda x: int(np.count_nonzero(x >= x.max() + temperature * math.log(1e-7)))
alone = np.full(V, -1000.0, dtype=np.float32)
alone[V // 2] = 0.0
artificial = np.random.default_rng(158).uniform(-10, 10, V).astype(np.float32)
add, address = llama2_numpy.load_kernels("simdkernel.so")["add_inplace"], logits.ctypes.data

def per_call(step, n=N):
    """µs a call of step(i), the mean of n"""
    began = clock()
    for i in range(n):
        step(i)
    return (clock() - began) / n * 1e6

def generated(prompt, steps):
    """generate() driven as the worker drives it: its stats, and the ms of Python's garbage collection in it"""
    collecting = [0.0, 0.0]
    def watch(phase, info):
        if phase == "start":
            collecting[1] = clock()
        else:
            collecting[0] += clock() - collecting[1]
    stops, llama.stop_tokens = llama.stop_tokens, ()
    gc.callbacks.append(watch)
    try:
        js_generate(llama.generate, prompt, dict(steps=steps, seed=1, **settings))
    finally:
        gc.callbacks.remove(watch)
        llama.stop_tokens = stops
    return llama.stats, collecting[0] * 1000

def sampling():
    """generate()'s steps after each forward pass, timed apart from it: penalize() and sample() on the real logits
    (the same text as generate(): the same random numbers), sample() once more on them with the caches warm again
    (another generator), µs a call, and C"""
    rng, again, token, history = np.random.default_rng(1), np.random.default_rng(2), llama.bos, [llama.bos]
    spent = {"penalize": 0.0, "sample": 0.0, "again": 0.0}
    past = []
    for pos in range(N):
        out = llama.forward(token, pos)
        began = clock()
        if penalty != 1.0:
            llama.penalize(out, history, penalty)
        penalized = clock()
        token = llama.sample(out, temperature, topp, rng)
        sampled = clock()
        llama.sample(out, temperature, topp, again)
        spent["again"] += clock() - sampled
        spent["sample"] += sampled - penalized
        spent["penalize"] += penalized - began
        past.append(past_floor(out))
        history.append(token)
    return {k: v / N * 1e6 for k, v in spent.items()}, statistics.median(past)

def after_forward(step):
    """step(logits) timed right after each forward pass, as generate() runs it: µs a call"""
    spent = 0.0
    for pos in range(N):
        out = llama.forward(llama.bos, pos)
        began = clock()
        step(out)
        spent += clock() - began
    return spent / N * 1e6

prompt = PROMPT
while len(llama.tokenizer.encode(prompt, llama.specials)) < 64:
    prompt += PROMPT
P = len(llama.tokenizer.encode(prompt, llama.specials))
cells = {}
for r in range(ROUNDS + 1):  # the first round warms up and is left out
    got = {"js": js_forward(engine, N)}
    stats, collected = generated("", N)
    got["generate"] = 1000 / stats["tokens_per_second"]
    got["outside"] = got["generate"] - got["js"]
    got["gc"] = collected / N * 1000
    got["copy"] = js_copy(logits, N) * 1000
    gc.disable()
    try:
        spent, got["past"] = sampling()
        got.update(spent)
        rng = np.random.default_rng(3)
        got["alone"] = after_forward(lambda out: llama.sample(alone, temperature, topp, rng))
        got["artificial"] = after_forward(lambda out: llama.sample(artificial, temperature, topp, rng))
        got["argmax"] = after_forward(lambda out: llama.sample(out, 0.0, topp, rng))
        got["crossing"] = per_call(lambda i: js_nothing(1, i, True))
        got["ctypes"] = per_call(lambda i: add(address, address, 0))
        got["encode"] = per_call(lambda i: llama.tokenizer.encode(prompt, llama.specials), 5) / 1000
    finally:
        gc.enable()
    got["many"] = js_prompt(engine, P)
    stats, _ = generated(prompt, P + 1)
    got["prompt"] = stats["prompt_seconds"] / stats["prompt_tokens"] * 1000
    if r:
        for k, v in got.items():
            cells.setdefault(k, []).append(v)
m = {k: statistics.median(v) for k, v in cells.items()}
outside = m["outside"]
rest = outside * 1000 - m["crossing"] - m["penalize"] - m["sample"] - m["gc"]
llama.release(); del llama, engine, logits; gc.collect()
[f"(vocabulary {V}) | {m['js']:.3f} | {m['copy']:.1f} | {m['generate']:.3f} "
 f"| {outside:.3f} ({outside / m['generate'] * 100:.1f}%; {min(cells['outside']):.3f} to {max(cells['outside']):.3f}) "
 f"| {m['crossing']:.1f} | {m['ctypes']:.1f} | {m['penalize']:.1f} | {m['sample']:.1f} ({m['past']:.0f}) | {m['again']:.1f} | {m['alone']:.1f} "
 f"| {m['artificial']:.1f} ({past_floor(artificial)}) | {m['argmax']:.1f} | {m['gc']:.1f} | {rest:.1f} |",
 f"| {P} | {m['encode']:.3f} | {m['many']:.3f} | {m['prompt']:.3f} ({m['prompt'] / m['many']:.2f}×) |"]
`);
  const [token, prompt] = rows.toJs();
  rows.destroy();
  tokenRows.push(`| ${entry.name} ${token}`);
  promptRows.push(`| ${entry.name} ${prompt}`);
}
console.log("A generated token: ms, and µs a call");
console.log("| model | JS forward (ms) | copy (µs) | generate() (ms) | outside (ms, share; range) | crossing (µs) | ctypes (µs) | penalize (µs) | sample (µs; C) | again (µs) | 1 past (µs) | T158's (µs; C) | argmax (µs) | GC (µs) | the rest (µs) |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const row of tokenRows) console.log(row);
console.log("A prompt: the tokenizer's ms for the whole of it, and ms a token");
console.log("| model | tokens | encode() (ms) | forwardMany from JS | through generate() |");
console.log("|---|---:|---:|---:|---:|");
for (const row of promptRows) console.log(row);
console.log(`load after: ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
process.exit(0);
