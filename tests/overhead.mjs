// T164: what a token costs outside forward.js, in Node's Pyodide, as the page runs it (the forward pass in
// public/forward.js, the sampling on simdkernel.so, generate() in Python). Per token, at the positions 0..N-1:
//
//   JS forward     engine.forward() in a loop of JavaScript: the forward pass alone (tests/profile.mjs's number)
//   Python call    the same called from a loop of Python: one crossing from Python to JavaScript per token
//   greedy         llama.forward() and np.argmax(): the logits as Python sees them, and the greedy choice
//   penalize, sample   the page's sampling kernels on the real logits of each position (the model's settings:
//                  temperature 0.7, top-p 0.9, its repetition penalty), and how many tokens pass the floor
//                  (the best's 1e-7) that sample() sorts from
//   generate()     128 sampled tokens through generate() with those settings, the stop tokens off: 1000 / tok/s,
//                  the number the page shows. Minus JS forward is what the token costs outside forward.js
//   prompt         64 tokens' forwardMany() from JavaScript, and a prompt of 64 tokens through generate()
//
// The measures take turns in each round (AGENTS.md: the same process, alternating), and each cell is the median of
// the rounds.
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
// the loops of JavaScript: one crossing from Python, then nothing but forward.js
py.globals.set("js_forward", (engine, n) => {
  const began = performance.now();
  for (let pos = 0; pos < n; pos++) engine.forward(1, pos, true);
  return (performance.now() - began) / n;
});
py.globals.set("js_prompt", (engine, n) => {
  const block = Number(engine.promptBlock ?? 16), fed = new Array(n).fill(1);
  const began = performance.now();
  for (let at = 0; at < n; at += block) engine.forwardMany(fed.slice(at, at + block), at);
  return (performance.now() - began) / n;
});
console.log(`${os.cpus()[0]?.model ?? "?"} × ${os.cpus().length}, Node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}; ${rounds} rounds, ${tokens} tokens`);
console.log("| model | JS forward (ms/token) | Python call | greedy: + logits and argmax | penalize (µs) | sample (µs; tokens past the floor) | generate() (ms/token) | outside forward.js | prompt: forwardMany from JS (ms/token) | prompt through generate() |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const id of ids.length ? ids : ["llm-jp-3-150m", "tiny-lm"]) {
  const entry = MODELS.find((m) => m.id === id);
  py.globals.set("CHECKPOINT", root + entry.checkpoint);
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
  py.globals.set("OPTIONS", py.toPy(entry.options));
  py.globals.set("GEN", py.toPy(entry.generation));
  py.globals.set("PROMPT", entry.prompt ?? "Once upon a time");
  const line = py.runPython(`
import time, math, gc, statistics
import numpy as np
N, ROUNDS = ${tokens}, ${rounds}
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
engine = llama._external[0]
temperature, topp, penalty = GEN["temperature"], GEN.get("topp", 0.9), GEN.get("repetition_penalty", 1.0)

def python_call():
    run, began = engine.forward, time.perf_counter()
    for pos in range(N):
        run(1, pos, True)
    return (time.perf_counter() - began) / N * 1000

def greedy():
    token, began = llama.bos, time.perf_counter()
    for pos in range(N):
        token = int(np.argmax(llama.forward(token, pos)))
    return (time.perf_counter() - began) / N * 1000

def sampling():
    # the page's steps on the logits of each position, timed apart from the forward pass
    rng, token, history = np.random.default_rng(1), llama.bos, [llama.bos]
    penalized = sampled = 0.0
    past = []
    for pos in range(N):
        logits = llama.forward(token, pos)
        began = time.perf_counter()
        if penalty != 1.0:
            llama.penalize(logits, history, penalty)
        middle = time.perf_counter()
        token = llama.sample(logits, temperature, topp, rng)
        ended = time.perf_counter()
        penalized += middle - began
        sampled += ended - middle
        past.append(int(np.count_nonzero(logits >= logits.max() + temperature * math.log(1e-7))))
        history.append(token)
    return penalized / N * 1e6, sampled / N * 1e6, statistics.median(past)

def generated(steps, prompt=""):
    stops, llama.stop_tokens = llama.stop_tokens, ()
    try:
        for _ in llama.generate(prompt, steps=steps, temperature=temperature, topp=topp, repetition_penalty=penalty, seed=1):
            pass
    finally:
        llama.stop_tokens = stops
    return llama.stats

prompt = PROMPT
while len(llama.tokenizer.encode(prompt, llama.specials)) < 64:
    prompt += PROMPT
prompt_tokens = len(llama.tokenizer.encode(prompt, llama.specials))

cells = {k: [] for k in ("js", "call", "greedy", "penalize", "sample", "past", "generate", "many", "prompt")}
for r in range(ROUNDS + 1):  # the first round warms up and is left out
    js = js_forward(engine, N)
    call = python_call()
    greedy_ms = greedy()
    penalize_us, sample_us, past = sampling()
    stats = generated(N + 1)
    generate_ms = 1000 / stats["tokens_per_second"]
    many = js_prompt(engine, 64)
    stats2 = generated(prompt_tokens + 1, prompt)
    prompt_ms = stats2["prompt_seconds"] / stats2["prompt_tokens"] * 1000
    if r:
        for k, v in zip(cells, (js, call, greedy_ms, penalize_us, sample_us, past, generate_ms, many, prompt_ms)):
            cells[k].append(v)
m = {k: statistics.median(v) for k, v in cells.items()}
outside = m["generate"] - m["js"]
llama.release(); del llama, engine; gc.collect()
(f"| {m['js']:.3f} | {m['call']:.3f} (+{m['call'] - m['js']:.3f}) | {m['greedy']:.3f} (+{m['greedy'] - m['call']:.3f}) "
 f"| {m['penalize']:.1f} | {m['sample']:.1f} ({m['past']:.0f}) | {m['generate']:.3f} "
 f"| {outside:.3f} ({outside / m['generate'] * 100:.1f}%) | {m['many']:.3f} | {m['prompt']:.3f} ({prompt_tokens} tokens, {m['prompt'] / m['many']:.2f}×) |")
`);
  console.log(`| ${entry.name} ${line}`);
}
console.log(`load after: ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
process.exit(0);
