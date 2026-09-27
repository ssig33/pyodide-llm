// T145, (7) and (8) of the review of T144 (2026-09-26): what lays out a checkpoint besides its header (its form,
// llama2_numpy.FORM: bias, arch, qk_norm, head_dim) goes from the converter's sink.open() through worker.js's
// weightsBuffer() to forward.js's footprint(), under the same names and with the same defaults. A name changed on
// one side only (head_dim, headDim) raises nothing: footprint() counts dim / heads, and a Qwen3 0.6B's keys and values
// come out 45% short (T124). Node only, with the native Python for FORM (numpy):
//
//   node tests/worker-sink-check.mjs          (PYTHON=.venv/bin/python to take another Python)
//
// worker.js runs in a vm context with a few stand-ins (no Pyodide, no kernels, a memory of one page): its top level
// only declares, and its functions are what is called.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import * as forward from "../public/forward.js";

const root = new URL("..", import.meta.url);
const FORM = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c",
  "import json, sys; sys.path.insert(0, 'public'); import llama2_numpy; print(json.dumps(llama2_numpy.FORM))"],
{ cwd: fileURLToPath(root) }).toString());
assert.deepEqual(Object.keys(FORM).sort(), ["arch", "bias", "head_dim", "qk_norm"],
  "FORM has other keys now: say here which of them footprint() reads");

// (7) footprint()'s defaults are FORM's: a form without arch or head_dim (the options of a model converted before
// T124, a manifest) is counted as the engine reads it
const defaults = forward.footprint.toString();
assert.equal(/\barch = "(\w+)"/.exec(defaults)?.[1], FORM.arch, "footprint()'s default arch is not FORM's");
assert.equal(Number(/\bhead_dim = (\d+)/.exec(defaults)?.[1]), FORM.head_dim, "footprint()'s default head_dim is not FORM's");
const QWEN3 = [1024, 3072, 28, 16, 8, 151936, 4096];  // Qwen3 0.6B: heads of 128, not 1024 / 16
const GPT2 = [768, 3072, 12, 12, 12, 50257, 1024];
for (const header of [QWEN3, GPT2, [288, 768, 6, 6, 6, 32000, 256]]) {
  for (const dtype of ["int8", "float32"]) {
    assert.equal(forward.footprint(header, 600e6, { dtype }),
      forward.footprint(header, 600e6, { dtype, arch: FORM.arch, head_dim: FORM.head_dim }),
      `footprint() without a form is not footprint() with FORM's defaults (${header}, ${dtype})`);
  }
}

// (8) sink.open() -> weightsBuffer() -> footprint()
const at = new URL("public/worker.js", root);
const source = fs.readFileSync(at, "utf8").replaceAll("import.meta.url", JSON.stringify(at.href));
const context = vm.createContext({
  self: { navigator: {}, location: { search: "" }, crossOriginIsolated: false },
  console, performance, URL, TextDecoder, TextEncoder, setTimeout, clearTimeout, WebAssembly, Atomics, postMessage() {},
});
vm.runInContext(source, context, { filename: fileURLToPath(at) });
const counted = [];
let destroyed = 0;
context.stand = {
  forward: {
    ...forward,
    footprint: (...args) => {
      counted.push(args);
      return forward.footprint(...args);
    },
    weightsMemory: () => ({ memory: new WebAssembly.Memory({ initial: 1 }), base: 0 }),
    growMemory() {},
  },
  // with ?without=kernels: the checkpoint in a Python bytearray
  pyodide: { globals: { get: () => () => ({ destroy: () => destroyed++, getBuffer: () => ({ data: new Uint8Array(8), release() {} }) }) } },
};
vm.runInContext("forwardModule = stand.forward; pyodide = stand.pyodide; jsKernels = { relaxed: true }; " +
  "llama2_numpy = { KV_START: 256, OUTLIER_CHANNELS: 8 }; disabled = [];", context);
const proxy = (value) => ({ toJs: () => value, destroy() {} });
const opened = (header, form, dtype = "int8") => {
  counted.length = 0;
  const into = vm.runInContext("checkpointSink()", context);
  into.sink.open(600e6, proxy(header), dtype, proxy(form));
  assert.equal(counted.length, 1, "sink.open() did not size the memory by footprint()");
  const [header2, size, options] = counted[0];
  assert.deepEqual([...header2], header);
  assert.equal(size, 600e6);
  assert.equal(options.dtype, dtype);
  for (const key of Object.keys(FORM)) assert.equal(options[key], form[key], `${key} of the form did not reach footprint()`);
  return into;
};
const qwen3 = { ...FORM, qk_norm: true, head_dim: 128 };
opened(QWEN3, qwen3);
// what the test stands on: the head's size changes what footprint() counts (a Qwen3 0.6B, T124)
assert.ok(forward.footprint(QWEN3, 600e6, { ...qwen3, dtype: "int8" }) > 1.5 * forward.footprint(QWEN3, 600e6, { ...FORM, dtype: "int8" }));
opened(GPT2, { ...FORM, bias: true, arch: "gpt2" });
assert.notEqual(forward.footprint(GPT2, 600e6, { ...FORM, arch: "gpt2", dtype: "int8" }), forward.footprint(GPT2, 600e6, { ...FORM, dtype: "int8" }));

// T160 (the review): the type of the keys and values the worker sized the memory for (keysInHalf) is what it hands the
// engine (external's halfKeys). Without it createForward takes float32 for every grouped-query model: the same answer
// where float32 fits a 32-bit memory, and not for Qwen2.5 3B (3.82 GiB with float16, 4.03 with float32: out of memory
// at the last growth of the cache, near position 2048) or Llama 3.2 3B (64-bit either way: 0.66 GB more). No other
// test runs this path: forward-check and gpu-check make their engines themselves.
{
  const handed = [];
  context.stand.forward.external = (args) => {
    handed.push(args.halfKeys);
    return {};
  };
  vm.runInContext("disabled = []; sharedKernels = {}; wideKernels = { plain: {}, shared: {} }; threadsRequest = undefined; " +
    "llama2_numpy.Llama = { callKwargs: () => ({}) };", context);
  const cases = [["llm-jp-3 150M", [512, 2048, 12, 8, 8, 99584, 4096], 160e6, {}, true],
    ["Qwen2.5 0.5B", [896, 4864, 24, 14, 2, 151936, 4096], 555992604, { bias: true }, false],
    ["Qwen2.5 3B", [2048, 11008, 36, 16, 2, 151936, 4096], 3472375836, { bias: true }, true],
    ["Llama 3.2 3B", [3072, 8192, 28, 24, 8, 128256, 4096], 3614847004, {}, true]];
  const handedFor = (header, size, form) => {
    handed.length = 0;
    const into = vm.runInContext("checkpointSink()", context);
    into.sink.open(size, proxy(header), "int8", proxy({ ...FORM, ...form }));
    into.weights.llama({}, {});
    assert.equal(handed.length, 1, "the engine was not made through external()");
    return handed[0];
  };
  for (const isolated of [true, false]) {
    context.self.crossOriginIsolated = isolated;
    for (const [name, header, size, form, half] of cases) {
      const want = isolated && half;
      assert.equal(handedFor(header, size, form), want,
        `${name}${isolated ? "" : " (not isolated)"}: the worker hands the engine ${want ? "float16" : "float32"} keys and values`);
    }
  }
  context.self.crossOriginIsolated = false;
  vm.runInContext("sharedKernels = undefined; wideKernels = undefined; delete llama2_numpy.Llama;", context);
  console.log("ok: the worker hands the engine the type of keys and values it sized the memory for");
}

// the Python buffer of ?without=kernels: let go once, by another open() (another tokenizer) or by release()
vm.runInContext('disabled = ["kernels"];', context);
const into = vm.runInContext("checkpointSink()", context);
into.sink.open(600e6, proxy(QWEN3), "int8", proxy(qwen3));
assert.equal(destroyed, 0);
into.sink.open(600e6, proxy(QWEN3), "int8", proxy(qwen3));
assert.equal(destroyed, 1, "a second open() kept the first buffer");
into.release();
into.release();
assert.equal(destroyed, 2, "release() let go of the buffer not once");
console.log("ok: FORM's keys and defaults reach footprint() from sink.open()");
