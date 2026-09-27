// Where the page's conversion spends its time (T89): converts a Hugging Face model directory inside Pyodide in Node,
// fed in 8 MiB pieces as the worker feeds it, under cProfile. No download: the files are on disk, so this is the
// converting alone, and the rest of a load in the browser is the fetching.
//
//   node tests/profile-convert.mjs <directory with config.json, model.safetensors and the tokenizer> [int8|int6|float32] [--numpy]
//   node tests/profile-convert.mjs <a Q8_0 .gguf file> [int8|int6|float32] [--numpy]
//   node tests/profile-convert.mjs <directory with config.json, the tokenizer and a Q8_0 .gguf> [...]
//       (T145: a GGUF with its original's vocabulary and configuration, T136's second stage; tests/hf_fetch.py makes it)
//
// int8 quantizes on the SIMD kernels, as the page does (T89: kernel_quantizer); int6 (T98) as the page does too.
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
const [dir, dtype = "int8"] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const { pyodide: py } = await pyodideWithEngine();
const convert = py.pyimport("llama2_convert");
const quantizeRows = py.pyimport("llama2_numpy").kernel_quantizer("simdkernel.so");
// T123: the widening of bfloat16 on the kernels, as the page does; --numpy: NumPy's, to compare
const bfloat16 = process.argv.includes("--numpy") ? undefined : py.pyimport("llama2_numpy").kernel_widener("simdkernel.so");
// T136: and of GGUF's Q8_0
const q8_0 = process.argv.includes("--numpy") ? undefined : py.pyimport("llama2_numpy").kernel_q8_0("simdkernel.so");
const gguf = dir.endsWith(".gguf");
const withVocabulary = !gguf && fs.readdirSync(dir).filter((name) => name.endsWith(".gguf")).sort()[0];
const weights = gguf ? dir : withVocabulary ? `${dir}/${withVocabulary}` : `${dir}/model.safetensors`;
const fd = fs.openSync(weights, "r"), size = fs.fstatSync(fd).size;
const range = (b, e) => { const x = new Uint8Array(e - b); fs.readSync(fd, x, 0, e - b, b); return x; };
let conversion, first;
if (gguf) {
  // the header (the vocabulary) is a few megabytes: all of it is in the first 16 MiB
  conversion = convert.Conversion.from_gguf.callKwargs(range(0, 16 << 20), { dtype, quantize_rows: quantizeRows, bfloat16, q8_0 });
  first = conversion.base;
} else {
  const tokName = ["tokenizer.json", "spiece.model", "tokenizer.model"].find((n) => fs.existsSync(`${dir}/${n}`));
  const config = fs.readFileSync(`${dir}/config.json`, "utf8");
  let header;
  if (withVocabulary) {
    // the GGUF's header as a safetensors one, as the worker makes it (llama2_convert.gguf_weights)
    const made = convert.gguf_weights(range(0, Math.min(64 << 20, size)), config);
    [header, first] = made.toJs();
    made.destroy();
  } else {
    const headerBytes = Number(new DataView(range(0, 8).buffer).getBigUint64(0, true));
    [header, first] = [new TextDecoder().decode(range(8, 8 + headerBytes)), 8 + headerBytes];
  }
  conversion = convert.Conversion.callKwargs(header, first, config, new Uint8Array(fs.readFileSync(`${dir}/${tokName}`)),
    tokName, { dtype, start: first, quantize_rows: quantizeRows, bfloat16, q8_0 });
}
py.globals.set("conversion", conversion);
py.runPython("import cProfile, pstats, io, time; profiler = cProfile.Profile(); began = time.perf_counter(); profiler.enable()");
let js = 0;
for (let at = first; at < size; at += 8 << 20) {
  const t = performance.now(); const chunk = range(at, Math.min(at + (8 << 20), size)); js += performance.now() - t;
  conversion.feed(chunk);
}
conversion.finish();
console.log(py.runPython(`
profiler.disable()
total = time.perf_counter() - began
out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(12)
f"converted {${size} / 1e6:.0f} MB in {total:.1f} s ({${size} / 1e6 / total:.0f} MB/s)\\n" + "\\n".join(line for line in out.getvalue().splitlines() if line.strip())[:3000]
`));
console.log(`reading the file in JS: ${(js / 1000).toFixed(1)} s`);
