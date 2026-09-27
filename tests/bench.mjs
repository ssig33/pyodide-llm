// src/bench.js: the table a visitor pastes into an issue (T45). A plain module, so this runs in Node alone.
//
//   node tests/bench.mjs
import assert from "node:assert/strict";
import { FULL_ROUNDS, PASTE, QUESTIONS, REPORT_LIMIT, ROUNDS, TOO_LONG, benchMarkdown, cpuBaseline, cpuSummary, cpuTable, deviceSummary, environmentOf, gpuSummary, lineSummary,
         loginUrl, parseReport, reportBody, shortReport, storageSummary,
         generateTable, gpuSkipped, layerCheckNumbers, layerStepsTable, layerTable, matVecTable, PATH_PROMPTS, PATH_WRITES, pathTable, reportTooLong, reportUrl, reportsTable, tableCell, threadCounts, threadsKey, threadsLine, times, timesFaster, tokenTable } from "../src/bench.js";
import fs from "node:fs";

const rows = [
  { name: "everything", without: [], tokens: 64, speed: 334.62, seconds: 8.4, backend: "SIMD kernels, int8, relaxed SIMD" },
  { name: "without the kernels", without: ["kernels"], tokens: 64, speed: 44.81, seconds: 8.1, backend: "NumPy (without kernels)" },
];
const environment = environmentOf({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (X11)" },
                                  { model: "tiny-lm 29M", pyodide: "314.0.7", build: "abc1234", site: "https://example.invalid/" });
const markdown = benchMarkdown(rows, environment);

// a table GitHub renders: a header, the separator, and one row per round
const lines = markdown.split("\n");
assert.equal(lines.filter((line) => line.startsWith("|")).length, rows.length + 2, "one row per round, and the header");
assert.ok(lines.some((line) => line.includes("|---|---|---|---|")), "the separator GitHub needs");
assert.ok(markdown.includes("| everything | 334.6 | 8.4 s | SIMD kernels, int8, relaxed SIMD |"), markdown);
assert.ok(markdown.includes("8 logical cores") && markdown.includes("8 GB or more"), "what the browser told us");
assert.ok(markdown.includes("Mozilla/5.0 (X11)"), "the user agent, for a report that means something");
assert.ok(markdown.includes("Pyodide 314.0.7 · site abc1234"), "the site's version (T176), for which shaders and kernels ran");

// a browser that says nothing about itself must not make anything up
const bare = benchMarkdown(rows, environmentOf({}, {}));
assert.ok(!bare.includes("site "), "no version, no word of it");
assert.ok(!bare.includes("undefined") && !bare.includes("NaN"), bare);
assert.ok(!bare.includes("cores"), "no invented core count");

// a round that failed to measure leaves a question mark, never a wrong number
const broken = benchMarkdown([{ name: "everything", speed: undefined, backend: "" }], environmentOf({}, {}));
assert.ok(broken.includes("| everything | ? |"), broken);

assert.equal(ROUNDS.length, 2, "?bench=1 runs with and without the kernels");
assert.equal(FULL_ROUNDS.length, 6, "?bench=full walks the steps of T52, and T110's");
assert.deepEqual(FULL_ROUNDS.at(-1).without, [], "the last step is everything switched on");
// T91: the issue the page opens, and the table the issues make
const url = new URL(reportUrl(markdown, environment));
assert.equal(url.searchParams.get("template"), "benchmark.md");
assert.equal(url.searchParams.get("title"), "Benchmark: tiny-lm 29M");
assert.ok(url.searchParams.get("body").endsWith(markdown), "the page's Markdown, unchanged, at the end");
// the questions of the page are the template's, word for word (the parser reads what either of them wrote)
const template = fs.readFileSync(new URL("../.github/ISSUE_TEMPLATE/benchmark.md", import.meta.url), "utf8");
for (const [name, example] of QUESTIONS) assert.ok(template.includes(`**${name}**: (${example})`), `${name} in the template`);
// a visitor who answered two of the three, and one who changed nothing
const answered = reportBody(markdown).replace(/\*\*Device\*\*: \(.*\)/, "**Device**: Pixel 8")
  .replace(/\*\*Browser\*\*: \(.*\)/, "**Browser**: Chrome 148");
const report = parseReport(answered);
assert.deepEqual([report.device, report.os, report.browser, report.model, report.cores], ["Pixel 8", "", "Chrome 148", "tiny-lm 29M", "8"]);
assert.deepEqual(report.rows.map((row) => row.speed), [334.6, 44.8]);
assert.equal(parseReport("no table here"), undefined);
// a report from before T91 (the Markdown alone, "8 threads") still reads
const old = markdown.replace("8 logical cores", "8 threads");
const table = reportsTable([{ number: 7, url: "https://github.com/x/y/issues/7", body: answered },
                            { number: 8, url: "https://github.com/x/y/issues/8", body: reportBody(old) },
                            { number: 9, url: "https://github.com/x/y/issues/9", body: "Something else entirely" }]);
const tableLines = table.split("\n");
assert.equal(tableLines.length, 4, "a header, the separator, and the two issues that are reports");
assert.equal(tableLines[2], "| Pixel 8 | ? | Chrome 148 | tiny-lm 29M | 8 | 334.6 | 44.8 | SIMD kernels, int8, relaxed SIMD | [#7](https://github.com/x/y/issues/7) |");
assert.ok(tableLines[3].startsWith("| ? | ? | ? | tiny-lm 29M | 8 | 334.6 |"), tableLines[3]);
// T134: /benchmark/ writes the model's table first and its other sections after it, with tables and bold lines of
// their own: the report reads as the model's table alone
const everything = [markdown, "#### This browser\n\n| feature | here |\n|---|---|\n| WebAssembly SIMD | yes |",
  "#### GPU\n\n**Adapter**: apple · metal-3; max binding 2048 MiB\n\n| int8 matrix × vector | GPU |\n|---|---:|\n| Llama 3.2 1B w1 | 51.0 GB/s |",
  // T168: the device's ceilings, and the share of them in the prompt's table (with a % in its cells)
  "**The device's ceilings** (each a loop of that alone):\n\n| ceiling | GPU |\n|---|---:|\n| f32 multiply-adds | 1520.3 GFLOPS |\n" +
  "| f16 multiply-adds | no shader-f16 here |\n| int8 dots (dot4I8Packed) | 2710.0 GOPS |\n| reading the workgroup's memory (16-byte reads) | 310 GB/s |\n| reading a buffer (128 MiB) | 38.2 GB/s |",
  // T146: the prompt's table names its shaders with × and parentheses
  "| shader | tokens at once on the GPU | GPU ms | GPU ms a token | GFLOPS | of the ceiling |\n|---|---:|---:|---:|---:|---:|\n| batched (T135) | 64 | 352.8 | 5.51 | 44 | 2.9% of f32 |\n" +
  "| ORT DP4A 64×64 | 64 | 60.0 | 0.94 | 260 | 9.6% of int8 dots |\n\n**Fastest on the GPU**: at 64 tokens ORT DP4A 64×64, 0.94 ms a token (260 GFLOPS)"].join("\n\n");
const whole = parseReport(reportBody(everything).replace(/\*\*Device\*\*: \(.*\)/, "**Device**: iPhone 15"));
assert.deepEqual([whole.device, whole.model, whole.cores], ["iPhone 15", "tiny-lm 29M", "8"]);
assert.deepEqual(whole.rows.map((row) => row.name), ["everything", "without the kernels"]);
// T134: a report without the model's section has no table and no model, so that reportsTable() leaves it out rather
// than writing a row of question marks under a model that did not run
const deviceOnly = [benchMarkdown([], environmentOf({ hardwareConcurrency: 4, userAgent: "UA" }, {})),
  "#### GPU\n\n| int8 matrix × vector | GPU |\n|---|---:|\n| Llama 3.2 1B w1 | 51.0 GB/s |"].join("\n\n");
assert.ok(!deviceOnly.includes("| what ran |"), deviceOnly);
assert.equal(parseReport(reportBody(deviceOnly)), undefined);
assert.equal(reportsTable([{ number: 10, url: "u", body: reportBody(deviceOnly) }]).split("\n").length, 2, "no row for it");
assert.equal(new URL(reportUrl(deviceOnly, environmentOf({}, {}))).searchParams.get("title"), "Benchmark: this device");
// T134: a report too long for the address goes by the clipboard; the address says so and stays short
assert.ok(!reportTooLong(everything, environment));
const long = [everything, `#### Line\n\n${"| x | y |\n".repeat(600)}`].join("\n\n");
assert.ok(reportTooLong(long, environment));
const longUrl = reportUrl(long, environment);
assert.ok(loginUrl(long, environment).length <= REPORT_LIMIT, `${loginUrl(long, environment).length}`);
// what GitHub's login drops is the address once more encoded: a report of pipes and × (a table's) is too long long
// before its own address is (the second review of T134: 4,935 characters of it were dropped by the login)
for (let rows = 1; rows < 120; rows++) {
  const markdown = `#### GPU\n\n${"| Llama 3.2 1B · 16 tokens | 12.3 ms | 45.6 GB/s | ok × |\n".repeat(rows)}`;
  if (!reportTooLong(markdown, environment)) assert.ok(reportUrl(markdown, environment).length < 4935, `${rows} rows`);
  assert.ok(loginUrl(markdown, environment).length <= REPORT_LIMIT, `${rows} rows`);
}
assert.ok(new URL(longUrl).searchParams.get("body").endsWith(TOO_LONG));
assert.ok(new URL(longUrl).searchParams.get("body").includes("**Device**: ("), "the three questions stay");

// T157: the GPU's token table against the CPU section's forward pass. The owner's Android (T134's first report): the
// CPU section 11.2, 10.3 and 7.3 ms a token with 1, 2 and 4 threads (18.9, 20.5, 28.7 GB/s), its prompt 5.80, 4.50
// and 2.77; the GPU's packed token of Llama 3.2 1B 48.4 ms, which the old column called 1.44×
const cpuSection = { status: "ok", data: { shared: true, rows: [
  { asked: 1, threads: 1, msPerToken: 11.2, GBps: 18.9, promptMsPerToken: 5.80 },
  { asked: 2, threads: 2, msPerToken: 10.3, GBps: 20.5, promptMsPerToken: 4.50 },
  { asked: 4, threads: 4, msPerToken: 7.3, GBps: 28.7, promptMsPerToken: 2.77 },
  { asked: 8, threads: 3, none: "the browser did not start that many software threads" }] } };
const baseline = cpuBaseline(cpuSection);
assert.deepEqual(baseline, { token: { threads: 4, msPerToken: 7.3, GBps: 28.7, isolated: true }, prompt: { threads: 4, msPerToken: 2.77, isolated: true } });
// the fastest of each on its own: a device whose prompt is fastest at another count than its token
assert.equal(cpuBaseline({ status: "ok", data: { rows: [{ threads: 1, msPerToken: 5, GBps: 40, promptMsPerToken: 3 },
  { threads: 2, msPerToken: 6, GBps: 35, promptMsPerToken: 2 }] } }).prompt.threads, 2);
// nothing to hold against: why, and no estimate in its place
assert.equal(cpuBaseline(undefined).why, "run the CPU section for it");
assert.equal(cpuBaseline({ status: "error", markdown: "stopped" }).why, "the CPU section failed");
assert.equal(cpuBaseline({ status: "wrong", data: cpuSection.data }).why, "the CPU section computed something wrong");
assert.equal(cpuBaseline({ status: "none", markdown: "no WebAssembly SIMD" }).why, "the CPU section found nothing to run on here");
assert.ok(cpuBaseline({ status: "ok", data: { rows: [{ asked: 1, none: "x" }] } }).why);
// T163: the CPU section's table with the ceilings beside it: a token's reads against reading alone with as many threads
// (211 MB of checkpoint and 23.4 MB of corrections: 18.9 GB/s of checkpoint reads 21.0, 56% of 37.8), the prompt's
// G MAC/s with one thread against relaxed_dot with its loads (121.6 M weights in the layers: 5.80 ms a token is 21.0)
const ceilingsOf = (read, extra = {}) => ({ read, dot: { GMACs: 43.4 }, dotRegisters: { GMACs: 67.9 }, fma: { GMACs: 8.3 }, ...extra });
const cpuResult = { backend: "SIMD kernels", shared: true, megabytes: 211, tokenMegabytes: 234.4, layerWeights: 121634816, rows: cpuSection.data.rows,
  ceilings: ceilingsOf([{ threads: 1, GBps: 37.8 }, { threads: 2, GBps: 41 }, { threads: 4, GBps: 57.4, unsteady: true }, { threads: 8, error: "x | y" }]) };
const cpuLines = cpuTable(cpuResult);
assert.ok(cpuLines.includes("| 1 | 11.2 | 18.9 (56%) | 89.3 | 5.80 | 21.0 (48%) |"), cpuLines.join("\n"));
assert.ok(cpuLines.includes("| 2 | 10.3 | 20.5 (56%) | 97.1 | 4.50 | 27.0 |"), "no share for the prompt past one thread");
assert.ok(cpuLines.includes("| 4 | 7.3 | 28.7 | 137.0 | 2.77 | 43.9 |"), "no share against an unsteady ceiling");
assert.ok(cpuLines.includes("| 8 | the browser did not start that many software threads | | | | |"));
assert.ok(cpuLines.includes("| reading alone | 4 | unsteady: 57.4 GB/s |"));
assert.ok(cpuLines.includes("| reading alone | 8 | failed: x \\| y |"), "a | in a cell");
assert.ok(cpuLines.includes("| relaxed_dot with its two loads (int8) | 1 | 43.4 G MAC/s |"));
assert.ok(cpuLines.includes("| relaxed_dot, registers only | 1 | 67.9 G MAC/s |"));
assert.ok(cpuLines.includes("| f32 multiply + add, registers only | 1 | 8.3 G MAC/s |"));
// Safari: no relaxed SIMD, so no dot ceilings, no share for the prompt, and no corrections read (matmul_q8)
const safari = cpuTable({ ...cpuResult, tokenMegabytes: 211, ceilings: ceilingsOf([{ threads: 1, GBps: 37.8 }], { dot: { none: "no relaxed SIMD in this browser" }, dotRegisters: { none: "no relaxed SIMD in this browser" } }) });
assert.ok(safari.includes("| relaxed_dot, registers only | 1 | not in this browser |"));
assert.ok(safari.includes("| relaxed_dot with its two loads (int8) | 1 | not in this browser |"));
assert.ok(safari.includes("| 1 | 11.2 | 18.9 (50%) | 89.3 | 5.80 | 21.0 |"), safari.join("\n"));
// the ceilings could not start: the forward pass stands, the ceilings say why
const unstarted = cpuTable({ ...cpuResult, ceilings: { error: "Error: the ceilings' loops could not be fetched" } });
assert.ok(unstarted.includes("| 1 | 11.2 | 18.9 | 89.3 | 5.80 | 21.0 |") && unstarted.at(-1).startsWith("Not measured: Error"), unstarted.join("\n"));
// a report from before T163 (no ceilings, no layerWeights) still makes a table
assert.ok(cpuTable({ ...cpuResult, layerWeights: undefined, ceilings: undefined }).includes("| 1 | 11.2 | 18.9 | 89.3 | 5.80 | ? |"));
// the counts of threads: doubling up to the logical cores, and the cores themselves
assert.deepEqual(threadCounts(8), [1, 2, 4, 8]);
assert.deepEqual(threadCounts(6), [1, 2, 4, 6]);
assert.deepEqual(threadCounts(1), [1]);
assert.deepEqual(threadCounts(undefined), [1, 2, 4]);
// ratios of one run each: one decimal from 1, two significant digits below
assert.equal(timesFaster(2.77, 5.51), "0.50×");
assert.equal(times(1.44), "1.4×");
assert.equal(times(0.0712), "0.071×");
assert.equal(times(277.3), "277×");
assert.equal(timesFaster(undefined, 5.51), "");
assert.equal(timesFaster(2.77, NaN), "");
const tokenSteps = [
  { name: "a token of Llama 3.2 1B", result: { kind: "widen", GB: 1.39, dispatches: 241, msPerToken: 81.6, tokPerSecond: 12.25 } },
  { name: "a token of Llama 3.2 1B, packed int8", result: { kind: "packed", GB: 1.39, dispatches: 241, msPerToken: 48.4, tokPerSecond: 20.66 } },
  { name: "a token of Llama 3.2 1B, chosen on the GPU", result: { kind: "widen", sample: true, GB: 1.39, dispatches: 242, msPerToken: 59.6, tokPerSecond: 16.78 } },
  { name: "a token of Llama 3.2 3B", error: "could not hold | the weights\nat all" },
  { name: "a token of llm-jp-3 150M", result: { model: "llm-jp-3 150M", error: "the GPU did not take 0.2 GB of weights" } }];
const right = { widen: { ok: true }, packed: { ok: true }, argmax: { ok: true } };
const variants = [["with the CPU", tokenTable(tokenSteps, baseline, { check: right })], ["without it", tokenTable(tokenSteps, cpuBaseline(undefined))],
  ["a lost device", tokenTable(tokenSteps, baseline, { lost: "lost" })], ["a fallback adapter", tokenTable(tokenSteps, baseline, { fallback: true })],
  ["a WRONG argmax", tokenTable(tokenSteps, baseline, { check: { ...right, argmax: { ok: false } } })]];
// the page's reading of a row (src/pages/benchmark.astro's rendered(), and GitHub's): a \| is a | of a cell
const cellsOf = (line) => line.split(/(?<!\\)\|/).slice(1, -1);
for (const [label, table] of variants) {
  // every row of the table as many cells as its header: a report that renders
  const rowsOf = table.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + tokenSteps.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!table.join("\n").includes("undefined") && !table.join("\n").includes("NaN"), label);
}
const ratios = (table) => table.filter((line) => /^\| Llama 3\.2 1B/.test(line)).map((line) => cellsOf(line).at(-1).trim());
const [withCpu, withoutCpu, lostTable, fallbackTable, wrongTable] = variants.map(([, table]) => table);
// 1.39 GB at 28.7 GB/s is 20.6 tok/s: the packed token the old column called 1.44× is 1.0×, GPU tok/s over CPU tok/s
assert.ok(withCpu.includes("| Llama 3.2 1B, packed int8 | 1.39 | 241 | 48.4 | 20.7 | 20.6 | 1.0× |"), withCpu.join("\n"));
assert.deepEqual(ratios(withCpu), ["0.59×", "1.0×", "0.81×"]);
assert.ok(withCpu[0].startsWith("The CPU (an estimate): each model's weights at the CPU section's fastest, 28.7 GB/s with 4 software threads."), withCpu[0]);
assert.ok(withoutCpu[0].includes("run the CPU section for it"), withoutCpu[0]);
assert.ok(withoutCpu.includes("| Llama 3.2 1B | 1.39 | 241 | 81.6 | 12.3 |  |  |"), withoutCpu.join("\n"));
// no ratio from a lost device (its times too fast), a fallback adapter, or a shader the check found wrong
assert.deepEqual(ratios(lostTable), ["", "", ""]);
assert.ok(lostTable.some((line) => line.includes("the device was lost")));
assert.deepEqual(ratios(fallbackTable), ["", "", ""]);
assert.deepEqual(ratios(wrongTable), ["0.59×", "1.0×", ""]);
assert.ok(wrongTable.some((line) => line.startsWith("| Llama 3.2 1B, chosen on the GPU (WRONG in the check) |")));
assert.deepEqual(ratios(tokenTable(tokenSteps, baseline, { check: { ...right, packed: { ok: false, error: "no dot4I8Packed" } } })), ["0.59×", "", "0.81×"]);
// a page that was not cross-origin isolated has one thread, and says so
const alone = cpuBaseline({ status: "ok", data: { shared: false, rows: [cpuSection.data.rows[0]] } });
assert.ok(tokenTable(tokenSteps, alone)[0].includes("18.9 GB/s with 1 software thread (not cross-origin isolated)."));
// an error's | and line breaks stay in their cell
assert.equal(tableCell("a | b\n c"), "a \\| b c");
assert.ok(withCpu.some((line) => line.includes("could not hold \\| the weights at all")));
// and the report with it still reads as the model's table alone
const withGpu = parseReport(reportBody([markdown, "#### GPU", ...withCpu].join("\n\n")));
assert.deepEqual(withGpu.rows.map((row) => row.name), ["everything", "without the kernels"]);
// T149: the matrix × vector table holds each GB/s against the buffer's reads, but not after a lost device (its times
// are no GPU's, and too fast: past 100%) nor on a fallback adapter; a failure's | stays in its cell
const matVecSteps = [{ name: "bandwidth: Llama 3.2 1B w1", result: { rows: [{ shader: "widened (T134)", check: "widen", GBps: 20 },
  { shader: "llama.cpp MMVQ, 4 rows", check: "llama.cpp MMVQ, 4 rows", error: "a | b" }], cpu: { GBps: 9 } } }];
const matVecCeilings = { global: { GBps: 40 } };
const matVecRow = (lines) => lines.find((line) => line.startsWith("| widened (T134) |"));
assert.equal(matVecRow(matVecTable(matVecSteps, {}, matVecCeilings)), "| widened (T134) | 20.0 GB/s (50.0%) |");
assert.equal(matVecRow(matVecTable(matVecSteps, {}, matVecCeilings, { lost: "lost" })), "| widened (T134) | 20.0 GB/s |");
assert.ok(matVecTable(matVecSteps, {}, matVecCeilings, { lost: "lost" }).some((line) => line.includes("the device was lost")));
assert.equal(matVecRow(matVecTable(matVecSteps, {}, { ...matVecCeilings, fallback: true })), "| widened (T134) | 20.0 GB/s |");
assert.ok(matVecTable(matVecSteps, {}, matVecCeilings).includes("| llama.cpp MMVQ, 4 rows | failed: a \\| b |"));
// T150: the layer's table: "faster than the separate steps" beside a fused row, against the separate steps of the
// same reduction, but none after a lost device, on a fallback adapter, for a row the check found WRONG or an unsteady
// one; the share of the buffer's reads beside GB/s, but not after a lost device or on a fallback adapter; every row
// as many cells as the header, a failure's | in its cell
const layerStep = { name: "a layer of a token", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, GB: 0.0684, rows: [
  { form: "llama.cpp, separate steps", check: "a layer, llama.cpp, separate steps", base: "llama.cpp", fused: false, subgroups: false, dispatches: 14, msPerLayer: 4.2, GBps: 16.3 },
  { form: "llama.cpp, fused (T150)", check: "a layer, llama.cpp, fused (T150)", base: "llama.cpp", fused: true, subgroups: false, dispatches: 5, msPerLayer: 2.1, GBps: 32.6 },
  { form: "llama.cpp, separate steps, subgroups", check: "a layer, llama.cpp, separate steps, subgroups", base: "llama.cpp", fused: false, subgroups: true, error: "a | b\nc" },
  { form: "llama.cpp, fused (T150), subgroups", check: "a layer, llama.cpp, fused (T150), subgroups", base: "llama.cpp", fused: true, subgroups: true, dispatches: 5, msPerLayer: 1.9, GBps: 36 },
  { form: "DP4A, separate steps", check: "a layer, DP4A, separate steps", base: "DP4A", fused: false, subgroups: false, dispatches: 18, msPerLayer: 3.0, GBps: 22.8 },
  { form: "DP4A, fused (T175), the norms apart", check: "a layer, DP4A, fused (T175), the norms apart", base: "DP4A", fused: true, normApart: true, subgroups: false, dispatches: 11, msPerLayer: 2.0, GBps: 34.2 },
  { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", base: "DP4A", fused: true, subgroups: false, dispatches: 9, msPerLayer: 1.5, GBps: 45.6 }] } };
const layerRight = { "a layer, llama.cpp, separate steps": { ok: true }, "a layer, llama.cpp, fused (T150)": { ok: true }, "a layer, DP4A, separate steps": { ok: true },
  "a layer, DP4A, fused (T175)": { ok: true } };
const layerCeilings = { global: { GBps: 40 } };
const fasterOf = (lines, form) => cellsOf(lines.find((line) => line.startsWith(`| ${form} |`) || line.startsWith(`| ${form} (WRONG`))).at(-1).trim();
const layerLines = layerTable(layerStep, layerRight, layerCeilings);
assert.ok(layerLines.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 (81.5%) | 33.6 | 2.0× |"), layerLines.join("\n"));
// T175: the DP4A forms against DP4A's separate steps (not llama.cpp's: 4.2 / 1.5 would say 2.8×)
assert.ok(layerLines.includes("| DP4A, fused (T175) | 9 | 1.50 | 45.6 (114.0%) | 24.0 | 2.0× |"), layerLines.join("\n"));
assert.ok(layerLines.includes("| DP4A, fused (T175), the norms apart | 11 | 2.00 | 34.2 (85.5%) | 32.0 | 1.5× |"), layerLines.join("\n"));
assert.equal(fasterOf(layerLines, "llama.cpp, separate steps"), "");
assert.equal(fasterOf(layerLines, "DP4A, separate steps"), "");
// no separate steps measured with subgroups: nothing to hold the fused one against
assert.equal(fasterOf(layerLines, "llama.cpp, fused (T150), subgroups"), "");
assert.ok(layerLines.includes("| llama.cpp, separate steps, subgroups | failed: a \\| b c | | | | |"), layerLines.join("\n"));
// a device without the packed int8 dot: the DP4A rows say so, and nothing is held against them
const noPackedStep = { ...layerStep, result: { ...layerStep.result, rows: layerStep.result.rows.map((row) => (row.base === "DP4A" ? { form: row.form, check: row.check, base: row.base, fused: row.fused, none: "no packed int8 dot here" } : row)) } };
const noPackedLines = layerTable(noPackedStep, layerRight, layerCeilings);
assert.ok(noPackedLines.includes("| DP4A, fused (T175) | not here: no packed int8 dot here | | | | |"), noPackedLines.join("\n"));
for (const [label, lines] of [["right", layerLines], ["lost", layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" })],
  ["fallback", layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true })], ["no check", layerTable(layerStep)], ["no packed", noPackedLines]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + layerStep.result.rows.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
}
const lostLayer = layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" });
assert.equal(fasterOf(lostLayer, "llama.cpp, fused (T150)"), "");
assert.ok(lostLayer.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), lostLayer.join("\n"));
assert.ok(lostLayer.some((line) => line.includes("the device was lost")));
const fallbackLayer = layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true });
assert.equal(fasterOf(fallbackLayer, "DP4A, fused (T175)"), "");
assert.ok(fallbackLayer.includes("| llama.cpp, fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), fallbackLayer.join("\n"));
const layerWrong = { ...layerRight, "a layer, DP4A, fused (T175)": { ok: false } };
assert.equal(fasterOf(layerTable(layerStep, layerWrong), "DP4A, fused (T175)"), "");
assert.ok(layerTable(layerStep, layerWrong).some((line) => line.startsWith("| DP4A, fused (T175) (WRONG in the check) |")));
// the separate steps WRONG: nothing right to hold the fused one against
assert.equal(fasterOf(layerTable(layerStep, { ...layerRight, "a layer, DP4A, separate steps": { ok: false } }), "DP4A, fused (T175)"), "");

// T202: where a layer's time goes. The times are multiples of 1/64 ms, so that every sum below is exact: the fused
// form's steps sum to 3.25 ms (matrices 2.75, attention 0.25, norms and quantizing 0.25), its layer 3.5 (the chain
// 0.25), the matrices alone 2.0 (fusing adds 0.75); the form with the norms apart sums to 3.375 against a layer of
// 3.25 (the chain less than nothing, as it can come out)
const NQ = "the norm with its quantizing (NORM_QUANTIZE)", NORM = "the norm (RMSNORM)", QD = "a vector of 2048 quantized (QUANTIZE)",
  QH = "a vector of 8192 quantized (QUANTIZE)", QKV = "q, k and v with RoPE and the cache", ATT = "the attention (flash attention's tile)",
  O = "o with the residual's add", GLU = "gate and up with SwiGLU", DOWN = "down with the residual's add";
const stepsStep = { name: "the steps of a layer", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, copies: 2, GB: 0.0684, dp4a: true,
  forms: [
    { form: "DP4A, fused (T175), the norms apart", check: "a layer, DP4A, fused (T175), the norms apart", normApart: true, dispatches: 11, ms: 3.25, n: 8,
      steps: [[NORM, 2], [QD, 3], [QKV, 1], [ATT, 1], [O, 1], [GLU, 1], [QH, 1], [DOWN, 1]].map(([step, count]) => ({ step, count })) },
    { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", normApart: false, dispatches: 9, ms: 3.5, n: 8,
      steps: [[NQ, 2], [QKV, 1], [ATT, 1], [QD, 1], [O, 1], [GLU, 1], [QH, 1], [DOWN, 1]].map(([step, count]) => ({ step, count })) }],
  steps: [
    { step: NORM, kind: "small", ms: 0.0625 }, { step: QD, kind: "small", ms: 0.0625 }, { step: QKV, kind: "matrix", matrix: "qkv", ms: 0.25 },
    { step: ATT, kind: "attention", ms: 0.25 }, { step: O, kind: "matrix", matrix: "o", ms: 0.25 }, { step: GLU, kind: "matrix", matrix: "gateUp", ms: 1.5 },
    { step: QH, kind: "small", ms: 0.0625 }, { step: DOWN, kind: "matrix", matrix: "down", ms: 0.75 }, { step: NQ, kind: "small", ms: 0.0625 },
    { step: "q, k and v alone", kind: "alone", matrix: "qkv", ms: 0.125 }, { step: "o alone", kind: "alone", matrix: "o", ms: 0.125 },
    { step: "gate and up alone", kind: "alone", matrix: "gateUp", ms: 1.25 }, { step: "down alone", kind: "alone", matrix: "down", ms: 0.5 },
    { step: "a dispatch of one workgroup (SMALL: a vector of dim added)", kind: "floor", ms: 0.015625 }] } };
const stepsLines = layerStepsTable(stepsStep, layerRight, layerCeilings);
const stepsText = stepsLines.join("\n");
assert.ok(stepsLines.includes(`| ${NQ} | 62.5 |  | 2 × = 125.0 |`), stepsText);
assert.ok(stepsLines.includes(`| ${QD} | 62.5 | 3 × = 187.5 | 1 × = 62.5 |`), stepsText);
assert.ok(stepsLines.includes("| gate and up alone | 1250.0 |  |  |"), stepsText);
assert.ok(stepsText.includes('- "DP4A, fused (T175)", 9 dispatches: the layer 3.50 ms. Its steps one by one 3.25 ms (the matrices with what they write 2.75, ' +
  "the attention 0.25, the norms and quantizing 0.25); the layer less them, what the chain costs beyond each step alone: 0.25 ms. " +
  "The matrices alone 2.00 ms (34.2 GB/s, 85.5% of the buffer's reads (40.0 GB/s)); the layer less them, 1.50 ms, is what fusing adds to the matrices 0.75 " +
  "+ the attention 0.25 + the norms and quantizing 0.25 + the chain 0.25. A dispatch of one workgroup takes 15.6 µs: 9 of them 0.14 ms."), stepsText);
assert.ok(stepsText.includes('- "DP4A, fused (T175), the norms apart", 11 dispatches: the layer 3.25 ms. Its steps one by one 3.38 ms'), stepsText);
assert.ok(stepsText.includes("what the chain costs beyond each step alone: −0.13 ms"), stepsText);
// the fewest MB read before the same weights again (T202's review: a matrix alone on ranges of all the copies), and
// without it a word, never "undefined"
assert.ok(layerStepsTable({ ...stepsStep, result: { ...stepsStep.result, cycleMB: 75.5 } }, layerRight, layerCeilings)[0].includes("(none read again before 76 MB of other weights: not from the GPU's caches)"));
assert.ok(stepsLines[0].includes("(none read again soon: not from the GPU's caches)"), stepsLines[0]);
// the two DP4A forms share their matrices and attention: their layers less their norms and quantizing (3.25 − 0.375
// and 3.5 − 0.25) are how far the run's times move; a form that failed, or forms of other matrices, have no such line
assert.ok(stepsLines.includes("The two forms run the same matrices and attention and differ only in their norms and quantizing, so their layers less those should be the same: " +
  "2.88 and 3.25 ms, 0.38 apart. That is about how far this run's times move: read the chain, and any part, only where it is larger."), stepsText);
// every row as wide as the head, nothing undefined, NaN or empty: right, a lost device, a fallback adapter, no check,
// a step that failed, one unsteady, and a form that failed
const stepsFailed = { ...stepsStep, result: { ...stepsStep.result, steps: stepsStep.result.steps.map((one) => (one.step === ATT ? { step: ATT, kind: "attention", error: "a | b\nc" } : one)) } };
const stepsUnsteady = { ...stepsStep, result: { ...stepsStep.result, steps: stepsStep.result.steps.map((one) => (one.step === GLU ? { ...one, unsteady: true } : one)) } };
const stepsFailedForm = { ...stepsStep, result: { ...stepsStep.result, forms: [stepsStep.result.forms[0], { form: "DP4A, fused (T175)", check: "a layer, DP4A, fused (T175)", error: "refused" }] } };
assert.ok(!layerStepsTable(stepsFailedForm, layerRight, layerCeilings).some((line) => line.startsWith("The two forms")));
const otherMatrices = { ...stepsStep, result: { ...stepsStep.result, forms: stepsStep.result.forms.map((form, i) => (i ? form : { ...form, steps: form.steps.map((s) => (s.step === QKV ? { ...s, step: O } : s)) })) } };
assert.ok(!layerStepsTable(otherMatrices, layerRight, layerCeilings).some((line) => line.startsWith("The two forms")));
for (const [label, lines] of [["right", stepsLines], ["lost", layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" })],
  ["fallback", layerStepsTable(stepsStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true })], ["no check", layerStepsTable(stepsStep)],
  ["a step failed", layerStepsTable(stepsFailed, layerRight, layerCeilings)], ["unsteady", layerStepsTable(stepsUnsteady, layerRight, layerCeilings)],
  ["a form failed", layerStepsTable(stepsFailedForm, layerRight, layerCeilings)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + stepsStep.result.steps.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
}
// a step that failed says so in its row, and the sums it is in are "?" rather than a number that leaves it out
const stepsFailedLines = layerStepsTable(stepsFailed, layerRight, layerCeilings);
assert.ok(stepsFailedLines.includes(`| ${ATT} | failed: a \\| b c | 1 × = ? | 1 × = ? |`), stepsFailedLines.join("\n"));
assert.ok(stepsFailedLines.join("\n").includes("Its steps one by one ? ms"), stepsFailedLines.join("\n"));
assert.ok(layerStepsTable(stepsUnsteady, layerRight, layerCeilings).includes(`| ${GLU} | unsteady: 1500.0 | 1 × = 1500.0 | 1 × = 1500.0 |`));
assert.ok(layerStepsTable(stepsUnsteady, layerRight, layerCeilings).some((line) => line.startsWith("Unsteady:")));
assert.ok(layerStepsTable(stepsFailedForm, layerRight, layerCeilings).includes('- "DP4A, fused (T175)": failed: refused'));
// no share of the buffer's reads after a lost device; a form the check found WRONG says so
assert.ok(!layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" }).join("\n").includes("of the buffer's reads"));
assert.ok(layerStepsTable(stepsStep, layerRight, layerCeilings, { lost: "lost" }).some((line) => line.includes("the device was lost")));
assert.ok(layerStepsTable(stepsStep, layerWrong, layerCeilings).join("\n").includes('in "DP4A, fused (T175)" (WRONG in the check), µs a layer'));
assert.deepEqual(layerStepsTable({ name: "the steps of a layer", error: "no | here" }), ["**Where a layer's time goes**: no \\| here"]);
// T208: the forms chosen by the layer table's fastest (named in the head, with the partner's norms), or by the packed
// int8 dot where it gave none (and why); the attention alone on its caches; the spares; the whole layer by timestamps
// in one line, or why not
const chosenStep = { ...stepsStep, result: { ...stepsStep.result, chosen: { by: "layer", fastest: "DP4A, fused (T175)" }, spares: 2, cycleMB: 137.2,
  caches: { count: 512, MB: 134.2 }, timestamps: { layers: 32, rounds: 5, forms: [{ form: "DP4A, fused (T175), the norms apart", ms: 3.125 }, { form: "DP4A, fused (T175)", ms: 3.375 }] } } };
const chosenLines = layerStepsTable(chosenStep, layerRight, layerCeilings), chosenText = chosenLines.join("\n");
assert.ok(chosenLines[0].includes('the fastest fused layer of the table above that the check found right, "DP4A, fused (T175)", as the engine chooses it (T152), ' +
  'and beside it the same with its norms apart: "DP4A, fused (T175), the norms apart" and "DP4A, fused (T175)"'), chosenLines[0]);
assert.ok(chosenLines[0].includes("and of 2 spare matrices of random weights (none read again before 137 MB of other weights"), chosenLines[0]);
assert.ok(chosenLines[0].includes("The attention alone reads 512 copies of the cache"), chosenLines[0]);
assert.ok(chosenText.includes('The whole layer by the GPU\'s own clock (timestamp-query), a check of the times above: "DP4A, fused (T175), the norms apart" 3.13 ms (above: 3.25), ' +
  '"DP4A, fused (T175)" 3.38 ms (above: 3.50) a layer, each the median of 5 submissions of 32 layers'), chosenText);
const oneForm = { ...chosenStep, result: { ...chosenStep.result, chosen: { by: "layer", fastest: "llama.cpp, fused (T150), subgroups" },
  forms: [{ ...stepsStep.result.forms[1], form: "llama.cpp, fused (T150), subgroups", check: "a layer, llama.cpp, fused (T150), subgroups" }], timestamps: { none: "no | clock" } } };
const oneFormLines = layerStepsTable(oneForm, layerRight, layerCeilings);
assert.ok(oneFormLines[0].includes('as the engine chooses it (T152): "llama.cpp, fused (T150), subgroups")'), oneFormLines[0]);
assert.ok(oneFormLines.includes("The whole layer by the GPU's own clock (timestamp-query), a check of the times above: not here: no \\| clock."), oneFormLines.join("\n"));
const byPacked = layerStepsTable({ ...stepsStep, result: { ...stepsStep.result, chosen: { by: "packed", why: "no layer table was timed before" } } }, layerRight, layerCeilings);
assert.ok(byPacked[0].includes("by the packed int8 dot alone (no layer table was timed before: ONNX Runtime's DP4A, the packed int8 dot is here)"), byPacked[0]);
assert.ok(layerStepsTable({ ...chosenStep, result: { ...chosenStep.result, timestamps: { error: "refused" } } }, layerRight, layerCeilings)
  .some((line) => line.endsWith("a check of the times above: failed: refused.")));
assert.ok(!stepsText.includes("GPU's own clock"), "no line where no timestamps were taken (a result before T208)");
for (const [label, lines] of [["chosen", chosenLines], ["one form", oneFormLines], ["by packed", byPacked],
  ["timestamps lost", layerStepsTable({ ...chosenStep, result: { ...chosenStep.result, timestamps: { layers: 32, rounds: 5, forms: [{ form: "DP4A, fused (T175)", ms: NaN }] } } }, layerRight, layerCeilings)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), `${label}: ${lines.join("\n")}`);
}
const unsteadyStep = { ...layerStep, result: { ...layerStep.result, rows: layerStep.result.rows.map((row) => (row.form === "llama.cpp, fused (T150)" ? { ...row, unsteady: true } : row)) } };
assert.ok(layerTable(unsteadyStep, layerRight).includes("| llama.cpp, fused (T150) | 5 | unsteady: 2.10 | 32.6 | 33.6 |  |"));
// a fallback adapter's few hundredths of a GB/s still show
const slowStep = { ...layerStep, result: { ...layerStep.result, rows: [{ ...layerStep.result.rows[0], msPerLayer: 2900, GBps: 0.0236 }] } };
assert.ok(layerTable(slowStep, layerRight, undefined, { fallback: true }).includes("| llama.cpp, separate steps | 14 | 2900.00 | 0.024 | 46400.0 |  |"));
assert.equal(layerTable({ name: "a layer of a token", error: "x | y" })[0], "**A layer of a token**: x \\| y");
// T151: the table of tokens generated on the GPU: what a submission costs besides its tokens and how many times faster
// several a submission are than one, neither after a lost device, on a fallback adapter or with the sampling WRONG
const generateStep = { name: "tokens generated on the GPU", result: { model: "Llama 3.2 1B's width", layers: 2, vocab: 32000, GB: 0.21,
  dispatches: 13, chunkDispatches: 16, tokens: 16, settings: { temperature: 0.7, topp: 0.9, penalty: 1.3 }, work: { ms: 9.1 }, chunkWork: { ms: 4.25 },
  rows: [{ perSubmission: 1, msPerToken: 14.2, fixedMs: 5.1, chunks: 9.3 }, { perSubmission: 4, msPerToken: 10.4, fixedMs: 5.2, chunks: 5.5 },
    { perSubmission: 16, msPerToken: 9.4, fixedMs: 4.8, chunks: 4.51 }],
  sampling: { vocab: 128256, msEach: 0.31, over: 5114, flat: { msEach: 1.2, over: 128256, unsteady: true } } } };
const generateRight = { sampling: { ok: true }, "sampling in chunks": { ok: true }, "tokens on the GPU": { ok: true } };
const generateLines = generateTable(generateStep, generateRight);
// T175: the layer the tokens ran, by its name in the layer table
assert.ok(generateTable({ ...generateStep, result: { ...generateStep.result, layer: "DP4A, fused (T175)" } }, generateRight)[0].includes(`the layer table's "DP4A, fused (T175)"`));
assert.ok(generateLines.includes("| 1, each read back as it comes | 14.20 | 9.30 | 5.10 | 5.10 |  |"), generateLines.join("\n"));
assert.ok(generateLines.includes("| 16, read back once | 9.40 | 4.51 | 4.80 | 0.30 | 1.5× |"), generateLines.join("\n"));
assert.ok(generateLines[0].includes("(16 dispatches a token)"), generateLines[0]);
assert.ok(generateLines.at(-1).includes("9.10 ms (with the sampling in chunks 4.25 ms)"), generateLines.at(-1));
// T191: the sampling alone in one workgroup and in chunks, and the chunks' column WRONG where their check is
const bothSamplers = generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, msEach: 5.893, over: 3695,
  chunks: { msEach: 0.812 }, flat: { msEach: 6.104, over: 128256, chunks: { error: "a | b" } } } } }, generateRight).at(-1);
assert.ok(bothSamplers.includes("one workgroup 5.893 ms (3695 tokens over the floor), in chunks 0.812 ms; on flat logits, one workgroup 6.104 ms " +
  "(128256 tokens over the floor), in chunks failed: a \\| b."), bothSamplers);
// T191's review: of the sampling in chunks, its last stage alone (SAMPLE's search), and a timing that failed on either
const withLast = generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, error: "c | d", over: 3695,
  chunks: { msEach: 0.812, pick: { msEach: 0.25, unsteady: true } } } } }, generateRight).at(-1);
assert.ok(withLast.includes("one workgroup failed: c \\| d (3695 tokens over the floor), in chunks 0.812 ms (of it the last stage, one workgroup's " +
  "nucleus and draw: unsteady: 0.250 ms)."), withLast);
assert.ok(withLast.includes("the same settings but no penalty"), withLast);
const chunksWrong = generateTable(generateStep, { ...generateRight, "sampling in chunks": { ok: false } });
assert.ok(chunksWrong.includes("| 4, read back once | 10.40 | 5.50 (WRONG in the check) | 5.20 | 1.30 | 1.4× |"), chunksWrong.join("\n"));
assert.ok(chunksWrong[0].includes("The check found the sampling in chunks WRONG."), chunksWrong[0]);
assert.ok(generateLines.at(-1).includes("9.10 ms") && generateLines.at(-1).includes("0.310 ms (5114 tokens over the floor)"), generateLines.at(-1));
assert.ok(generateLines.at(-1).includes("on flat logits, unsteady: 1.200 ms (128256 tokens over the floor)."), generateLines.at(-1));
assert.ok(generateTable({ ...generateStep, result: { ...generateStep.result, sampling: { vocab: 128256, msEach: 0.31 } } }, generateRight).at(-1).endsWith("0.310 ms."));
for (const [label, lines] of [["right", generateLines], ["lost", generateTable(generateStep, generateRight, { lost: "lost" })],
  ["fallback", generateTable(generateStep, generateRight, { fallback: true })], ["wrong", generateTable(generateStep, { ...generateRight, sampling: { ok: false } })],
  ["no check", generateTable(generateStep)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + generateStep.result.rows.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
  if (label !== "right" && label !== "no check") {
    assert.ok(rowsOf.slice(2).every((line) => cellsOf(line).slice(3).every((cell) => cell.trim() === "")), `${label}: no derived numbers\n${lines.join("\n")}`);
  }
}
const noisy = { ...generateStep, result: { ...generateStep.result, rows: [generateStep.result.rows[0], { perSubmission: 8, msPerToken: 9.0, fixedMs: -0.8 }] } };
assert.ok(generateTable(noisy, generateRight).includes("| 8, read back once | 9.00 |  | under the noise | under the noise | 1.6× |"), generateTable(noisy, generateRight).join("\n"));
assert.ok(generateTable(generateStep, { ...generateRight, "tokens on the GPU": { ok: false } }).some((line) => line.startsWith("| 4, read back once (WRONG in the check) |")));
assert.ok(generateTable(generateStep, generateRight, { fallback: true }).some((line) => line.includes("none on a fallback adapter")));
assert.equal(generateTable({ name: "x", error: "a | b" })[0], "**Tokens generated on the GPU**: a \\| b");
const unmeasured = generateTable({ ...generateStep, result: { ...generateStep.result, work: undefined, sampling: undefined, rows: [{ perSubmission: 1, msPerToken: 900 }, { perSubmission: 2, msPerToken: 800 }] } }, generateRight, { fallback: true });
assert.ok(unmeasured.at(-1).includes("not measured here"), unmeasured.at(-1));
// T184: the model page's own path, one table: a device whose GPU took the prompts, one whose GPU stopped while timed,
// CI's fallback adapter, a model the GPU does not take, a failure
const cell = (speed, low, high, gpuTokens, unsteady = false) => ({ speed, low, high, gpuTokens, unsteady });
const real = { threads: 4, gpu: { seconds: 4.23, matrices: "TF.js 64×64", attention: "llama.cpp flash" }, status: "prompts of 32 tokens and more on WebGPU, answers on the CPU (faster here)",
  rows: [{ what: "prompt", tokens: 64, chosen: cell(820.4, 800, 830, 64), cpu: cell(612, 600, 640, 0, true), gpu: cell(830, 810, 840, 64) },
         { what: "prompt", tokens: 256, chosen: cell(1300, 1280, 1310, 192), cpu: cell(650, 640, 655, 0), gpu: cell(1310, 1300, 1320, 256) },
         { what: "generation", tokens: 64, chosen: { same: "cpu" }, cpu: cell(80.44, 79.1, 81.2), gpu: { skip: "not on the GPU yet" } }] };
const pathLines = pathTable(real, "llm-jp-3 150M").split("\n");
assert.equal(pathLines[0], "**The model page's path** (llm-jp-3 150M): 4 software threads · WebGPU ready in 4.2 s · matrices by TF.js 64×64 · attention by llama.cpp flash · prompts of 32 tokens and more on WebGPU, answers on the CPU (faster here)");
assert.ok(pathLines.includes("| a prompt of 64 tokens | 820 tok/s (800–830), GPU | 612 tok/s (600–640, unsteady) | 830 tok/s (810–840) | 1.4× |"), pathLines.join("\n"));
assert.ok(pathLines.includes("| a prompt of 256 tokens | 1300 tok/s (1280–1310), GPU 192 of 256 | 650 tok/s (640–655) | 1310 tok/s (1300–1320) | 2.0× |"), pathLines.join("\n"));
assert.ok(pathLines.includes("| writing 64 tokens | same as CPU only | 80.4 tok/s (79.1–81.2) | not on the GPU yet |  |"), pathLines.join("\n"));
const pathRows = pathLines.filter((line) => line.startsWith("|"));
assert.equal(pathRows.length, 2 + real.rows.length, "a header, the separator, a row each: 3 to 6 rows (T185 keeps the report short)");
for (const line of pathRows) assert.equal(cellsOf(line).length, cellsOf(pathRows[0]).length, line);
// the review's must-fix: a GPU that stopped while the sides were timed says so, and no ratio stands (T157)
const stopped = pathTable({ ...real, gpu: { ...real.gpu, lost: "the GPU failed on a block of the prompt" },
  rows: [real.rows[0], { ...real.rows[1], chosen: cell(640, 630, 650, 0), gpu: { skip: "the GPU failed on a block of the prompt" } }] });
assert.ok(stopped.includes("WebGPU stopped while timed: the GPU failed on a block of the prompt") && !stopped.includes("ready in"), stopped);
assert.ok(!stopped.includes("×"), `no ratio where the GPU stopped\n${stopped}`);
assert.ok(stopped.includes("| 650 tok/s (640–655) | the GPU failed on a block of the prompt |  |"), stopped);
// the GPU's cells say why in a few words; the line has the reason where it is no such few words
const fallback = "a fallback adapter: the CPU in the GPU's place";
const onFallback = pathTable({ threads: 2, gpu: { why: fallback },
  rows: [{ what: "prompt", tokens: 64, chosen: { same: "cpu" }, cpu: cell(505, 500, 510, 0), gpu: { skip: fallback } }] });
assert.ok(onFallback.includes("WebGPU: not on a fallback adapter") && onFallback.includes("| a prompt of 64 tokens | same as CPU only | 505 tok/s (500–510) | not on a fallback adapter |  |"), onFallback);
assert.equal(gpuSkipped("no WebGPU in a worker here"), "not here");
assert.equal(gpuSkipped("no GPU adapter here"), "not here");
const float32 = pathTable({ threads: 1, gpu: { why: "float32 weights are not on the GPU yet" },
  rows: [{ what: "prompt", tokens: 64, chosen: { same: "cpu" }, cpu: cell(51, 50, 52, 0), gpu: { skip: "float32 weights are not on the GPU yet" } }] });
assert.ok(float32.includes("WebGPU: float32 weights are not on the GPU yet") && float32.includes("| 51.0 tok/s (50.0–52.0) | not used |  |"), float32);
assert.equal(pathTable({ error: "a | b" }, "x"), "**The model page's path** (x): failed: a \\| b");
assert.ok(!pathTable(undefined).includes("undefined"));
for (const text of [pathTable(real), stopped, onFallback, float32]) assert.ok(!text.includes("undefined") && !text.includes("NaN"), text);
// T190: the page path's threads are the model page's: the count it remembers (same key on both pages), or the search's
// verdicts here, or why there is one thread
const nav = { hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Linux; Android 15)" };
assert.equal(threadsKey("llm-jp-3-150m", nav), "threads:llm-jp-3-150m:8:8:Mozilla/5.0 (Linux; Android 15)");
assert.equal(threadsKey("x", { hardwareConcurrency: 4, userAgent: "u" }), "threads:x:4::u", "no deviceMemory (Safari, Firefox)");
for (const page of ["index.astro", "benchmark.astro"]) {
  assert.ok(/threadsKey as sharedThreadsKey|threadsKey\(entry\.id, navigator\)/.test(fs.readFileSync(new URL(`../src/pages/${page}`, import.meta.url), "utf8")),
    `${page} keys the remembered threads with src/bench.js's threadsKey`);
}
const headOf = (how, threads = 4) => pathTable({ ...real, threads, how }).split("\n")[0];
assert.ok(headOf({ remembered: true }).includes("4 software threads, as the model page remembers ·"), headOf({ remembered: true }));
assert.ok(headOf({ searched: [[8, 4, 4], [4, 2, 4]] }).includes("4 software threads, searched here (8 or 4: 4, 4 or 2: 4) ·"));
assert.ok(headOf({ unfinished: 120 }).includes("4 software threads: the search had not ended after 120 s ·"));
assert.ok(headOf({ alone: "no shared memory here" }, 1).includes("1 software thread: no shared memory here ·"));
assert.ok(headOf({ searched: [] }).includes("4 software threads ·"), "no search: nothing more");
// T190's review: a software thread that stopped says so (worker.js reads engine.lostThreads: found is 1 then, and the
// search's verdicts would name another count); one that stopped after the count was found, too
assert.ok(headOf({ alone: "a software thread stopped" }, 1).includes("1 software thread: a software thread stopped ·"));
assert.ok(headOf({ remembered: true, stopped: true }).includes("4 software threads, as the model page remembers; a software thread stopped while timed, and one thread went on ·"));
assert.ok(headOf({ searched: [[8, 4, 4]], stopped: true }).includes("searched here (8 or 4: 4); a software thread stopped while timed"));
// the writing on each number of threads, the page's marked, under the table; nothing with one count or none
const counted = pathTable({ ...real, perCount: [{ threads: 1, speed: 107.2, low: 105, high: 108 }, { threads: 2, speed: 151, low: 149, high: 160, unsteady: true },
  { threads: 4, speed: 163.4, low: 160, high: 165 }, { threads: 8, speed: 120, low: 118, high: 121 }] });
assert.equal(counted.split("\n").at(-1), "Writing on each number of software threads (CPU only): 1: 107 tok/s (105–108) · 2: 151 tok/s (149–160, unsteady) · " +
  "4 (the page's): 163 tok/s (160–165) · 8: 120 tok/s (118–121)");
assert.ok(!pathTable({ ...real, perCount: [{ threads: 1, speed: 107, low: 105, high: 108 }] }).includes("Writing on each"));
assert.ok(!pathTable(real).includes("Writing on each"));
assert.equal(threadsLine([]), "");
assert.deepEqual(PATH_PROMPTS, [64, 256]);
assert.equal(PATH_WRITES, 64);
// in the report, under the rounds' table: parseReport() still reads the rounds alone
const withPath = [markdown, pathTable(real, "tiny-lm 29M")].join("\n\n");
assert.deepEqual(parseReport(withPath).rows.map((row) => row.name), ["everything", "without the kernels"]);
// T186: a layer's verdict in one short line: the four quantized vectors summed, the fused form against the norms apart
const quantized = [{ point: "q, k, v", wrong: null, scale: 1.2e-7, apart: 3, of: 2112 }, { point: "o", wrong: null, scale: 2.5e-7, apart: 1, of: 2112 },
  { point: "gate, up", wrong: null, scale: 0, apart: 0, of: 2112 }, { point: "down", wrong: null, scale: 1e-8, apart: 2, of: 2080 }];
assert.equal(layerCheckNumbers({ quantized }), "quantized: scales 2.5e-7, 6 of 8416 off by 1");
assert.equal(layerCheckNumbers({ quantized, sameAsNormsApart: { ulps: 2, apart: 1, stream: 3e-7, bitForBit: false } }),
  "quantized: scales 2.5e-7, 6 of 8416 off by 1; norms apart: 2 ulp, 1 off by 1, stream 3.0e-7");
assert.equal(layerCheckNumbers({ sameAsNormsApart: { ulps: 0, apart: 0, stream: 0, bitForBit: true } }), "norms apart: bit for bit");
assert.equal(layerCheckNumbers({ quantized: [quantized[0], { ...quantized[1], wrong: "far from quantize_x's" }] }), "quantized: o far from quantize_x's");
assert.equal(layerCheckNumbers({ worstRelative: 1e-6, ok: true }), "");
// T185: a report of a real GPU is about four times the link's limit (the owner's Android's shape: packed int8 dot,
// shader-f16 and subgroups, 8 logical cores, the page's path; the numbers made up). Where the whole is too long the link
// holds the head and a line a section, and asks for the whole from the clipboard; parseReport() reads it as before.
const android = environmentOf({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Mobile Safari/537.36" },
  { model: "tiny-lm 29M", pyodide: "314.0.7", build: "abc1234", site: "https://takano32.github.io/pyodide-llm/benchmark/" });
const aDevice = { cores: 8, memoryGB: 8, simd: true, relaxedSimd: true, memory64: true, crossOriginIsolated: true, sharedMemory: true, webgpu: true, opfs: true, syncHandle: true };
const aCpu = { backend: "SIMD kernels, int8, relaxed SIMD", shared: true, megabytes: 211, tokenMegabytes: 234, layerWeights: 121.6e6,
  rows: [1, 2, 4, 8].map((threads, i) => ({ threads, msPerToken: [11.2, 8.1, 7.6, 7.9][i], GBps: [18.9, 26.0, 27.8, 26.7][i], promptMsPerToken: [7.9, 4.6, 3.9, 4.4][i] })),
  ceilings: { read: [1, 2, 4, 8].map((threads, i) => ({ threads, GBps: [24.3, 31.2, 33.0, 32.1][i] })), dot: { GMACs: 44.2 }, dotRegisters: { GMACs: 61.5 }, fma: { GMACs: 17.3 } } };
const MATVEC = ["widened (T134)", "packed int8 (T134)", "llama.cpp mul_mat_vec, 4 rows", "llama.cpp mul_mat_vec, 4 rows, subgroups", "llama.cpp MMVQ, 4 rows",
  "llama.cpp MMVQ, 4 rows, subgroups", "ORT MatMulNBits, 8 rows", "ORT DP4A small M, 4 rows"];
const matVecChecks = ["widen", "packed", ...MATVEC.slice(2)];
const aCheck = Object.fromEntries([...matVecChecks, "llama.cpp tiles 32×32, f16", "llama.cpp tiles 64×64, f16", "TF.js tiles 32×32, vec4", "ORT DP4A 64×64",
  "ORT DP4A 64×64, subgroups", "argmax", "a layer, llama.cpp, separate steps", "a layer, llama.cpp, fused (T150)", "a layer, DP4A, separate steps",
  "a layer, DP4A, fused (T175)", "sampling", "tokens on the GPU"].map((k) => [k, { ok: true, worstRelative: 3.2e-8 }]));
const aBandwidths = ["llm-jp-3 150M w1", "Llama 3.2 1B w1", "Llama 3.2 1B classifier"].map((name, j) => ({ name: `bandwidth: ${name}`,
  result: { rows: MATVEC.map((shader, i) => ({ shader, check: matVecChecks[i], GBps: 3.1 + i * 2.37 + j * 4.11 })), cpu: { GBps: 9.8 + j }, quantize: { msEach: 0.041 + j * 0.013 } } }));
const aCeilings = { global: { GBps: 48.3, MiB: 128 }, f32: { GFLOPS: 812 }, f16: { GFLOPS: 1530 }, dot4: { GOPS: 1210 }, shared: { GBps: 402 } };
const aTokens = ["llm-jp-3 150M", "llm-jp-3 150M, packed int8", "Llama 3.2 1B", "Llama 3.2 1B, packed int8", "Llama 3.2 1B, chosen on the GPU"].map((n, i) => ({
  name: `a token of ${n}`, result: { kind: n.includes("packed") ? "packed" : "widen", sample: n.includes("chosen"), GB: [0.17, 0.17, 1.4, 1.4, 1.4][i],
    dispatches: [120, 132, 240, 256, 242][i], msPerToken: 12.34 + i * 17.1, tokPerSecond: 81.0 - i * 13.3 } }));
const layerForms = ["llama.cpp, separate steps", "llama.cpp, fused (T150), the norms apart", "llama.cpp, fused (T150)"];
const aLayer = { name: "a layer of a token", result: { model: "Llama 3.2 1B", pos: 127, layers: 16, GB: 0.0612, rows: [
  ...[false, true].flatMap((subgroups) => layerForms.map((f) => ({ form: `${f}${subgroups ? ", subgroups" : ""}`, check: `a layer, ${f}`, base: "llama.cpp", fused: f !== layerForms[0], subgroups }))),
  ...["DP4A, separate steps", "DP4A, fused (T175), the norms apart", "DP4A, fused (T175)"].map((f, i) => ({ form: f, check: `a layer, ${f}`, base: "DP4A", fused: i > 0, subgroups: false }))]
  .map((row, i) => ({ ...row, dispatches: [14, 7, 5, 14, 7, 5, 18, 11, 9][i], msPerLayer: 3.1 - i * 0.211, GBps: 21.3 + i * 1.7 })) } };
const aGenerate = { name: "tokens generated on the GPU", result: { model: "Llama 3.2 1B", layers: 16, vocab: 128256, GB: 1.36, dispatches: 83, tokens: 16, layer: "DP4A, fused (T175)",
  settings: { penalty: 1.1, temperature: 0.7, topp: 0.9 }, work: { ms: 41.23 }, rows: [1, 4, 8, 16].map((perSubmission, i) => ({ perSubmission, msPerToken: 58.12 - i * 5.3, fixedMs: 16.87 + i * 3.1 })),
  sampling: { vocab: 128256, msEach: 1.234, over: 312, flat: { msEach: 4.567, over: 12045 } } } };
const PROMPT_SHADERS = ["batched (T135)", "llama.cpp tiles 32×32, f16", "llama.cpp tiles 64×64, f16", "TF.js tiles 32×32, vec4", "ORT DP4A 64×64", "ORT DP4A 64×64, subgroups"];
const aPrompt = { name: "a prompt all at once", result: { GB: 0.14, weights: 121.6e6, rows: [...PROMPT_SHADERS.flatMap((shader, i) => [1, 16, 64].map((tokens, j) =>
  ({ shader, packed: shader.startsWith("ORT"), tokens, ms: 12.3 + i * 7 + j * 11, msPerToken: 3.21 - j * 0.9 - i * 0.1, GFLOPS: 34 + i * 20 + j * 40 }))),
  { shader: "batched (T135), again at the end", again: true, tokens: 64, ms: 101.2, msPerToken: 0.5, GFLOPS: 154 }] } };
const aSteps = [{ name: "the adapter", result: { adapter: "arm · valhall", fallback: false, packed: true, features: ["shader-f16", "subgroups"], subgroupSizes: [16, 16] } },
  { name: "the shaders against JavaScript", result: aCheck }, ...aBandwidths, ...aTokens, aLayer, aGenerate, { name: "the device's ceilings", result: aCeilings }, aPrompt];
const aStorage = { mib: 256, pieces: 32, sequential: { seconds: 0.95 }, scattered: { seconds: 1.15 }, scatteredFlushEach: { seconds: 1.64 }, read: { seconds: 0.12 } };
const aLine = { site: { bytes: 8.4e6, firstByteMs: 162, MBps: 12.7 }, hf: { bytes: 33.6e6, firstByteMs: 863, MBps: 8.4 },
  paced: [1, 4, 8].map((rate, i) => ({ rate, MBps: [1, 4, 7.6][i] })) };
const aBaseline = cpuBaseline({ status: "ok", data: aCpu });
const aHead = [benchMarkdown(rows, android), pathTable(real, "tiny-lm 29M")].join("\n\n");
const aWhole = [aHead, "#### CPU", ...cpuTable(aCpu), "#### GPU", ...matVecTable(aBandwidths, aCheck, aCeilings), ...tokenTable(aTokens, aBaseline),
  ...layerTable(aLayer, aCheck, aCeilings), ...generateTable(aGenerate, aCheck)].join("\n");
const aLines = [...deviceSummary(aDevice), ...cpuSummary(aCpu), ...gpuSummary(aSteps, aBaseline), ...storageSummary(aStorage), ...lineSummary(aLine)];
const aSummary = shortReport(aHead, aLines);
assert.ok(reportTooLong(aWhole, android), "a GPU's whole report is too long for the link");
assert.ok(loginUrl(aWhole, android, aSummary).length <= REPORT_LIMIT, `${loginUrl(aWhole, android, aSummary).length}`);
const aBody = new URL(reportUrl(aWhole, android, aSummary)).searchParams.get("body");
assert.ok(aBody.endsWith(`#### Summary\n\n${aLines.map((line) => `- ${line}`).join("\n")}\n\n${PASTE}`), aBody);
assert.ok(aBody.includes("**Device**: (") && aBody.includes(aHead), "the questions and the head as they are");
assert.deepEqual(parseReport(aBody), parseReport(reportBody(aWhole)), "parseReport() reads the summary as the whole");
assert.deepEqual(parseReport(aBody).rows.map((row) => row.name), ["everything", "without the kernels"]);
assert.equal(reportsTable([{ number: 1, url: "u", body: aBody }]).split("\n").length, 3, "a row of reportsTable()");
// every section's line, with the fastest of each and the CPU beside the GPU
assert.deepEqual(aLines, [
  "Browser: SIMD yes, relaxed SIMD yes, 64-bit memory yes, cross-origin isolated yes, WebGPU in a worker yes, private file system yes",
  "CPU: 7.6 ms a token with 4 software threads, 27.8 GB/s (93% of reading alone); a prompt 31.2 G MAC/s with 4; ceilings: reading alone 33.0 GB/s, relaxed_dot 44.2 G MAC/s, f32 17.3 G MAC/s",
  "GPU: arm · valhall; packed int8 dot yes, shader-f16 yes, subgroups yes (16 wide); the check 20 of 20 ok",
  "GPU, a token: llm-jp-3 150M 81.0 tok/s (0.50× the CPU); Llama 3.2 1B 54.4 tok/s (2.7× the CPU); the fastest matrix × vector of Llama 3.2 1B classifier ORT DP4A small M, 4 rows, 27.9 GB/s (58% of reading a buffer)",
  "GPU, the fastest layer DP4A, fused (T175), 9 dispatches, 1.41 ms; generated 58.12 ms a token one a submission, 42.22 with 16 (1.4×)",
  "GPU, a prompt of 64 tokens: ORT DP4A 64×64, subgroups, 214 GFLOPS (4.3× the CPU)",
  "Storage: writes 283 MB/s in order, 233 far apart, 164 with a flush each piece; read back 2237 MB/s",
  "Line: this site 12.7 MB/s (first byte 162 ms), huggingface.co 8.4 MB/s (first byte 863 ms); huggingface.co read no faster than 1, 4, 8 MB/s: 1.00, 4.00, 7.60"]);
// no ratio on a fallback adapter or after a lost device, and a WRONG row is never the fastest nor held against the CPU
const onFallbackAdapter = gpuSummary([{ ...aSteps[0], result: { ...aSteps[0].result, fallback: true } }, ...aSteps.slice(1)], aBaseline);
assert.ok(onFallbackAdapter[0].includes("a fallback adapter: nothing timed") && !onFallbackAdapter.slice(1).join().includes("the CPU)") && !onFallbackAdapter.join().includes("% of"), onFallbackAdapter.join("\n"));
const afterLost = gpuSummary(aSteps, aBaseline, { lost: "gone" });
assert.ok(afterLost[0].endsWith("; the device was lost (gone)") && !/\(\d[\d.]*×/.test(afterLost.join()) && !afterLost.join().includes("% of"), afterLost.join("\n"));
// T202: the summary says where the fastest broken-down layer's time goes (the norms apart: 3.25 ms, its small steps
// 0.375, its chain −0.125), not after a lost device
const withSteps = gpuSummary([...aSteps, stepsStep], aBaseline);
assert.ok(withSteps.includes("GPU, where the time of a layer DP4A, fused (T175), the norms apart goes: 3.25 ms, the matrices alone 2.00, fusing adds 0.75, " +
  "the attention 0.25, the norms and quantizing 0.38, the chain -0.13"), withSteps.join("\n"));
assert.ok(!gpuSummary([...aSteps, stepsStep], aBaseline, { lost: "gone" }).join().includes("where the time"));
const wrongDp4a = gpuSummary(aSteps.map((s) => (s.name === "the shaders against JavaScript" ? { ...s, result: { ...aCheck, "ORT DP4A small M, 4 rows": { ok: false, worstRelative: 1 }, "ORT DP4A 64×64, subgroups": { ok: false, worstRelative: 1 } } } : s)), aBaseline);
assert.ok(wrongDp4a[0].includes("the check 18 of 20 ok (ORT DP4A small M, 4 rows WRONG, ORT DP4A 64×64, subgroups WRONG)"), wrongDp4a[0]);
assert.ok(!wrongDp4a[1].includes("ORT DP4A small M") && !wrongDp4a[3].includes("ORT DP4A 64×64, subgroups"), wrongDp4a.join("\n"));
// a GPU section with only its adapter and check (the fallback's), or with no check: no undefined, no NaN
for (const lines of [gpuSummary(aSteps.slice(0, 2), {}), gpuSummary([{ name: "the adapter", result: {} }, { name: "the shaders against JavaScript", error: "lost" }], {}),
  cpuSummary({ ...aCpu, ceilings: { error: "no" } }), cpuSummary({ ...aCpu, shared: false, rows: aCpu.rows.slice(0, 1), ceilings: {} }),
  lineSummary({ site: { error: "x" }, hf: { error: "y" }, paced: [] }), lineSummary({ ...aLine, paced: [{ rate: 1, slower: true }] })]) {
  assert.ok(!lines.join().includes("undefined") && !lines.join().includes("NaN"), lines.join("\n"));
}
// a report that fits is in the link whole, summary or not; a summary too long as well leaves the request alone
assert.equal(new URL(reportUrl(everything, environment, aSummary)).searchParams.get("body"), reportBody(everything));
assert.ok(new URL(reportUrl(long, environment, long)).searchParams.get("body").endsWith(TOO_LONG));
console.log("ok");
