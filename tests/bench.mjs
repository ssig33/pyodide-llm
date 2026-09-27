// src/bench.js: the table a visitor pastes into an issue (T45). A plain module, so this runs in Node alone.
//
//   node tests/bench.mjs
import assert from "node:assert/strict";
import { FULL_ROUNDS, QUESTIONS, REPORT_LIMIT, ROUNDS, TOO_LONG, benchMarkdown, cpuBaseline, cpuTable, environmentOf, loginUrl, parseReport, reportBody,
         generateTable, layerTable, matVecTable, reportTooLong, reportUrl, reportsTable, tableCell, threadCounts, times, timesFaster, tokenTable } from "../src/bench.js";
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
  { form: "separate steps", check: "a layer, separate steps", fused: false, subgroups: false, dispatches: 14, msPerLayer: 4.2, GBps: 16.3 },
  { form: "fused (T150), the norms apart", check: "a layer, fused (T150), the norms apart", fused: true, normApart: true, subgroups: false, dispatches: 7, msPerLayer: 2.8, GBps: 24.4 },
  { form: "fused (T150)", check: "a layer, fused (T150)", fused: true, subgroups: false, dispatches: 5, msPerLayer: 2.1, GBps: 32.6 },
  { form: "separate steps, subgroups", check: "a layer, separate steps, subgroups", fused: false, subgroups: true, error: "a | b\nc" },
  { form: "fused (T150), subgroups", check: "a layer, fused (T150), subgroups", fused: true, subgroups: true, dispatches: 5, msPerLayer: 1.9, GBps: 36 }] } };
const layerRight = { "a layer, separate steps": { ok: true }, "a layer, fused (T150)": { ok: true } };
const layerCeilings = { global: { GBps: 40 } };
const fasterOf = (lines, form) => cellsOf(lines.find((line) => line.startsWith(`| ${form} |`) || line.startsWith(`| ${form} (WRONG`))).at(-1).trim();
const layerLines = layerTable(layerStep, layerRight, layerCeilings);
assert.ok(layerLines.includes("| fused (T150) | 5 | 2.10 | 32.6 (81.5%) | 33.6 | 2.0× |"), layerLines.join("\n"));
assert.ok(layerLines.includes("| fused (T150), the norms apart | 7 | 2.80 | 24.4 (61.0%) | 44.8 | 1.5× |"), layerLines.join("\n"));
assert.equal(fasterOf(layerLines, "separate steps"), "");
// no separate steps measured with subgroups: nothing to hold the fused one against
assert.equal(fasterOf(layerLines, "fused (T150), subgroups"), "");
assert.ok(layerLines.includes("| separate steps, subgroups | failed: a \\| b c | | | | |"), layerLines.join("\n"));
for (const [label, lines] of [["right", layerLines], ["lost", layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" })],
  ["fallback", layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true })], ["no check", layerTable(layerStep)]]) {
  const rowsOf = lines.filter((line) => line.startsWith("|"));
  assert.equal(rowsOf.length, 2 + layerStep.result.rows.length, label);
  for (const line of rowsOf) assert.equal(cellsOf(line).length, cellsOf(rowsOf[0]).length, `${label}: ${line}`);
  assert.ok(!lines.join("\n").includes("undefined") && !lines.join("\n").includes("NaN"), label);
}
const lostLayer = layerTable(layerStep, layerRight, layerCeilings, { lost: "lost" });
assert.equal(fasterOf(lostLayer, "fused (T150)"), "");
assert.ok(lostLayer.includes("| fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), lostLayer.join("\n"));
assert.ok(lostLayer.some((line) => line.includes("the device was lost")));
const fallbackLayer = layerTable(layerStep, layerRight, { ...layerCeilings, fallback: true }, { fallback: true });
assert.equal(fasterOf(fallbackLayer, "fused (T150)"), "");
assert.ok(fallbackLayer.includes("| fused (T150) | 5 | 2.10 | 32.6 | 33.6 |  |"), fallbackLayer.join("\n"));
const layerWrong = { ...layerRight, "a layer, fused (T150)": { ok: false } };
assert.equal(fasterOf(layerTable(layerStep, layerWrong), "fused (T150)"), "");
assert.ok(layerTable(layerStep, layerWrong).some((line) => line.startsWith("| fused (T150) (WRONG in the check) |")));
// the separate steps WRONG: nothing right to hold the fused one against
assert.equal(fasterOf(layerTable(layerStep, { ...layerRight, "a layer, separate steps": { ok: false } }), "fused (T150)"), "");
const unsteadyStep = { ...layerStep, result: { ...layerStep.result, rows: layerStep.result.rows.map((row) => (row.fused && !row.subgroups && !row.normApart ? { ...row, unsteady: true } : row)) } };
assert.ok(layerTable(unsteadyStep, layerRight).includes("| fused (T150) | 5 | unsteady: 2.10 | 32.6 | 33.6 |  |"));
// a fallback adapter's few hundredths of a GB/s still show
const slowStep = { ...layerStep, result: { ...layerStep.result, rows: [{ ...layerStep.result.rows[0], msPerLayer: 2900, GBps: 0.0236 }] } };
assert.ok(layerTable(slowStep, layerRight, undefined, { fallback: true }).includes("| separate steps | 14 | 2900.00 | 0.024 | 46400.0 |  |"));
assert.equal(layerTable({ name: "a layer of a token", error: "x | y" })[0], "**A layer of a token**: x \\| y");
// T151: the table of tokens generated on the GPU: what a submission costs besides its tokens and how many times faster
// several a submission are than one, neither after a lost device, on a fallback adapter or with the sampling WRONG
const generateStep = { name: "tokens generated on the GPU", result: { model: "Llama 3.2 1B's width", layers: 2, vocab: 32000, GB: 0.21,
  dispatches: 13, tokens: 16, settings: { temperature: 0.7, topp: 0.9, penalty: 1.3 }, work: { ms: 9.1 },
  rows: [{ perSubmission: 1, msPerToken: 14.2, fixedMs: 5.1 }, { perSubmission: 4, msPerToken: 10.4, fixedMs: 5.2 }, { perSubmission: 16, msPerToken: 9.4, fixedMs: 4.8 }],
  sampling: { vocab: 128256, msEach: 0.31, over: 5114, flat: { msEach: 1.2, over: 128256, unsteady: true } } } };
const generateRight = { sampling: { ok: true }, "tokens on the GPU": { ok: true } };
const generateLines = generateTable(generateStep, generateRight);
assert.ok(generateLines.includes("| 1, each read back as it comes | 14.20 | 5.10 | 5.10 |  |"), generateLines.join("\n"));
assert.ok(generateLines.includes("| 16, read back once | 9.40 | 4.80 | 0.30 | 1.5× |"), generateLines.join("\n"));
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
    assert.ok(rowsOf.slice(2).every((line) => cellsOf(line).slice(2).every((cell) => cell.trim() === "")), `${label}: no derived numbers\n${lines.join("\n")}`);
  }
}
const noisy = { ...generateStep, result: { ...generateStep.result, rows: [generateStep.result.rows[0], { perSubmission: 8, msPerToken: 9.0, fixedMs: -0.8 }] } };
assert.ok(generateTable(noisy, generateRight).includes("| 8, read back once | 9.00 | under the noise | under the noise | 1.6× |"), generateTable(noisy, generateRight).join("\n"));
assert.ok(generateTable(generateStep, { ...generateRight, "tokens on the GPU": { ok: false } }).some((line) => line.startsWith("| 4, read back once (WRONG in the check) |")));
assert.ok(generateTable(generateStep, generateRight, { fallback: true }).some((line) => line.includes("none on a fallback adapter")));
assert.equal(generateTable({ name: "x", error: "a | b" })[0], "**Tokens generated on the GPU**: a \\| b");
const unmeasured = generateTable({ ...generateStep, result: { ...generateStep.result, work: undefined, sampling: undefined, rows: [{ perSubmission: 1, msPerToken: 900 }, { perSubmission: 2, msPerToken: 800 }] } }, generateRight, { fallback: true });
assert.ok(unmeasured.at(-1).includes("not measured here"), unmeasured.at(-1));
console.log("ok");
