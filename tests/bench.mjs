// src/bench.js: the table a visitor pastes into an issue (T45). A plain module, so this runs in Node alone.
//
//   node tests/bench.mjs
import assert from "node:assert/strict";
import { FULL_ROUNDS, QUESTIONS, REPORT_LIMIT, ROUNDS, TOO_LONG, benchMarkdown, cpuBaseline, environmentOf, loginUrl, parseReport, reportBody,
         reportTooLong, reportUrl, reportsTable, tableCell, threadCounts, times, timesFaster, tokenTable } from "../src/bench.js";
import fs from "node:fs";

const rows = [
  { name: "everything", without: [], tokens: 64, speed: 334.62, seconds: 8.4, backend: "SIMD kernels, int8, relaxed SIMD" },
  { name: "without the kernels", without: ["kernels"], tokens: 64, speed: 44.81, seconds: 8.1, backend: "NumPy (without kernels)" },
];
const environment = environmentOf({ hardwareConcurrency: 8, deviceMemory: 8, userAgent: "Mozilla/5.0 (X11)" },
                                  { model: "tiny-lm 29M", pyodide: "314.0.7", site: "https://example.invalid/" });
const markdown = benchMarkdown(rows, environment);

// a table GitHub renders: a header, the separator, and one row per round
const lines = markdown.split("\n");
assert.equal(lines.filter((line) => line.startsWith("|")).length, rows.length + 2, "one row per round, and the header");
assert.ok(lines.some((line) => line.includes("|---|---|---|---|")), "the separator GitHub needs");
assert.ok(markdown.includes("| everything | 334.6 | 8.4 s | SIMD kernels, int8, relaxed SIMD |"), markdown);
assert.ok(markdown.includes("8 logical cores") && markdown.includes("8 GB or more"), "what the browser told us");
assert.ok(markdown.includes("Mozilla/5.0 (X11)"), "the user agent, for a report that means something");

// a browser that says nothing about itself must not make anything up
const bare = benchMarkdown(rows, environmentOf({}, {}));
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
console.log("ok");
