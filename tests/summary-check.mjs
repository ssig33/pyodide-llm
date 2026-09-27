// tests/summary.mjs: the table a CI job writes into its summary (T82). Node alone.
//
//   node tests/summary-check.mjs
import assert from "node:assert/strict";
import { readResults, summaryMarkdown } from "./summary.mjs";

const lines = [
  { engine: "chromium", browserVersion: "148.0", model: "tiny-lm", os: "linux x64", ok: true, timedOut: false,
    readySeconds: 8.42, tokPerSecond: 307.2, backend: "tiny-lm 29M · SIMD kernels, int8, relaxed SIMD", failures: [],
    threadReports: 1, then: [{ model: "llm-jp-3-150m", threadReports: 0 }], offline: { threadReports: 2 } },
  { engine: "webkit", browserVersion: "26.4", model: "tiny-lm", os: "linux x64", ok: false, timedOut: true,
    failures: ["timed out after 900s"] },
  { engine: "firefox", browserVersion: "150", model: "llm-jp-3-150m", os: "linux x64", ok: false, timedOut: false,
    readySeconds: 12, tokPerSecond: null, backend: "a | pipe", failures: ["no speed line under the answer"] },
].map((entry) => JSON.stringify(entry));
// a run killed while writing leaves half a line: it is skipped, not fatal
const results = readResults(lines.join("\n") + "\n{\"engine\": \"chro");
assert.equal(results.length, 3);

const table = summaryMarkdown(results);
const rows = table.split("\n").filter((line) => line.startsWith("|"));
assert.equal(rows.length, 2 + 3, "a header, the separator GitHub needs, and a row per run");
assert.ok(table.includes("| tiny-lm | chromium 148.0 | 8.4 | 307.2 | 1, 0, 2 | tiny-lm 29M · SIMD kernels, int8, relaxed SIMD | ok |  |"), table);
// T172: a record from before the count (or a run that never answered) leaves the cell empty, not "undefined"
assert.ok(table.includes("| webkit 26.4 |  |  |  |  | **timed out** |"), table);
assert.ok(table.includes("**timed out**") && table.includes("timed out after 900s"), "a timeout says so, and why");
assert.ok(table.includes("a \\| pipe"), "a | in a cell must not break the table");
assert.ok(!table.includes("undefined") && !table.includes("NaN") && !table.includes("null"), table);
assert.ok(table.includes("1 of 3 ran on linux x64. 2 failed or timed out"), table);
assert.ok(table.includes("stock-firefox.mjs writes no line here"), "the table says what it does not count");
assert.equal(summaryMarkdown([]), "No results were recorded.\n");
console.log("ok");
