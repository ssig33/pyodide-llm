// T173: the page-memory section of /benchmark/ in Chromium, on a limit a runner holds well below its own: (1) ?run=memory
// holds all it is asked to and removes its mark, "Run all" leaves it out; (2) a tab ended while it grew is played by
// loading the page again in the middle of a run: the page reports the last step it held, from the mark, and does not
// run the section again although the address still says ?run=memory. Never on the development machine with a large
// limit (AGENTS.md: gigabytes held there take the machine down). Meant for CI (preview.yml).
//
//   node tests/page-memory-check.mjs [url of a site | --dist]
//     --dist serves dist/ (npm run build) itself, as tests/bench-check.mjs does
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as playwright from "playwright-core";

const PROFILE = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", ".tmp", `page-memory-check-${process.pid}`);
const MiB = 2 ** 20;
const args = process.argv.slice(2);
const dist = args.includes("--dist");
let site = args.find((a) => a !== "--dist") ?? "https://takano32.github.io/pyodide-llm/";
let server;
if (dist) {
  const base = "/pyodide-llm/", root = new URL("../dist/", import.meta.url).pathname;
  const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".py": "text/plain", ".wasm": "application/wasm" };
  server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    let file = path.join(root, pathname.slice(base.length));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!pathname.startsWith(base) || !fs.existsSync(file)) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  site = `http://localhost:${server.address().port}${base}`;
}

let failed = false;
const check = (ok, what) => {
  console.log(`${ok ? "ok" : "FAILED"}: ${what}`);
  if (!ok) failed = true;
};
// a wait that starts again when the page loads again (the service worker's first reload, T93, or the test's own)
async function waitFor(page, fn, arg, timeout = 300000) {
  for (;;) {
    try {
      return await page.waitForFunction(fn, arg, { timeout, polling: 50 });
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
}
const MARK = "benchmark-memory";

fs.rmSync(PROFILE, { recursive: true, force: true });
const context = await playwright.chromium.launchPersistentContext(PROFILE, {});
const page = context.pages()[0] ?? await context.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(String(error.message)));
try {
  // (1) all it is asked to
  await page.goto(`${site}benchmark/?run=memory&memoryMB=256`);
  await waitFor(page, () => window.__benchmark?.done);
  let b = await page.evaluate(() => ({ ...window.__benchmark, mark: sessionStorage.getItem("benchmark-memory") }));
  const one = b.results.memory;
  console.log(one?.markdown ?? "no result");
  check(one?.status === "ok" && one.data.stop === "limit" && one.data.held === 256 * MiB, `held 256 MiB and stopped at the limit (${one?.data?.stop}, ${one?.data?.held / MiB} MiB)`);
  check(b.mark === null, "the mark is gone after a run that ended");
  check(Object.keys(b.results).join() === "memory", `only the page memory ran (${Object.keys(b.results).join()})`);
  check(b.markdown.includes("#### Page memory") && b.markdown.includes("| 256 MiB |"), "the report holds the section");
  check(Array.isArray(b.all) && !b.all.includes("memory") && b.all.includes("cpu"), `"Run all" leaves it out (${b.all})`);

  // (2) a tab ended while it grew, played by loading the page again
  await page.goto(`${site}benchmark/?run=memory&memoryMB=2048`);
  await waitFor(page, ([mark, least]) => {
    const text = sessionStorage.getItem(mark);
    return text && JSON.parse(text).held >= least;
  }, [MARK, 128 * MiB]);
  const left = JSON.parse(await page.evaluate((mark) => sessionStorage.getItem(mark), MARK));
  console.log(`loading again at the mark: held ${left.held / MiB} MiB, trying ${left.trying / MiB} MiB`);
  await page.reload();
  await waitFor(page, () => window.__benchmark?.done);
  // a moment more: a section run again by mistake would have begun by now
  await page.waitForTimeout(2000);
  b = await page.evaluate(() => ({ ...window.__benchmark, mark: sessionStorage.getItem("benchmark-memory"),
    state: document.querySelector('section[data-section="memory"] .state')?.textContent }));
  const two = b.results.memory;
  console.log(two?.markdown ?? "no result");
  check(two?.data?.stop === "reloaded", `the page found the mark (${two?.data?.stop})`);
  check(two?.data?.held >= left.held && two.data.held < 2048 * MiB && two.data.trying === two.data.held + 64 * MiB,
        `it reports the last step held (${two?.data?.held / MiB} MiB, trying ${two?.data?.trying / MiB} MiB)`);
  check(b.mark === null, "the mark was read once and removed");
  check(/loaded again/.test(b.state ?? ""), `the section says so and did not run again (state: ${b.state})`);
  check(b.markdown.includes("#### Page memory") && b.markdown.includes("loaded again while it grew"), "the report holds it");
} catch (error) {
  console.log(`failed: ${String(error.message).split("\n")[0]}`);
  failed = true;
}
if (errors.length) {
  console.log(`page errors: ${errors.join(" / ")}`);
  failed = true;
}
// T141: never wait on a browser's close for long
await Promise.race([context.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
fs.rmSync(PROFILE, { recursive: true, force: true });
server?.close();
process.exit(failed ? 1 : 0);
