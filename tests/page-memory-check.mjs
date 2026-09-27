// T173: the page-memory section of /benchmark/ in Chromium, on a limit a runner holds well below its own: (1) ?run=memory
// holds all it is asked to and removes its mark, "Run all" leaves it out; (2) a page left while it grows (a reload)
// leaves no mark: a visitor's reload is no limit; (3) a page hidden while it grows stops at once and says so; (4) a
// tab ended while it grows (the renderer crashed by DevTools' Page.crash, no event in the page, as a phone's jetsam or
// low-memory killer ends it) is loaded again in the same tab: the page reports the last step it held, from the mark,
// and does not run the section again although the address still says ?run=memory.
// With --ended <MiB> (T173's review) the run is the section's own 4 GiB and the tab is ended by the kernel for its
// memory: the workflow puts the whole browser in a cgroup of that many MiB with no swap (preview.yml), and the page
// must report a number below it. Never on the development machine (AGENTS.md: gigabytes held there take the machine
// down). Meant for CI (preview.yml).
//
//   node tests/page-memory-check.mjs [url of a site | --dist] [--ended <MiB>]
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
const endedAt = args.includes("--ended") ? Number(args[args.indexOf("--ended") + 1]) : 0;
let site = args.find((a, i) => !a.startsWith("--") && args[i - 1] !== "--ended") ?? "https://takano32.github.io/pyodide-llm/";
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
const markAt = (page, least) => waitFor(page, ([mark, least]) => {
  const text = sessionStorage.getItem(mark);
  return text && JSON.parse(text).held >= least;
}, [MARK, least]);
const state = (page) => page.evaluate((mark) => ({ ...window.__benchmark, mark: sessionStorage.getItem(mark),
  state: document.querySelector('section[data-section="memory"] .state')?.textContent }), MARK);
// the page's section button, once the page has set itself up (it reads a mark before it says __benchmark.all)
async function pressRun(page, query) {
  await page.goto(`${site}benchmark/?${query}`);
  await waitFor(page, () => Array.isArray(window.__benchmark?.all));
  await page.click('section[data-section="memory"] .head button');
}

// a tab ended while it grew: the page's own session crashes it (or the kernel ends it) and loads it again in the same
// tab, where Playwright's page is gone (a crashed page takes no more calls): what the loaded page says is read by the
// DevTools session, which lives on in the tab
async function endedAndLoaded(context, page, url, crash) {
  const cdp = await context.newCDPSession(page);
  const ended = new Promise((resolve) => page.once("crash", resolve));
  if (crash) cdp.send("Page.crash").catch(() => {});  // the renderer never answers it
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("the tab was not ended")), (crash ? 30 : 600) * 1000));
  await Promise.race([ended, timeout]);
  await cdp.send("Page.navigate", { url });
  const read = `(() => { const b = window.__benchmark; if (!b?.done) return null;
    return JSON.stringify({ results: b.results, markdown: b.markdown, mark: sessionStorage.getItem(${JSON.stringify(MARK)}),
      state: document.querySelector('section[data-section="memory"] .state')?.textContent }); })()`;
  for (const begun = Date.now(); Date.now() - begun < 120000; await new Promise((resolve) => setTimeout(resolve, 250))) {
    try {
      const { result } = await cdp.send("Runtime.evaluate", { expression: read, returnByValue: true });
      if (result?.value) {
        // a moment more: a section run again by mistake would have begun by now
        await new Promise((resolve) => setTimeout(resolve, 2000));
        return JSON.parse((await cdp.send("Runtime.evaluate", { expression: read, returnByValue: true })).result.value);
      }
    } catch {
      // the new page has no context yet
    }
  }
  throw new Error("the page loaded again never said it was done");
}
function reportsTheMark(b, least, below) {
  const d = b.results.memory;
  console.log(d?.markdown ?? "no result");
  check(d?.status === "ok" && d.data?.stop === "reloaded", `the page found the mark (${d?.data?.stop})`);
  check(d?.data?.held >= least && d.data.held < below && d.data.trying === d.data.held + 64 * MiB,
        `it reports the last step held (${d?.data?.held / MiB} MiB, trying ${d?.data?.trying / MiB} MiB, below ${below / MiB} MiB)`);
  check(b.mark === null, "the mark was read once and removed");
  check(/loaded again/.test(b.state ?? ""), `the section says so and did not run again (state: ${b.state})`);
  check(b.markdown?.includes("#### Page memory") && b.markdown.includes("the tab ended while it grew"), "the report holds it");
}

fs.rmSync(PROFILE, { recursive: true, force: true });
const context = await playwright.chromium.launchPersistentContext(PROFILE, {});
let page = context.pages()[0] ?? await context.newPage();
const errors = [];
const watch = (p) => p.on("pageerror", (error) => errors.push(String(error.message)));
watch(page);
try {
  if (endedAt) {
    // (5) the tab ended by the kernel for its memory: the whole browser in a cgroup of endedAt MiB
    const url = `${site}benchmark/?run=memory`;
    await page.goto(url);
    await markAt(page, 64 * MiB);
    const b = await endedAndLoaded(context, page, url, false);
    reportsTheMark(b, 256 * MiB, endedAt * MiB);
    console.log(`the tab ended by the kernel in a cgroup of ${endedAt} MiB held ${b.results.memory?.data?.held / MiB} MiB`);
  } else {
    // (1) all it is asked to
    await page.goto(`${site}benchmark/?run=memory&memoryMB=256`);
    await waitFor(page, () => window.__benchmark?.done);
    let b = await state(page);
    const one = b.results.memory;
    console.log(one?.markdown ?? "no result");
    check(one?.status === "ok" && one.data.stop === "limit" && one.data.held === 256 * MiB, `held 256 MiB and stopped at the limit (${one?.data?.stop}, ${one?.data?.held / MiB} MiB)`);
    check(b.mark === null, "the mark is gone after a run that ended");
    check(Object.keys(b.results).join() === "memory", `only the page memory ran (${Object.keys(b.results).join()})`);
    check(b.markdown.includes("#### Page memory") && b.markdown.includes("| 256 MiB |"), "the report holds the section");
    check(Array.isArray(b.all) && !b.all.includes("memory") && b.all.includes("cpu"), `"Run all" leaves it out (${b.all})`);
    check(one?.data?.maximum === 256 * MiB, `the memory was made with room for the limit (${one?.data?.maximum})`);

    // (2) a page left while it grew (a visitor's reload): no mark, and nothing reported
    await pressRun(page, "memoryMB=2048");
    await markAt(page, 128 * MiB);
    await page.reload();
    await waitFor(page, () => Array.isArray(window.__benchmark?.all));
    await page.waitForTimeout(2000);
    b = await state(page);
    check(b.mark === null && !b.results.memory, `a page left leaves no mark and reports nothing (mark ${b.mark}, ${b.results.memory?.data?.stop})`);

    // (3) a page hidden while it grew: the run stops at once and says why
    await pressRun(page, "memoryMB=2048");
    await markAt(page, 128 * MiB);
    await page.evaluate(() => {
      Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true });
      document.dispatchEvent(new Event("visibilitychange"));
    });
    await waitFor(page, () => window.__benchmark?.done, undefined, 30000);
    b = await state(page);
    const three = b.results.memory;
    console.log(three?.markdown ?? "no result");
    check(three?.status === "error" && three.data?.stop === "hidden" && three.data.held >= 128 * MiB && three.data.held < 2048 * MiB,
          `a hidden page stops the run (${three?.data?.stop}, held ${three?.data?.held / MiB} MiB)`);
    check(b.mark === null, "and leaves no mark");

    // (4) a tab ended while it grew: crashed, and loaded again in the same tab with ?run=memory still in the address
    const url = `${site}benchmark/?run=memory&memoryMB=2048`;
    await page.goto(url);
    await markAt(page, 128 * MiB);
    const marked = JSON.parse(await page.evaluate((mark) => sessionStorage.getItem(mark), MARK));
    console.log(`crashing the tab at the mark: held ${marked.held / MiB} MiB, trying ${marked.trying / MiB} MiB`);
    reportsTheMark(await endedAndLoaded(context, page, url, true), marked.held, 2048 * MiB);
  }
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
