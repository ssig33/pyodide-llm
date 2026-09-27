// The page as the owner sees it before a change to it goes out (AGENTS.md: a drawing task shows a phone-width and a
// desktop-width screenshot first). Serves dist/ like tests/e2e.mjs, or takes a deployed site, and saves one PNG per
// scene and width into a directory. Meant for CI (preview.yml): the development machine runs no browser.
//
//   node tests/screenshots.mjs <directory> [url of a deployed site]
//
// The scenes: the page with the smallest model ready; the settings sheet open at its engine switches (T75); the
// benchmark's bubble (T76); the sheet that opens any repository, and a repository the converter refuses (T88);
// /benchmark/ before a run, while "Run all" runs the CPU section (T177: its progress) and after (T134, whole pages: it is
// a page that scrolls).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";

const [directory = "screenshots", deployed] = process.argv.slice(2);
const base = "/pyodide-llm/";
const root = new URL("../dist/", import.meta.url).pathname;
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".py": "text/plain",
  ".wasm": "application/wasm" };
let server, site = deployed;
if (!site) {
  server = http.createServer((req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    let file = path.join(root, pathname.slice(base.length) || "index.html");
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");  // /benchmark/
    if (!pathname.startsWith(base) || !fs.existsSync(file)) {
      res.writeHead(404);
      return res.end();
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  site = `http://localhost:${server.address().port}${base}`;
}
fs.mkdirSync(directory, { recursive: true });

const browser = await playwright.chromium.launch({ headless: true });
const widths = { phone: { width: 390, height: 844 }, desktop: { width: 1280, height: 800 } };
// a wait that the first visit's reload (coi.js, T93) interrupts starts again
const acrossReload = async (wait) => {
  for (;;) {
    try {
      return await wait();
    } catch (error) {
      if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
    }
  }
};
const shot = (page, name, width) => page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });

// Every scene in a context of its own: a second document in the same tab could not allocate its WebAssembly memory
// (T96's failure, seen here first with the benchmark as the second document).
const fresh = async (viewport) => {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage();
  page.on("dialog", (dialog) => dialog.accept());
  return { context, page };
};
for (const [width, viewport] of Object.entries(widths)) {
  let { context, page } = await fresh(viewport);
  const ready = () => acrossReload(() => page.waitForFunction(() => document.getElementById("run")?.disabled === false || document.querySelector(".error"), null, { timeout: 600000 }));

  // 1. the page, the smallest model ready
  await page.goto(`${site}?model=stories260K`);
  await ready();
  await shot(page, "home", width);
  // T128: the order of the list. A browser draws an open <select> outside the page, where a screenshot does not
  // reach, so the select is shown as a list box of every entry (the order is the same; its look is not the phone's)
  await page.evaluate(() => {
    const select = document.getElementById("model");
    select.size = select.querySelectorAll("option, optgroup").length;
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(directory, `list-${width}.png`), fullPage: true });
  await page.evaluate(() => { document.getElementById("model").size = 0; });

  // 2. the settings sheet, the engine's switches open (T75)
  await page.click("#settings-open");
  await page.evaluate(() => { document.getElementById("engine").open = true; });
  await page.waitForTimeout(300);
  await shot(page, "settings-engine", width);
  // T119 (5): Engine and More both open: the sheet stays under the top of the screen and scrolls
  await page.evaluate(() => { for (const part of document.querySelectorAll("#settings details")) part.open = true; });
  await page.waitForTimeout(300);
  await shot(page, "settings-both", width);
  await page.keyboard.press("Escape");

  // 3. the sheet that opens any repository, with a name typed (T88)
  await page.selectOption("#model", "hf-other");
  await page.waitForTimeout(300);
  await page.fill("#repository-name", "Qwen/Qwen2.5-0.5B-Instruct");
  await shot(page, "repository", width);
  // T119 (1): a wrong revision is marked on the revision's field, not the name's
  await page.fill("#repository-revision", "not a revision!");
  await page.click("#repository-form button[type=submit]");
  await page.waitForTimeout(300);
  await shot(page, "repository-revision", width);
  const marks = await page.evaluate(() => ["repository-name", "repository-revision"].map((id) => document.getElementById(id).validationMessage));
  console.log(`${width}: the name's field says "${marks[0]}", the revision's "${marks[1]}"`);
  await page.keyboard.press("Escape");

  // 4. the benchmark's bubble (T76)
  await context.close();
  ({ context, page } = await fresh(viewport));
  // T214: at the phone's width the browser does not say its memory, as an iPhone's Safari does not: the round without
  // the kernels must be a row that says why (and nothing measured), where the desktop's runs both
  const unsaid = width === "phone";
  if (unsaid) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "deviceMemory", { get: () => undefined, configurable: true }));
  await page.goto(`${site}?model=stories260K&bench=1`);
  await acrossReload(() => page.waitForFunction(() => window.__bench || document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "bench", width);
  const benched = await page.evaluate(() => window.__bench?.rows ?? []);
  console.log(`${width}: ?bench=1${unsaid ? " without deviceMemory" : ""}: ${benched.map((row) => `${row.name} ${row.skip !== undefined ? `skipped (${row.skip})` : `${row.speed?.toFixed(1)} tok/s`}`).join(", ")}`);
  const [all, plain] = benched;
  if (!(all?.speed > 0) || all.skip !== undefined || !plain || (plain.skip !== undefined) !== unsaid ||
      (unsaid ? !/does not say how much memory/.test(plain.skip) || plain.speed !== undefined : !(plain.speed > 0))) {
    throw new Error(`${width}: ?bench=1 did not ${unsaid ? "skip the round without the kernels, in words" : "run both rounds"}`);
  }

  // 5. a repository the converter refuses, in words (T88)
  await context.close();
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=microsoft/phi-1_5`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "refused", width);

  // T119 (2): a gated repository, in words
  await context.close();
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=meta-llama/Llama-3.2-1B`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector(".error"), null, { timeout: 600000 }));
  await page.waitForTimeout(300);
  await shot(page, "gated", width);

  // T134: /benchmark/, before a run and after every section ran (the smallest model, a small file for the storage)
  await context.close();
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}benchmark/`);
  await acrossReload(() => page.waitForFunction(() => document.querySelector("section .head button"), null, { timeout: 120000 }));
  await page.screenshot({ path: path.join(directory, `benchmark-${width}.png`), fullPage: true });
  await page.goto(`${site}benchmark/?run=all&model=stories260K&size=64`);
  // T177: the CPU section under way, its stage counted and its seconds going (the window, not the whole page)
  await acrossReload(() => page.waitForFunction(() => {
    const box = document.querySelector('section[data-section="cpu"] .progress');
    return box && !box.hidden && /\/ \d+ ·/.test(box.querySelector(".stage").textContent)
      && box.querySelector(".elapsed").textContent !== "0 s";
  }, null, { timeout: 300000, polling: 100 }));
  await page.locator('section[data-section="cpu"]').scrollIntoViewIfNeeded();
  await shot(page, "benchmark-running", width);
  await acrossReload(() => page.waitForFunction(() => window.__benchmark?.done, null, { timeout: 900000 }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(directory, `benchmark-done-${width}.png`), fullPage: true });
  console.log(`${width}: /benchmark/ ${await page.evaluate(() => Object.entries(window.__benchmark.results).map(([n, r]) => `${n} ${r.status}`).join(", "))}`);
  // the page itself never scrolls sideways at a phone's width: a wide table scrolls inside its section
  console.log(`${width}: /benchmark/ is ${await page.evaluate(() => document.documentElement.scrollWidth)} px wide in a window of ${viewport.width}`);
  // T186: "This browser" at 360 px (a narrow Android): its table's values wrap instead of running out on the right
  if (width === "phone") {
    await page.setViewportSize({ width: 360, height: viewport.height });
    await page.locator('section[data-section="device"]').scrollIntoViewIfNeeded();
    await shot(page, "benchmark-device", 360);
    console.log(`360: "This browser"'s table is ${await page.evaluate(() => {
      const out = document.querySelector('section[data-section="device"] table');
      return out ? `${out.scrollWidth} px wide in a box of ${out.parentElement.clientWidth}` : "not there";
    })}, the page ${await page.evaluate(() => document.documentElement.scrollWidth)} px`);
    await page.setViewportSize(viewport);
  }

  // T119 (6): a conversion under way says what has arrived, how fast, and what is converted
  await context.close();
  ({ context, page } = await fresh(viewport));
  await page.goto(`${site}?hf=HuggingFaceTB/SmolLM2-360M-Instruct`);
  await acrossReload(() => page.waitForFunction(() => /MB\/s/.test(document.getElementById("status-text")?.textContent ?? ""), null, { timeout: 600000 }));
  await shot(page, "converting", width);
  console.log(`${width}: ${await page.evaluate(() => document.getElementById("status-text").textContent)}`);
  await context.close();
}
await browser.close();
server?.close();
console.log(fs.readdirSync(directory).join("\n"));
