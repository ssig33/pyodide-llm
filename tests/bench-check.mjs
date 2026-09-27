// /benchmark/ (T134) in real browsers: every section run once (?run=all), the Markdown printed, and a failure where a
// result cannot be right. A feature a browser lacks (no WebGPU, no private file system in a worker) is no failure: the
// page says so and goes on. The runners are not the owner's devices, so the numbers only show that the page works;
// Playwright's Firefox runs under a debugger and its numbers are not Firefox's (AGENTS.md). Meant for CI.
//
//   node tests/bench-check.mjs [url of a site | --dist] [engine ...] [--model id] [--size MiB] [--run sections]
//     engines: chromium firefox webkit chrome msedge (default: the first three)
//     --dist serves dist/ (npm run build) itself, as tests/screenshots.mjs does: a branch before it goes out
//     --run the page's ?run= (default all; "gpu" alone: the GPU section takes minutes on SwiftShader, T146)
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";

const args = process.argv.slice(2);
const option = (name, value) => {
  const at = args.indexOf(name);
  return at >= 0 ? args.splice(at, 2)[1] : value;
};
const model = option("--model", "llm-jp-3-150m"), size = option("--size", "256"), sections = option("--run", "all");
const dist = args.includes("--dist") ? args.splice(args.indexOf("--dist"), 1) : null;
let [site = "https://takano32.github.io/pyodide-llm/", ...engines] = dist ? [undefined, ...args] : args;
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
// Chromium's WebGPU without a GPU: SwiftShader's Vulkan, where the build has it (as tests/gpu-check.mjs had it)
const WEBGPU = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"];
// the sections whose failure is this page's: the others depend on the runner (its GPU, its disk, its line), and what
// they could not do is printed
const OURS = ["device", "cpu", "model"];
// T177: every stage a section showed while it ran, as the page drew it (a watcher in the page from its start, so that a
// stage shown for a moment is seen too). A section of more than one stage that measured and never changed its stage
// on the page would look frozen to a visitor: that fails (the page's own "starting" is not counted: the stages come from
// the section). The browser's features are one quick stage, and a section
// that found no such thing here stops at its first.
const STAGED = ["cpu", "model", "gpu", "storage", "line"];
function watchStages() {
  const seen = (window.__stages = {});
  const look = () => {
    for (const box of document.querySelectorAll("section[data-section] .progress")) {
      const text = box.querySelector(".stage")?.textContent;
      const name = box.closest("section").dataset.section;
      if (box.hidden || !text) continue;
      const list = (seen[name] ??= []);
      if (list.at(-1) !== text) list.push(text);
    }
  };
  document.addEventListener("DOMContentLoaded", () => {
    new MutationObserver(look).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true });
  });
}

let failed = false;
for (const engine of engines.length ? engines : ["chromium", "firefox", "webkit"]) {
  const branded = ["chrome", "msedge"].includes(engine);
  const type = branded ? playwright.chromium : playwright[engine];
  console.log(`### ${engine}`);
  let browser;
  try {
    browser = await type.launch({ ...(branded ? { channel: engine } : {}), args: branded || engine === "chromium" ? WEBGPU : [] });
  } catch (error) {
    console.log(`could not start: ${String(error.message).split("\n")[0]}\n`);
    continue;
  }
  const page = await (await browser.newContext()).newPage();
  await page.addInitScript(watchStages);
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error.message)));
  try {
    await page.goto(`${site}benchmark/?run=${sections}&model=${model}&size=${size}`);
    // the first visit reloads once, when the service worker takes the page over (T93): the wait starts again
    for (;;) {
      try {
        await page.waitForFunction(() => window.__benchmark?.done, null, { timeout: 1200000 });
        break;
      } catch (error) {
        if (!/destroyed|navigat|detached/i.test(String(error.message))) throw error;
      }
    }
    console.log(`${browser.version()}\n`);
    const { results, markdown } = await page.evaluate(() => window.__benchmark);
    console.log(markdown, "\n");
    const statuses = Object.entries(results).map(([name, result]) => `${name} ${result.status}`);
    console.log(`sections: ${statuses.join(", ")}`);
    for (const [name, result] of Object.entries(results)) {
      if (result.status === "wrong" || (result.status === "error" && OURS.includes(name))) failed = true;
    }
    const stages = await page.evaluate(() => window.__stages);
    for (const [name, result] of Object.entries(results)) {
      const shown = stages[name] ?? [];
      console.log(`${name} stages on the page: ${shown.join(" → ") || "none"}`);
      if (STAGED.includes(name) && ["ok", "wrong"].includes(result.status) && new Set(shown.filter((text) => text !== "starting")).size < 2) {
        console.log(`${name}: its stage on the page never changed while it ran`);
        failed = true;
      }
    }
    // the shaders against JavaScript are the GPU section's check of its own results: with an adapter they must run
    // (and every tiled shader of T146 among them: one the runner's adapter could not make fails here too)
    const check = results.gpu?.data?.steps?.find((s) => s.name === "the shaders against JavaScript");
    if (check?.error || Object.values(check?.result ?? {}).some((v) => v.error)) failed = true;
    // T157: after the CPU section, the GPU section's token table holds the GPU against it (and not the old estimate)
    if (results.cpu?.status === "ok" && results.gpu?.data && !results.gpu.markdown.includes("The CPU (an estimate): each model's weights at the CPU section's fastest")) {
      console.log("the GPU section's token table does not hold the GPU against the CPU section");
      failed = true;
    }
  } catch (error) {
    console.log(`failed: ${String(error.message).split("\n")[0]}`);
    failed = true;
  }
  if (errors.length) console.log(`page errors: ${errors.join(" / ")}`);
  console.log("");
  await browser.close();
}
server?.close();
process.exit(failed ? 1 : 0);
