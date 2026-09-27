// The same check as tests/e2e.mjs, in the Firefox that is installed on the machine, driven by Selenium and
// geckodriver instead of Playwright. Playwright can only drive its own patched Firefox, and it drives it through the
// debugger, which makes SpiderMonkey compile WebAssembly for debugging: baseline only, 13 to 17 times slower than
// Chromium even on a bare SIMD loop with nothing of Pyodide (T59). Marionette, which geckodriver talks to, does not.
//
//   node tests/stock-firefox.mjs [model id = stories260K] [url of the site]
import { Builder, Key } from "selenium-webdriver";
import firefox from "selenium-webdriver/firefox.js";

const [model = "stories260K", site = "https://takano32.github.io/pyodide-llm/"] = process.argv.slice(2);
const expected = { stories260K: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park." };

const driver = await new Builder().forBrowser("firefox").setFirefoxOptions(new firefox.Options().addArguments("-headless")).build();
const failures = [];
try {
  const capabilities = await driver.getCapabilities();
  const until = async (script, seconds) => {
    for (const end = Date.now() + seconds * 1000; Date.now() < end; await new Promise((r) => setTimeout(r, 250))) {
      // T93: the first visit reloads once under the service worker (coi.js); a script the reload cuts off is tried again
      if (await driver.executeScript(script).catch(() => false)) {
        return;
      }
    }
    throw new Error(`timed out waiting for: ${script}`);
  };
  const started = Date.now();
  await driver.get(`${site}?model=${model}`);
  await until('return !document.getElementById("run").disabled || !!document.querySelector(".error")', 600);
  const readySeconds = (Date.now() - started) / 1000;
  // 256 tokens, as in every other number of the documents
  await driver.executeScript('const steps = document.getElementById("steps"); steps.value = "256"; steps.dispatchEvent(new Event("input"));');
  await driver.findElement({ id: "prompt" }).sendKeys(Key.chord(Key.CONTROL, Key.ENTER));
  await until('return !!document.querySelector(".model .meta") || !!document.querySelector(".error")', 900);
  const result = await driver.executeScript(`return {
    text: document.querySelector(".model .bubble")?.textContent ?? "",
    meta: document.querySelector(".model .meta summary")?.textContent ?? "",
    error: document.querySelector(".error .bubble")?.textContent ?? "",
    status: document.getElementById("status-text")?.textContent ?? "" }`);
  if (result.error) failures.push(`the page reported: ${result.error}`);
  if (!/tok\/s/.test(result.meta)) failures.push("no speed line under the answer");
  if (expected[model] && !result.text.startsWith(expected[model])) failures.push(`unexpected text: ${result.text.slice(0, 120)}`);
  console.log(`stock firefox ${capabilities.get("browserVersion")} (${capabilities.get("moz:geckodriverVersion")}), ${model}: ready in ${readySeconds.toFixed(1)}s, ${result.meta}`);
  console.log(`status: ${result.status}`);
  console.log(result.text.slice(0, 160).replace(/\n/g, " / "));
} finally {
  // T145: given up on after 15 s, as tests/e2e.mjs gives up on browser.close() (T141): the verdict is known by then
  await Promise.race([driver.quit(), new Promise((resolve) => setTimeout(resolve, 15000))]).catch(() => {});
}
if (failures.length) {
  console.error("FAILED\n- " + failures.join("\n- "));
  process.exit(1);
}
// T141: end here, as tests/e2e.mjs does: what the browser leaves behind must not keep the job waiting
process.stdout.write("ok\n", () => process.exit(0));
