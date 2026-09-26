// The benchmark of T45: what this browser does with this model, as a table someone can paste into an issue.
// A plain module, so that Node can import it and test it (the page and tests/bench.mjs both use it).

/** One row per switch combination the benchmark ran. */
export const ROUNDS = [
  { name: "everything", without: [] },
  { name: "without the kernels", without: ["kernels"] },
];

/** The steps of T52, for ?bench=full: each optimization added to the one before it (T110 added the sixth). */
export const FULL_ROUNDS = [
  { name: "NumPy only", without: ["kernels"] },
  { name: "the kernels, int8 widened", without: ["int8", "relaxed", "sampler", "kv16"] },
  { name: "int8 kept as int8", without: ["relaxed", "sampler", "kv16"] },
  { name: "relaxed SIMD", without: ["sampler", "kv16"] },
  { name: "sampling in the kernel", without: ["kv16"] },
  { name: "float16 keys and values", without: [] },
];

const number = (value, digits = 1) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "?");

/** The environment the page can see. Whatever a browser does not tell is left out, never guessed. */
export function environmentOf(navigatorLike = globalThis.navigator, extra = {}) {
  const { hardwareConcurrency, deviceMemory, userAgent } = navigatorLike ?? {};
  return {
    userAgent: userAgent ?? "unknown",
    threads: typeof hardwareConcurrency === "number" ? hardwareConcurrency : undefined,
    memory: typeof deviceMemory === "number" ? deviceMemory : undefined,
    ...extra,
  };
}

/**
 * The Markdown of a benchmark: a line about the machine, then a row per round.
 * rows: [{ name, without, speed, backend, seconds }], environment: what environmentOf() made.
 */
export function benchMarkdown(rows, environment) {
  const machine = [
    environment.model && `**${environment.model}**`,
    // hardwareConcurrency counts logical cores: the software threads the engine used are in the backend column
    environment.threads !== undefined && `${environment.threads} logical cores`,
    environment.memory !== undefined && `${environment.memory} GB or more`,
    environment.pyodide && `Pyodide ${environment.pyodide}`,
    environment.site,
  ].filter(Boolean).join(" · ");
  // no rounds, no table: a /benchmark/ report without its model section (T134) is no row of reportsTable()
  const table = rows.length ? ["| what ran | tok/s | ready | backend |", "|---|---|---|---|", ...rows.map((row) => {
    const ready = row.seconds === undefined ? "" : `${number(row.seconds)} s`;
    return `| ${row.name} | ${number(row.speed)} | ${ready} | ${row.backend ?? ""} |`;
  }), ""] : [];
  return [`### Pyodide LLM benchmark`, "", machine, "", ...table, `<sub>${environment.userAgent}</sub>`].join("\n");
}

// T91: where a visitor sends the result, and the one table the results make.
export const REPOSITORY = "takano32/pyodide-llm";
/** The questions of .github/ISSUE_TEMPLATE/benchmark.md that the page cannot answer. The answer goes after the colon;
 * what is in parentheses is an example, and is not an answer when it is left there. */
export const QUESTIONS = [
  ["Device", "e.g. Pixel 8, MacBook Air M2, a desktop with a Ryzen 7 7700"],
  ["OS", "e.g. Android 16, macOS 26, Windows 11"],
  ["Browser", "e.g. Chrome 148, Safari 26, Firefox 150"],
];

/** The body of a benchmark issue: the three questions, then the page's Markdown as it is. */
export function reportBody(markdown) {
  return [...QUESTIONS.map(([name, example]) => `**${name}**: (${example})`), "",
          "<!-- the page's Markdown, as the page wrote it: please leave it as it is -->", markdown].join("\n");
}

/** The longest address of GitHub's login a new issue may be sent through. A visitor who is not signed in is sent to
 * https://github.com/login?return_to=<the new issue's address, encoded once more: every % becomes %25>, and from
 * about 7,700 characters of that GitHub drops return_to without a word: the visitor signs in, lands on the dashboard,
 * and the report is gone (the second review of T134, 2026-09-26, by curl: an address of 4,485 characters of a
 * report's table was kept, one of 4,935 dropped; the 302 that "6,799 went through" once read was that drop). GitHub
 * also answers 500 from about 7,000 characters of the address itself, and 414 to anyone from about 8,100. A report
 * whose login address would pass this goes to the clipboard, and the issue asks for it to be pasted. */
export const REPORT_LIMIT = 7000;
export const TOO_LONG = "The results were too long for the link and are on your clipboard: please paste them here.";

const issueUrl = (body, environment) => `https://github.com/${REPOSITORY}/issues/new?${new URLSearchParams(
  { template: "benchmark.md", title: `Benchmark: ${environment.model ?? "this device"}`, body: reportBody(body) })}`;

const throughLogin = (url) => `https://github.com/login?return_to=${encodeURIComponent(url)}`;

/** Whether the results are too long for the address of a new issue (reportUrl() then leaves them out). */
export const reportTooLong = (markdown, environment) => throughLogin(issueUrl(markdown, environment)).length > REPORT_LIMIT;

/** The address of a new issue with the template, the title and the body filled in: the page's Markdown, or where it
 * is too long, a line that asks for it from the clipboard. */
export function reportUrl(markdown, environment) {
  return issueUrl(reportTooLong(markdown, environment) ? TOO_LONG : markdown, environment);
}
/** reportUrl() as GitHub's login gets it (tests/bench.mjs) */
export const loginUrl = (markdown, environment) => throughLogin(reportUrl(markdown, environment));

const cells = (line) => line.split("|").slice(1, -1).map((cell) => cell.trim());

/** What one issue says: the answers, the model, the logical cores, and the rows of its table. undefined when the body
 * has no table of the page's. */
export function parseReport(body) {
  const lines = body.replace(/\r/g, "").split("\n");
  const answer = (name) => {
    const line = lines.find((text) => text.startsWith(`**${name}**:`));
    const value = line?.slice(`**${name}**:`.length).trim() ?? "";
    return /^\(e\.g\./.test(value) ? "" : value;  // the example left in place is no answer
  };
  const head = lines.findIndex((line) => line.startsWith("| what ran |"));
  if (head < 0) return undefined;
  const rows = [];
  for (const line of lines.slice(head + 2)) {
    if (!line.startsWith("|")) break;
    const [name, speed, ready, backend] = cells(line);
    rows.push({ name, speed: Number(speed), ready, backend });
  }
  const machine = lines.find((line) => line.startsWith("**") && line.includes(" · ") && !line.includes("**:")) ?? "";
  return { device: answer("Device"), os: answer("OS"), browser: answer("Browser"),
           model: /^\*\*(.+?)\*\*/.exec(machine)?.[1] ?? "", cores: /(\d+) logical cores|(\d+) threads/.exec(machine)?.slice(1).find(Boolean) ?? "",
           rows };
}

/** The table of the visitors' reports (T83's 30-measurements.md): one row per issue, from its "everything" round. */
export function reportsTable(issues) {
  const out = ["| device | OS | browser | model | logical cores | tok/s | without the kernels | backend | report |",
               "|---|---|---|---|---:|---:|---:|---|---|"];
  for (const { number: id, url, body } of issues) {
    const report = parseReport(body ?? "");
    if (!report) continue;
    const row = (name) => report.rows.find((r) => r.name === name);
    const all = row("everything") ?? report.rows.at(-1), plain = row("without the kernels") ?? row("NumPy only");
    out.push(`| ${report.device || "?"} | ${report.os || "?"} | ${report.browser || "?"} | ${report.model || "?"} | ` +
             `${report.cores || "?"} | ${number(all?.speed)} | ${number(plain?.speed)} | ${all?.backend ?? ""} | [#${id}](${url}) |`);
  }
  return out.join("\n");
}

// T157: the GPU section of /benchmark/ against the CPU section's forward pass (forward.js itself, relaxed SIMD and the
// software threads, as the model page runs it). Before T157 the CPU of the GPU's token table was one thread of
// matmul_q8 doubled, which read low: 1.44× on the owner's Android where the CPU section's numbers make about 1.0×.
const STATES = { none: "found nothing to run on here", wrong: "computed something wrong", error: "failed" };

/**
 * The CPU the GPU is held against: of the CPU section's rows, the fastest count of software threads for a token and,
 * each on its own, for a prompt's tokens 16 at once. section: the page's result of the CPU section ({status, data}),
 * or undefined where it has not run. Where it gives nothing to hold against, { why } says so, and no estimate stands
 * in for it.
 */
export function cpuBaseline(section) {
  if (!section) return { why: "run the CPU section for it" };
  if (section.status !== "ok") return { why: `the CPU section ${STATES[section.status] ?? section.status}` };
  const rows = (section.data?.rows ?? []).filter((row) => Number.isFinite(row.msPerToken) && row.msPerToken > 0);
  const fastest = (key) => rows.filter((row) => Number.isFinite(row[key]) && row[key] > 0).sort((a, b) => a[key] - b[key])[0];
  const token = fastest("msPerToken"), prompt = fastest("promptMsPerToken");
  if (!token) return { why: "the CPU section measured no token" };
  return { token: { threads: token.threads, msPerToken: token.msPerToken, GBps: token.GBps },
           ...(prompt ? { prompt: { threads: prompt.threads, msPerToken: prompt.promptMsPerToken } } : {}) };
}

/** How many times faster the GPU is than the CPU on the same work ("0.52×": the CPU is faster); "" where either is
 * missing. */
export const timesFaster = (cpuMs, gpuMs) => (Number.isFinite(cpuMs) && Number.isFinite(gpuMs) && cpuMs > 0 && gpuMs > 0
  ? `${(cpuMs / gpuMs).toFixed(2)}×` : "");

/**
 * The GPU section's table of a token, with the CPU beside it. steps: the GPU section's steps whose name begins with
 * "a token of " ({name, result: {GB, dispatches, msPerToken, tokPerSecond}} or {name, error}); baseline: cpuBaseline().
 * The CPU section's model (two layers of Llama 3.2 1B's width) is not the GPU's, so the CPU's time of a token is its
 * weights read at the GB/s the CPU section measured: a token of the model page is its weights read once (T93).
 */
export function tokenTable(steps, baseline) {
  const cpu = baseline.token;
  const lines = [cpu ? `The CPU: the CPU section's forward pass with ${cpu.threads} software thread${cpu.threads === 1 ? "" : "s"}, its fastest ` +
      `(${number(cpu.GBps)} GB/s), each model's weights read at that. Above 1× the GPU is faster.`
    : `The CPU: not measured (${baseline.why}).`, "",
    "| a token (weights, dispatches, logits back) | GB | dispatches | GPU ms | GPU tok/s | CPU ms | GPU ÷ CPU |",
    "|---|---:|---:|---:|---:|---:|---:|"];
  for (const s of steps) {
    const t = s.result ?? {};
    const name = s.name.replace(/^a token of /, "");
    if (s.error || t.error) {
      // an error's words in one cell: a | or a line break of it would break the table
      lines.push(`| ${name} | | | ${String(s.error ?? t.error).replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " ")} | | | |`);
      continue;
    }
    const cpuMs = cpu && Number.isFinite(t.GB) ? (t.GB / cpu.GBps) * 1000 : undefined;
    lines.push(`| ${name} | ${number(t.GB, 2)} | ${t.dispatches} | ${number(t.msPerToken)} | ${number(t.tokPerSecond)} | ` +
               `${cpuMs === undefined ? "" : number(cpuMs)} | ${timesFaster(cpuMs, t.msPerToken)} |`);
  }
  return lines;
}
