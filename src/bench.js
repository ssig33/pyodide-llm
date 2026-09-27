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
    // T176: the site's version (the commit it was built from), so that a report says which shaders and kernels ran
    environment.build && `site ${environment.build}`,
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

/** The counts of software threads the CPU section measures: 1, 2, 4 and on, doubling, up to the logical cores, and
 * the cores themselves where they are no power of two (T157's review: the owner's Android has 8, and 2 to 4 threads
 * still gave 1.41 to 1.46 times). */
export function threadCounts(cores) {
  const most = Number.isInteger(cores) && cores > 0 ? cores : 4;
  const counts = [];
  for (let n = 1; n <= most; n *= 2) counts.push(n);
  if (counts.at(-1) !== most) counts.push(most);
  return counts;
}

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
  const isolated = section.data?.shared !== false;
  return { token: { threads: token.threads, msPerToken: token.msPerToken, GBps: token.GBps, isolated },
           ...(prompt ? { prompt: { threads: prompt.threads, msPerToken: prompt.promptMsPerToken, isolated } } : {}) };
}

/** "4 software threads", "1 software thread (not cross-origin isolated)" */
export const threadsOf = ({ threads, isolated }) =>
  `${threads} software thread${threads === 1 ? "" : "s"}${isolated ? "" : " (not cross-origin isolated)"}`;

/** A ratio as the tables write it: to one decimal from 1 up, to two significant digits below ("0.50×", "0.071×"),
 * whole from 100; "" where it is no number. The measurements are one run each: more digits would be noise. */
export function times(value) {
  if (!Number.isFinite(value) || value <= 0) return "";
  return `${value >= 100 ? value.toFixed(0) : value >= 1 ? value.toFixed(1) : value.toPrecision(2)}×`;
}

/** How many times faster the GPU is than the CPU on the same work ("0.50×": the CPU is faster); "" where either is
 * missing. */
export const timesFaster = (cpuMs, gpuMs) => (Number.isFinite(cpuMs) && Number.isFinite(gpuMs) && cpuMs > 0 && gpuMs > 0
  ? times(cpuMs / gpuMs) : "");

/** Why the GPU section shows no ratios at all, whatever the CPU: a lost device (its later times are no GPU's, and
 * too fast: a lost device answers every wait at once) or a fallback adapter (the CPU in a GPU's place). undefined
 * where the ratios stand. gpu: { fallback, lost }. */
export function noRatios({ fallback, lost } = {}) {
  if (lost) return "none: the device was lost, and the times after it are no GPU's";
  if (fallback) return "none on a fallback adapter: its times are no GPU's";
  return undefined;
}

/** Words for one cell of a table: a | or a line break of them would break the row (the page and GitHub both read
 * \| as a | in a cell). */
export const tableCell = (text) => String(text).replace(/\|/g, "\\|").replace(/\s*\n\s*/g, " ");

/** Whether the check of the shaders found the one a token's row ran with wrong (or could not run it): widened or
 * packed, and the choosing on the GPU where it chose. */
function tokenWrong(t, check) {
  if (!check) return false;
  const bad = (key) => check[key] !== undefined && !check[key].ok;
  return bad(t.kind === "packed" ? "packed" : "widen") || (Boolean(t.sample) && bad("argmax"));
}

/** T149: the GPU section's table of an int8 matrix times a vector: a row a shader (T134's two and those of llama.cpp and
 * ONNX Runtime), a column a shape (bandwidths: the steps named "bandwidth: <shape>", {name, result: {rows, cpu,
 * quantize}} or {name, error}), with each GB/s's share of what a loop that only reads a buffer reads (ceilings: T168's).
 * No share where the device was lost (its later times are no GPU's, and too fast: past 100%), on a fallback adapter,
 * or where that read was unsteady. A shader the check found WRONG says so and is never the fastest, nor is a row
 * measured again at the end. gpu: { lost }. */
const QUANTIZED_A_TOKEN = 16 * 4 + 1;  // Llama 3.2 1B: q, o, gate and down a layer read an input of their own, and the classifier
export function matVecTable(bandwidths, check, ceilings, gpu = {}) {
  const percent = (part, whole) => `${number((100 * part) / whole)}%`;
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  const shape = (s) => s.name.replace(/^bandwidth: /, "");
  const shaders = [...new Map(bandwidths.flatMap((s) => (s.result?.rows ?? []).map((row) => [row.shader, row]))).values()];
  const cell = (s, name) => {
    const row = s.result?.rows?.find((one) => one.shader === name);
    if (s.error || !row) return tableCell(s.error ?? "");
    if (row.none || row.error) return tableCell(row.none ?? `failed: ${row.error}`);
    return `${row.unsteady ? "unsteady: " : ""}${number(row.GBps)} GB/s${reads ? ` (${percent(row.GBps, reads)})` : ""}`;
  };
  const fastest = (s) => {
    const best = (s.result?.rows ?? []).filter((row) => row.GBps && !row.again && !row.unsteady && !wrong(row)).sort((a, b) => b.GBps - a.GBps)[0];
    return best ? `${shape(s)} ${best.shader}, ${number(best.GBps)} GB/s` : "";
  };
  const quantized = bandwidths.filter((s) => s.result?.quantize?.msEach);
  const wide = quantized.find((s) => s.name.includes("1B"));
  const quantizing = quantized.length ? "**Quantizing the vector** of a packed row (QUANTIZE, one dispatch; not in the packed rows): " +
    quantized.map((s) => `${shape(s)} ${number(s.result.quantize.msEach, 3)} ms`).join(", ") +
    (wide ? `; a token of Llama 3.2 1B quantizes ${QUANTIZED_A_TOKEN} vectors (16 of them 8192 wide), ${number(QUANTIZED_A_TOKEN * wide.result.quantize.msEach, 2)} ms or more` : "") : "";
  return ["**An int8 matrix × vector** (a generated token's). A shader's name says whose form it takes and the rows a workgroup takes. " +
    "Each is timed as a submission of 2n of it less one of n: what waiting for a submission costs is not in it (a token pays it once, in the table of what it costs besides the weights below), " +
    "and each matrix is read from copies of it that make 128 MiB in turn, as a token reads it once, not from the GPU's caches: not to be compared with reports from before T149. " +
    `The packed rows take the vector quantized already${reads ? `. In parentheses, the share of what a loop that only reads a buffer reads, ${number(reads)} GB/s below` : ""}.`,
    ...(gpu.lost ? ["No share of the buffer's reads: the device was lost, and the times after it are no GPU's."] : []), "",
    `| shader | ${bandwidths.map(shape).join(" | ")} |`, `|---|${bandwidths.map(() => "---:|").join("")}`,
    ...shaders.map((row) => `| ${row.shader}${wrong(row) ? " (WRONG in the check)" : ""} | ${bandwidths.map((s) => cell(s, row.shader)).join(" | ")} |`),
    // Safari's kernel on one thread, for scale; the page's forward on the CPU is the CPU section's (T157)
    `| CPU matmul_q8, one thread (not the page's forward) | ${bandwidths.map((s) => (s.result?.cpu ? `${number(s.result.cpu.GBps)} GB/s` : "")).join(" | ")} |`,
    ...(quantizing ? ["", quantizing] : []),
    "", `**Fastest on the GPU** (of the shaders the check found right): ${bandwidths.map(fastest).filter(Boolean).join("; ") || "nothing measured"}`];
}

/**
 * The GPU section's table of a token, with the CPU beside it. steps: the GPU section's steps whose name begins with
 * "a token of " ({name, result: {kind, sample, GB, dispatches, msPerToken, tokPerSecond}} or {name, error});
 * baseline: cpuBaseline(); gpu: { fallback, lost, check } (check: the shaders against JavaScript).
 * The CPU section's model (two layers of Llama 3.2 1B's width) is not the GPU's, so the CPU's speed on each model is
 * an estimate: its weights read at the GB/s the CPU section measured (a token of the model page reads its weights once,
 * T93). GPU ÷ CPU is then the GPU's tok/s over that.
 */
export function tokenTable(steps, baseline, gpu = {}) {
  const cpu = baseline.token, none = noRatios(gpu);
  const lines = [cpu ? `The CPU (an estimate): each model's weights at the CPU section's fastest, ${number(cpu.GBps)} GB/s with ${threadsOf(cpu)}. ` +
      "Its model is as wide as Llama 3.2 1B; a narrower one such as llm-jp-3 150M runs slower than this says. Above 1× the GPU is faster. " +
      "Each number is one run of this page (the same device has differed by more than twice from one run to another)."
    : `The CPU: not measured (${baseline.why}).`,
    ...(none && cpu ? [`GPU ÷ CPU: ${none}.`] : []), "",
    "| a token (weights, dispatches, logits back) | GB | dispatches | GPU ms | GPU tok/s | CPU tok/s (estimate) | GPU ÷ CPU |",
    "|---|---:|---:|---:|---:|---:|---:|"];
  for (const s of steps) {
    const t = s.result ?? {};
    const name = s.name.replace(/^a token of /, "");
    if (s.error || t.error) {
      lines.push(`| ${name} | | | ${tableCell(s.error ?? t.error)} | | | |`);
      continue;
    }
    const cpuTok = cpu && Number.isFinite(t.GB) && t.GB > 0 ? cpu.GBps / t.GB : undefined;
    const wrong = tokenWrong(t, gpu.check);
    const ratio = none || wrong || cpuTok === undefined ? "" : times(t.tokPerSecond / cpuTok);
    lines.push(`| ${name}${wrong ? " (WRONG in the check)" : ""} | ${number(t.GB, 2)} | ${t.dispatches} | ${number(t.msPerToken)} | ` +
               `${number(t.tokPerSecond)} | ${cpuTok === undefined ? "" : number(cpuTok)} | ${ratio} |`);
  }
  return lines;
}

/**
 * T150: the GPU section's table of one layer of a token: its fourteen steps each a dispatch of their own, the same
 * fused into five (public/shaders.js's fusedMatVec), and fused but for the norms (seven). step: {name, result: {model,
 * pos, layers, GB, rows: [{form, check, fused, normApart, subgroups, dispatches, msPerLayer, GBps, unsteady}, or {form,
 * check, error}]}} or {name, error}; check: the shaders against JavaScript (a form's verdict under row.check);
 * ceilings: T168's (the buffer's reads, for the share of it the layer's weights are read at); gpu: { fallback, lost }.
 * "Faster than the separate steps" beside a fused row, against the separate steps of the same reduction (measured in
 * turn with it): not where noRatios() says none, nor for a row the check found WRONG or that was unsteady. The share
 * of the reads: not after a lost device, on a fallback adapter, or where that read was unsteady (as matVecTable).
 */
export function layerTable(step, check, ceilings, gpu = {}) {
  if (!step) return [];
  if (step.error || !step.result) return [`**A layer of a token**: ${tableCell(step.error ?? "not measured")}`];
  const r = step.result, none = noRatios(gpu);
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  const usable = (row) => Number.isFinite(row?.msPerLayer) && row.msPerLayer > 0 && !row.unsteady && !wrong(row);
  const faster = (row) => {
    if (none || !row.fused || !usable(row)) return "";
    const separate = r.rows.find((one) => !one.fused && one.subgroups === row.subgroups);
    return usable(separate) ? times(separate.msPerLayer / row.msPerLayer) : "";
  };
  // two significant digits under 1 GB/s (a fallback adapter reads a layer at a few hundredths)
  const GBps = (row) => (Number.isFinite(row.GBps) ? `${row.GBps < 1 ? row.GBps.toPrecision(2) : number(row.GBps)}${reads ? ` (${number((100 * row.GBps) / reads)}%)` : ""}` : "");
  return [`**A layer of a token** (${r.model}'s width, at position ${r.pos}, ${number(r.GB * 1000, 1)} MB of weights): its fourteen steps each a dispatch of its own ` +
    "(the norm, q, k, v, RoPE and the cache, the attention, o, the residual's add, the norm, gate, up, SwiGLU, down, the add), the same fused into five " +
    "(q, k and v with the norm, RoPE and the cache; the attention; o with the add; gate and up with the norm and SwiGLU; down with the add), " +
    "and fused but for the two norms (seven). The forms are timed in turn, each as a submission of 2n layers less one of n, the weights read from copies of them in turn, as the matrix × vector is. " +
    `GB/s: the layer's weights over its time${reads ? `; in parentheses, the share of what a loop that only reads a buffer reads, ${number(reads)} GB/s below` : ""}.`,
    ...(none ? [`Faster than the separate steps: ${none}.`] : []), ...(gpu.lost ? ["No share of the buffer's reads: the device was lost."] : []), "",
    `| a layer | dispatches | GPU ms | GB/s | its ${r.layers} layers, ms | faster than the separate steps |`, "|---|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.error ? `| ${tableCell(row.form)} | ${tableCell(`failed: ${row.error}`)} | | | | |`
      : `| ${tableCell(row.form)}${wrong(row) ? " (WRONG in the check)" : ""} | ${row.dispatches} | ${row.unsteady ? "unsteady: " : ""}${number(row.msPerLayer, 2)} | ` +
        `${GBps(row)} | ${number(row.msPerLayer * r.layers)} | ${faster(row)} |`))];
}

/**
 * T151: the GPU section's table of tokens generated on the GPU (public/shaders.js's EMBED, fusedMatVec and SAMPLE: the
 * sampling on the GPU too, the state carried from token to token there). step: {name, result: {model, layers, vocab,
 * GB, dispatches, tokens, settings, work: {ms, unsteady}, rows: [{perSubmission, msPerToken, fixedMs}], sampling:
 * {vocab, msEach, over, unsteady, flat: {msEach, over, unsteady}}}} or {name, error}; check: the shaders against
 * JavaScript ("sampling" and "tokens on the GPU"); gpu: { fallback, lost }. A row a way of submitting: one token a submission, each read back as it comes, and
 * several in one submission, read back once. "A submission besides its tokens" is what a submission costs past its
 * tokens' work (the wait and the reading back: a row's ms a token times its tokens, less a token's work, the
 * difference of 2n tokens in one submission and n), and "faster than one a submission" the first row's ms a token
 * over the row's: neither where noRatios() says none, where the check found the sampling or the run WRONG, nor
 * (the first) where a token's work was unsteady. A submission's cost below 0 (a token's work read higher than the
 * row's time allows) is "under the noise".
 */
export function generateTable(step, check, gpu = {}) {
  if (!step) return [];
  if (step.error || !step.result) return [`**Tokens generated on the GPU**: ${tableCell(step.error ?? "not measured")}`];
  const r = step.result, none = noRatios(gpu);
  const wrong = ["sampling", "tokens on the GPU"].some((key) => check?.[key] && !check[key].ok);
  const derived = !none && !wrong;
  const one = r.rows.find((row) => row.perSubmission === 1);
  const s = r.settings ?? {};
  return [`**Tokens generated on the GPU** (${tableCell(r.model)}: ${r.layers} layers, a vocabulary of ${r.vocab}, ${number(r.GB, 2)} GB of weights a token; ` +
    `a token is ${r.dispatches} dispatches: its row of the embedding, the layers fused as T150's, the classifier, and the sampling on the GPU, ` +
    `penalty ${s.penalty}, temperature ${s.temperature}, top-p ${s.topp}). The same ${r.tokens} tokens each way, the ways in turn; ` +
    "a token's work alone is a submission of 2n tokens less one of n. A submission besides its tokens: what it costs past their work (submitting, waiting, reading the ids back)." +
    `${wrong ? " The check found the sampling on the GPU WRONG." : ""}`,
    ...(none ? [`Besides its tokens and faster: ${none}.`] : []), "",
    "| tokens a submission | GPU ms a token | a submission besides its tokens, ms | faster than one a submission |", "|---|---:|---:|---:|",
    ...r.rows.map((row) => `| ${row.perSubmission === 1 ? "1, each read back as it comes" : `${row.perSubmission}, read back once`}${wrong ? " (WRONG in the check)" : ""} | ` +
      `${number(row.msPerToken, 2)} | ${derived && Number.isFinite(row.fixedMs) ? (row.fixedMs < 0 ? "under the noise" : number(row.fixedMs, 2)) : ""} | ` +
      `${derived && row !== one && one ? times(one.msPerToken / row.msPerToken) : ""} |`),
    "", `A token's work: ${r.work ? `${r.work.unsteady ? "unsteady: " : ""}${number(r.work.ms, 2)} ms` : "not measured here"}; ` +
    `the sampling alone (made-up logits of ${r.sampling?.vocab ?? "Llama 3's"} tokens): ${r.sampling ? samplingCell(r.sampling) : "not measured here"}` +
    `${r.sampling?.flat ? `; on flat logits, ${samplingCell(r.sampling.flat)}` : ""}.`];
}
// the ms of the sampling alone, with how many tokens were over the nucleus's floor (what SAMPLE gathers and reads each
// round of its searches: T151)
const samplingCell = (s) => `${s.unsteady ? "unsteady: " : ""}${number(s.msEach, 3)} ms${Number.isFinite(s.over) ? ` (${s.over} tokens over the floor)` : ""}`;
