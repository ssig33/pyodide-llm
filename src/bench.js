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

/** T214: why a round that widens the weights is skipped where the browser does not say how much memory the device has
 * (navigator.deviceMemory: Safari and Firefox never say). Without the kernels NumPy widens every weight to float32
 * (four bytes a weight, where the kernels keep one): with llm-jp-3 150M the worker's Pyodide grew to 958 MB and the
 * renderer by 0.81 GB in CI's Chromium, and an iPhone's tab went down in that round (T205's review). A page that is
 * not told the memory cannot tell whether it fits (the owner, 2026-09-27: such a device skips the NumPy round). */
export const MEMORY_UNSAID = "this browser does not say how much memory the device has, and without the kernels " +
  "NumPy widens every weight to float32 (four bytes a weight: about 1.2 GB in all with llm-jp-3 150M), which took an iPhone's tab down";

/** The rounds as this device runs them (T214): a round without the kernels carries skip (the words of its row) where
 * deviceMemory is not a number. The page decides (both ?bench= and /benchmark/), and the worker writes the row. */
export function roundsHere(rounds, deviceMemory) {
  const unsaid = typeof deviceMemory !== "number";
  return rounds.map((round) => (unsaid && round.without.includes("kernels") ? { ...round, skip: MEMORY_UNSAID } : round));
}

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
  const table = rows.length ? [...roundsTable(rows), ""] : [];
  return [`### Pyodide LLM benchmark`, "", machine, "", ...table, `<sub>${environment.userAgent}</sub>`].join("\n");
}

/** The rounds' table (its lines): a row per round; a skipped round (T214) says "skipped" for its speed, and why in
 * the last column. benchMarkdown() and /benchmark/'s model section both write it. */
export function roundsTable(rows) {
  return ["| what ran | tok/s | ready | backend |", "|---|---|---|---|", ...rows.map((row) => {
    if (row.skip !== undefined) return `| ${row.name} | skipped |  | ${tableCell(row.skip)} |`;
    const ready = row.seconds === undefined ? "" : `${number(row.seconds)} s`;
    return `| ${row.name} | ${number(row.speed)} | ${ready} | ${row.backend ?? ""} |`;
  })];
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
/** T185: the last line of a summary (shortReport()), which the link holds where the whole report is too long */
export const PASTE = "The whole report is on your clipboard: please paste it below.";

const issueUrl = (body, environment) => `https://github.com/${REPOSITORY}/issues/new?${new URLSearchParams(
  { template: "benchmark.md", title: `Benchmark: ${environment.model ?? "this device"}`, body: reportBody(body) })}`;

const throughLogin = (url) => `https://github.com/login?return_to=${encodeURIComponent(url)}`;

/** Whether the results are too long for the address of a new issue (reportUrl() then leaves them out). */
export const reportTooLong = (markdown, environment) => throughLogin(issueUrl(markdown, environment)).length > REPORT_LIMIT;

/** The address of a new issue with the template, the title and the body filled in: the page's Markdown; where it is
 * too long, the summary (shortReport(): T185, a GPU's report is about four times the limit), which asks for the whole
 * from the clipboard; where that is too long as well, or there is none (the model page's), that request alone. */
export function reportUrl(markdown, environment, summary) {
  const body = !reportTooLong(markdown, environment) ? markdown
    : summary && !reportTooLong(summary, environment) ? summary : TOO_LONG;
  return issueUrl(body, environment);
}
/** reportUrl() as GitHub's login gets it (tests/bench.mjs) */
export const loginUrl = (markdown, environment, summary) => throughLogin(reportUrl(markdown, environment, summary));

/** T185: the report the link holds where the whole is too long: the top of the report as it is (the machine's line and
 * the rounds' table, which parseReport() reads, and T184's table of the model page's path), a line for each section,
 * and the request to paste the whole below. */
export function shortReport(head, lines) {
  return [head, "#### Summary", lines.map((line) => `- ${line}`).join("\n"), PASTE].filter(Boolean).join("\n\n");
}

// a | that tableCell() escaped stays in its cell (T214: a skipped round's reason is the words of a cell)
const cells = (line) => line.split(/(?<!\\)\|/).slice(1, -1).map((cell) => cell.trim().replace(/\\\|/g, "|"));

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
    // T214: a skipped round has no speed, and its last column is why
    rows.push(speed === "skipped" ? { name, speed: NaN, ready, backend: "", skip: backend } : { name, speed: Number(speed), ready, backend });
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
             `${report.cores || "?"} | ${number(all?.speed)} | ${plain?.skip !== undefined ? "skipped" : number(plain?.speed)} | ${all?.backend ?? ""} | [#${id}](${url}) |`);
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

/** A CPU ceiling's value, where it was measured and steady (T163) */
const steadyCeiling = (ceiling, key) => (ceiling && !ceiling.error && !ceiling.none && !ceiling.unsteady && ceiling[key] > 0 ? ceiling[key] : undefined);

// what a CPU row reads a token, GB/s (before T163's review a result had no tokenMegabytes: the checkpoint's bytes),
// and a prompt's G MAC/s (a multiply-add for each weight of the layers and token)
const tokenReads = (r, row) => row.GBps * (r.tokenMegabytes ?? r.megabytes) / r.megabytes;
const promptGMACsOf = (r, row) => (r.layerWeights && row.promptMsPerToken > 0 ? r.layerWeights / (row.promptMsPerToken / 1000) / 1e9 : undefined);

/**
 * The CPU section's Markdown (the lines), from its result r ({backend, shared, megabytes, tokenMegabytes, layerWeights,
 * rows, ceilings}): the forward pass at each count of software threads, and T163's ceilings beside it. A token's share
 * of reading alone with the same count counts all it reads (tokenMegabytes: with relaxed SIMD matmul_q8r's corrections
 * too), while its GB/s column stays the checkpoint's (T157 turns that into other models' speeds). A prompt's G MAC/s
 * with one thread is held against relaxed_dot with its two loads, on one thread; with more threads it is not (threads
 * on cores of different speeds do not add up to a count times one). That loop is the one-token kernel's form (a weight
 * and an activation loaded for each dot); a prompt's tile (T159, four rows by four tokens) loads half a vector of data
 * a dot, so the share can pass 100%, and the text says where its bound lies (T198). No share against a ceiling that
 * failed or was unsteady.
 */
export function cpuTable(r) {
  const c = r.ceilings ?? {};
  const share = (value, whole) => (whole && Number.isFinite(value) ? ` (${number((100 * value) / whole, 0)}%)` : "");
  const readAt = (threads) => steadyCeiling((c.read ?? []).find((one) => one.threads === threads), "GBps");
  const dot = steadyCeiling(c.dot, "GMACs");
  const read = (row) => tokenReads(r, row);
  const promptGMACs = (row) => promptGMACsOf(r, row);
  const ceiling = (one, key, unit) => {
    if (!one) return "not measured";
    if (one.none) return tableCell(one.none === "no relaxed SIMD in this browser" ? "not in this browser" : one.none);
    if (one.error) return tableCell(`failed: ${one.error}`);
    return `${one.unsteady ? "unsteady: " : ""}${number(one[key])} ${unit}`;
  };
  const lines = [`${r.backend}, ${number(r.megabytes, 0)} MB a token${r.shared ? "" : " (not cross-origin isolated: one thread only)"}. ` +
    "In parentheses, the share of this device's ceiling below: a token's reads (with the kernel's corrections) of reading alone with as many threads, " +
    "a prompt's G MAC/s (a multiply-add for each weight of the layers and token) of relaxed_dot with its two loads, one thread. " +
    "A token reads each weight once, so its bound is reading. A prompt's 16 tokens go through tiles of four rows by four tokens (T159): " +
    "for each group of 32 weights a tile loads the four rows' weights once for the four tokens and the four tokens' activations once for the four rows, " +
    "32 relaxed_dots on 16 loads of data (and 12 of scales and corrections), where the loop below loads two for each dot. " +
    "So a prompt can pass 100% of that loop. Its bound lies between that loop and relaxed_dot on registers alone, " +
    "and below the second by the tile's own work: each row, token and group's sum is turned, scaled and added (on arm64, 32 of the 96 vector instructions a tile spends on a group are dots).", "",
    "| software threads | ms a token | GB/s | tok/s | ms a token of a prompt, 16 at once | G MAC/s of the prompt |", "|---:|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.none ? `| ${row.asked} | ${tableCell(row.none)} | | | | |`
      : `| ${row.threads} | ${number(row.msPerToken)} | ${number(row.GBps)}${share(read(row), readAt(row.threads))} | ${number(1000 / row.msPerToken)} | ` +
        `${number(row.promptMsPerToken, 2)} | ${number(promptGMACs(row))}${row.threads === 1 ? share(promptGMACs(row), dot) : ""} |`))];
  lines.push("", "**Ceilings** (T163): loops of one kind of instruction, each timed as 2n passes less n. " +
    "Reading alone reads the model's weights above, a megabyte at a time taken in turn by the threads. " +
    "relaxed_dot with its two loads reads 8 KB that stay in the first cache, a weight and an activation for each dot, as the one-token kernel (matmul_q8r) does. " +
    "On registers alone it is the instruction's own rate, which no kernel that loads its weights and tokens reaches; a prompt's tiles, with half a load of data for each dot, lie between the two.");
  if (c.error) return [...lines, "", tableCell(`Not measured: ${c.error}`)];
  lines.push("", "| loop | software threads | ceiling |", "|---|---:|---:|",
    ...(c.read ?? []).map((one) => `| reading alone | ${one.threads} | ${ceiling(one, "GBps", "GB/s")} |`),
    `| relaxed_dot with its two loads (int8) | 1 | ${ceiling(c.dot, "GMACs", "G MAC/s")} |`,
    `| relaxed_dot, registers only | 1 | ${ceiling(c.dotRegisters, "GMACs", "G MAC/s")} |`,
    `| f32 multiply + add, registers only | 1 | ${ceiling(c.fma, "GMACs", "G MAC/s")} |`);
  return lines;
}

/**
 * The CPU the GPU is held against: of the CPU section's rows, the fastest count of software threads for a token and,
 * each on its own, for a prompt's tokens 16 at once. section: the page's result of the CPU section ({status, data}),
 * or undefined where it has not run. Where it gives nothing to hold against, { why } says so, and no estimate stands
 * in for it.
 */
/** T156: where /benchmark/ keeps its CPU section's fastest reading of the weights (cpuBaseline().token: { GBps,
 * threads }) for the model page, which holds a model on the GPU alone against it (the same estimate as the GPU
 * section's GPU ÷ CPU, T157) */
export const CPU_SPEED_KEY = "benchmark:cpu";
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

/** T186: the numbers under a layer's verdict in the check (T175), in one short line for the page and the report: its
 * quantized vectors held to quantize_x (quantized: [{point, wrong, scale, apart, of}]; the worst scale's relative
 * difference and the values off by 1 of all four, or the first that was wrong) and, for the DP4A fused form, the one
 * with the norms apart (sameAsNormsApart: {ulps, apart, stream, bitForBit}). "" where the verdict has neither. */
export function layerCheckNumbers({ quantized, sameAsNormsApart: apart } = {}) {
  const parts = [];
  if (quantized?.length) {
    const wrong = quantized.find((q) => q.wrong);
    const sum = (key) => quantized.reduce((total, q) => total + q[key], 0);
    parts.push(wrong ? `quantized: ${wrong.point} ${wrong.wrong}`
      : `quantized: scales ${Math.max(...quantized.map((q) => q.scale)).toExponential(1)}, ${sum("apart")} of ${sum("of")} off by 1`);
  }
  if (apart) {
    parts.push(`norms apart: ${apart.bitForBit ? "bit for bit" : `${apart.ulps} ulp, ${apart.apart} off by 1, stream ${apart.stream.toExponential(1)}`}`);
  }
  return parts.join("; ");
}

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
/** Of a shape's matrix × vector rows, the fastest the check did not find WRONG (not one measured again, nor unsteady) */
function fastestMatVec(s, check) {
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  return (s.result?.rows ?? []).filter((row) => row.GBps && !row.again && !row.unsteady && !wrong(row)).sort((a, b) => b.GBps - a.GBps)[0];
}
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
    const best = fastestMatVec(s, check);
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
 * T150: the GPU section's table of one layer of a token: its fourteen steps each a dispatch of their own and the same
 * fused into five (public/shaders.js's fusedMatVec), on llama.cpp's matrix × vector; T175: the same on ONNX Runtime's
 * DP4A for small M, its vector quantized before each matrix: separate steps (eighteen), fused but for the norms
 * (eleven) and fused (nine: fusedDp4aMatVec, the norm with the quantizer). step: {name, result: {model, pos, layers,
 * GB, rows: [{form, check, base, fused, normApart, subgroups, dispatches, msPerLayer, GBps, unsteady}, or {form, check,
 * error}, or {form, none}]}} or {name, error}; check: the shaders against JavaScript (a form's verdict under
 * row.check); ceilings: T168's (the buffer's reads, for the share of it the layer's weights are read at); gpu: {
 * fallback, lost }. "Faster than the separate steps" beside a fused row, against the separate steps of the same matrix
 * × vector (base) and reduction (measured in turn with it): not where noRatios() says none, nor for a row the check
 * found WRONG or that was unsteady. The share of the reads: not after a lost device, on a fallback adapter, or where
 * that read was unsteady (as matVecTable). none: a form this device cannot run (no packed int8 dot), and why.
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
    const separate = r.rows.find((one) => !one.fused && one.subgroups === row.subgroups && one.base === row.base);
    return usable(separate) ? times(separate.msPerLayer / row.msPerLayer) : "";
  };
  // two significant digits under 1 GB/s (a fallback adapter reads a layer at a few hundredths)
  const GBps = (row) => (Number.isFinite(row.GBps) ? `${row.GBps < 1 ? row.GBps.toPrecision(2) : number(row.GBps)}${reads ? ` (${number((100 * row.GBps) / reads)}%)` : ""}` : "");
  return [`**A layer of a token** (${r.model}'s width, at position ${r.pos}, ${number(r.GB * 1000, 1)} MB of weights): on llama.cpp's matrix × vector, its fourteen steps each a dispatch of its own ` +
    "(the norm, q, k, v, RoPE and the cache, the attention, o, the residual's add, the norm, gate, up, SwiGLU, down, the add) and the same fused into five " +
    "(q, k and v with the norm, RoPE and the cache; the attention; o with the add; gate and up with the norm and SwiGLU; down with the add); " +
    "on ONNX Runtime's DP4A, the vector quantized to 8 bits before each matrix, as separate steps (eighteen), fused but for the two norms (eleven), " +
    "and fused (nine: the norm with its quantizing, q, k and v with RoPE and the cache, the attention, its quantizing, o with the add, the norm with its quantizing, " +
    "gate and up with SwiGLU, its quantizing, down with the add). The forms are timed in turn, each as a submission of 2n layers less one of n, the weights read from copies of them in turn, as the matrix × vector is. " +
    `GB/s: the layer's weights over its time${reads ? `; in parentheses, the share of what a loop that only reads a buffer reads, ${number(reads)} GB/s below` : ""}.`,
    ...(none ? [`Faster than the separate steps: ${none}.`] : []), ...(gpu.lost ? ["No share of the buffer's reads: the device was lost."] : []), "",
    `| a layer | dispatches | GPU ms | GB/s | its ${r.layers} layers, ms | faster than the separate steps |`, "|---|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.none ? `| ${tableCell(row.form)} | ${tableCell(`not here: ${row.none}`)} | | | | |`
      : row.error ? `| ${tableCell(row.form)} | ${tableCell(`failed: ${row.error}`)} | | | | |`
      : `| ${tableCell(row.form)}${wrong(row) ? " (WRONG in the check)" : ""} | ${row.dispatches} | ${row.unsteady ? "unsteady: " : ""}${number(row.msPerLayer, 2)} | ` +
        `${GBps(row)} | ${number(row.msPerLayer * r.layers)} | ${faster(row)} |`))];
}

// T202: a time of layerSteps() in ms (undefined where it failed or is not a positive number)
const stepMs = (one) => (one && !one.error && Number.isFinite(one.ms) && one.ms > 0 ? one.ms : undefined);
const less = (a, b) => (a === undefined || b === undefined ? undefined : a - b);
// T202: a form's layer (r: layerSteps()'s result) and its parts in ms: the steps one by one (steps) as the matrices
// with what they write (matrices), the attention and the norms and quantizing (small); the four matrices alone
// (alone); the chain (the layer less its steps). undefined where a part was not timed.
function layerSplit(r, form) {
  const byName = new Map(r.steps.map((one) => [one.step, one]));
  const sum = (kinds) => form.steps.filter(({ step }) => kinds.includes(byName.get(step)?.kind))
    .reduce((total, { step, count }) => (total === undefined || stepMs(byName.get(step)) === undefined ? undefined : total + count * byName.get(step).ms), 0);
  const aloneRows = r.steps.filter((one) => one.kind === "alone");
  const alone = aloneRows.length && aloneRows.every((one) => stepMs(one) !== undefined) ? aloneRows.reduce((total, one) => total + one.ms, 0) : undefined;
  const layer = stepMs(form), steps = sum(["matrix", "attention", "small"]);
  return { layer, steps, matrices: sum(["matrix"]), attention: sum(["attention"]), small: sum(["small"]), alone, chain: less(layer, steps) };
}

/**
 * T202: the GPU section's table of where a layer's time goes (public/benchmark/gpu.js's layerSteps()): each step of
 * the fused forms a token would run on this device alone, the four matrices alone, a dispatch of one workgroup, and
 * the whole layer, timed in turn. step: {name, result: {model, pos, layers, GB, dp4a, forms: [{form, check, normApart,
 * dispatches, steps: [{step, count}], ms, unsteady} or {form, check, error}], steps: [{step, kind ("matrix",
 * "attention", "small", "alone", "floor"), matrix, ms, unsteady} or {step, kind, error}]}} or {name, error}; check:
 * the shaders against JavaScript (a form's verdict under its check); ceilings: T168's (the share of the buffer's reads
 * the matrices alone read the weights at); gpu: {fallback, lost}. Under the table, for each form: the layer, its
 * steps' sum and what the chain costs beyond it, the matrices alone and the layer less them split into what fusing adds
 * to the matrices (their writes; the norm on the read and gate with up in one workgroup where so), the attention, the
 * norms and quantizing, and the chain. cycleMB: the fewest MB of weights read before the same ones again. T208: chosen,
 * how the forms were chosen ({by: "layer", fastest}: the layer table's fastest fused form the check found right, and
 * beside it the same with the norms folded or apart; {by: "packed", why}: as T202, by the packed int8 dot); caches,
 * the attention's ({count, MB}: the step alone reads them in turn); timestamps, the whole layer by the GPU's own clock
 * ({layers, rounds, forms: [{form, ms}]}, {none} or {error}), one line under the forms'.
 */
export function layerStepsTable(step, check, ceilings, gpu = {}) {
  if (!step) return [];
  const title = "**Where a layer's time goes**";
  if (step.error || !step.result) return [`${title}: ${tableCell(step.error ?? "not measured")}`];
  const r = step.result, none = noRatios(gpu);
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (form) => Boolean(check && check[form.check] && !check[form.check].ok);
  const microseconds = (one) => (one?.error ? `failed: ${tableCell(one.error)}` : `${one?.unsteady ? "unsteady: " : ""}${stepMs(one) === undefined ? "?" : number(1000 * one.ms, 1)}`);
  const floor = r.steps.find((one) => one.kind === "floor"), forms = r.forms.filter((form) => !form.error);
  const GBps = (took) => (took === undefined ? "?" : `${number(r.GB / (took / 1000))} GB/s${reads ? `, ${number((100 * r.GB) / (took / 1000) / reads)}% of the buffer's reads (${number(reads)} GB/s)` : ""}`);
  const named = forms.map((form) => `"${tableCell(form.form)}"`).join(" and ");
  const which = r.chosen?.by === "layer"
    ? `the layer a token would run here: the fastest fused layer of the table above that the check found right, "${tableCell(r.chosen.fastest)}", as the engine chooses it (T152)` +
      `${forms.length > 1 ? `, and beside it the same with its norms ${forms.find((form) => form.form !== r.chosen.fastest)?.normApart ? "apart" : "folded in"}` : ""}`
    : `the layers a token would run here by the packed int8 dot alone (${r.chosen?.why ? `${tableCell(r.chosen.why)}: ` : ""}` +
      `${r.dp4a ? "ONNX Runtime's DP4A, the packed int8 dot is here" : "llama.cpp's matrix × vector, no packed int8 dot here"})`;
  const lines = [`${title} (T202, T208; ${r.model}'s width at position ${r.pos}, ${number(r.GB * 1000, 1)} MB of weights a layer, ${which}${named ? `: ${named}` : ""}). ` +
    "Each step of a layer alone, as a submission of 2n of its one dispatch less one of n; the four matrices alone (the plain matrix × vector over all of a matrix's rows, " +
    "nothing folded into what it writes: what the matrices themselves take); a dispatch of one workgroup that does next to nothing (what a step costs for being a dispatch in a chain); " +
    "and the whole layer: all timed in turn, the residual stream written back before every submission. " +
    `The layer reads its copies of the weights in turn, a matrix alone or a step with a matrix the ranges of its size of all of them in turn${r.spares ? ` and of ${r.spares} spare matrices of random weights` : ""}` +
    `${Number.isFinite(r.cycleMB) ? ` (none read again before ${number(r.cycleMB, 0)} MB of other weights` : " (none read again soon"}: not from the GPU's caches). ` +
    "Each step alone, not the layer less that step: a small step is 1 or 2% of a layer, about what a layer's time moves from one pair of submissions to the next, " +
    "so the difference of two layers could not tell it; alone it is repeated until a submission takes long enough. What the steps alone leave out is the layer less their sum: " +
    "what a chain of different dispatches, each waiting for the one before, costs beyond each on its own. " +
    "The fused matrices less the matrices alone is what fusing adds to them: the writes folded in (RoPE and the cache, the residual's add, SwiGLU), in llama.cpp's fused form the norm folded into the read, " +
    "and gate with up in one workgroup, which can take less than the two alone (so this can come out less than nothing). " +
    (r.caches?.count > 1
      ? `The attention alone reads ${r.caches.count} copies of the cache (the keys and values of the positions up to this one) in turn, ${number(r.caches.MB, 0)} MB, so not from the GPU's caches, as a token's layers each read their own; ` +
        "in the whole layer it reads one cache, with a copy of the layer's weights read between two reads of it, as in the layer table."
      : "The attention reads one cache (the positions up to this one) every time, in the GPU's caches as in the layer table."),
  ...(none ? [`Read nothing from these times: ${none.replace(/^none:? /, "")}.`] : []), "",
  `| step | µs each | ${forms.map((form) => `in "${tableCell(form.form)}"${wrong(form) ? " (WRONG in the check)" : ""}, µs a layer`).join(" | ")} |`,
  `|---|---:|${forms.map(() => "---:|").join("")}`,
  ...r.steps.map((one) => `| ${tableCell(one.step)} | ${microseconds(one)} | ${forms.map((form) => {
    const count = form.steps.find((s) => s.step === one.step)?.count;
    return count ? `${count} × = ${stepMs(one) === undefined ? "?" : number(1000 * count * one.ms, 1)}` : "";
  }).join(" | ")} |`), ""];
  for (const form of r.forms) {
    if (form.error) {
      lines.push(`- "${tableCell(form.form)}": failed: ${tableCell(form.error)}`);
      continue;
    }
    const { layer, steps, matrices, attention, small, alone: aloneMs, chain } = layerSplit(r, form);
    const signed = (value) => (value === undefined ? "?" : `${value < 0 ? "−" : ""}${number(Math.abs(value), 2)}`);
    lines.push(`- "${tableCell(form.form)}"${wrong(form) ? " (WRONG in the check)" : ""}, ${form.dispatches} dispatches: the layer ${form.unsteady ? "(unsteady) " : ""}${signed(layer)} ms. ` +
      `Its steps one by one ${signed(steps)} ms (the matrices with what they write ${signed(matrices)}, the attention ${signed(attention)}, the norms and quantizing ${signed(small)}); ` +
      `the layer less them, what the chain costs beyond each step alone: ${signed(chain)} ms. ` +
      `The matrices alone ${signed(aloneMs)} ms (${GBps(aloneMs)}); the layer less them, ${signed(less(layer, aloneMs))} ms, is ` +
      `what fusing adds to the matrices ${signed(less(matrices, aloneMs))} + the attention ${signed(attention)} + the norms and quantizing ${signed(small)} + the chain ${signed(chain)}. ` +
      `A dispatch of one workgroup takes ${microseconds(floor)} µs: ${form.dispatches} of them ${stepMs(floor) === undefined ? "?" : number(form.dispatches * floor.ms, 2)} ms.`);
  }
  // T202's review: how far this run's times move. Two forms of the same matrices and attention (DP4A's) differ only in
  // their norms and quantizing, so their layers less those should come out the same; on lavapipe in CI (steady by the
  // pairs) they came out 0.4 and 4.2 ms apart in 48, where the chain was −2.4 to 2.7
  const bigSteps = (form) => form.steps.filter(({ step }) => ["matrix", "attention"].includes(r.steps.find((one) => one.step === step)?.kind))
    .map(({ step, count }) => `${step} ${count}`).sort().join("\n");
  if (forms.length === 2 && bigSteps(forms[0]) === bigSteps(forms[1])) {
    const [a, b] = forms.map((form) => { const s = layerSplit(r, form); return less(s.layer, s.small); });
    if (a !== undefined && b !== undefined) {
      lines.push("", `The two forms run the same matrices and attention and differ only in their norms and quantizing, so their layers less those should be the same: ` +
        `${number(a, 2)} and ${number(b, 2)} ms, ${number(Math.abs(a - b), 2)} apart. That is about how far this run's times move: read the chain, and any part, only where it is larger.`);
    }
  }
  if (r.steps.some((one) => one.unsteady) || r.forms.some((form) => form.unsteady)) lines.push("", "Unsteady: the pairs of a submission of n and of 2n were not about twice each other, so those times are rough.");
  // T208: the whole layer once more by the GPU's own clock
  const t = r.timestamps;
  if (t) {
    const head = "The whole layer by the GPU's own clock (timestamp-query), a check of the times above";
    if (t.none || t.error) lines.push("", `${head}: ${t.none ? "not here" : "failed"}: ${tableCell(t.none ?? t.error)}.`);
    else {
      const ms = (value) => (Number.isFinite(value) && value > 0 ? number(value, 2) : "?");
      const got = t.forms.map(({ form, ms: mean, span }) => {
        const above = stepMs(r.forms.find((one) => one.form === form));
        return `"${tableCell(form)}" ${ms(mean)} ms (first to last ${ms(span)}; above: ${above === undefined ? "?" : number(above, 2)})`;
      });
      lines.push("", `${head}: ${got.join(", ")} a layer, each the median of ${t.rounds} submissions of ${t.layers} layers, each layer a compute pass that writes a timestamp as it begins and as it ends: ` +
        `the mean of the passes, and in parentheses from the first pass's beginning to the last one's end over the ${t.layers} layers. ` +
        "The times above are a submission of 2n layers less one of n, by the page's clock; these hold only what runs on the GPU. " +
        "Chrome cuts every timestamp down to a multiple of 65.5 µs (its own words say 100 µs; unless its developer features are on), so a layer's time is off by up to that, " +
        `the mean of ${t.layers} by 6 µs or less where the passes start anywhere on that clock's steps, and first to last by ${number(65.5 / t.layers, 0)} µs or less whatever they do. ` +
        "First to last also holds what comes between two passes; a mean above it says the passes overlapped (one stamped as begun before the one before it ended) and is not the layers' work. " +
        "Where these agree with the times above to about 0.1 ms, the times above are the layers' work; where these are well below, the times above hold something besides it (what comes between one layer and the next, or of the submissions).");
    }
  }
  return lines;
}

/**
 * T151: the GPU section's table of tokens generated on the GPU (public/shaders.js's EMBED, fusedMatVec and SAMPLE: the
 * sampling on the GPU too, the state carried from token to token there). step: {name, result: {model, layers, vocab,
 * GB, dispatches, chunkDispatches, tokens, settings, work: {ms, unsteady}, chunkWork, rows: [{perSubmission, msPerToken,
 * fixedMs, chunks}], sampling: {vocab, msEach, over, unsteady, chunks: {msEach, unsteady, pick} (pick: its last stage
 * alone, the same form), flat: {msEach, over, unsteady, chunks}}}} or {name, error} (a time {error} where it failed);
 * check: the shaders against JavaScript ("sampling", "sampling in chunks"
 * and "tokens on the GPU"); gpu: { fallback, lost }. T191: each row also the ms a token with the sampling in chunks
 * (shaders.js's SAMPLER_STAGES, many workgroups) where the rest is SAMPLE's one workgroup. A row a way of submitting: one token a submission, each read back as it comes, and
 * several in one submission, read back once. "A submission besides its tokens" (F) is what a submission costs past its
 * tokens' work (the wait and the reading back: a row's ms a token times its tokens, less a token's work, the
 * difference of 2n tokens in one submission and n), "of it a token" F over the row's tokens (what is left of F a
 * token), and "faster than one a submission" the first row's ms a token
 * over the row's: neither where noRatios() says none, where the check found the sampling or the run WRONG, nor
 * (the first) where a token's work was unsteady. A submission's cost below 0 (a token's work read higher than the
 * row's time allows) is "under the noise".
 */
export function generateTable(step, check, gpu = {}) {
  if (!step) return [];
  if (step.error || !step.result) return [`**Tokens generated on the GPU**: ${tableCell(step.error ?? "not measured")}`];
  const r = step.result, none = noRatios(gpu);
  const wrong = ["sampling", "tokens on the GPU"].some((key) => check?.[key] && !check[key].ok);
  const chunksWrong = check?.["sampling in chunks"] && !check["sampling in chunks"].ok;
  const derived = !none && !wrong;
  const one = r.rows.find((row) => row.perSubmission === 1);
  // a submission's cost besides its tokens, or its share a token (what N tokens a submission leave of it: F / N)
  const fixed = (row, ms) => (derived && Number.isFinite(ms) ? (ms < 0 ? "under the noise" : number(ms, 2)) : "");
  const s = r.settings ?? {};
  return [`**Tokens generated on the GPU** (${tableCell(r.model)}: ${r.layers} layers, a vocabulary of ${r.vocab}, ${number(r.GB, 2)} GB of weights a token; ` +
    `a token is ${r.dispatches} dispatches: its row of the embedding, the layers as ${r.layer ? `the layer table's "${tableCell(r.layer)}"` : "T150's fused"}, the classifier, and the sampling on the GPU in one workgroup (T151), ` +
    `penalty ${s.penalty}, temperature ${s.temperature}, top-p ${s.topp}). The same ${r.tokens} tokens each way, the ways in turn; ` +
    "a token's work alone is a submission of 2n tokens less one of n. A submission besides its tokens: what it costs past their work (submitting, waiting, reading the ids back)." +
    " With the sampling in chunks: the same tokens sampled by a workgroup a chunk of the vocabulary" +
    `${r.chunkDispatches ? ` (${r.chunkDispatches} dispatches a token)` : ""}, the ways and the samplers in turn (T191).` +
    `${wrong ? " The check found the sampling on the GPU WRONG." : ""}${chunksWrong ? " The check found the sampling in chunks WRONG." : ""}`,
    ...(none ? [`Besides its tokens and faster: ${none}.`] : []), "",
    "| tokens a submission | GPU ms a token | with the sampling in chunks, ms | a submission besides its tokens, ms | of it a token, ms | faster than one a submission |",
    "|---|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => `| ${row.perSubmission === 1 ? "1, each read back as it comes" : `${row.perSubmission}, read back once`}${wrong ? " (WRONG in the check)" : ""} | ` +
      `${number(row.msPerToken, 2)} | ${Number.isFinite(row.chunks) ? `${number(row.chunks, 2)}${chunksWrong ? " (WRONG in the check)" : ""}` : ""} | ` +
      `${fixed(row, row.fixedMs)} | ${fixed(row, row.fixedMs / row.perSubmission)} | ` +
      `${derived && row !== one && one ? times(one.msPerToken / row.msPerToken) : ""} |`),
    "", `A token's work: ${r.work ? `${r.work.unsteady ? "unsteady: " : ""}${number(r.work.ms, 2)} ms` : "not measured here"}` +
    `${r.chunkWork ? ` (with the sampling in chunks ${r.chunkWork.unsteady ? "unsteady: " : ""}${number(r.chunkWork.ms, 2)} ms)` : ""}; ` +
    `the sampling alone (made-up logits of ${r.sampling?.vocab ?? "Llama 3's"} tokens, the same settings but no penalty: ` +
    `it would change them in place, sampling after sampling): ${r.sampling ? samplingCell(r.sampling) : "not measured here"}` +
    `${r.sampling?.flat ? `; on flat logits, ${samplingCell(r.sampling.flat)}` : ""}.`];
}
// the ms of the sampling alone, with how many tokens were over the nucleus's floor (what SAMPLE gathers and reads each
// round of its searches: T151), in one workgroup and (T191) in chunks, and of those the last stage alone (T191's
// review: one workgroup's nucleus and draw, SAMPLE's search; the rest of SAMPLE is its reading of the vocabulary)
const samplingTime = (s) => (s.error ? `failed: ${tableCell(s.error)}` : `${s.unsteady ? "unsteady: " : ""}${number(s.msEach, 3)} ms`);
const samplingCell = (s) => `${s.chunks ? "one workgroup " : ""}${samplingTime(s)}` +
  `${Number.isFinite(s.over) ? ` (${s.over} tokens over the floor)` : ""}` +
  `${s.chunks ? `, in chunks ${samplingTime(s.chunks)}` : ""}` +
  `${s.chunks?.pick ? ` (of it the last stage, one workgroup's nucleus and draw: ${samplingTime(s.chunks.pick)})` : ""}`;

/** T93: where the model page remembers the number of threads it found for a model on this device (localStorage).
 * T190: /benchmark/ reads the same, so that the page path is timed on the model page's count. nav: the navigator */
export const threadsKey = (id, nav) => `threads:${id}:${nav.hardwareConcurrency}:${nav.deviceMemory ?? ""}:${nav.userAgent}`;

/** T190: how the page path's number of threads came about (worker.js's timedPaths), in a few words */
export function threadsHow(how) {
  if (!how) return "";
  if (how.alone) return `: ${how.alone}`;
  // T190's review: a software thread that stopped after the count was found (T120): the later times are one thread's
  const stopped = how.stopped ? "; a software thread stopped while timed, and one thread went on" : "";
  if (how.unfinished) return `: the search had not ended after ${how.unfinished} s${stopped}`;
  if (how.remembered) return `, as the model page remembers${stopped}`;
  const verdicts = (how.searched ?? []).map(([best, candidate, kept]) => `${best} or ${candidate}: ${kept}`);
  return `${verdicts.length ? `, searched here (${verdicts.join(", ")})` : ""}${stopped}`;
}

/** T190's review: the writing on each number of software threads (worker.js's timedPaths: in turn, CPU only), the page's
 * marked; "" where there is one count or none. perCount: [{ threads, speed, low, high, unsteady }] */
export function threadsLine(perCount = [], page) {
  if (perCount.length < 2) return "";
  const rate = (value) => number(value, value >= 100 ? 0 : 1);
  const cells = perCount.map((c) => `${c.threads}${c.threads === page ? " (the page's)" : ""}: ${rate(c.speed)} tok/s` +
    ` (${rate(c.low)}–${rate(c.high)}${c.unsteady ? ", unsteady" : ""})`);
  return `Writing on each number of software threads (CPU only): ${cells.join(" · ")}`;
}

/** T184: the prompts the model page's path is timed on: one block of the GPU's (forward.js's GPU_BLOCK) and four; and
 * the tokens it writes after a prompt */
export const PATH_PROMPTS = [64, 256];
export const PATH_WRITES = 64;

/** T184: why a GPU cell of the page path is empty, in a few words: "not here" (no WebGPU, no adapter), "not on a
 * fallback adapter" (the CPU in a GPU's place: the page refuses it, T148, and CI's are all such, T182), else the words of
 * forward.js (a model the GPU does not take yet, too little memory, a failure). */
export function gpuSkipped(why = "") {
  if (/no WebGPU|no GPU adapter/.test(why)) return "not here";
  if (/fallback adapter/.test(why)) return "not on a fallback adapter";
  return why || "not measured";
}

/** T184: the model page's own path on this device, as one table (the model section times it on its first load,
 * worker.js's timedPaths and forward.js's timePrompts): prompts as the page chooses between the GPU and the CPU (T148),
 * on the CPU only and on the GPU only, how many times faster the GPU is, and the writing after a prompt. paths:
 * { threads, how (threadsHow()), perCount (threadsLine()), gpu: { seconds, matrices, attention, lost? } or { why },
 * status (the status line's words of the GPU),
 * rows: [{ what: "prompt" | "generation", tokens, chosen, cpu, gpu }] } where a cell is { speed, low, high, gpuTokens,
 * unsteady }, { same: "cpu" } (one path: timed once) or { skip: why }; or { error }. The writing's GPU cells (T152: the
 * steps of a generation on the GPU) are timed where the GPU takes them, else the reason. A GPU that stopped while the sides were timed
 * (gpu.lost) leaves no ratio anywhere: its later times are no GPU's (T157). */
export function pathTable(paths, name = "") {
  const title = `**The model page's path**${name ? ` (${name})` : ""}`;
  if (!paths || paths.error) return `${title}: failed: ${tableCell(paths?.error ?? "no answer")}`;
  const { gpu = {}, rows = [] } = paths;
  const facts = [paths.threads !== undefined && `${paths.threads} software thread${paths.threads === 1 ? "" : "s"}${threadsHow(paths.how)}`];
  if (gpu.why !== undefined) facts.push(`WebGPU: ${tableCell(gpuSkipped(gpu.why))}`);
  else if (gpu.lost) facts.push(`WebGPU stopped while timed: ${tableCell(gpu.lost)}`);
  else {
    facts.push(`WebGPU ready in ${number(gpu.seconds)} s`, `matrices by ${tableCell(gpu.matrices ?? "?")}`, `attention by ${tableCell(gpu.attention ?? "?")}`);
    if (paths.status) facts.push(tableCell(paths.status));
  }
  // a cell the GPU did not run: the few words of gpuSkipped(), or "not used" where the line above has the reason
  const skipped = (why) => (why === gpu.why && gpuSkipped(why) === why ? "not used" : tableCell(gpuSkipped(why)));
  const rate = (value) => number(value, value >= 100 ? 0 : 1);
  const speed = (cell) => {
    if (!cell) return "?";
    if (cell.skip !== undefined) return skipped(cell.skip);
    if (cell.same) return "same as CPU only";
    const spread = cell.low !== undefined ? ` (${rate(cell.low)}–${rate(cell.high)}${cell.unsteady ? ", unsteady" : ""})` : "";
    return `${rate(cell.speed)} tok/s${spread}`;
  };
  // where the page's choice put the tokens: all on one side, or a part on the GPU (a GPU, then the CPU: T148)
  const side = (cell, tokens) => (cell?.gpuTokens === undefined || cell.same ? "" : cell.gpuTokens >= tokens ? ", GPU"
    : cell.gpuTokens > 0 ? `, GPU ${cell.gpuTokens} of ${tokens}` : ", CPU");
  const lines = [`${title}: ${facts.filter(Boolean).join(" · ")}`, "",
    "| the page | as chosen | CPU only | GPU only | GPU ÷ CPU |", "|---|---|---|---|---|",
    ...rows.map((row) => {
      const what = row.what === "prompt" ? `a prompt of ${row.tokens} tokens` : `writing ${row.tokens} tokens`;
      const ratio = !gpu.lost && row.gpu?.speed && row.cpu?.speed ? times(row.gpu.speed / row.cpu.speed) : "";
      return `| ${what} | ${speed(row.chosen)}${side(row.chosen, row.tokens)} | ${speed(row.cpu)} | ${speed(row.gpu)} | ${ratio} |`;
    })];
  const counts = threadsLine(paths.perCount, paths.threads);
  if (counts) lines.push("", counts);
  return lines.join("\n");
}

// ---- T185: a line of the summary for each section (shortReport()), from the data the section's tables are made of.
// Each says the fastest and the ceilings; the tables with every row are in the whole report.

const yesNo = (value) => (value ? "yes" : "no");

/** The device section's line: what the head's line does not say (the cores and the memory are there) */
export function deviceSummary(r) {
  return [`Browser: SIMD ${yesNo(r.simd)}, relaxed SIMD ${yesNo(r.relaxedSimd)}, 64-bit memory ${yesNo(r.memory64)}, ` +
    `cross-origin isolated ${yesNo(r.crossOriginIsolated && r.sharedMemory)}, WebGPU in a worker ${yesNo(r.webgpu)}, ` +
    `private file system ${yesNo(r.opfs && r.syncHandle)}`];
}

/** The CPU section's line: its fastest token and prompt (cpuTable()'s rows), and the ceilings */
export function cpuSummary(r) {
  const c = r.ceilings ?? {};
  const rows = (r.rows ?? []).filter((row) => row.msPerToken > 0);
  const token = [...rows].sort((a, b) => a.msPerToken - b.msPerToken)[0];
  const prompt = rows.filter((row) => row.promptMsPerToken > 0).sort((a, b) => a.promptMsPerToken - b.promptMsPerToken)[0];
  if (!token) return ["CPU: no token measured"];
  const isolated = r.shared !== false;
  const readAlone = steadyCeiling((c.read ?? []).find((one) => one.threads === token.threads), "GBps");
  const parts = [`${number(token.msPerToken)} ms a token with ${threadsOf({ threads: token.threads, isolated })}, ${number(token.GBps)} GB/s` +
    (readAlone ? ` (${number((100 * tokenReads(r, token)) / readAlone, 0)}% of reading alone)` : "")];
  if (prompt) parts.push(`a prompt ${number(promptGMACsOf(r, prompt))} G MAC/s with ${prompt.threads}`);
  const reads = (c.read ?? []).map((one) => steadyCeiling(one, "GBps")).filter(Boolean);
  const ceilings = [reads.length && `reading alone ${number(Math.max(...reads))} GB/s`,
    steadyCeiling(c.dot, "GMACs") && `relaxed_dot ${number(c.dot.GMACs)} G MAC/s`,
    steadyCeiling(c.fma, "GMACs") && `f32 ${number(c.fma.GMACs)} G MAC/s`].filter(Boolean);
  if (ceilings.length) parts.push(`ceilings: ${ceilings.join(", ")}`);
  return [`CPU: ${parts.join("; ")}`];
}

/**
 * The GPU section's lines, from its steps (as the page's gpuMarkdown() takes them): the adapter and the check; a token
 * of each model with the CPU beside it (tokenTable()'s estimate) and the fastest matrix × vector of the widest shape;
 * the fastest layer and the gain of several tokens a submission; the fastest prompt at the most tokens. No ratio where
 * noRatios() says none or the check found the row WRONG; a step that failed or was not measured has no line (the whole
 * report says why). baseline: cpuBaseline(); gpu: { lost }.
 */
export function gpuSummary(steps, baseline = {}, gpu = {}) {
  const find = (name) => steps.find((s) => s.name === name);
  const a = find("the adapter")?.result ?? {};
  const checkStep = find("the shaders against JavaScript"), check = checkStep?.result;
  const none = noRatios({ fallback: a.fallback, lost: gpu.lost });
  const verdicts = Object.entries(check ?? {});
  const bad = verdicts.filter(([, v]) => v.error || !v.ok).map(([k, v]) => `${k} ${v.error ? "FAILED" : "WRONG"}`);
  const sizes = a.subgroupSizes ? ` (${a.subgroupSizes[0] === a.subgroupSizes[1] ? a.subgroupSizes[0] : a.subgroupSizes.join(" to ")} wide)` : "";
  const lines = [`GPU: ${a.adapter ?? "?"}${a.fallback ? " (a fallback adapter: nothing timed)" : ""}; packed int8 dot ${yesNo(a.packed)}, ` +
    `shader-f16 ${yesNo(a.features?.includes("shader-f16"))}, subgroups ${yesNo(a.features?.includes("subgroups"))}${sizes}; ` +
    (check ? `the check ${verdicts.length - bad.length} of ${verdicts.length} ok${bad.length ? ` (${bad.join(", ")})` : ""}` : `not checked (${checkStep?.error ?? "not run"})`) +
    (gpu.lost ? `; the device was lost (${gpu.lost})` : "")];
  // a token of each model, widened (the table's first row of each): its tok/s and the CPU's estimate beside it
  const cpu = baseline.token;
  const tokens = steps.filter((s) => /^a token of [^,]+$/.test(s.name) && s.result?.tokPerSecond > 0).map((s) => {
    const t = s.result, cpuTok = cpu && t.GB > 0 ? cpu.GBps / t.GB : undefined;
    const ratio = none || tokenWrong(t, check) || !cpuTok ? "" : ` (${times(t.tokPerSecond / cpuTok)} the CPU)`;
    return `${s.name.replace(/^a token of /, "")} ${number(t.tokPerSecond)} tok/s${ratio}`;
  });
  const widest = steps.filter((s) => s.name.startsWith("bandwidth: ")).at(-1);
  const best = widest && fastestMatVec(widest, check);
  const reads = find("the device's ceilings")?.result?.global;
  const share = best && !gpu.lost && !a.fallback && reads?.GBps > 0 && !reads.unsteady ? ` (${number((100 * best.GBps) / reads.GBps, 0)}% of reading a buffer)` : "";
  if (tokens.length || best) {
    lines.push(`GPU, a token: ${[...tokens, best && `the fastest matrix × vector of ${widest.name.replace(/^bandwidth: /, "")} ${best.shader}, ${number(best.GBps)} GB/s${share}`].filter(Boolean).join("; ")}`);
  }
  const layer = find("a layer of a token")?.result;
  const fastestLayer = (layer?.rows ?? []).filter((row) => row.msPerLayer > 0 && !row.unsteady && !(check?.[row.check] && !check[row.check].ok))
    .sort((x, y) => x.msPerLayer - y.msPerLayer)[0];
  const generated = find("tokens generated on the GPU")?.result;
  const one = generated?.rows?.find((row) => row.perSubmission === 1);
  const most = generated?.rows?.filter((row) => row.perSubmission > 1 && row.msPerToken > 0).sort((x, y) => x.msPerToken - y.msPerToken)[0];
  const samplingWrong = ["sampling", "tokens on the GPU"].some((key) => check?.[key] && !check[key].ok);
  const layerParts = [fastestLayer && `the fastest layer ${fastestLayer.form}, ${fastestLayer.dispatches} dispatches, ${number(fastestLayer.msPerLayer, 2)} ms`,
    one && most && `generated ${number(one.msPerToken, 2)} ms a token one a submission, ${number(most.msPerToken, 2)} with ${most.perSubmission}` +
      (none || samplingWrong ? "" : ` (${times(one.msPerToken / most.msPerToken)})`)].filter(Boolean);
  if (layerParts.length) lines.push(`GPU, ${layerParts.join("; ")}`);
  // T202: where the time of the fastest layer the steps' table broke down goes
  const split = find("the steps of a layer")?.result;
  const splitForm = (split?.forms ?? []).filter((form) => stepMs(form) !== undefined && !form.unsteady && !(check?.[form.check] && !check[form.check].ok))
    .sort((x, y) => x.ms - y.ms)[0];
  if (splitForm && !none) {
    const parts = layerSplit(split, splitForm), two = (value) => (value === undefined ? "?" : number(value, 2));
    lines.push(`GPU, where the time of a layer ${splitForm.form} goes: ${two(parts.layer)} ms, the matrices alone ${two(parts.alone)}, fusing adds ${two(less(parts.matrices, parts.alone))}, ` +
      `the attention ${two(parts.attention)}, the norms and quantizing ${two(parts.small)}, the chain ${two(parts.chain)}`);
  }
  const prompt = find("a prompt all at once")?.result;
  const promptTokens = Math.max(0, ...(prompt?.rows ?? []).filter((row) => row.tokens).map((row) => row.tokens));
  const fastestPrompt = (prompt?.rows ?? []).filter((row) => row.tokens === promptTokens && !row.again && row.msPerToken > 0 && !(check?.[row.shader] && !check[row.shader].ok))
    .sort((x, y) => x.msPerToken - y.msPerToken)[0];
  if (fastestPrompt) {
    const ratio = none ? "" : timesFaster(baseline.prompt?.msPerToken, fastestPrompt.msPerToken);
    lines.push(`GPU, a prompt of ${promptTokens} tokens: ${fastestPrompt.shader}, ${number(fastestPrompt.GFLOPS, 0)} GFLOPS${ratio ? ` (${ratio} the CPU)` : ""}`);
  }
  return lines;
}

/** The storage section's line: the writes' MB/s and the read back */
export function storageSummary(r) {
  const mb = (r.mib * 2 ** 20) / 1e6, rate = (part) => number(mb / part.seconds, 0);
  return [`Storage: writes ${rate(r.sequential)} MB/s in order, ${rate(r.scattered)} far apart, ${rate(r.scatteredFlushEach)} with a flush each piece; ` +
    `read back ${rate(r.read)} MB/s${r.read.wrong ? ` (${r.read.wrong} pieces WRONG)` : ""}`];
}

/** The line section's line: the two fetches and the paced reads */
export function lineSummary(r) {
  const fetched = (f) => (f?.error ? "failed" : f ? `${number(f.MBps)} MB/s (first byte ${number(f.firstByteMs, 0)} ms)` : "not measured");
  const paced = (r.paced ?? []).map((p) => (p.slower ? "slower" : p.error ? "failed" : number(p.MBps, 2)));
  return [`Line: this site ${fetched(r.site)}, huggingface.co ${fetched(r.hf)}` +
    (paced.length ? `; huggingface.co read no faster than ${r.paced.map((p) => p.rate).join(", ")} MB/s: ${paced.join(", ")}` : "")];
}
