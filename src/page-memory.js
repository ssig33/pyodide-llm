// T173: the page-memory section of /benchmark/: how much memory the page can hold before the browser refuses it or
// ends the tab. A plain module, so that Node tests it alone (tests/page-memory.mjs); the page's worker is
// public/benchmark/memory.js.
//
// iOS ends a tab past about 2 to 3 GB (jetsam) without an error, so the limit is found by a mark: before each step
// the page writes in sessionStorage what it held and what it is about to try; a tab the browser ended and loaded again
// finds the mark (readMark) and reports the last step it held. A normal end removes the mark. The section is never
// part of "Run all" (it may end the tab): only its own button or ?run=memory runs it, and a page that finds a mark
// does not run it again (a ?run=memory in the address would otherwise end the tab over and over).
import { tableCell } from "./bench.js";

export const MEMORY_MARK = "benchmark-memory";
// the steps: 64 MiB of random bytes, a quarter of a second apart (time for the device to end the tab after a step, so
// that a step the page went on from was held for a while); up to 4 GiB, the most one 32-bit WebAssembly memory holds
export const MEMORY_STEP = 64 * 2 ** 20, MEMORY_MOST = 4 * 2 ** 30, MEMORY_PAUSE_MS = 250;

const mib = (bytes) => `${Math.round(bytes / 2 ** 20)} MiB`;

/** The limit the page is asked for (?memoryMB=, for tests: a runner stops short of its own limit), in whole steps,
 * at least one step and at most MEMORY_MOST. */
export function memoryLimit(asked) {
  const bytes = Number(asked) > 0 ? Number(asked) * 2 ** 20 : MEMORY_MOST;
  return Math.min(MEMORY_MOST, Math.max(MEMORY_STEP, Math.floor(bytes / MEMORY_STEP) * MEMORY_STEP));
}

/** Grow the worker's memory step by step up to limit, writing the mark before each step. ask(message) answers as the
 * worker does; storage is the tab's sessionStorage (null: none here, and nothing is grown: a tab ended without a
 * mark would report nothing). stage({stage, at, of}) is the page's progress (T177). What it returns is the section's
 * data: { stop: "limit" | "refused" | "error" | "none", held, trying?, limit, why?, seconds }. */
export async function holdMemory({ ask, storage, limit = MEMORY_MOST, stage = () => {}, pause = MEMORY_PAUSE_MS,
                                   sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), now = Date.now }) {
  if (!storage) return { stop: "none", held: 0, limit, why: "no sessionStorage in this tab: a tab the browser ended could not say how far it got" };
  const began = now(), of = limit / MEMORY_STEP;
  const seconds = () => (now() - began) / 1000;
  let held = 0;
  try {
    while (held < limit) {
      const to = held + MEMORY_STEP;
      storage.setItem(MEMORY_MARK, JSON.stringify({ held, trying: to, limit, began }));
      stage({ stage: `holding ${mib(to)}`, at: to / MEMORY_STEP, of });
      const answer = await ask({ step: "grow", to, limit });
      if (answer.error) return { stop: "error", held, trying: to, limit, why: answer.error, seconds: seconds() };
      if (answer.result.refused) return { stop: "refused", held: answer.result.held, trying: to, limit, why: answer.result.refused, seconds: seconds() };
      held = answer.result.held;
      await sleep(pause);
    }
    return { stop: "limit", held, limit, seconds: seconds() };
  } catch (error) {
    return { stop: "error", held, limit, why: `${error?.name ?? "Error"}: ${error?.message ?? error}`, seconds: seconds() };
  } finally {
    try {
      storage.removeItem(MEMORY_MARK);
    } catch {
      // nothing to remove it from
    }
  }
}

/** A mark left by a run that never ended (the tab was ended, or left, while it grew): the section's data for it, and
 * the mark removed. undefined when there is none. */
export function readMark(storage, now = Date.now) {
  let text;
  try {
    text = storage?.getItem(MEMORY_MARK);
    if (text) storage.removeItem(MEMORY_MARK);
  } catch {
    return undefined;
  }
  if (!text) return undefined;
  try {
    const m = JSON.parse(text);
    if (!(m.held >= 0 && m.trying > m.held)) return undefined;
    return { stop: "reloaded", held: m.held, trying: m.trying, limit: m.limit, seconds: Number.isFinite(m.began) ? (now() - m.began) / 1000 : undefined };
  } catch {
    return undefined;
  }
}

// how a run ended, in a few words (provisional English: the owner chooses the words, TODO.md T173)
function ending(d) {
  switch (d.stop) {
    case "reloaded": return `the page was loaded again while it grew to ${mib(d.trying)}: the browser ended the tab for its memory, or the tab was left`;
    case "refused": return `the browser refused to grow it to ${mib(d.trying)} (${d.why})`;
    case "limit": return d.limit >= MEMORY_MOST ? "it held the most one 32-bit WebAssembly memory holds" : "it held all it was asked to";
    default: return d.why ?? d.stop;
  }
}

/** The section's result for the page: { status, markdown, data } */
export function memoryResult(d) {
  if (d.stop === "none") return { status: "none", markdown: d.why, data: d };
  if (d.stop === "error") return { status: "error", markdown: `${d.why} (after holding ${mib(d.held)})`, data: d };
  return { status: "ok", data: d, markdown: [
    `One WebAssembly memory grown ${mib(MEMORY_STEP)} at a time and filled with random bytes, ${MEMORY_PAUSE_MS / 1000} s apart, up to ${mib(d.limit)}. ` +
      "What the page used before it began is not counted.", "",
    "| held | how it ended |", "|---:|---|",
    `| ${mib(d.held)} | ${tableCell(ending(d))} |`,
  ].join("\n") };
}

/** The section's line of the report's summary (T185) */
export function memorySummary(d) {
  if (!d || d.stop === "none" || d.stop === "error") return [];
  return [`Page memory: held ${mib(d.held)}; ${ending(d)}`];
}
