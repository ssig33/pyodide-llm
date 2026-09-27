// src/page-memory.js (T173): the page-memory section's steps and its mark, with a made-up worker and storage. Node alone.
//
//   node tests/page-memory.mjs
import assert from "node:assert/strict";
import { MEMORY_MARK, MEMORY_MOST, MEMORY_STEP, forgetMark, holdMemory, memoryLimit, memoryResult, memorySummary, readMark } from "../src/page-memory.js";

const MiB = 2 ** 20;
// sessionStorage as the page has it; `log` keeps every mark written, in order
function storage() {
  const items = new Map(), log = [];
  return { log, items, getItem: (k) => items.get(k) ?? null, setItem: (k, v) => { items.set(k, String(v)); log.push(JSON.parse(v)); }, removeItem: (k) => items.delete(k) };
}
// a worker that holds what it is asked to, and refuses past `most` bytes (as a RangeError of Memory.grow); maximum is
// what memory.js says of the room its memory was made with (undefined: says nothing, as before T173's review)
function worker(most = Infinity, error, maximum) {
  let held = 0;
  const asked = [];
  const room = maximum === undefined ? {} : { maximum };
  return { asked, ask: async (m) => {
    asked.push(m);
    if (error && m.to > error) return { error: "RangeError: out of memory" };
    if (m.to > most) return { result: { held, ...room, refused: "RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded" } };
    held = m.to;
    return { result: { held, ...room } };
  } };
}
const quick = { sleep: async () => {}, now: () => 1000 };

// the limit asked for, in whole steps, at least one and at most 4 GiB
assert.equal(memoryLimit(null), MEMORY_MOST);
assert.equal(memoryLimit("256"), 256 * MiB);
assert.equal(memoryLimit("100"), 64 * MiB, "down to whole steps");
assert.equal(memoryLimit("10"), MEMORY_STEP, "at least one step");
assert.equal(memoryLimit("99999"), MEMORY_MOST, "no more than one 32-bit memory");

// up to the limit: a mark before every step, holding what the step before held, and none left at the end
{
  const s = storage(), w = worker(), stages = [];
  const d = await holdMemory({ ask: w.ask, storage: s, limit: 256 * MiB, stage: (x) => stages.push(x), ...quick });
  assert.deepEqual({ stop: d.stop, held: d.held }, { stop: "limit", held: 256 * MiB });
  assert.deepEqual(s.log.map((m) => [m.held, m.trying]), [[0, 64], [64, 128], [128, 192], [192, 256]].map(([a, b]) => [a * MiB, b * MiB]));
  assert.deepEqual(w.asked.map((m) => m.to), [64, 128, 192, 256].map((x) => x * MiB), "the mark is written before the worker is asked");
  assert.equal(s.getItem(MEMORY_MARK), null, "a run that ended removes its mark");
  assert.deepEqual(stages.map((x) => [x.at, x.of]), [[1, 4], [2, 4], [3, 4], [4, 4]]);
  const r = memoryResult(d);
  assert.equal(r.status, "ok");
  assert.ok(r.markdown.includes("| 256 MiB | it held all it was asked to |"), r.markdown);
  assert.deepEqual(memorySummary(d), ["Page memory: held 256 MiB; it held all it was asked to"]);
}

// the mark is written before the worker is asked, every time: the worker sees the mark of its own step
{
  const s = storage();
  let seen = [];
  const ask = async (m) => { seen.push(JSON.parse(s.getItem(MEMORY_MARK)).trying === m.to); return { result: { held: m.to } }; };
  await holdMemory({ ask, storage: s, limit: 192 * MiB, ...quick });
  assert.deepEqual(seen, [true, true, true]);
}

// the browser refused: the held bytes are the worker's, and the mark is gone
{
  const s = storage(), w = worker(130 * MiB);
  const d = await holdMemory({ ask: w.ask, storage: s, limit: 512 * MiB, ...quick });
  assert.deepEqual({ stop: d.stop, held: d.held, trying: d.trying }, { stop: "refused", held: 128 * MiB, trying: 192 * MiB });
  assert.equal(s.getItem(MEMORY_MARK), null);
  const r = memoryResult(d);
  assert.equal(r.status, "ok");
  assert.ok(r.markdown.includes("| 128 MiB | the browser refused to grow it to 192 MiB (RangeError: WebAssembly.Memory.grow(): Maximum memory size exceeded) |"), r.markdown);
  // the whole limit: the most one 32-bit memory holds
  assert.ok(memoryResult({ stop: "limit", held: MEMORY_MOST, limit: MEMORY_MOST }).markdown.includes("the most one 32-bit WebAssembly memory holds"));
}

// the worker failed: an error, and the section's state says it
{
  const s = storage(), w = worker(Infinity, 64 * MiB);
  const d = await holdMemory({ ask: w.ask, storage: s, limit: 512 * MiB, ...quick });
  assert.equal(d.stop, "error");
  assert.equal(memoryResult(d).status, "error");
  assert.ok(memoryResult(d).markdown.includes("after holding 64 MiB"), memoryResult(d).markdown);
  assert.deepEqual(memorySummary(d), [], "the page's summary says the state and why (summaryOf)");
}

// no sessionStorage: nothing is grown (a tab ended then could not say how far it got)
{
  const w = worker();
  const d = await holdMemory({ ask: w.ask, storage: null, ...quick });
  assert.equal(d.stop, "none");
  assert.equal(w.asked.length, 0);
  assert.equal(memoryResult(d).status, "none");
}

// a mark that cannot be written stops the run before anything more is grown
{
  const w = worker(), s = storage();
  let writes = 0;
  s.setItem = () => { if (++writes > 2) throw new Error("QuotaExceededError"); };
  const d = await holdMemory({ ask: w.ask, storage: s, limit: 512 * MiB, ...quick });
  assert.equal(d.stop, "error");
  assert.equal(w.asked.length, 2, "no step without its mark");
}

// a tab ended while it grew: the mark left behind is what it held, then it is gone (read once)
{
  const s = storage(), w = worker();
  let asked = 0;
  // the run is left in the middle of its third step: the worker never answers, as a tab the device ended
  const ask = (m) => (++asked === 3 ? new Promise(() => {}) : w.ask(m));
  holdMemory({ ask, storage: s, limit: 512 * MiB, sleep: async () => {}, now: () => 1000 });
  await new Promise((resolve) => setTimeout(resolve, 10));
  const d = readMark(s, () => 4000);
  assert.deepEqual(d, { stop: "reloaded", held: 128 * MiB, trying: 192 * MiB, limit: 512 * MiB, seconds: 3 });
  assert.equal(s.getItem(MEMORY_MARK), null, "the mark is read once: a ?run=memory does not end the tab again");
  assert.equal(readMark(s), undefined);
  const r = memoryResult(d);
  assert.equal(r.status, "ok");
  assert.ok(r.markdown.includes("| 128 MiB | the tab ended while it grew to 192 MiB and the page was loaded again: the browser ended it, most likely for its memory |"), r.markdown);
  assert.deepEqual(memorySummary(d), ["Page memory: held 128 MiB; the tab ended while it grew to 192 MiB and the page was loaded again: the browser ended it, most likely for its memory"]);
}

// T173's review: the page hidden or left while a step grows: the run stops at once, holds no more, and leaves no mark
// (the page removes it in the event: a phone ends a hidden page far sooner, so a tab ended then is no limit)
{
  const s = storage(), w = worker();
  let hide, asked = 0;
  const hidden = new Promise((resolve) => (hide = resolve));
  // the third step's worker is ended by the page (it never answers), as the page's visibilitychange does
  const ask = (m) => {
    if (++asked < 3) return w.ask(m);
    forgetMark(s);
    hide();
    return new Promise(() => {});
  };
  const d = await holdMemory({ ask, storage: s, limit: 512 * MiB, hidden, ...quick });
  assert.deepEqual({ stop: d.stop, held: d.held, trying: d.trying }, { stop: "hidden", held: 128 * MiB, trying: 192 * MiB });
  assert.equal(asked, 3, "no step after the page was hidden");
  assert.equal(s.getItem(MEMORY_MARK), null);
  assert.equal(readMark(s), undefined, "a page loaded again after it was hidden finds no mark");
  const r = memoryResult(d);
  assert.equal(r.status, "error", "a run stopped by a hidden page is no measurement");
  assert.ok(r.markdown.startsWith("stopped: the page was hidden or left") && r.markdown.includes("while it grew to 192 MiB, after holding 128 MiB") &&
            r.markdown.includes("this is not the limit"), r.markdown);
  assert.deepEqual(memorySummary(d), [], "the page's summary says the state and why (summaryOf)");
}

// hidden in the pause between steps: no next step
{
  const s = storage(), w = worker();
  let hide, sleeps = 0;
  const hidden = new Promise((resolve) => (hide = resolve));
  const sleep = () => (++sleeps === 2 ? (hide(), new Promise(() => {})) : Promise.resolve());
  const d = await holdMemory({ ask: w.ask, storage: s, limit: 512 * MiB, hidden, sleep, now: () => 1000 });
  assert.deepEqual({ stop: d.stop, held: d.held, trying: d.trying }, { stop: "hidden", held: 128 * MiB, trying: 192 * MiB });
  assert.equal(w.asked.length, 2);
  assert.equal(s.getItem(MEMORY_MARK), null);
}

// hidden before it began: nothing is held
{
  const s = storage(), w = worker();
  const d = await holdMemory({ ask: (m) => new Promise(() => w.asked.push(m)), storage: s, limit: 512 * MiB, hidden: Promise.resolve(), ...quick });
  assert.deepEqual({ stop: d.stop, held: d.held }, { stop: "hidden", held: 0 });
  assert.equal(s.getItem(MEMORY_MARK), null);
}

// the room the memory was made with: a browser that refused the maximum is said, one that took it is not
{
  const refused = await holdMemory({ ask: worker(130 * MiB, undefined, null).ask, storage: storage(), limit: 512 * MiB, ...quick });
  assert.equal(refused.maximum, null);
  assert.ok(memoryResult(refused).markdown.includes("The browser refused a memory with room for 512 MiB, so it was made without a maximum."), memoryResult(refused).markdown);
  const held = await holdMemory({ ask: worker(Infinity, undefined, 512 * MiB).ask, storage: storage(), limit: 256 * MiB, ...quick });
  assert.equal(held.maximum, 512 * MiB);
  assert.ok(!memoryResult(held).markdown.includes("room for"), memoryResult(held).markdown);
  assert.ok(!memoryResult(readMark((() => { const s = storage(); s.items.set(MEMORY_MARK, JSON.stringify({ held: 0, trying: MEMORY_STEP, limit: MEMORY_MOST })); return s; })())).markdown.includes("room for"),
            "a mark says nothing of the room");
}

// a mark that is not one of ours, or no storage at all, is no result
{
  const s = storage();
  s.items.set(MEMORY_MARK, "{not json");
  assert.equal(readMark(s), undefined);
  s.items.set(MEMORY_MARK, JSON.stringify({ held: 5 }));
  assert.equal(readMark(s), undefined);
  assert.equal(readMark(null), undefined);
}

console.log("page-memory: ok");
