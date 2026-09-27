// src/wake.js (T194): the wake lock of /benchmark/ and the seconds its page was hidden, with a made-up document,
// navigator and clock. Node alone.
//
//   node tests/wake.mjs
import assert from "node:assert/strict";
import { hiddenLine, wakeKeeper } from "../src/wake.js";

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

// a document whose visibility the test sets; a navigator whose lock is granted, refused, or absent
function page({ lock = "granted", visible = true } = {}) {
  const listeners = new Set();
  const document = {
    visibilityState: visible ? "visible" : "hidden",
    addEventListener: (type, f) => type === "visibilitychange" && listeners.add(f),
  };
  let t = 0;
  const asked = [], released = [];
  const request = async (type) => {
    asked.push(type);
    if (lock === "refused") throw new DOMException("not allowed", "NotAllowedError");
    const sentinel = { released: false, on: [], addEventListener: (e, f) => e === "release" && sentinel.on.push(f),
                       release: async () => { sentinel.released = true; released.push(sentinel); sentinel.on.forEach((f) => f()); } };
    return sentinel;
  };
  const navigator = lock === "none" ? {} : { wakeLock: { request } };
  const show = (state) => {
    document.visibilityState = state;
    // the browser lets the lock go when the page is hidden (and says so)
    listeners.forEach((f) => f());
  };
  return { document, navigator, asked, released, show, now: () => t, pass: (ms) => (t += ms) };
}

// never hidden: a lock asked once, let go at the end, and no line
{
  const p = page();
  const awake = wakeKeeper(p);
  awake.start();
  await settle();
  assert.deepEqual(p.asked, ["screen"]);
  assert.ok(awake.held());
  p.pass(60000);
  awake.stop();
  assert.equal(p.released.length, 1, "let go when the run ends");
  assert.ok(!awake.held());
  assert.equal(awake.hidden(), 0);
  assert.equal(hiddenLine(awake.hidden()), "", "no line when it was never hidden");
}

// hidden 12.4 s in the middle: counted, asked again once shown, and the line says 12 s
{
  const p = page();
  const awake = wakeKeeper(p);
  awake.start();
  await settle();
  p.pass(5000);
  p.show("hidden");
  assert.ok(!awake.held(), "a hidden page holds no lock");
  p.pass(7000);
  assert.equal(awake.hidden(), 7000, "the seconds hidden so far, while still hidden");
  p.pass(5400);
  p.show("visible");
  await settle();
  assert.deepEqual(p.asked, ["screen", "screen"], "asked again when shown");
  assert.ok(awake.held());
  p.pass(3000);
  awake.stop();
  assert.equal(awake.hidden(), 12400);
  assert.match(hiddenLine(awake.hidden()), /^\*\*hidden for 12 s\*\*/);
  // hidden again after the run: not counted, nothing asked
  p.show("hidden");
  p.pass(9000);
  p.show("visible");
  await settle();
  assert.equal(awake.hidden(), 12400, "no count between runs");
  assert.equal(p.asked.length, 2, "no lock between runs");
  // a second run adds to the first (the report holds both runs' sections)
  awake.start();
  await settle();
  p.show("hidden");
  p.pass(2000);
  awake.stop();  // a run that ends while hidden counts up to its end
  assert.equal(awake.hidden(), 14400);
}

// started while hidden: counted from the start, no lock asked until shown
{
  const p = page({ visible: false });
  const awake = wakeKeeper(p);
  awake.start();
  await settle();
  assert.equal(p.asked.length, 0, "a hidden page does not ask");
  p.pass(3000);
  p.show("visible");
  await settle();
  assert.equal(p.asked.length, 1);
  awake.stop();
  assert.equal(awake.hidden(), 3000);
}

// no wake lock in this browser, and a lock refused: the count goes on all the same, and nothing throws
for (const lock of ["none", "refused"]) {
  const p = page({ lock });
  const awake = wakeKeeper(p);
  awake.start();
  await settle();
  assert.ok(!awake.held(), lock);
  p.show("hidden");
  p.pass(4000);
  p.show("visible");
  await settle();
  awake.stop();
  assert.equal(awake.hidden(), 4000, lock);
  assert.equal(p.asked.length, lock === "none" ? 0 : 2, lock);
}

// a run that ends before the browser answers lets the late lock go
{
  const p = page();
  const awake = wakeKeeper(p);
  awake.start();
  awake.stop();
  await settle();
  assert.equal(p.released.length, 1, "the late lock let go");
  assert.ok(!awake.held());
}

// under half a second rounds to no line
assert.equal(hiddenLine(400), "");
assert.match(hiddenLine(600), /hidden for 1 s/);

console.log("wake: ok");
