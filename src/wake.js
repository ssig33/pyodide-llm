// src/wake.js (T194): keep the screen on while /benchmark/ runs, and count the seconds the page was hidden.
//
// A phone that turns its screen off puts the page away: its timers and workers slow down or stop, and a section that
// takes minutes then measures a device asleep. While sections run, the page asks for a Screen Wake Lock
// (navigator.wakeLock.request("screen")); a browser without one, or one that refuses it (a battery saver, a page not
// shown), changes nothing. The browser lets the lock go by itself when the page is hidden, so the page asks again
// when it is shown. Whatever the lock did, the seconds the page was hidden while it ran go at the head of the report
// ("hidden for N s"), so that a reader knows which numbers were taken on a device put away. T173's page-memory
// section stops by itself when the page is hidden (benchmark.astro), as before.
//
// A plain module with its document, navigator and clock handed in, so that tests/wake.mjs runs it in Node alone.

/** Starts counting and holding the screen on; `stop()` lets the lock go and ends the count. `hidden()` is the
 * milliseconds hidden so far, over every run of this page (a report holds the sections of every run). */
export function wakeKeeper({ document, navigator, now = () => performance.now() }) {
  let hiddenMs = 0, hiddenSince, running = false, lock, asking;

  async function ask() {
    if (!running || asking || lock || document.visibilityState !== "visible" || !navigator?.wakeLock?.request) return;
    asking = true;
    try {
      const got = await navigator.wakeLock.request("screen");
      // a run that ended while the browser answered: let the lock go at once
      if (!running) got.release?.().catch?.(() => {});
      else {
        lock = got;
        got.addEventListener?.("release", () => { if (lock === got) lock = undefined; });
      }
    } catch {
      // refused (a battery saver, a page not shown, a policy): the run goes on as it would without one
    } finally {
      asking = false;
    }
  }

  function onVisibility() {
    if (!running) return;
    if (document.visibilityState === "hidden") {
      if (hiddenSince === undefined) hiddenSince = now();
      lock = undefined;  // the browser has let it go
    } else {
      if (hiddenSince !== undefined) hiddenMs += now() - hiddenSince;
      hiddenSince = undefined;
      ask();
    }
  }
  document.addEventListener("visibilitychange", onVisibility);

  return {
    start() {
      if (running) return;
      running = true;
      if (document.visibilityState === "hidden") hiddenSince = now();
      ask();
    },
    stop() {
      if (!running) return;
      if (hiddenSince !== undefined) hiddenMs += now() - hiddenSince;
      hiddenSince = undefined;
      running = false;
      const held = lock;
      lock = undefined;
      held?.release?.().catch?.(() => {});
    },
    hidden: () => hiddenMs + (running && hiddenSince !== undefined ? now() - hiddenSince : 0),
    held: () => lock !== undefined,
  };
}

/** The report's line of the seconds hidden, or "" when the page was never hidden while it ran. */
export function hiddenLine(ms) {
  const seconds = Math.round(ms / 1000);
  if (!(seconds > 0)) return "";
  return `**hidden for ${seconds} s** while the sections ran: the page was not on the screen (another app, the screen off), and the browser may have slowed or stopped it then; the numbers of those sections may be low.`;
}
