// T163: a reading thread of the CPU section's ceilings (sections.js): each time the section says go, it takes chunks
// of the made-up model's weights in turn with the section (NEXT, until TOTAL) and reads each with the loop of
// kernels/ceilings.ts, on the section's shared memory, then counts itself done. It sleeps between (Atomics.wait) and
// never spins. The control words are sections.js's; the first message is claimed before anything is awaited (T109).
const GO = 0, DONE = 1, COUNT = 2, TOTAL = 3, STOP = 4, NEXT = 5;

self.onmessage = ({ data: { module, memory, control, base, chunk, chunks, share } }) => {
  self.onmessage = null;
  const read = new WebAssembly.Instance(module, { env: { memory } }).exports.read;
  const ctl = new Int32Array(control);
  let seen = Atomics.load(ctl, GO);
  postMessage("ready");
  for (;;) {
    Atomics.wait(ctl, GO, seen);
    seen = Atomics.load(ctl, GO);
    if (Atomics.load(ctl, STOP)) break;
    const count = Atomics.load(ctl, COUNT);
    if (share >= count) continue;  // not one of this run's threads
    const total = Atomics.load(ctl, TOTAL);
    for (let k; (k = Atomics.add(ctl, NEXT, 1)) < total;) read(base + (k % chunks) * chunk, chunk, 1);
    if (Atomics.add(ctl, DONE, 1) + 1 === count - 1) Atomics.notify(ctl, DONE);
  }
};
