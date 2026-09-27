// T163: a reading thread of the CPU section's ceilings (sections.js): it reads its part of the made-up model's weights
// with the loop of kernels/ceilings.ts each time the section says go, on the section's shared memory, and counts itself
// done. It sleeps between (Atomics.wait) and never spins. The control words are sections.js's (GO, DONE, COUNT, PASSES,
// STOP); the first message is claimed before anything is awaited (T109).
const GO = 0, DONE = 1, COUNT = 2, PASSES = 3, STOP = 4;

self.onmessage = ({ data: { module, memory, control, base, size, share } }) => {
  self.onmessage = null;
  const read = new WebAssembly.Instance(module, { env: { memory } }).exports.read;
  const ctl = new Int32Array(control);
  let seen = Atomics.load(ctl, GO), checksums = 0;
  postMessage("ready");
  for (;;) {
    Atomics.wait(ctl, GO, seen);
    seen = Atomics.load(ctl, GO);
    if (Atomics.load(ctl, STOP)) break;
    const count = Atomics.load(ctl, COUNT);
    if (share >= count) continue;  // not one of this run's threads
    // the same parts as sections.js: equal, of whole 64-byte steps
    const part = Math.floor(size / count / 64) * 64;
    checksums ^= read(base + share * part, part, Atomics.load(ctl, PASSES));
    if (Atomics.add(ctl, DONE, 1) + 1 === count - 1) Atomics.notify(ctl, DONE);
  }
  postMessage({ checksums });
};
