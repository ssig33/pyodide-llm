// The page-memory section of /benchmark/ (T173): how much memory this page can hold before the browser refuses it or
// ends the tab. iOS ends a tab past about 2 to 3 GB of its own (jetsam) and throws nothing, so this cannot find the
// limit by catching an error: the page (src/page-memory.js) asks for one step at a time and writes in sessionStorage,
// before each step, how much was held; a tab ended and loaded again finds that mark and reports it.
//
//   { step: "grow", to, limit }   grow one WebAssembly memory to `to` bytes (a multiple of its 64 KiB pages) and fill the
//                                 new part with random bytes; limit: the most it will be asked to hold (its maximum)
//     -> { result: { held } }                     held: the bytes now held and filled
//     -> { result: { held, refused: "…" } }       the browser refused to grow it (RangeError): held is what it had
//
// One WebAssembly memory, as the models' weights are held (forward.js): a browser may refuse a memory that large before
// the tab runs out. The bytes are random so that a device that compresses its memory (iOS does) counts them all; a
// page of zeros, or of one byte a page, would compress to nothing and the tab would seem to hold far more. The memory
// is freed when the page ends this worker. The section runs alone: no other WebAssembly memory is alive in the page
// (T96: Chromium refused a page its third).
const PAGE = 65536;
let memory, filled = 0, seed = 0x9e3779b9 | 0;

onmessage = ({ data }) => {
  try {
    if (data.step === "grow") postMessage({ result: grow(data.to, data.limit) });
    else postMessage({ error: `no step ${data.step} here` });
  } catch (error) {
    postMessage({ error: `${error?.name ?? "Error"}: ${error?.message ?? error}` });
  }
};

function grow(to, limit) {
  if (!memory) {
    // the maximum of the limit, where the browser takes it; else none (the browser's own)
    try {
      memory = new WebAssembly.Memory({ initial: 0, maximum: Math.ceil(limit / PAGE) });
    } catch {
      memory = new WebAssembly.Memory({ initial: 0 });
    }
  }
  try {
    memory.grow((to - memory.buffer.byteLength) / PAGE);
  } catch (error) {
    return { held: filled, refused: `${error?.name ?? "Error"}: ${error?.message ?? error}` };
  }
  // xorshift32 over the new part, a word at a time
  const words = new Uint32Array(memory.buffer, filled, (to - filled) / 4);
  let x = seed;
  for (let i = 0; i < words.length; i++) {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    words[i] = x;
  }
  seed = x;
  filled = to;
  return { held: filled };
}
