// The storage section of /benchmark/ (T134; T137's stage 0 until then, at /opfs-test/): how fast this browser's origin
// private file system (OPFS) takes the writes a resumable conversion would make, measured in a worker with a
// synchronous access handle (the only kind that writes at an offset). The page (src/pages/benchmark.astro) asks for one
// step at a time; nothing here touches the model page or what it keeps.
//
//   { step: "info" }              whether there is an OPFS and a sync handle here, and navigator.storage.estimate()
//   { step: "run", mib }          a file of mib MiB written 8 MiB at a time: in order, far apart, far apart with a flush
//                                 (and a small progress file rewritten) after every piece, then read back and checked;
//                                 the time to open a handle, whether a second handle on the same file is refused, and
//                                 whether the file opens again once the first is closed. The file is removed at the end.

const PIECE = 8 << 20;
// T177: the stage a run is at, for the page to show ({stage, at, of}; it answers none of the page's questions)
const STAGES = 4;
const stage = (name, at) => postMessage({ stage: name, at, of: STAGES });
const FOLDER = "benchmark", FILE = "pieces.bin", PROGRESS = "progress.json";

// the bytes of piece i: a pattern of its own, so that the read-back can tell the pieces apart
function fill(bytes, i) {
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  let x = ((i + 1) * 2654435761) >>> 0;
  for (let j = 0; j < words.length; j += 1024) {
    x = (x ^ (x << 13)) >>> 0; x = (x ^ (x >>> 17)) >>> 0; x = (x ^ (x << 5)) >>> 0;
    words[j] = x;
  }
  words[0] = i;
}
// far apart: every piece about half the file away from the one before (a stride coprime with the count, so that every
// piece comes once). A conversion writes each tensor where the checkpoint has it, in the order of the source file
function scattered(count) {
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  let stride = Math.floor(count / 2) + 1;
  while (gcd(stride, count) !== 1) stride++;
  return Array.from({ length: count }, (_, k) => (k * stride) % count);
}
const seconds = (began) => (performance.now() - began) / 1000;

async function info() {
  const out = { opfs: Boolean(navigator.storage?.getDirectory), syncHandle: typeof FileSystemFileHandle !== "undefined"
    && typeof FileSystemFileHandle.prototype.createSyncAccessHandle === "function" };
  try {
    const { usage, quota } = await navigator.storage.estimate();
    Object.assign(out, { usage, quota });
  } catch (error) {
    out.estimate = String(error);
  }
  return out;
}

async function run(mib) {
  const count = Math.max(2, Math.floor(mib * (1 << 20) / PIECE)), size = count * PIECE;
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(FOLDER, { create: true });
  const file = await dir.getFileHandle(FILE, { create: true });
  const out = { mib: size / (1 << 20), pieces: count };
  let began = performance.now();
  const handle = await file.createSyncAccessHandle();
  out.openSeconds = seconds(began);
  try {
    // T137 (7): a second tab opening the same model.bin must be refused, not share the file
    try {
      const second = await file.createSyncAccessHandle();
      second.close();
      out.secondHandle = "opened (NOT refused)";
    } catch (error) {
      out.secondHandle = `refused (${error.name})`;
    }
    began = performance.now();
    handle.truncate(size);
    handle.flush();
    out.truncateSeconds = seconds(began);
    const bytes = new Uint8Array(PIECE);
    const put = (i) => {
      fill(bytes, i);
      const written = handle.write(bytes, { at: i * PIECE });
      // T180: Chromium hands back a file error as a negative count read unsigned (4294967288 is -8, no space)
      if (written !== PIECE) {
        const code = written > 0x7fffffff ? written - 0x100000000 : written;
        throw new Error(code < 0 ? `piece ${i}: the browser could not write it (error ${code}${code === -8 ? ": no space left" : ""})`
          : `piece ${i}: ${written} of ${PIECE} bytes written`);
      }
    };
    // (a) in order and (b) far apart, one flush at the end
    const once = (order) => {
      const start = performance.now();
      for (const i of order) put(i);
      const f = performance.now();
      handle.flush();
      return { seconds: seconds(start), flushSeconds: seconds(f) };
    };
    const apart = scattered(count);
    stage("writing in order", 1);
    out.sequential = once(Array.from({ length: count }, (_, i) => i));
    stage("writing far apart", 2);
    out.scattered = once(apart);
    // (c) far apart with a flush after every piece and the small record rewritten, as a resumable conversion would
    // at every boundary of a tensor (T137, 2: the record comes after the flush of the output)
    const progressFile = await dir.getFileHandle(PROGRESS, { create: true });
    const progress = await progressFile.createSyncAccessHandle();
    const encoder = new TextEncoder();
    stage("far apart, a flush after each piece", 3);
    try {
      const start = performance.now();
      let flushing = 0;
      for (const i of apart) {
        put(i);
        const f = performance.now();
        handle.flush();
        const record = encoder.encode(JSON.stringify({ step: i, begin: i * PIECE, done: i, saved: Date.now() }));
        progress.truncate(0);
        progress.write(record, { at: 0 });
        progress.flush();
        flushing += performance.now() - f;
      }
      out.scatteredFlushEach = { seconds: seconds(start), flushSeconds: flushing / 1000 };
    } finally {
      progress.close();
    }
    // (d) read back in order, and check that every piece is the one written there
    stage("reading back", 4);
    began = performance.now();
    let wrong = 0;
    for (let i = 0; i < count; i++) {
      const read = handle.read(bytes, { at: i * PIECE });
      if (read !== PIECE || new Uint32Array(bytes.buffer, 0, 1)[0] !== i) wrong++;
    }
    out.read = { seconds: seconds(began), wrong };
  } finally {
    handle.close();
  }
  // the review of T137 (5): once closed, the file opens again (a handle left open would look like another tab)
  try {
    const again = await file.createSyncAccessHandle();
    again.close();
    out.reopen = "opened";
  } catch (error) {
    out.reopen = `refused (${error.name})`;
  }
  await root.removeEntry(FOLDER, { recursive: true });
  return out;
}

onmessage = async ({ data }) => {
  try {
    postMessage({ result: data.step === "info" ? await info() : await run(data.mib) });
  } catch (error) {
    // what is left of the file goes, so that a failed run never keeps a gigabyte
    try {
      await (await navigator.storage.getDirectory()).removeEntry(FOLDER, { recursive: true });
    } catch {
      // nothing was made
    }
    postMessage({ error: `${error.name}: ${error.message}` });
  }
};
