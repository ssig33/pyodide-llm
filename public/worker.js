// Pyodide lives in this worker, so the page stays responsive while the model is loading and generating.
// model is an entry of src/models.js, or one with {file, tokenizerFile}: two files of the visitor's own disk,
// which are read where they are and go nowhere. Or one with {hf: {weights, config, tokenizer}}: a Hugging Face
// model, of that disk (Files) or of huggingface.co ({repo, revision} and file names), which
// public/llama2_convert.py converts in here as it arrives.
// The page sends   {type: "init", search, model, load},  {type: "load", search, model, load},
//                  {type: "generate", prompt, ...options}  and  {type: "stop"}; /benchmark/'s model section also
//                  {type: "bench", model, load, rounds, prompt, steps} and (T184) {type: "paths", load, prompt, counts, sampled}
// and receives     {type: "status" | "progress" | "ready" | "token" | "done" | "bench" | "paths" | "error"
//                         | "threads" | "threads-compared" | "gpu", ...}
// Of these, only ready (a load, and each round of the benchmark), bench (the benchmark), paths (the page's path timed,
// T184), done (a text, also one stopped) and error (any of them) end a piece of this worker's work: the page reads them, and nothing else, as the
// worker being idle (T172). status and progress come during a load, token during a text, threads and
// threads-compared from the search for the number of threads, which runs inside a text, and gpu once the GPU is
// ready or refused, whatever else is going on (T148). A new message is one of the ends or not, and the page's list
// of the ends says so.
// load is a number the page counts up: a newer load cancels the one that is going on, and whatever this worker
// reports about a load carries its number, so that the page can tell a late report of a cancelled one.

// the version becomes part of a CDN URL, so accept nothing but a plain version number
const PYODIDE_VERSION_PATTERN = /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/;

// the latest release on npm (the "latest" tag never points at an alpha), or ?pyodide=<version> to force one
async function resolvePyodideVersion(search) {
  const forced = new URLSearchParams(search).get("pyodide");
  if (PYODIDE_VERSION_PATTERN.test(forced)) {
    return forced;
  }
  const res = await fetch("https://data.jsdelivr.com/v1/packages/npm/pyodide/resolved?specifier=latest");
  const version = (await res.json()).version;
  if (!PYODIDE_VERSION_PATTERN.test(version)) {
    throw new Error(`Unexpected Pyodide version: ${version}`);
  }
  return version;
}

// Pyodide asks for the NumPy wheel only once it has started, several seconds after its own files: until then it
// does not know the name. That name is in pyodide-lock.json, which Pyodide fetches anyway, so this reads the lock
// as well and puts the wheel into the HTTP cache while Pyodide is still coming up. Nothing is written down here:
// the version is the one resolved at run time, the file name comes from the lock. Anything unexpected (another
// shape of the lock, a CDN that says no) only means no head start, so every error is dropped.
async function prefetchNumpy(base) {
  try {
    const lock = await (await fetch(`${base}pyodide-lock.json`)).json();
    const name = lock.packages?.numpy?.file_name;
    if (typeof name !== "string" || !/^[A-Za-z0-9._+-]+\.whl$/.test(name)) {
      return;
    }
    const wheel = await fetch(base + name);
    // read it to the end so that the browser keeps it, and drop every chunk: this copy is never used
    await (wheel.body ? wheel.body.pipeTo(new WritableStream()) : wheel.arrayBuffer());
  } catch {
    // no head start
  }
}

// A model without a model.safetensors is split over several files (model-00001-of-00002.safetensors, ...), or
// published under the name of a shard even when there is only one (T78). model.safetensors.index.json says which
// file every tensor is in: the files, in the order of their names, which is the order they are fed in (T105).
function shardsOf(index) {
  try {
    const map = (typeof index === "string" ? JSON.parse(index) : index)?.weight_map;
    return [...new Set(Object.values(map ?? {}))].sort();
  } catch {
    return [];
  }
}

// Checkpoints are deployed in parts of 8 MiB (see the Makefile). Several parts download at once, which is
// about twice as fast as one stream, and the download runs while Pyodide is still loading: until the Python
// buffer exists the chunks wait in a queue, after that every chunk is written straight into it.
const PART_BYTES = 8 * 1024 * 1024;
const CONNECTIONS = 8;

// GitHub Pages lets the browser keep a file for ten minutes only, so the parts also go into the Cache API: the
// next visit starts without downloading the model again. The size is part of the key, so a rebuilt model of
// another size is fetched anew. Without the Cache API (some private modes) this is a plain fetch.
// v2: llm-jp-3-150m got its whole context of 4096 tokens, which changed its header and not its size
const MODEL_CACHE = "models-v2";

// what an earlier version of this page stored
globalThis.caches?.delete("models-v1").catch(() => {});

// A part's bytes: PART_BYTES, the last one less
const partBytes = (model, part) => Math.min(PART_BYTES, model.bytes - part * PART_BYTES);
const partKey = (url, model) => `${url}?bytes=${model.bytes}`;

async function fetchPart(url, model, part, signal) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  const key = partKey(url, model);
  // a Cache API that fails here (T117 met it in the service worker) leaves the network to answer
  const cached = await cache?.match(key).catch(() => undefined);
  if (cached) {
    return cached;
  }
  const res = await fetch(url, { signal });
  if (res.ok && cache) {
    // stored while the other copy streams into Python; a full disk must not stop the download. A part that is
    // cancelled half way is not stored at all, the finished ones stay for the next time. Nor is one whose body
    // ends short without an error (the review of T97): it would come back from here on every visit
    const expected = partBytes(model, part);
    let count = 0;
    const whole = new TransformStream({
      transform(chunk, out) {
        count += chunk.byteLength;
        out.enqueue(chunk);
      },
      flush() {
        if (count !== expected) throw new Error(`part ${part}: ${count} of ${expected} bytes`);
      },
    });
    cache.put(key, new Response(res.clone().body.pipeThrough(whole))).catch(() => {});
  }
  return res;
}
// a part that broke is not read from the cache again: the next try asks the network
async function forgetPart(url, model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  await cache?.delete(partKey(url, model)).catch(() => {});
}

// parts of this checkpoint that were cached for another size are of no use any more
async function dropStaleParts(model) {
  const cache = await globalThis.caches?.open(MODEL_CACHE).catch(() => undefined);
  for (const request of (await cache?.keys().catch(() => undefined)) ?? []) {
    const url = new URL(request.url);
    if (url.pathname.includes(`/models/${model.checkpoint}.`) && url.searchParams.get("bytes") !== String(model.bytes)) {
      cache.delete(request).catch(() => {});
    }
  }
}

function download(model, signal, load) {
  const parts = Math.ceil(model.bytes / PART_BYTES);
  const queue = [];
  const started = performance.now();
  let sink, next = 0, received = 0, reported = -1;
  // T115: the checkpoint's first bytes (its header), as soon as the first part brings them
  let head = new Uint8Array(0), tell;
  const header = new Promise((resolve) => { tell = resolve; });
  // T97: Firefox on Windows breaks the body of a part now and then ("Error in input stream", 1 load in 12 on the CI
  // runners, with the service worker and without it alike): the part is fetched again, twice at most. Its chunks go
  // to the same offsets, so what arrived before the break is written over with the same bytes.
  const partUrl = (part) => new URL(`models/${model.checkpoint}.${String(part).padStart(3, "0")}`, import.meta.url).href;
  const fetchOnce = async (part) => {
    const res = await fetchPart(partUrl(part), model, part, signal);
    if (!res.ok) {
      throw Object.assign(new Error(`Could not fetch part ${part} of ${model.checkpoint}: ${res.status}`), { final: true });
    }
    const reader = res.body.getReader();
    let got = 0;
    try {
      for (let offset = part * PART_BYTES; ;) {
        const { done, value } = await reader.read();
        if (done) {
          // a body that ends short without an error is a break too (the review of T97)
          if (got !== partBytes(model, part)) throw new Error(`part ${part} of ${model.checkpoint} ended after ${got} of ${partBytes(model, part)} bytes`);
          return;
        }
        // a chunk that was already on its way when the load was cancelled: its buffer is gone
        signal.throwIfAborted();
        sink ? sink(offset, value) : queue.push([offset, value]);
        if (sink) await weightsRoom();  // (T156: a model on the GPU alone, its worker no more than FLOW_BYTES behind)
        // the header's bytes that this chunk brings (a part fetched again brings some a second time)
        if (head.length < HEADER_BYTES && offset <= head.length && offset + value.length > head.length) {
          head = new Uint8Array([...head, ...value.subarray(head.length - offset, HEADER_BYTES - offset)]);
          if (head.length === HEADER_BYTES) tell(head);
        }
        offset += value.length;
        got += value.length;
        received += value.length;
        // one message per percent is plenty
        const percent = Math.floor((received / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received, total: model.bytes });
        }
      }
    } catch (error) {
      received -= got;  // counted again when the part comes again
      throw error;
    }
  };
  const connection = async () => {
    while (next < parts) {
      const part = next++;
      for (let attempt = 0; ; attempt++) {
        try {
          await fetchOnce(part);
          break;
        } catch (error) {
          if (signal.aborted || error.final || attempt === 2) {
            throw error;
          }
          console.warn(`part ${part} of ${model.checkpoint} broke off (${error.message ?? error}): fetched again`);
          await forgetPart(partUrl(part), model);  // it may have come from the cache: the next try is the network's
        }
      }
    }
  };
  // The download runs while Pyodide loads, so it usually ends long before into() can write anything into Python.
  // Its own seconds are the time until the last byte arrived, not the time until the waiting was over as well.
  const source = { overlapped: true };
  const finished = Promise.all(Array.from({ length: Math.min(CONNECTIONS, parts) }, connection))
    .then(() => { source.seconds = since(started); });
  // a load that is cancelled while Pyodide still loads never gets to into(): that is no unhandled rejection
  finished.catch(() => {});
  // a download that fails before the header came fails the wait for it
  source.header = Promise.race([header, finished.then(() => head)]);
  source.header.catch(() => {});
  // write(offset, chunk) receives everything queued so far, and every later chunk
  source.into = async (write) => {
    sink = write;
    queue.splice(0).forEach(([offset, chunk]) => write(offset, chunk));
    await finished;
    if (received !== model.bytes) {
      throw new Error(`${model.checkpoint}: got ${received} bytes instead of ${model.bytes}`);
    }
  };
  return source;
}

// The same for a file of the visitor's own disk: read in chunks straight into the Python buffer, never as a whole.
function readFile(model, signal, load) {
  // a file is read only once there is somewhere to put it, so these seconds begin here and not at the choice
  const source = {
    async into(write) {
      const started = performance.now();
      const reader = model.file.stream().getReader();
      let reported = -1;
      for (let offset = 0; ;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (signal.aborted) {
          reader.cancel();
          signal.throwIfAborted();
        }
        write(offset, value);
        await weightsRoom();  // (T156)
        offset += value.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }
      source.seconds = since(started);
    },
  };
  return source;
}

// A llama2.c checkpoint at a URL (?checkpoint=&tokenizer=): range requests in parallel, written where they belong
function readUrl(model, signal, load) {
  const source = {
    async into(write) {
      const started = performance.now();
      let offset = 0, reported = -1;
      await inOrder(model.url.checkpoint, 0, model.bytes, (bytes) => {
        write(offset, bytes);
        offset += bytes.length;
        const percent = Math.floor((offset / model.bytes) * 100);
        if (percent !== reported) {
          reported = percent;
          postMessage({ type: "progress", load, received: offset, total: model.bytes });
        }
      }, signal);
      source.seconds = since(started);
    },
  };
  return source;
}

// The header of the legacy format: 7 ints, the shape of the model. T115: they also say what the forward pass puts
// after the checkpoint (forward.js's footprint()), which the memory is chosen by.
const HEADER_BYTES = 28;
const headerInts = (bytes) => [...new Int32Array(bytes.slice(0, HEADER_BYTES).buffer)];

// The legacy format carries no metadata, but its header fixes the size of a float32, a float16 and an int8 file.
// A file that is none of them is refused before it is read, and so is a tokenizer.bin of another vocabulary.
async function localOptions(model, vocabulary, head) {
  const header = pyodide.toPy(headerInts(head));
  const pieces = pyodide.toPy(vocabulary);
  try {
    // what the file cannot say and the settings may: a Qwen2 has biases, a GPT-2 or GPT-NeoX another set of tensors,
    // a Qwen3 the norms of q and k and maybe heads of another size than dim / heads (T124): the form of the options
    const dtype = llama2_numpy.checkpoint_dtype(header, model.bytes, model.options ?? {});
    llama2_numpy.check_tokenizer(pieces, header);
    return { ...model.options, dtype };
  } finally {
    header.destroy();
    pieces.destroy();
  }
}

let pyodide, llama2_numpy, llama2_convert, llama, kernels;
// the models kept from earlier conversions (kept.js, T99), imported when the first conversion comes
let keptModule;
// T93: the forward pass in JavaScript (forward.js) and its kernels, compiled once; the memory of the model loaded now
let forwardModule, jsKernels, weightsNow;
// T93 stage 2: the kernels for a shared memory (only where the page is cross-origin isolated), what the page asked
// about the number of threads ({ fixed, remembered, hint }), and the forward pass of the model loaded now
let sharedKernels, threadsRequest, outsideNow;
// T101: the kernels for a 64-bit memory, for a model past 4 GiB ({ plain, shared }), where the browser has Memory64
let wideKernels;
// ?wide=on: a 64-bit memory for every model, to try that path on a small one (measuring, tests), as ?offline=on says
let forceWide = false;
// T148: a prompt's tokens through the layers on the GPU (gpu.js) by default, wherever this worker has WebGPU and
// forward.js can put the model there, and the GPU is faster than the CPU here (AGENTS.md's policy 9: no option).
// ?gpuTest=on, for the tests only: a fallback adapter (SwiftShader, the only WebGPU of CI) taken as a GPU, and every
// block of a prompt it can take given to it (its speed is no GPU's: the tests look at its numbers, not at its time),
// and the first right shader of the matrices taken untimed (SwiftShader timed Llama 3.2 1B's past gpu.js's 180 s)
const hasWebGpu = Boolean(self.navigator?.gpu);
let gpuForce = {};
// T156: the adapter, asked for before a model is loaded ({ fallback, limits } or null): whether a model the device
// cannot hold twice goes on the GPU alone is decided before its bytes come (forward.js's weightsPlace, gpuOnlyUnfit)
let gpuAdapter = null;
// (and its key, T148's: what the page kept of the model on this device holds only for the same, shaders.js's deviceKey)
const adapterAsked = !hasWebGpu ? Promise.resolve() : navigator.gpu.requestAdapter().then(async (adapter) => {
  if (!adapter) return;
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  gpuAdapter = { fallback: Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter),
    limits: { maxStorageBufferBindingSize, maxBufferSize } };
  try {
    const wgsl = await import(new URL(`shaders.js${self.location.search}`, import.meta.url));
    gpuAdapter.key = wgsl.deviceKey(adapter);
  } catch {
    // no key: nothing the page kept of this device holds (a model on the GPU alone is weighed again)
  }
}).catch(() => {});
// T156: the models whose GPU failed while they were on it alone, loaded again on the CPU from then on (this visit)
const cpuOnly = new Set();
const modelKey = (model) => model.hf ? `hf:${model.hf.repo}@${model.hf.revision}` : model.id ?? model.name;
// ?bench= (T45): the page measures the CPU's combinations, and no GPU starts beside them
let benchPage = false;
// what the page kept of the GPU's shaders for the model asked for ({ remembered }), as threadsRequest
let gpuRequest;
// the optimizations this session leaves out (T52): ?without=relaxed,sampler, and ?kernel=off as it always was
let disabled = [];
// what the page's own URL said, to come back to after a benchmark has tried other combinations (T77)
let pageSwitches = [];

// Only what the engine has a fallback for. A name it does not know is refused there, and the page says so.
function switchesOf(search) {
  const parameters = new URLSearchParams(search);
  const names = (parameters.get("without") ?? "").split(",").map((name) => name.trim()).filter(Boolean);
  if (parameters.get("kernel") === "off" && !names.includes("kernels")) {
    names.push("kernels");
  }
  return names;
}
// init() as a promise: every load waits for it, also the one that replaces the first
let initialized;
// the AbortController of the load that is going on, and a promise that settles once it has cleaned up
let loading, unloaded = Promise.resolve();

// how long the load took, in seconds: Pyodide once per session, the other two per model. The download of a model
// of this site runs while Pyodide loads and usually ends first, so its seconds are counted until the last byte
// arrives (not until the bytes reach Python, which has to wait for Pyodide). The page says that the two overlap
// instead of adding them up, but only for the model that was loaded while Pyodide was still coming.
const loadSeconds = {};
// when Pyodide became usable, to tell that first model from the ones chosen afterwards
let pyodideAt = 0;
const since = (started) => (performance.now() - started) / 1000;

// T118: whether Pyodide's loading still moves. Its large files (pyodide.asm.wasm 3.4 MB, the standard library 2.5 MB,
// NumPy's wheel 2.9 MB) come by this worker's fetch, which, while Pyodide loads, hands out responses whose bodies are
// counted as they arrive; a file that is finished (the two imports too) counts as well. A step is given up when
// nothing has arrived for QUIET_SECONDS, not when it takes long: on a line of 128 kbps to 1 Mbps (a phone's plan past
// its limit) the 9 MB took longer than T113's limits of 60 to 90 seconds, and a sound load fell back to no service
// worker, then timed out again. Nothing arriving for this long is a stop, not a slow line (128 kbps is 16 kB a second).
const QUIET_SECONDS = 30;
function watchArrivals() {
  const watch = { arrived: 0 };
  const plain = self.fetch;
  self.fetch = async (...args) => {
    const res = await plain(...args);
    watch.arrived += 1;
    if (!res.body || [101, 204, 205, 304].includes(res.status)) {
      return res;
    }
    const counted = res.body.pipeThrough(new TransformStream({
      transform(chunk, out) {
        watch.arrived += chunk.byteLength;
        out.enqueue(chunk);
      },
    }));
    return new Response(counted, { status: res.status, statusText: res.statusText, headers: res.headers });
  };
  let observer;
  try {
    observer = new PerformanceObserver((list) => { watch.arrived += list.getEntries().length; });
    observer.observe({ type: "resource" });
  } catch {
    observer = undefined;  // no PerformanceObserver here: the fetches alone
  }
  watch.stop = () => {
    self.fetch = plain;
    observer?.disconnect();
  };
  /** { promise, cancel }: the promise settles once nothing has arrived for seconds */
  watch.quiet = (seconds) => {
    let timer;
    const promise = new Promise((resolve) => {
      let seen = -1, since = 0;
      timer = setInterval(() => {
        const now = performance.now();
        if (watch.arrived !== seen) {
          [seen, since] = [watch.arrived, now];
        } else if (now - since >= seconds * 1000) {
          clearInterval(timer);
          resolve();
        }
      }, 1000);
    });
    return { promise, cancel: () => clearInterval(timer) };
  };
  return watch;
}

async function init(search) {
  const started = performance.now();
  const asked = new URLSearchParams(search);
  const parts = Number(asked.get("hfParts")), connections = Number(asked.get("hfConnections"));
  if (parts >= 1 && parts <= 64) hfPartBytes = Math.round(parts * 1024 * 1024);
  if (connections >= 1 && connections <= 32) hfConnections = Math.floor(connections);
  forceWide = asked.get("wide") === "on";
  // (T156: ?gpuTest=only, the tests' too: a model the GPU can take on the GPU alone, whatever its size)
  gpuForce = ["on", "only"].includes(asked.get("gpuTest")) ? { fallback: true, always: true, quick: true, only: asked.get("gpuTest") === "only" } : {};
  benchPage = asked.has("bench");
  const version = await resolvePyodideVersion(search);
  const base = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  // Each step says its name, and ends in an error rather than never: loadPyodide() does not fail when a fetch of
  // its files fails, it waits for ever (AGENTS.md), and a phone that stopped at "Loading Pyodide" said nothing else.
  // It ends when nothing has arrived for QUIET_SECONDS (T118), however long it takes while bytes keep coming.
  const watch = watchArrivals();
  const step = async (name, promise) => {
    postMessage({ type: "status", text: `Loading Pyodide ${version}: ${name}...` });
    const quiet = watch.quiet(QUIET_SECONDS);
    const stalled = quiet.promise.then(() => {
      const error = new Error(`Pyodide ${version}: "${name}" got nothing from the network for ${QUIET_SECONDS} seconds`);
      error.pyodide = true;  // the page may try again without the service worker (isolation made this hang on iOS)
      throw error;
    });
    try {
      return await Promise.race([promise, stalled]);
    } finally {
      quiet.cancel();
    }
  };
  try {
    // while Pyodide starts, not after: loadPackage("numpy") below finds the wheel in the HTTP cache
    const numpy = prefetchNumpy(base);
    const { loadPyodide } = await step("the loader", import(`${base}pyodide.mjs`));
    pyodide = await step("the runtime", loadPyodide());
    // a prefetch that is still running would otherwise be raced by loadPackage, and the wheel fetched twice. One
    // that stopped is not waited for past a quiet spell (T118): loadPackage then fetches the wheel itself
    const quiet = watch.quiet(QUIET_SECONDS);
    await Promise.race([numpy, quiet.promise]);
    quiet.cancel();
    await step("NumPy", pyodide.loadPackage("numpy"));
  } finally {
    watch.stop();
  }

  // with the ?v=<build> of this worker, so that both always come from the same deployment
  const res = await fetch(new URL(`llama2_numpy.py${self.location.search}`, import.meta.url));
  if (!res.ok) {
    throw new Error(`Could not fetch llama2_numpy.py: ${res.status}`);
  }
  pyodide.FS.writeFile("llama2_numpy.py", await res.text());
  llama2_numpy = pyodide.pyimport("llama2_numpy");

  // The WASM SIMD kernels (kernels/*.ts), which llama2_numpy.py loads with ctypes. They are optional: without
  // them, or with ?kernel=off, NumPy does the math, several times slower. They are read even with ?kernel=off: the
  // switches say what is used (disabled), and the benchmark's rounds with the kernels need them there (T119: its
  // "everything" round sampled on NumPy after the page's switch had turned them off, and said nothing).
  disabled = pageSwitches = switchesOf(search);
  for (const name of ["simdkernel.so", "simdkernel_relaxed.wasmlib"]) {
    const kernel = await fetch(new URL(`${name}${self.location.search}`, import.meta.url)).catch(() => undefined);
    if (kernel?.ok) {
      pyodide.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(await kernel.arrayBuffer()));
      kernels = "/home/pyodide/simdkernel.so";
    }
  }
  // T93: the forward pass runs in forward.js, on the plain build of the same kernels (simdkernel.so stays for the
  // sampling, which works on Python's logits). Without them (no WebAssembly SIMD) the engine runs NumPy.
  try {
    forwardModule = await import(new URL(`forward.js${self.location.search}`, import.meta.url));
    // one build of the kernels: simdkernel_<kind>.wasm and simdkernel_relaxed_<kind>.wasm, or null
    const build = async (kind, wide = false) => {
      const [plain, relaxed] = await Promise.all([`simdkernel_${kind}.wasm`, `simdkernel_relaxed_${kind}.wasm`].map((name) =>
        fetch(new URL(`${name}${self.location.search}`, import.meta.url)).then((res) => (res.ok ? res.arrayBuffer() : null)).catch(() => null)));
      return plain ? forwardModule.compileKernels(plain, relaxed, wide) : null;
    };
    jsKernels = await build("plain");
    if (jsKernels && self.crossOriginIsolated) sharedKernels = await build("shared");
    // T101: on their own, so that a browser that says it makes 64-bit memories and then cannot compile their kernels
    // (Playwright's WebKit did, 2026-09-25) keeps the 32-bit ones: in the same try it lost every kernel and ran NumPy
    if (jsKernels && forwardModule.memory64()) {
      try {
        wideKernels = { plain: await build("plain64", true), shared: self.crossOriginIsolated ? await build("shared64", true) : null };
      } catch {
        wideKernels = undefined;
      }
    }
  } catch {
    jsKernels = null;
  }
  loadSeconds.pyodide = since(started);
  pyodideAt = performance.now();
}

// a Python bytearray that JavaScript fills in place
function pythonBuffer(size) {
  const buffer = pyodide.globals.get("bytearray")(size);
  const write = (offset, chunk) => {
    // the view is taken anew every time: it dies when the WebAssembly memory grows
    const view = buffer.getBuffer("u8");
    view.data.set(chunk, offset);
    view.release();
  };
  return { buffer, write };
}

// Where the weights of a model go (T93): the WebAssembly memory of forward.js, where the forward pass runs, or,
// without the kernels (?without=kernels, or no WebAssembly SIMD), a Python bytearray for the NumPy engine.
// write(offset, bytes) fills it, slice(begin, end) copies a stretch out (to keep a conversion), llama() makes the
// engine on it, destroy() lets go of the Python buffer (a memory of forward.js goes with the engine).
// a software thread of forward.js (stage 2): a module worker of its own, ready once its kernels are warm
const spawnThread = (data) => new Promise((resolve, reject) => {
  const worker = new Worker(new URL(`helper.js${self.location.search}`, import.meta.url), { type: "module" });
  worker.onmessage = () => resolve({ terminate: () => worker.terminate() });
  worker.onerror = (event) => reject(new Error(event.message ?? "a software thread did not start"));
  worker.postMessage(data);
});

// T135: the GPU's worker of forward.js, a module worker of its own; forward.js starts it
const openGpu = () => new Worker(new URL(`gpu.js${self.location.search}`, import.meta.url), { type: "module" });

// T96: one memory of forward.js, kept from model to model. A browser reserves address space for every WebAssembly
// memory, shared or not and whatever its maximum, and Chromium refused the third one of a page: the benchmark's
// first round, or a visitor's second change of model, then found no memory at all. So a memory is kept and grown
// as long as the next model fits under its maximum, and made anew only for a larger one. The maximum comes from the
// model (weightsMemory: what it needs and a gigabyte): asking for 4 GB up front left a phone no room for Pyodide's
// own memory, and "Loading Pyodide" never ended (2026-09-25).
// "Fits" is the checkpoint and what the forward pass puts after it (after: forward.js's footprint(), T115). Pythia
// 1B's checkpoint (1.1 GB) fitted the memory made for tiny-lm (1.2 GB) and its corrections did not ("Maximum memory
// size exceeded"; opened first, it ran; the review of T96). Where a new memory would get no more room than this one
// (the browser gave less than it asked for, or it is not shared and grows as far as any would), the checkpoint
// fitting is enough. shared is what was asked for, not what the browser gave: a device without shared memories
// made one for every model.
let weightsPool;
function pooledWeights(size, after, shared, wide) {
  const pages = (bytes) => Math.ceil(bytes / 65536);
  const needs = (base) => pages(base + size + (weightsPool.limited ? 0 : after)) + 1;
  const fits = weightsPool && weightsPool.asked === shared && weightsPool.wide === wide && needs(weightsPool.base) <= weightsPool.maximum;
  if (!fits) {
    // nothing may hold the old memory while the new one is made (T96: Chromium refused a page's third)
    weightsPool = weightsNow = undefined;
    let memory, base;
    if (shared) {
      try {
        ({ memory, base } = forwardModule.weightsMemory(size, { shared: true, wide, after }));
      } catch {
        memory = undefined;  // no shared memory here: one thread
      }
    }
    if (!memory) ({ memory, base } = forwardModule.weightsMemory(size, { wide }));
    const isShared = shared && memory.buffer instanceof SharedArrayBuffer;
    // a memory without a maximum (not shared) grows as far as the browser allows: 4 GB of pages, 16 GB when wide
    const most = wide ? 262144 : 65536;
    weightsPool = { memory, base, wide, asked: shared, shared: isShared, maximum: isShared ? memory.maximum ?? most : most,
                    limited: !isShared || Boolean(memory.limited) };
  }
  const { memory, base } = weightsPool;
  const more = pages(base + size) + 1 - memory.buffer.byteLength / 65536;
  if (more > 0) forwardModule.growMemory(memory, more, wide);
  return weightsPool;
}

// T115: what the forward pass of a checkpoint of size bytes puts after it, at most (forward.js's footprint()): from
// its header (the 7 ints) and the options it is loaded with (their form: head_dim where a head is not dim / heads,
// T124: the keys and values of a Qwen3 0.6B are twice what the header says), on a shared memory (an int8 model's keys
// and values may be float16 there, T110; forward.js's keysInHalf says whether they are, T160) or not.
// What footprint() takes (the worker asks keysInHalf the same).
function forwardOptions(options, shared) {
  const { dtype = "float32" } = options;
  const int8 = !disabled.includes("int8"), quantized = dtype === "int8" || dtype === "int6";
  return {
    ...options, dtype, int8, relaxed: Boolean(jsKernels?.relaxed) && !disabled.includes("relaxed"),
    halfKV: shared && quantized && int8 && !disabled.includes("kv16"),
    kvStart: llama2_numpy.KV_START, outliers: llama2_numpy.OUTLIER_CHANNELS, gpu: hasWebGpu,
  };
}
const afterCheckpoint = (header, size, options, shared) => forwardModule.footprint(header, size, forwardOptions(options, shared));
// the page cross-origin isolated (stage 3), shared memories to be had, and not ?threads=1: the memory is shared
const sharedWanted = () => Boolean(sharedKernels && self.crossOriginIsolated && threadsRequest?.fixed !== 1);

// T115: the bits of a model converted with none asked for (weightsFor() in src/models.js asks for six bits where the
// device says it has too little memory): int8 unless its forward pass does not fit a 32-bit memory and this browser
// has no 64-bit one (T133), then six bits (T98: 7/9 of int8's memory, and about half as fast). The converter calls
// this once it knows the header: the size of either (sizes) and what the forward pass puts after them depend on it.
// form: what else sizes the forward pass (llama2_numpy.FORM, llama2_convert.Stream.form).
function automaticBits(header, form, sizes) {
  const ints = header.toJs(), int8 = sizes.toJs({ dict_converter: Object.fromEntries }).int8;
  const given = form.toJs({ dict_converter: Object.fromEntries });
  header.destroy();
  sizes.destroy();
  form.destroy();
  if (!forwardModule) return "int8";  // no forward.js (no WebAssembly SIMD): NumPy widens every weight anyway
  const shared = sharedWanted();
  return forwardModule.automaticDtype(int8, afterCheckpoint(ints, int8, { ...given, dtype: "int8" }, shared),
    Boolean(wideKernels?.plain));
}

// header: the checkpoint's 7 ints, options: what it is loaded with (its dtype and arch): what the forward pass puts
// after the checkpoint follows from them (T115). T156, keep: where a conversion is kept as it comes (kept.js's
// keeper()), for a model on the GPU alone, whose weights nothing holds whole to keep afterwards
function weightsBuffer(size, header, options, keep) {
  if (jsKernels && !disabled.includes("kernels")) {
    // a shared memory where the page is cross-origin isolated (stage 3), unless ?threads=1; else one thread
    const wanted = sharedWanted();
    // T101: a model past 4 GiB with its forward pass goes on a 64-bit memory (about a tenth slower: only when it has
    // to). Where a shared one is refused after all, the plain one keeps float32 keys and values, twice what was
    // counted: a model at the edge then runs out of memory near the end of its context (T115)
    const after = afterCheckpoint(header, size, options, wanted);
    const halfKeys = forwardModule.keysInHalf(header, size, forwardOptions(options, wanted));  // T160: what after counts
    // T148: the layers on the GPU are a second copy of them, in the same memory where the GPU is a phone's or an
    // Apple's: both, with the rest of this model, within half of what the device says it has (as src/models.js's
    // weightsFor asks for six bits past half). T156: a model that does not fit so goes on the GPU alone where it can
    // (the owner's B, 2026-09-27: forward.js's weightsPlace; the double copy held to 6.5 GiB where the device says 8),
    // decided before any memory is made for it (a memory of the whole checkpoint would be one too many, T96)
    const deviceMemory = self.navigator?.deviceMemory ?? 4;
    // T205: a browser that does not say (Safari, Firefox) keeps a generation's steps (the classifier and the embedding
    // on the GPU as well) on the CPU: nor does it put a model on the GPU alone, whose steps are there
    const memoryUnsaid = self.navigator?.deviceMemory === undefined;
    const onGpu = gpuOnlyWeightsFor(size, header, options, after, deviceMemory);
    if (onGpu.mode === "gpu") return gpuOnlyBuffer(size, header, options, onGpu, keep);
    const gpuRoom = onGpu.gpuRoom;
    const wide = forceWide || forwardModule.needsWide(size, after);
    if (wide && !wideKernels?.plain) {
      throw new Error("This model needs more than 4 GB of memory, which this browser cannot give a web page (no 64-bit " +
        "WebAssembly memory: Safari has none yet). Chrome and Firefox can.");
    }
    const { memory, base, shared } = pooledWeights(size, after, wanted && (!wide || Boolean(wideKernels.shared)), wide);
    const kernels = wide ? (shared ? wideKernels.shared : wideKernels.plain) : (shared ? sharedKernels : jsKernels);
    const spawn = shared ? spawnThread : undefined;
    weightsNow = memory;
    return {
      write: (offset, chunk) => new Uint8Array(memory.buffer, base + offset, chunk.length).set(chunk),
      slice: (begin, end) => new Uint8Array(memory.buffer, base + begin, end - begin).slice(),
      llama: (tokenizer, options) => {
        // (not on the benchmark's page, ?bench=, T45: it times the CPU's combinations, which a GPU starting beside
        // them would slow down with its upload and compilation; its first load is one of them)
        // (nor under the rounds of /benchmark/'s model section, T184: the same; its page path, timed on the first load
        // before the rounds, has the GPU, and T205: the rounds' loads wait for that GPU to let go of its device)
        outsideNow = forwardModule.external({ memory, base, size, kernels, spawn, gpu: hasWebGpu && !benchPage && !benching ? openGpu : undefined, gpuRoom, memoryUnsaid,
          gpuRemembered: gpuRequest?.remembered, gpuForce, halfKeys });
        return llama2_numpy.Llama.callKwargs(null, tokenizer, { ...options, external: outsideNow });
      },
      destroy() {},
    };
  }
  weightsNow = undefined;
  outsideNow = undefined;
  const { buffer, write } = pythonBuffer(size);
  return {
    write,
    slice(begin, end) {
      const view = buffer.getBuffer("u8");
      const copy = view.data.slice(begin, end);
      view.release();
      return copy;
    },
    llama: (tokenizer, options) => llama2_numpy.Llama.callKwargs(buffer, tokenizer, options),
    destroy: () => buffer.destroy(),
  };
}

// T156: what a model on the GPU alone needs of this page and device, whatever the model (gpuOnlyUnfit has the rest): a
// memory the device says (T205), the GPU's worker (not the benchmark's rounds), a shared memory and the 64-bit kernels,
// the int8 kernels, an adapter that is no fallback (but for the tests), and no failure of the GPU under this model before
// (cpuOnly). convert() asks it too before it opens a file to keep a conversion in as it comes (the review of T156)
// (the owner, 2026-09-27: and no verdict the page kept that the CPU is faster here: then the CPU at once)
const gpuOnlyPossible = () => gpuOnlyAllowed() && !aloneKept();
const gpuOnlyAllowed = () => hasWebGpu && self.navigator?.deviceMemory !== undefined && !benchPage && !benching && sharedWanted() &&
  Boolean(wideKernels?.shared) && !disabled.includes("int8") && !cpuOnly.has(loadingKey) &&
  Boolean(gpuAdapter) && (!gpuAdapter.fallback || Boolean(gpuForce.fallback));
// T156: whether the page kept that the CPU was faster here than this model on the GPU alone (forward.js's aloneHolds)
const aloneKept = () => Boolean(gpuAdapter) && forwardModule.aloneHolds(gpuRequest?.remembered?.alone, gpuAdapter.key, gpuRequest?.cpu);
// T156: where a model goes ({ mode: "both" | "gpu" | "cpu", gpuRoom }, forward.js's weightsPlace): the GPU alone only
// for a Llama of int8 the GPU's steps take (gpuOnlyUnfit), where the page and the device may (gpuOnlyPossible)
function gpuOnlyWeightsFor(size, header, options, after, deviceMemory) {
  const { dtype = "float32" } = options, form = { arch: options.arch, bias: options.bias, qk_norm: options.qk_norm, head_dim: options.head_dim };
  const cpu = size + after, gpu = forwardModule.gpuBytes(header, form);
  const fit = gpuOnlyAllowed() && !forwardModule.gpuOnlyUnfit(header, dtype, form, gpuAdapter, gpuForce);
  const eligible = fit && !aloneKept();
  // (the second review of T156: the status line then says only the memory's reason where the layers do not fit beside
  // the CPU's copy; the console says the kept verdict, and what asks again)
  if (fit && !eligible) {
    console.info("gpu: the CPU as /benchmark/ measured it was faster here than this model on the GPU alone, as the page kept it: " +
      "on the CPU (a new run of /benchmark/'s CPU section, another browser version or new shaders weigh the two again)");
  }
  if (!eligible) return forwardModule.weightsPlace({ cpu, gpu, deviceMemory });
  const tensors = placesOf(header, dtype, form), stored = size - forwardModule.gpuHoles(tensors).reduce((sum, [a, b]) => sum + b - a, 0);
  const gpuOnly = stored + afterCheckpoint(header, stored, { ...options, direct: true }, true);
  return { ...forwardModule.weightsPlace({ cpu, gpuOnly, gpu, deviceMemory, eligible, forced: gpuForce.only }), tensors, stored, gpuOnly };
}
// llama2_numpy.external_tensors(): where every tensor of the checkpoint is, from its header alone
function placesOf(header, dtype, form) {
  const proxy = llama2_numpy.external_tensors(header, dtype, form);
  try {
    return proxy.toJs({ dict_converter: Object.fromEntries });
  } finally {
    proxy.destroy();
  }
}
// T156: the weights of a model on the GPU alone: the layers' matrices to the GPU's worker as they come (forward.js's
// gpuOnlyWeights), the rest into a memory of their size, and every byte to keep's file where the conversion is kept
let gpuOnlyNow;
function gpuOnlyBuffer(size, header, options, { tensors, stored, gpuOnly }, keep) {
  const after = gpuOnly - stored, wide = forwardModule.needsWide(stored, after);
  const { memory, base, shared } = pooledWeights(stored, after, true, wide);
  if (!shared) throw new Error("This browser gave no shared memory for a model on the GPU alone.");
  const worker = openGpu();
  // (forward.js listens from start() on. Here only "ended", beside it: T205, the next load after the GPU's worker let
  // go of its device and its buffers, which it makes for every layer as it opens, before a byte comes)
  let ended;
  const ending = new Promise((resolve) => { ended = resolve; });
  worker.addEventListener("message", ({ data }) => data?.type === "ended" && ended());
  worker.addEventListener("error", () => ended());
  // the GPU's worker stopped, and once it said "ended" (or after GPU_END_MS: terminated), resolved; as release() does
  const end = () => {
    worker.postMessage({ type: "stop" });
    let timer;
    const late = new Promise((resolve) => { timer = setTimeout(resolve, forwardModule.GPU_END_MS); });
    return Promise.race([ending, late]).then(() => {
      clearTimeout(timer);
      worker.terminate();
    });
  };
  const weights = forwardModule.gpuOnlyWeights({ memory, base, size, tensors, worker });
  worker.postMessage({ type: "open", plan: forwardModule.gpuOnlyPlan(header, tensors, gpuForce, gpuRequest?.remembered), flow: weights.flow });
  // (T156: the checkpoint's size, its layers' multiply-adds, /benchmark/'s CPU reading and the page's use, for the
  // estimate the GPU is held against: forward.js's aloneVerdict)
  const direct = gpuOnlyNow = { worker, lost: null, stored, size, place: weights.place,
    layerWeights: forwardModule.layerWeightsOf(header, options), cpu: gpuRequest?.cpu, usage: gpuRequest?.usage ?? forwardModule.USAGE_UNKNOWN,
    room: weights.room, drained: weights.drained, onLost: (why) => { direct.lost = why; }, end };
  weightsNow = memory;
  const kernels = wide ? wideKernels.shared : sharedKernels;
  let built = false;
  return {
    direct,
    write(offset, chunk) {
      weights.write(offset, chunk);
      keep?.write(offset, chunk);
    },
    room: weights.room,
    drained: weights.drained,  // every byte of the layers on the GPU (or the GPU failed: start() says so)
    llama: (tokenizer, engineOptions) => {
      outsideNow = forwardModule.external({ memory, base, size, kernels, spawn: spawnThread, gpu: () => worker, gpuForce, halfKeys: true, direct,
        gpuRemembered: gpuRequest?.remembered });
      const made = llama2_numpy.Llama.callKwargs(null, tokenizer, { ...engineOptions, external: outsideNow });
      built = true;
      return made;
    },
    // (the review of T156: a load cancelled or failed before its engine was built, or another try of the converter's
    // (sink.open() again), let go of the GPU's worker: nothing else would, and it held the device and a buffer for
    // every layer's matrices, 4.1 GB for Llama 3.2 3B, for the rest of the visit. The next load waits for it)
    destroy() {
      if (!built) gpuOnlyEnding = end();
    },
  };
}
// the GPU's worker of a model on the GPU alone let go before its engine was built (destroy() above): the next load
// waits for it, as release() waits for a built one's (T205)
let gpuOnlyEnding = Promise.resolve();
// T156: where the loops that write the weights can wait for the GPU's worker of a model on the GPU alone (room), and
// before its engine is built (drained)
const weightsRoom = () => gpuOnlyNow?.room?.();
const weightsDrained = () => gpuOnlyNow?.drained?.();
// T156: after a model on the GPU alone is built: its GPU ready (true), or failed (false: the worker let go of it, and
// the model goes on the CPU from now on this visit)
async function gpuOnlyReady(model, id) {
  const direct = gpuOnlyNow;
  if (!direct || outsideNow?.engine === undefined) return true;
  await outsideNow.engine.gpu;
  if (!direct.lost) return true;
  console.warn(`gpu: ${direct.lost}: the model was on the GPU alone, and is loaded again on the CPU`);
  cpuOnly.add(modelKey(model));
  // (the CPU was faster: the page keeps it for this model and device once the load on the CPU is ready, and the next
  // visit loads on the CPU at once. The load's id: the page reads no word of a load another choice cancelled, the
  // second review of T156: a verdict that came late was kept for the model chosen since)
  if (direct.verdict) postMessage({ type: "gpu-alone", load: id, alone: direct.verdict });
  // (the GPU's worker let go of its device: stopped by forward.js where it started it, and here where it did not, as a
  // model whose GPU forward.js gave up as the engine was built, before start(): its release() has no "ended" to wait
  // for then. The review of T156: the load on the CPU begins after it, T205)
  await direct.end();
  gpuOnlyNow = undefined;
  return false;
}
// the key of the model being loaded (cpuOnly)
let loadingKey;

// T93: where the converter writes the checkpoint, piece by piece, straight into where the engine will read it (a
// Python buffer on the way would stay: Pyodide's memory never shrinks). sink is what the converter calls
// (llama2_convert.Writer), weights what sink.open() made (weightsBuffer), bytes its size; release() lets go of it once
// (T145: a conversion that failed after sink.open() left a Python buffer of the model's size behind with
// ?without=kernels). Out of convert() so that tests/worker-sink-check.mjs can follow sink.open() to footprint().
function checkpointSink(keep) {
  const into = {
    weights: undefined, bytes: 0,
    release() {
      into.weights?.destroy();
      into.weights = undefined;
    },
  };
  into.sink = {
    // form: what lays out the checkpoint and sizes the forward pass besides the header (llama2_numpy.FORM, T115, T144)
    open(bytes, header, dtype, form) {
      into.release();  // an earlier try (another tokenizer) that got this far
      into.weights = weightsBuffer(bytes, header.toJs(), { dtype, ...form.toJs({ dict_converter: Object.fromEntries }) }, keep);
      header.destroy();
      form.destroy();
      into.bytes = bytes;
    },
    write(offset, array) {
      const view = array.getBuffer("u8");
      into.weights.write(offset, view.data);
      view.release();
    },
  };
  return into;
}

// Every await in here may end with the AbortError of signal: a newer load has taken over, and this one must
// leave nothing behind, least of all a Python buffer as large as its model.
// A Hugging Face model, from the visitor's disk ({weights, config, tokenizer} are Files) or from huggingface.co
// ({repo, revision, weights, config, tokenizer} are names): model.safetensors arrives in the order of the file, a few
// megabytes at a time, and the Python code that builds the models of this site converts every tensor as it comes and
// writes it to its place in a buffer of the final size. Reading in the order of the output instead would mean
// hundreds of range requests, and each one takes a second.
// 16 MiB over 6 connections (T107, measured in CI against huggingface.co): parts of 8 MiB took 1.36 times as long
// for Qwen2.5 0.5B, of 4 MiB 2.8 times; more connections gained 6% at most. But the first bytes then come late on a
// slow line, and a phone has less room for what waits in the queue (two parts per connection: 192 MB at 16 MiB),
// so the first part is 8 MiB wherever the size is not fixed by the URL, and the rest follow what it measured
// (the owner's ask, 2026-09-25): 16 MiB where that part came in at 4 MB/s or more and the device says nothing of a
// small memory, 8 MiB otherwise.
const HF_PART_BYTES = 16 * 1024 * 1024;
const HF_SMALL_PART_BYTES = 8 * 1024 * 1024;
const HF_FAST_BYTES_PER_SECOND = 4e6;
const HF_CONNECTIONS = 6;
const HF_HEADER_BYTES = 512 * 1024;  // the JSON header of a safetensors file is a few dozen kilobytes
// T107: ?hfParts=<MiB>&hfConnections=<N> fix the two, to measure; the page offers no way to them
let hfPartBytes = 0, hfConnections = HF_CONNECTIONS;  // 0: not fixed, decided per file from its first part

// The size of a file, for the few places that need it (the whole of a model: how many parts to ask for). A range
// response says it in Content-Range, but that header is not one CORS shows by default: huggingface.co exposes it by
// name, its CDN by "*", and a browser that does not honour "*" (WebKit; T112) sees none and the fetch never began
// ("The file ended before all of its tensors were read"). Content-Length of a HEAD is always shown.
async function fileSize(url, signal) {
  const res = await fetch(url, { method: "HEAD", signal });
  const length = Number(res.headers.get("Content-Length"));
  if (!res.ok || !Number.isFinite(length) || length <= 0) {
    throw new Error(`Could not learn the size of ${url}: ${res.status}`);
  }
  return length;
}
// the size a range response reported, or the file's size asked for separately when it did not
const sized = async (url, result, signal) => (Number.isFinite(result.total) && result.total > 0 ? result : { ...result, total: await fileSize(url, signal) });

// Bytes from..to of a response's body, taken as they stream past and no further: the rest is cancelled.
// arriving(count) is told of every stretch kept. Fewer bytes than asked for when the body ends first.
async function bodyBetween(res, from, to, arriving) {
  const bytes = new Uint8Array(to - from);
  let at = 0, kept = 0;  // at: where in the body the next chunk begins
  const reader = res.body.getReader();
  try {
    while (at < to) {
      const { done, value } = await reader.read();
      if (done) break;
      const start = Math.max(from - at, 0), stop = Math.min(to - at, value.length);
      if (stop > start) {
        bytes.set(value.subarray(start, stop), at + start - from);
        kept += stop - start;
        arriving?.(stop - start);
      }
      at += value.length;
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  return bytes.subarray(0, kept);
}

// arriving(count): told of every stretch of the body as it comes, for a progress line before a whole part is in
// A fetch that huggingface.co refused, in the visitor's words (T119). The status alone does not tell: a gated
// repository and one that is not there (or private) both answer 401; X-Error-Code (which CORS shows) says which.
// Such an answer is the same the next time: it is not asked again (status says so).
function refused(url, res) {
  const [, repository, revision, file] = /^https:\/\/huggingface\.co\/(.+?)\/resolve\/([^/]+)\/(.+)$/.exec(url) ?? [];
  const code = res.headers.get("X-Error-Code");
  const error = new Error(!repository ? `Could not fetch ${url}: ${res.status}`
    : code === "GatedRepo" ? `${repository} is gated on huggingface.co: its owner lets it be fetched only after a login and an accepted license, which this page cannot do. A copy of it that someone else published openly may work.`
    : code === "RevisionNotFound" ? `${repository} has no revision ${revision} on huggingface.co.`
    : code === "EntryNotFound" ? `${repository} has no ${file} at ${revision} on huggingface.co` +
      // a commit that does not exist is answered so too (only a branch or tag that does not is RevisionNotFound)
      (/^[0-9a-f]{40}$/.test(revision) ? `, or has no commit ${revision}.` : ".")
    : res.status === 401 || res.status === 404 ? `huggingface.co has no public repository ${repository}: check its name.`
    : `huggingface.co answered ${res.status} for ${file} of ${repository}.`);
  error.status = res.status;
  return error;
}

async function fetchRange(url, begin, end, signal, arriving) {
  for (let attempt = 0; ; attempt++) {
    // the bytes of a body that broke come again with the next try: they are taken back (the review of T119)
    let counted = 0;
    const counting = arriving && ((count) => { counted += count; arriving(count); });
    try {
      const res = await fetch(url, { headers: { Range: `bytes=${begin}-${end - 1}` }, signal });
      if (res.status !== 206 && res.status !== 200) {
        throw refused(url, res);
      }
      // 200: the server ignored the range and sends the whole file (T112: a browser whose stack does this is one to
      // know about). What was asked for is cut out as it streams past, and the rest is never fetched: taking the
      // whole file for every part fetched SmolLM2's 145 MB ten times over, and held it whole for each (the review)
      const whole = res.status === 200;
      if (whole) {
        console.warn(`${url} answered a range request with the whole file`);
      }
      const bytes = await bodyBetween(res, whole ? begin : 0, whole ? end : end - begin, counting);
      const total = Number(whole ? res.headers.get("Content-Length") : (res.headers.get("Content-Range") ?? "").split("/")[1]);
      return { bytes, total };
    } catch (error) {
      if (counted) arriving(-counted);
      if (signal.aborted || attempt === 2 || (error.status >= 400 && error.status < 500)) {
        throw error;
      }
    }
  }
}

// feed(bytes) gets the file from position start to its end, in order, although the parts arrive as they like.
// The parts are cut as they are asked for: the first small, the rest by what the first one measured (see above).
// arriving(bytes): how much of the file is in so far, told as it comes (the page shows it until the conversion of
// the first part gives it percentages: on a slow line the first part alone takes a while, and a line that says
// nothing looks stuck).
async function inOrder(url, start, size, feed, signal, arriving = () => {}) {
  const small = navigator.deviceMemory !== undefined && navigator.deviceMemory <= 4;
  let partBytes = hfPartBytes || HF_SMALL_PART_BYTES;
  const ranges = [];  // [begin, end] of every part asked for so far, in the order of the file
  const arrived = new Map();
  let scheduled = start, fed = 0, waiting = [], received = start;
  const connection = async () => {
    for (;;) {
      // no more than two parts per connection wait in memory for an earlier one
      while (ranges.length - fed >= 2 * hfConnections) {
        await new Promise((resolve) => waiting.push(resolve));
      }
      if (scheduled >= size) {
        return;
      }
      const part = ranges.length, begin = scheduled, end = Math.min(begin + partBytes, size);
      ranges.push([begin, end]);
      scheduled = end;
      const began = performance.now();
      const { bytes } = await fetchRange(url, begin, end, signal, (count) => { received += count; arriving(received); });
      // every part lies inside the file: a short one would feed the converter a file with a hole in it, which it
      // would convert without a word (the review of T112; the header's fetches may ask past the end, these not)
      if (bytes.length !== end - begin) {
        throw new Error(`${url} gave ${bytes.length} of the ${end - begin} bytes asked for at ${begin}`);
      }
      arrived.set(part, bytes);
      if (part === 0 && !hfPartBytes) {
        const rate = (end - begin) / ((performance.now() - began) / 1000);
        partBytes = rate >= HF_FAST_BYTES_PER_SECOND && !small ? HF_PART_BYTES : HF_SMALL_PART_BYTES;
      }
      while (arrived.has(fed)) {
        signal.throwIfAborted();
        feed(arrived.get(fed));
        arrived.delete(fed++);
        // the conversion of a part takes a moment: let messages in (T156: and the GPU's worker catch up)
        await breathe();
        await weightsRoom();
      }
      waiting.splice(0).forEach((resolve) => resolve());
    }
  };
  await Promise.all(Array.from({ length: hfConnections }, connection));
}

// What a conversion made is kept for the next visit (kept.js): in the origin private file system where there is one
// (T99), else in the Cache API. The original is twice as large, and fetching and converting it again on every visit
// would be no way to use a model. The page lists what is kept and deletes it.
// Returns what the page needs of it, or { miss } with why nothing kept could be used (for tests/e2e.mjs).
async function loadConverted(model, signal, id) {
  let kept;
  try {
    // under the bits asked for, or either the worker may choose (T115), by this converter (T116)
    kept = await keptModule.openKept(model);
  } catch (error) {
    return { miss: `could not open what is kept: ${error.message ?? error}` };
  }
  if (!kept) {
    return { miss: "nothing kept for this model" };
  }
  const { manifest } = kept;
  const started = performance.now();
  // the first part holds the header, which the memory is chosen by (T115)
  const parts = kept.parts()[Symbol.asyncIterator]();
  const unreadable = (error) => {
    if (signal.aborted) {
      throw error;
    }
    return { miss: `could not read what is kept: ${error.message ?? error}` };  // evicted: convert again
  };
  let part;
  try {
    part = await parts.next();
  } catch (error) {
    return unreadable(error);
  }
  if (part.done || part.value.length < HEADER_BYTES) {
    return { miss: "what is kept is empty" };
  }
  const weights = weightsBuffer(manifest.bytes, headerInts(part.value), manifest.options);
  let tokenizer;
  try {
    let offset = 0;
    try {
      for (; !part.done; part = await parts.next()) {
        signal.throwIfAborted();
        weights.write(offset, part.value);
        await weightsRoom();  // (T156)
        offset += part.value.length;
        postMessage({ type: "progress", load: id, received: offset, total: manifest.bytes });
      }
    } catch (error) {
      return unreadable(error);
    }
    const vocabulary = await kept.tokenizer();
    loadSeconds.download = since(started);
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    // template is for the page, not for the engine (see convert())
    const engineOptions = { ...manifest.options };
    delete engineOptions.template;
    await weightsDrained();  // (T156: a model on the GPU alone: every byte of its layers there)
    llama = weights.llama(tokenizer.buffer, { kernels, disable: disabled, ...engineOptions, ...model.options });
    loadSeconds.construct = since(constructStarted);
    return { template: manifest.options.template, keptIn: kept.where };
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
}

// checkpoint: the weights (weightsBuffer), bytes long; tokenizer: a PyProxy of the converted tokenizer. Returns why
// nothing was kept, or undefined.
// T156, kept: the file of a model on the GPU alone, written as the conversion came (kept.js's keeper)
async function keepConverted(model, checkpoint, bytes, tokenizer, options, signal, kept) {
  const view = tokenizer.getBuffer("u8");
  const vocabulary = view.data.slice();
  view.release();
  const manifest = { id: model.id, name: model.name, repo: model.hf.repo, revision: model.hf.revision, bytes, options, saved: Date.now() };
  // under the bits it was converted to, which the worker may have chosen (T115)
  const converted = { ...model, conversion: { ...model.conversion, dtype: options.dtype } };
  // T136: what this model was kept as before its source changed is never used again, and takes the room it needs
  for (const old of await keptModule.replaced(model).catch(() => [])) await keptModule.forget(old).catch(() => {});
  // slice() copies: the memory it comes from may grow (and so move) while an await waits
  if (checkpoint.direct) return kept ? kept.finish(manifest, vocabulary) : "the weights went to the GPU alone, and there is no file system here to keep them in as they came";
  return keptModule.keep(converted, manifest, (begin, end) => checkpoint.slice(begin, end), vocabulary, signal);
}

async function convert(model, signal, id) {
  const remote = typeof model.hf.repo === "string";
  // with the ?v=<build> of this worker, like every file it reads (AGENTS.md)
  keptModule ??= await import(new URL(`kept.js${self.location.search}`, import.meta.url));
  const kept = remote ? await loadConverted(model, signal, id) : undefined;
  if (kept && !kept.miss) {
    return { fromCache: true, keptIn: kept.keptIn, template: kept.template };
  }
  const keptMiss = kept?.miss;
  if (!llama2_convert) {
    // fetched when it is first needed: most visitors never convert anything
    const res = await fetch(new URL(`llama2_convert.py${self.location.search}`, import.meta.url), { signal });
    if (!res.ok) {
      throw new Error(`Could not fetch llama2_convert.py: ${res.status}`);
    }
    pyodide.FS.writeFile("llama2_convert.py", await res.text());
    llama2_convert = pyodide.pyimport("llama2_convert");
  }
  const started = performance.now();
  const at = (name) => `https://huggingface.co/${model.hf.repo}/resolve/${model.hf.revision}/${name}`;
  const text = async (url) => {
    const res = await fetch(url, { signal });
    if (!res.ok) {
      throw refused(url, res);
    }
    return res;
  };
  let first, size, base, conversion, shards;
  // T156: a model that goes on the GPU alone is kept as it comes (nothing holds its weights whole afterwards): a file
  // opened for its int8 conversion where it may (the choice is the sink's, once the header is known), let go otherwise
  const mayKeep = remote && gpuOnlyPossible() && (model.conversion?.dtype ?? "int8") === "int8";
  let keep = mayKeep ? await keptModule.keeper({ ...model, conversion: { ...model.conversion, dtype: "int8" } }).catch(() => undefined) : undefined;
  const into = checkpointSink(keep), { sink } = into;
  let keptAsItCame = false;  // keep went to keepConverted, which keeps it or lets it go
  // T115: no bits asked for (weightsFor() in src/models.js asks for six only where the device says it has too little
  // memory): int8 where its forward pass fits a 32-bit memory or the browser has a 64-bit one, six bits where neither
  // (T133), once the header is known
  const converting = { ...model.conversion, dtype: model.conversion?.dtype ?? automaticBits };
  // T89: quantize() on the SIMD kernels, the same bytes six times faster (none with ?without=kernels); T123: the
  // widening of bfloat16 too, the same float32 three times faster; T136: and of GGUF's Q8_0
  const onKernels = kernels && !disabled.includes("kernels");
  const quantizeRows = onKernels ? llama2_numpy.kernel_quantizer(kernels) : undefined;
  const bfloat16 = onKernels ? llama2_numpy.kernel_widener(kernels) : undefined;
  const q8_0 = onKernels ? llama2_numpy.kernel_q8_0(kernels) : undefined;
  // T136: a GGUF's weights with the vocabulary and config.json of the original repository (a sentencepiece vocabulary
  // in a GGUF says neither its kind nor its normalization): those files come from there, the weights from the GGUF
  const vocabulary = remote ? model.hf.vocabulary : undefined;
  const from = (name) => vocabulary ? `https://huggingface.co/${vocabulary.repo}/resolve/${vocabulary.revision}/${name}` : at(name);
  if (remote && model.hf.weights.endsWith(".gguf") && !vocabulary) {
    // T74: a GGUF holds the configuration and the vocabulary in its header, before the tensors: no config.json and
    // no tokenizer to fetch. The header is a few megabytes (the vocabulary), so it is fetched in growing pieces
    // until the converter can read all of it.
    for (let bytes = 4 * HF_HEADER_BYTES; ; bytes *= 4) {
      ({ bytes: first, total: size } = await sized(at(model.hf.weights), await fetchRange(at(model.hf.weights), 0, bytes, signal), signal));
      try {
        conversion = llama2_convert.Conversion.from_gguf.callKwargs(first, { ...converting, sink, quantize_rows: quantizeRows, bfloat16, q8_0 });
        break;
      } catch (error) {
        if (error.type !== "Incomplete" || bytes >= size) {
          into.release();
          throw error;
        }
      }
    }
    base = conversion.base;
  } else {
    // the beginning of a file: 8 bytes that say how long the JSON header is, then the header
    const head = async (name) => {
      let { bytes, total } = remote ? await sized(at(name), await fetchRange(at(name), 0, HF_HEADER_BYTES, signal), signal)
        : { bytes: new Uint8Array(await name.slice(0, HF_HEADER_BYTES).arrayBuffer()), total: name.size };
      const headerBytes = bytes.length >= 8 ? Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true)) : -1;
      if (!(headerBytes >= 2 && headerBytes <= 100e6)) {
        throw new Error("This is not a safetensors file.");
      }
      const start = 8 + headerBytes;
      if (start > bytes.length) {
        bytes = remote ? (await fetchRange(at(name), 0, start, signal)).bytes : new Uint8Array(await name.slice(0, start).arrayBuffer());
      }
      return { name, header: new TextDecoder().decode(bytes.subarray(8, start)), base: start, total };
    };
    let header;
    const config = remote ? await (await text(from(model.hf.config ?? "config.json"))).text() : await model.hf.config.text();
    if (vocabulary) {
      // the GGUF's header as a safetensors one, once the original's config.json agrees with it; the header is a few
      // megabytes (the GGUF's own vocabulary), fetched in growing pieces as above
      for (let bytes = 4 * HF_HEADER_BYTES; ; bytes *= 4) {
        ({ bytes: first, total: size } = await sized(at(model.hf.weights), await fetchRange(at(model.hf.weights), 0, bytes, signal), signal));
        try {
          const made = llama2_convert.gguf_weights(first, config);
          [header, base] = made.toJs();
          made.destroy();
          break;
        } catch (error) {
          if (error.type !== "Incomplete" || bytes >= size) {
            throw error;
          }
        }
      }
    } else {
      try {
        ({ header, base, total: size } = await head(model.hf.weights));
      } catch (error) {
        if (!remote) {
          throw error;
        }
        signal.throwIfAborted();
        const index = await text(at(`${model.hf.weights}.index.json`)).then((res) => res.text())
          .catch(() => { throw error; });
        const files = shardsOf(index);
        if (!files.length) {
          throw error;
        }
        if (files.length === 1) {
          model = { ...model, hf: { ...model.hf, weights: files[0] } };
          ({ header, base, total: size } = await head(files[0]));
        } else {
          // T105: the shards' headers joined into the header of one file made of their data one after another, which
          // the converter reads as it reads any file. Each shard is then fed from its own base, the next after it.
          shards = [];
          for (const name of files) {
            shards.push(await head(name));
          }
          const joined = llama2_convert.joined_shards(shards.map((shard) => shard.header));
          let lengths;
          [header, lengths] = joined.toJs();
          joined.destroy();
          shards.forEach((shard, i) => { shard.length = lengths[i]; });
          base = 0;
          size = shards.reduce((sum, shard) => sum + shard.length, 0);
        }
      }
    }
    // The format of one turn, when the model publishes a chat_template (T73). It is small, and a model without
    // one (or with one the converter cannot read) simply keeps the format src/models.js has for it.
    const tokenizerConfig = await (remote ? text(from("tokenizer_config.json")).then((r) => r.text())
      : model.hf.tokenizerConfig?.text() ?? Promise.resolve("")).catch(() => "");
    // T127: newer repositories keep the template in chat_template.jinja instead. Asked for only where
    // tokenizer_config.json has none: most repositories have no such file, and WebKit reports each 404 as an error
    const hasTemplate = (() => {
      try {
        return Boolean(JSON.parse(tokenizerConfig).chat_template);
      } catch {
        return false;
      }
    })();
    const chatTemplate = hasTemplate ? "" : await (remote ? text(from("chat_template.jinja")).then((r) => r.text())
      : model.hf.chatTemplate?.text() ?? Promise.resolve("")).catch(() => "");
    // For a repository nobody has looked at (?hf=), the tokenizer is whichever of these it has and the converter can read.
    // Where none will do, the converter's refusal of one that is there says why; a file that is not there (a 404 of the
    // first candidate) is said only where no other was there either (T144). Only a 404 moves on: a fetch that failed
    // otherwise (the line, 429, 5xx) must neither hide behind a later refusal nor let a later candidate be converted
    // and kept in its place (the review of T144)
    let refusal, missing;
    for (const candidate of [].concat(vocabulary?.tokenizer ?? model.hf.tokenizer)) {
      let tokenizer;
      try {
        tokenizer = new Uint8Array(remote ? await (await text(from(candidate))).arrayBuffer() : await candidate.arrayBuffer());
      } catch (error) {
        if (signal.aborted || !remote || error.status !== 404) {
          throw error;
        }
        missing ??= error;
        continue;
      }
      signal.throwIfAborted();
      try {
        conversion = llama2_convert.Conversion.callKwargs(header, base, config, tokenizer, remote ? candidate : candidate.name,
          { start: base, tokenizer_config: tokenizerConfig, chat_template: chatTemplate || null, ...converting, sink,
            quantize_rows: quantizeRows, bfloat16, q8_0 });
        break;
      } catch (error) {
        refusal ??= error;
      }
    }
    if (!conversion) {
      into.release();
      throw refusal ?? missing;
    }
  }
  let template;
  try {
    // T119: the page hears both how much has arrived (and how fast) and how much is converted. With the converted
    // share alone, a slow line looked like a slow conversion: the share moves as fast as the bytes arrive (8 MB/s
    // from Japan, T123), the conversion itself is about 500 MB/s
    let converting = 0, told = 0, arrived = 0, converted = 0, firstAt = 0, firstBytes = 0;
    const tell = (now = performance.now()) => {
      if (now - told < 250 && converted < 1) {
        return;
      }
      told = now;
      const perSecond = firstAt && now > firstAt + 1000 ? ((arrived - firstBytes) / (now - firstAt)) * 1000 : undefined;
      postMessage({ type: "progress", load: id, received: arrived, total: size, converted, perSecond });
    };
    const arriving = (received) => {
      if (!firstAt) {
        [firstAt, firstBytes] = [performance.now(), received];
      }
      arrived = received;
      tell();
    };
    const feed = (bytes) => {
      // T84: the time Python spends converting, apart from the time spent waiting for the download
      const began = performance.now();
      converted = conversion.feed(bytes);
      converting += performance.now() - began;
      tell();
    };
    if (shards) {
      // what has arrived counts across the shards, one after another (the review of T119: it went back to 0 with each)
      let before = 0;
      for (const shard of shards) {
        await inOrder(at(shard.name), shard.base, shard.base + shard.length, feed, signal,
          (received) => arriving(before + received - shard.base));
        before += shard.length;
      }
    } else if (remote) {
      await inOrder(at(model.hf.weights), base, size, feed, signal, arriving);
    } else {
      const reader = model.hf.weights.slice(base).stream().getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        if (signal.aborted) {
          reader.cancel();
          signal.throwIfAborted();
        }
        feed(value);
        await weightsRoom();  // (T156)
      }
    }
    conversion.finish();
    loadSeconds.download = since(started);
    loadSeconds.convert = converting / 1000;

    const constructStarted = performance.now();
    // every one of these proxies keeps its Python object alive: none may be left behind (the checkpoint went to
    // weights, through the sink)
    const proxies = [conversion.options, conversion.tokenizer];
    let kept;
    try {
      const options = proxies[0].toJs({ dict_converter: Object.fromEntries });
      // template is for the page (the format of one turn), not for the engine
      ({ template } = options);
      const engineOptions = { ...options };
      delete engineOptions.template;
      await weightsDrained();  // (T156)
      llama = into.weights.llama(proxies[1], { kernels, disable: disabled, ...engineOptions, ...model.options });
      loadSeconds.construct = since(constructStarted);
      // (the review of T156: a model not on the GPU alone is kept from its memory, as before (keep()): the file opened
      // for it as it came goes first. Its open handle refused keep()'s, and its drop() in the finally below removed what
      // keep() wrote: every conversion of a browser with a GPU adapter was kept no more, run 36344754761)
      if (keep && !into.weights.direct) {
        await keep.drop();
        keep = undefined;
      }
      if (remote) {
        postMessage({ type: "status", load: id, text: `${model.name}: keeping the converted model...` });
        keptAsItCame = Boolean(into.weights.direct && keep);
        kept = await keepConverted(model, into.weights, into.bytes, proxies[1], options, signal, into.weights.direct ? keep : undefined);
      }
    } finally {
      proxies.forEach((proxy) => proxy.destroy());
    }
    return { fromCache: false, notKept: kept, keptMiss, template };
  } finally {
    // (T156: the file kept as it came is let go where it was not the one kept)
    if (!keptAsItCame) await keep?.drop();
    // the engine keeps what it needs of the checkpoint alive, the rest goes with this; and a feed that failed (the
    // line, a refusal on the way) leaves no Python buffer of the model's size behind (T145)
    into.release();
    conversion.destroy();
    quantizeRows?.destroy();
    bfloat16?.destroy();
    q8_0?.destroy();
  }
}

async function load(model, signal, id) {
  signal.throwIfAborted();
  loadingKey = modelKey(model);
  gpuOnlyNow = undefined;
  await adapterAsked;  // (T156: before any weights are placed)
  await gpuOnlyEnding;  // (the review of T156: the GPU's worker of a model on the GPU alone let go before it was built)
  // let go of the previous model first, so that two never have to fit in memory
  if (llama) {
    // what forward.js holds of Python's, and its software threads (T93); T205: and the GPU's worker, whose buffers and
    // device the next model waits for (up to forward.js's GPU_END_MS): an iPhone's tab went down in /benchmark/'s rounds
    // where the one before still held them as the next came from the cache
    const released = llama.release?.();
    llama.destroy();
    llama = undefined;
    outsideNow = undefined;  // the engine goes; the memory stays for the next model (T96)
    // the engine's closures and the model refer to each other, so only the cycle collector frees the weights
    pyodide.runPython("import gc; gc.collect()");
    await released;
    signal.throwIfAborted();
  }
  if (model.hf) {
    postMessage({ type: "status", load: id, text: `${model.name}: ${model.hf.repo ? "fetching from Hugging Face and converting" : "converting"}...` });
    await initialized;
    signal.throwIfAborted();
    const converted = await convert(model, signal, id);
    if (!(await gpuOnlyReady(model, id))) return load(model, signal, id);  // T156: the GPU failed on it alone
    await startThreads(model);
    postMessage({
      type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
      seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id), ...converted,
    });
    return;
  }
  postMessage({ type: "status", load: id, text: `${model.file ? "Reading" : "Downloading"} ${model.name}...` });
  const downloadStarted = performance.now();
  let head;  // the checkpoint's header (T115); a download of this site's tells it when its first part comes
  if (model.url) {
    // the size of a file somewhere else is what its server says
    const first = await fetchRange(model.url.checkpoint, 0, HEADER_BYTES, signal);
    head = first.bytes;
    model.bytes = (await sized(model.url.checkpoint, first, signal)).total;
    if (!(model.bytes > 28)) {
      throw new Error(`${model.url.checkpoint} does not answer range requests, so its size is unknown.`);
    }
  }
  const checkpoint = model.file ? readFile(model, signal, id) : model.url ? readUrl(model, signal, id) : download(model, signal, id);
  const tokenizerBytes = model.file ? model.tokenizerFile.arrayBuffer()
    : fetch(model.url ? model.url.tokenizer : new URL(`models/${model.tokenizer}`, import.meta.url), { signal }).then((res) => {
      if (!res.ok) {
        throw new Error(`Could not fetch ${model.url?.tokenizer ?? model.tokenizer}: ${res.status}`);
      }
      return res.arrayBuffer();
    });
  tokenizerBytes.catch(() => {});
  await initialized;
  signal.throwIfAborted();
  if (model.file) {
    head = new Uint8Array(await model.file.slice(0, HEADER_BYTES).arrayBuffer());
  }
  const options = model.file || model.url ? await localOptions(model, new Uint8Array(await tokenizerBytes), head) : model.options;
  head ??= await checkpoint.header;
  signal.throwIfAborted();

  const weights = weightsBuffer(model.bytes, headerInts(head), options);
  let tokenizer;
  try {
    await checkpoint.into(weights.write);
    const vocabulary = new Uint8Array(await tokenizerBytes);
    await weightsDrained();  // (T156: a model on the GPU alone: every byte of its layers there)
    signal.throwIfAborted();
    // what the source itself measured: the bytes, without the wait for Pyodide that into() may have spent
    loadSeconds.download = checkpoint.seconds ?? since(downloadStarted);

    // from here to the end nothing waits, so no other message gets in between
    const constructStarted = performance.now();
    tokenizer = pythonBuffer(vocabulary.length);
    tokenizer.write(0, vocabulary);
    try {
      llama = weights.llama(tokenizer.buffer, { kernels, disable: disabled, ...options });
    } catch (err) {
      if (!model.file) {
        throw err;
      }
      // the checkpoint has passed its check, so it is the second file or the settings
      const reason = String(err.message ?? err).trim().split("\n").pop();
      throw new Error(`${model.tokenizerFile.name} does not work as the tokenizer.bin of ${model.file.name} (${reason})`);
    }
    loadSeconds.construct = since(constructStarted);
  } finally {
    weights.destroy();
    tokenizer?.buffer.destroy();
  }
  if (!(await gpuOnlyReady(model, id))) return load(model, signal, id);  // T156: the GPU failed on it alone
  await startThreads(model);
  postMessage({
    type: "ready", load: id, pyodide: pyodide.version, backend: llama.backend, seq_len: llama.seq_len,
    seconds: { ...loadSeconds }, heap: heapBytes(), threads: threadsNow(), gpu: watchGpu(id),
    overlapped: checkpoint.overlapped === true && pyodideAt > downloadStarted,
  });
  if (!model.file && !model.url) {
    dropStaleParts(model);
  }
}

// The number of threads of the model just loaded (stage 2): ?threads=N fixes it; a count the page remembers for
// this device and model is used (and checked now and then); else forward.js finds it while generating, from the
// number of logical cores, and the page is told the answer to remember. The software threads of the starting count
// are started and warmed here, before the model is ready.
async function startThreads(model) {
  const engine = outsideNow?.engine;
  if (!engine?.findThreads) {
    return 1;
  }
  const { fixed = 0, remembered = 0, hint = 1 } = threadsRequest ?? {};
  try {
    if (fixed) {
      return await engine.setThreads(fixed);
    }
    return await engine.findThreads({ from: hint, remembered,
      chose: (count) => postMessage({ type: "threads", model: model.id, count, from: remembered ? "remembered" : "hint", hint }),
      compared: (verdict) => postMessage({ type: "threads-compared", model: model.id, ...verdict }) });
  } catch {
    engine.stopThreads?.();
    return 1;
  }
}
const threadsNow = () => outsideNow?.engine?.threads ?? 1;

// T135, T148: what the status line says of the GPU as the model is ready (nothing waits for the GPU: the model runs on
// the CPU until the GPU is ready, and the prompts go where they are faster from then on, forward.js). Once the layers
// are on the GPU, or it is known that they will not be, the page is told { type: "gpu", note, ... } (what the GPU
// chose, and how long it took: the page shows it and remembers the shaders for the next visit). A load let go of
// meanwhile says it too; the page drops what is not of its latest load.
function watchGpu(id) {
  const engine = outsideNow?.engine;
  if (!engine) return "prompts on the CPU (the NumPy engine runs this model)";
  if (!engine.gpu) return `prompts on the CPU (${benchPage ? "the benchmark times the CPU" : "no WebGPU in a worker here"})`;
  engine.gpu.then((note) => outsideNow?.engine === engine && postMessage({ type: "gpu", load: id, note, ...(engine.gpuReady ?? {}) }));
  return engine.gpuStatus;
}

// T45: a warm-up, then the measured run: the prompt, greedy, steps tokens (fewer where the model ends it first). The
// warm-up runs its 8 steps (T184's review: return() on a generator that has not started runs none of it)
function timedGeneration(prompt, steps) {
  warmUp(prompt);
  const begin = performance.now();
  const pieces = llama.generate.callKwargs(prompt, { steps, temperature: 0, echo: false });
  let tokens = 0;
  try {
    while (!pieces.next().done) tokens += 1;
  } finally {
    pieces.destroy();
  }
  return { tokens, speed: tokens / ((performance.now() - begin) / 1000) };
}
function warmUp(prompt) {
  const pieces = llama.generate.callKwargs(prompt, { steps: 8, temperature: 0, echo: false });
  try {
    while (!pieces.next().done);
  } finally {
    pieces.destroy();
  }
}

// T184: the model page's own path on the model loaded now (src/bench.js's pathTable() writes it). The GPU is waited for
// first, GPU_WAIT_S at most (the page does not wait: its first prompts go on the CPU meanwhile; a GPU not ready by then
// is said as such), then the number of threads is the model page's (T190: the count the model page remembers, which
// /benchmark/ passes as the model page does, or else the search run to its end: forward.js's endSearch()), and every
// side is timed on it (the CPU's times, and so the choice, are per number of threads). The prompts:
// forward.js's timePrompts(). The writing: sampled tokens after the prompt at the tok/s the page's status line says
// (the engine's stats); T152: where the GPU takes a generation's steps, as the page chooses, on the CPU only and on the
// GPU only (engine.gpuSide), the sides in turn, a run each after the warm-up.
const GPU_WAIT_S = 240, WRITING_RUNS = 3;
async function timedPaths({ prompt, counts, sampled }) {
  const engine = outsideNow?.engine;
  if (!engine) return { error: `${llama.backend} runs this model here: the page's path is the NumPy engine's` };
  if (engine.gpu && !engine.gpuReady) {
    postMessage({ type: "status", text: "the GPU gets ready" });
    let timer;
    await Promise.race([engine.gpu, new Promise((resolve) => (timer = setTimeout(resolve, GPU_WAIT_S * 1000)))]);
    clearTimeout(timer);
  }
  const chosen = engine.gpuReady;
  const gpu = chosen ? { seconds: chosen.seconds, matrices: chosen.matrices, attention: chosen.attention }
    : { why: engine.gpuWhyNot ?? `not ready after ${GPU_WAIT_S} s` };
  postMessage({ type: "status", text: "the software threads" });
  const { threads, found, ended } = await forwardModule.endSearch(engine, () => timedGeneration(prompt, 64));
  // how the count came about, for the table's head (T190): src/bench.js's pathTable() says it. A software thread that
  // stopped (T120: the engine gave its helpers up and runs on one) is said first: found is 1 then, and the search's
  // verdicts or the remembered count would name another count (T190's review)
  const remembered = threadsRequest?.remembered || 0;
  const how = !weightsPool?.shared ? { alone: "no shared memory here" }
    : engine.lostThreads ? { alone: "a software thread stopped" }
    : threads < found ? { alone: `not the ${found} asked for: its software threads did not start` }
    : !ended ? { unfinished: forwardModule.SEARCH_SECONDS }
    : remembered && threads !== remembered ? { alone: `not the ${remembered} the model page remembers: its software threads did not start` }
    : remembered ? { remembered: true }
    : { searched: engine.searchLog.map(({ best, candidate, faster }) => [best, candidate, faster ? candidate : best]) };
  const encoded = llama.tokenizer.encode(prompt);
  const words = encoded.toJs();
  encoded.destroy();
  postMessage({ type: "status", text: `prompts of ${counts.join(" and ")} tokens` });
  const rows = forwardModule.timePrompts(engine, { words: words.length ? words : [llama.bos], counts: counts.filter((n) => n <= llama.seq_len) });
  const writes = Math.min(sampled, llama.seq_len - words.length);  // steps counts the prompt's positions too
  postMessage({ type: "status", text: `writing ${writes} tokens` });
  warmUp(prompt);
  let fewest = writes;  // a stop token may end a run first: its tok/s stands, and the row says the fewest
  // a run of the writing on side ("cpu", "gpu", or null: as the page chooses, T152); whether the GPU took every step
  const written = (side = null) => {
    engine.gpuSide = side;
    const before = engine.gpuSampled;
    try {
      const pieces = llama.generate.callKwargs(prompt, { steps: words.length + writes, temperature: 0, echo: false });
      try {
        while (!pieces.next().done);
      } finally {
        pieces.destroy();
      }
      const stats = llama.stats.toJs({ dict_converter: Object.fromEntries });
      fewest = Math.min(fewest, stats.sampled);
      const whole = engine.gpuSampled - before >= stats.sampled;
      // as if each had written them all, at its tok/s
      return { ms: (1000 * writes) / stats.tokens_per_second, gpuTokens: whole ? writes : engine.gpuSampled - before };
    } finally {
      engine.gpuSide = null;
    }
  };
  // T152: the sides, where the GPU takes a generation's steps (else the CPU's alone), in turn
  const tokens = Boolean(chosen?.tokens);
  const sides = tokens ? { chosen: null, cpu: "cpu", gpu: "gpu" } : { cpu: "cpu" };
  const runs = Object.fromEntries(Object.keys(sides).map((name) => [name, []]));
  for (let run = 0; run < WRITING_RUNS; run++) {
    for (const [name, side] of Object.entries(sides)) runs[name].push(written(side));
  }
  const row = { what: "generation", tokens: fewest, chosen: tokens ? forwardModule.timedCell(runs.chosen, writes) : { same: "cpu" },
    cpu: forwardModule.timedCell(runs.cpu, writes), gpu: tokens ? forwardModule.timedCell(runs.gpu, writes) : { skip: chosen ? engine.gpuTokensWhyNot ?? "not on the GPU" : gpu.why } };
  // a GPU side the GPU did not take whole (it failed, or was lost, on the way: its time is the CPU's)
  if (tokens && runs.gpu.some((run) => run.gpuTokens < writes)) row.gpu = { skip: engine.gpuTokensWhyNot ?? "the GPU did not take every step" };
  rows.push(row);
  // T190's review: the writing on each number of threads the search goes through (1, 2, 4, ... up to the logical cores,
  // and the page's), in turn: the page's count against the others on this very model. A count the model page remembers
  // is not searched here, and the CPU section's made-up model (2 layers: 11 waits between phases a token) says little
  // of a model like llm-jp-3 150M (12 layers: 61 waits, most on a phase of about 1 MB)
  const hint = Math.max(1, threadsRequest?.hint || 1);
  const tried = [...new Set([1, ...Array.from({ length: Math.floor(Math.log2(hint)) }, (_, i) => 2 ** (i + 1)), hint, threads])].sort((a, b) => a - b);
  const byCount = new Map(tried.map((n) => [n, []]));
  if (weightsPool?.shared && !engine.lostThreads && tried.length > 1) {
    for (let run = 0; run < WRITING_RUNS; run++) {
      for (const n of tried.filter((c) => byCount.has(c))) {
        postMessage({ type: "status", text: `writing on ${n} software thread${n === 1 ? "" : "s"}` });
        // a count whose software threads did not start runs on fewer: no times of it
        if ((await engine.setThreads(n)) !== n) {
          byCount.delete(n);
          continue;
        }
        warmUp(prompt);  // the helpers a switch wakes, out of the time
        byCount.get(n).push(written("cpu"));  // (T152: the CPU's steps, whichever side the page chooses)
      }
    }
    await engine.setThreads(threads);
  }
  const perCount = engine.lostThreads ? [] : [...byCount].filter(([, list]) => list.length).map(([n, list]) => ({ threads: n, ...forwardModule.timedCell(list, writes) }));
  // a software thread that stopped while the sides were timed: the times after it are one thread's
  if (engine.lostThreads && !how.alone) how.stopped = true;
  // the rounds (T45) that follow load the model again: on this count too, not searching again while they are timed
  if (weightsPool?.shared && !engine.lostThreads) threadsRequest = { ...threadsRequest, remembered: threads };
  // whatever stopped the GPU while the sides were timed (a failure, a lost device): its cells are empty (timePrompts)
  if (chosen && engine.gpuWhyNot) gpu.lost = engine.gpuWhyNot;
  return { threads, how, perCount, gpu, status: engine.gpuStatus, rows };
}

// the run that is going on, and whether the page asked it to stop
let generating, stopped = false;
let lastLoad;  // the load message of the model now (T156)
let benching = false;  // the benchmark's rounds are running (T45); see the bench message
let pathing = false;  // T184: the benchmark's page path is being timed; see the paths message

// Hand the event loop a turn, so that a message sent meanwhile is delivered. setTimeout would cost 4ms per
// call (the browsers clamp it), a MessageChannel comes back in the same millisecond.
function breathe() {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  });
}

async function generate({ type, prompt, ...options }) {
  outsideNow?.engine?.newGeneration?.();  // now and then the remembered number of threads is checked again
  // a Python generator: every step of the iteration runs one forward pass and hands over one piece of text
  const pieces = llama.generate.callKwargs(prompt, options);
  try {
    let breathed = performance.now();
    while (!stopped) {
      const { done, value } = pieces.next();
      if (done) {
        break;
      }
      postMessage({ type: "token", text: value });
      // by the clock and not by the token: often enough for the button to feel immediate with a model that
      // writes 50 tokens a second, rarely enough not to cost one that writes 900 a measurable tok/s
      if (performance.now() - breathed > 50) {
        await breathe();
        breathed = performance.now();
      }
    }
    // close the generator here, so that the engine has written its stats before the "done" below
    pieces.return();
  } finally {
    pieces.destroy();
  }
  postMessage({ type: "done", threads: threadsNow(), gpuTokens: outsideNow?.engine?.gpuTokens ?? 0, gpuSampled: outsideNow?.engine?.gpuSampled ?? 0,
                gpu: outsideNow?.engine?.gpuStatus,
                ...llama.stats.toJs({ dict_converter: Object.fromEntries }) });
}

/** The size of Pyodide's WebAssembly memory, which only grows; undefined before Pyodide is there. */
function heapBytes() {
  const python = pyodide?._module?.HEAPU8?.length;
  // T93: the weights and the forward pass have a memory of their own, outside Pyodide's
  return python === undefined ? undefined : python + (weightsNow?.buffer.byteLength ?? 0);
}

self.onmessage = async ({ data }) => {
  let signal;
  try {
    if (data.type === "init" || data.type === "load") {
      lastLoad = data;  // (T156: loaded again on the CPU where its GPU fails while the model is on it alone)
      threadsRequest = data.threads;
      gpuRequest = data.gpu;
      // The latest choice wins: the download that is going on stops, and its parts that are complete stay in
      // the cache. Pyodide is loaded once, whatever happens to the model that was asked for first.
      loading?.abort();
      loading = new AbortController();
      signal = loading.signal;
      // the model downloads while Pyodide loads
      initialized ??= init(data.search);
      // never take the model away from a run that is going on
      stopped = generating !== undefined;
      // The cancelled load frees its buffer a few turns of the event loop after the abort. Without waiting for
      // that the next buffer is allocated first, and the WebAssembly memory, which never shrinks, grows by a
      // whole model with every change of mind (587 MB after four of them).
      const previous = unloaded;
      const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
      unloaded = current.catch(() => {});
      await current;
    } else if (data.type === "bench") {
      // T45: the same model, measured again for every combination of switches the page asked for. The model is
      // built once per round from the checkpoint that is already in the Cache API, so only the engine changes.
      // One at a time: a second request while the rounds run would end a round's threads under its coordinator
      if (benching) {
        return;
      }
      benching = true;
      const rows = [];
      try {
        for (const round of data.rounds) {
          // T214: a round the page skips (src/bench.js's roundsHere(): NumPy's float32 weights where the browser does
          // not say its memory) is a row that says why, with nothing loaded
          if (round.skip !== undefined) {
            rows.push({ name: round.name, without: round.without, skip: round.skip });
            continue;
          }
          // every round is a load of its own, and it cancels whatever went before, exactly like a change of model
          loading?.abort();
          loading = new AbortController();
          signal = loading.signal;
          disabled = round.without;
          const started = performance.now();
          const previous = unloaded;
          const current = previous.then(() => generating?.catch(() => {})).then(() => load(data.model, signal, data.load));
          unloaded = current.catch(() => {});
          await current;
          const ready = since(started);
          const { tokens, speed } = timedGeneration(data.prompt, data.steps);
          rows.push({ name: round.name, without: round.without, tokens, speed, backend: llama.backend, seconds: ready });
        }
      } finally {
        // also when a round failed or a change of model cancelled it (the review of T76): the next model must not
        // load with a round's switches off while the panel shows them on
        disabled = pageSwitches;  // not self.location.search: that is the worker's own URL (?v=hash)
        benching = false;
      }
      postMessage({ type: "bench", load: data.load, rows, pyodide: pyodide.version });
    } else if (data.type === "paths") {
      // T184: the model page's own path, on the model as the first load left it (the GPU too), before the rounds:
      // its prompts and its writing as the page chooses, on the CPU only and on the GPU only
      if (benching || pathing || !llama) {
        return;
      }
      pathing = true;
      try {
        postMessage({ type: "paths", load: data.load, ...(await timedPaths(data)) });
      } finally {
        pathing = false;
      }
    } else if (data.type === "generate") {
      if (!llama) {
        throw new Error("The model is not ready.");
      }
      // messages keep arriving while this runs, hence the promise the other branches look at
      stopped = false;
      generating = generate(data);
      try {
        await generating;
      } catch (err) {
        // T156: the GPU stopped under a model on it alone: said, and the model loaded again on the CPU
        const lost = gpuOnlyNow?.lost;
        if (!lost) throw err;
        cpuOnly.add(modelKey(lastLoad.model));
        postMessage({ type: "error", load: lastLoad.load, reloading: true,
          message: `The GPU stopped (${lost}). This model was on the GPU alone, and is loaded again on the CPU.` });
        generating = undefined;
        self.onmessage({ data: lastLoad });
      } finally {
        generating = undefined;
      }
    } else if (data.type === "stop") {
      // a stop that arrives before a run starts, or after it ended, must not cut the next one short
      stopped = generating !== undefined;
    }
  } catch (err) {
    // a cancelled load has nothing to report: the one that replaced it speaks for itself
    if (!signal?.aborted) {
      // A JavaScript error thrown inside a call from Python comes back as a PythonError of type JsException, its own
      // name and message on the last line: the memory of the weights, which the converter opens through the sink
      // (weightsBuffer: Safari's refusal of a model past 4 GB, a memory the browser will not make or grow), and
      // forward.js growing the keys and values. It is told as itself (the review of T133: it was a traceback)
      const [, name = err?.name, text = err?.message] = err?.type === "JsException"
        ? /^pyodide\.ffi\.JsException: (\w+): (.*)$/.exec(err.message.trim().split("\n").pop()) ?? [] : [];
      // a ValueError of the engine is a message for the reader (wrong file, prompt too long): no traceback
      const message = err.type === "ValueError" ? err.message.trim().split("\n").pop().replace(/^ValueError: /, "")
        : name === "Error" ? text : String(err);
      // T90: the memory ran out, in Python (MemoryError: malloc could not grow the WebAssembly memory) or in
      // JavaScript (RangeError: an ArrayBuffer or WebAssembly.Memory.grow was refused). The page says so in words
      // a visitor understands, with how much memory the page had when it happened.
      // (V8 also raises RangeError for a stack overflow, which is not this; Firefox says InternalError: out of memory)
      const memory = err?.type === "MemoryError" || name === "InternalError" ||
        (name === "RangeError" && !/call stack/i.test(text ?? ""));
      // where it happened goes to the page's console (T96): tests/e2e.mjs keeps the console of a failed run
      postMessage({ type: "error", load: data.load, message: memory ? String(text ?? err).trim().split("\n").pop() : message,
                    stack: String(err?.stack ?? err), weights: weightsNow?.buffer.byteLength ?? 0, pyodide: Boolean(err?.pyodide),
                    ...(memory && { memory: true, heap: heapBytes() }) });
    }
  }
};
