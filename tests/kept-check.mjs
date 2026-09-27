// kept-check.mjs (T99): public/kept.js in Node, on small stand-ins for the origin private file system and the Cache
// API (Node has neither). What a browser does with them is for CI (tests/e2e.mjs with E2E_TWICE); this holds the
// logic: a kept model reads back the same bytes, OPFS comes before the Cache API, the Cache API is the fallback and
// still read, a write that fails leaves nothing, and forget() removes what the list shows.
//
//   node tests/kept-check.mjs
import assert from "node:assert/strict";

// ---- stand-ins: a directory tree of files, and a cache of responses
class File {
  constructor() { this.bytes = new Uint8Array(0); }
}
class FileHandle {
  constructor(file, limit) { this.file = file; this.limit = limit; }
  async getFile() {
    const bytes = this.file.bytes;
    const blob = { size: bytes.length, slice: (a, b) => ({ arrayBuffer: async () => bytes.slice(a, b).buffer }),
      text: async () => new TextDecoder().decode(bytes), arrayBuffer: async () => bytes.slice().buffer };
    return blob;
  }
  async createSyncAccessHandle() {
    const file = this.file, limit = this.limit;
    return {
      truncate(n) { file.bytes = file.bytes.slice(0, n); },
      write(data, { at }) {
        const end = at + data.length;
        if (end > limit.bytes) throw new Error("QuotaExceededError");
        if (end > file.bytes.length) { const grown = new Uint8Array(end); grown.set(file.bytes); file.bytes = grown; }
        file.bytes.set(data, at);
        return data.length;
      },
      flush() {}, close() {},
    };
  }
}
class DirectoryHandle {
  constructor(limit) { this.children = new Map(); this.limit = limit; }
  async getDirectoryHandle(name, { create = false } = {}) {
    if (!this.children.has(name)) { if (!create) throw new Error("NotFoundError"); this.children.set(name, new DirectoryHandle(this.limit)); }
    return this.children.get(name);
  }
  async getFileHandle(name, { create = false } = {}) {
    if (!this.children.has(name)) { if (!create) throw new Error("NotFoundError"); this.children.set(name, new File()); }
    return new FileHandle(this.children.get(name), this.limit);
  }
  async removeEntry(name) { if (!this.children.delete(name)) throw new Error("NotFoundError"); }
  async *entries() { for (const [name, child] of this.children) if (child instanceof DirectoryHandle) yield [name, child]; }
}
class Cache {
  constructor() { this.map = new Map(); }
  async put(key, response) { this.map.set(key, new Uint8Array(await response.arrayBuffer())); }
  async match(key) { const bytes = this.map.get(typeof key === "string" ? key : key.url); return bytes && new Response(bytes); }
  async keys() { return [...this.map.keys()].map((url) => ({ url })); }
  async delete(request) { return this.map.delete(request.url); }
}

function browser({ opfs = true, room = Infinity } = {}) {
  const limit = { bytes: room };
  const root = new DirectoryHandle(limit), cache = new Cache();
  globalThis.self = { location: { origin: "https://example.test" } };
  globalThis.caches = { open: async () => cache };
  Object.defineProperty(globalThis, "navigator", { value: { storage: { getDirectory: async () => root } }, configurable: true });
  globalThis.FileSystemFileHandle = opfs ? FileHandle : undefined;
  if (!opfs) globalThis.navigator.storage.getDirectory = async () => { throw new Error("no OPFS"); };
  return { root, cache };
}

const kept = await import("../public/kept.js");
const model = { id: "m", name: "M", hf: { repo: "a/b", revision: "0123" }, conversion: {} };
const bytes = Uint8Array.from({ length: 20 * 1024 * 1024 + 5 }, (_, i) => (i * 7) & 255);  // three parts, the last short
const manifest = { id: "m", name: "M", repo: "a/b", revision: "0123", bytes: bytes.length, options: { dtype: "int8" } };
const vocabulary = Uint8Array.of(1, 2, 3);
const readBack = async (found) => {
  const out = new Uint8Array(found.manifest.bytes);
  let at = 0;
  for await (const part of found.parts()) { out.set(part, at); at += part.length; }
  return out;
};

// OPFS: kept, listed, read back the same, forgotten
{
  const { root, cache } = browser();
  assert.equal(await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  assert.equal(cache.map.size, 0, "nothing goes to the Cache API where OPFS is");
  const list = await kept.keptModels();
  assert.deepEqual(list.map(({ where, manifest: m }) => [where, m.bytes]), [["opfs", bytes.length]]);
  const found = await kept.openKept(model);
  assert.deepEqual(await readBack(found), bytes);
  assert.deepEqual(await found.tokenizer(), vocabulary);
  await kept.forget(list[0]);
  assert.equal((await kept.keptModels()).length, 0);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
}
// no OPFS: the Cache API, as before T99, and read back the same
{
  const { cache } = browser({ opfs: false });
  assert.equal(await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  assert.equal(cache.map.size, 3 + 2, "three parts, the tokenizer and the manifest");
  const list = await kept.keptModels();
  assert.deepEqual(list.map(({ where }) => where), ["cache"]);
  assert.deepEqual(await readBack(await kept.openKept(model)), bytes);
  await kept.forget(list[0]);
  assert.equal(cache.map.size, 0);
}
// a model kept in the Cache API before T99 is still found where OPFS is now, and not moved
{
  const { cache } = browser({ opfs: false });
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  const entries = [...cache.map];
  const { cache: later } = browser();
  entries.forEach(([key, value]) => later.map.set(key, value));
  assert.deepEqual((await kept.keptModels()).map(({ where }) => where), ["cache"]);
  assert.deepEqual(await readBack(await kept.openKept(model)), bytes);
}
// a write that runs out of room leaves nothing, and says why
{
  const { root } = browser({ room: 10 * 1024 * 1024 });
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  assert.match(why, /Quota/);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
  assert.equal(await kept.openKept(model), undefined);
}
// a write the browser says it did but did not keep: found out, and nothing is left
{
  const { root } = browser();
  const original = FileHandle.prototype.createSyncAccessHandle;
  FileHandle.prototype.createSyncAccessHandle = async function () {
    const handle = await original.call(this);
    return { ...handle, write: (data) => data.length };  // says it wrote, holds nothing
  };
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  FileHandle.prototype.createSyncAccessHandle = original;
  assert.match(why, /holds 0 of/);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
}
// a write that comes back short (the room ran out without an exception): found out, and nothing is left
{
  const { root } = browser();
  const original = FileHandle.prototype.createSyncAccessHandle;
  FileHandle.prototype.createSyncAccessHandle = async function () {
    const handle = await original.call(this);
    return { ...handle, write: (data, at) => { handle.write(data.subarray(0, data.length - 1), at); return data.length - 1; } };
  };
  const why = await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  FileHandle.prototype.createSyncAccessHandle = original;
  assert.match(why, /only part of a write/);
  assert.equal((await root.getDirectoryHandle("converted-v1")).children.size, 0);
}
// a folder without its manifest (a write that did not finish) is not a kept model
{
  const { root } = browser();
  const folder = await (await root.getDirectoryHandle("converted-v1", { create: true })).getDirectoryHandle(kept.keptName(model), { create: true });
  await folder.getFileHandle("model.bin", { create: true });
  assert.equal((await kept.keptModels()).length, 0);
  assert.equal(await kept.openKept(model), undefined);
}
// T116: the bits. A model kept as int8 serves one that asks for int8 or for no bits (the worker chooses, T115), never
// one that asks for six bits: that one is converted again, and the page does not call it kept
{
  browser();
  const as = (dtype) => ({ ...model, conversion: dtype ? { dtype } : {} });
  assert.equal(await kept.keep(as("int8"), manifest, (a, b) => bytes.slice(a, b), vocabulary), undefined);
  const [one] = await kept.keptModels();
  assert.equal(one.manifest.converter, kept.CONVERTER, "the manifest says which converter made it");
  assert.ok(kept.serves(one, as("int8")) && kept.serves(one, as(undefined)) && !kept.serves(one, as("int6")));
  assert.equal(await kept.openKept(as("int6")), undefined);
  assert.deepEqual(await readBack(await kept.openKept(as(undefined))), bytes);
  // one the worker chose six bits for is found with no bits asked too
  await kept.forget(one);
  await kept.keep(as("int6"), manifest, (a, b) => bytes.slice(a, b), vocabulary);
  assert.deepEqual(await readBack(await kept.openKept(as(undefined))), bytes);
  assert.equal(await kept.openKept(as("int8")), undefined);
}
// T116: the converter. What an older one kept is not used and is deleted; what a newer one kept (a tab of an older
// page next to a newer one) is not used either, and is left alone
{
  const { root } = browser();
  const manifestOf = async () => (await (await root.getDirectoryHandle("converted-v1")).getDirectoryHandle(kept.keptName(model))).children.get("manifest.json");
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  (await manifestOf()).bytes = new TextEncoder().encode(JSON.stringify(manifest));  // as before T116: no converter
  const [old] = await kept.keptModels();
  assert.ok(kept.outdated(old) && !kept.serves(old, model));
  assert.equal(await kept.openKept(model), undefined);
  assert.equal((await kept.keptModels()).length, 0, "the older one is deleted");
  await kept.keep(model, manifest, (a, b) => bytes.slice(a, b), vocabulary);
  (await manifestOf()).bytes = new TextEncoder().encode(JSON.stringify({ ...manifest, converter: kept.CONVERTER + 1 }));
  const [newer] = await kept.keptModels();
  assert.ok(!kept.outdated(newer) && !kept.serves(newer, model));
  assert.equal(await kept.openKept(model), undefined);
  assert.equal((await kept.keptModels()).length, 1, "the newer one stays");
}
// T136: a GGUF's weights with another repository's vocabulary and config.json is kept under both: the same GGUF with
// another vocabulary is another conversion. Without one, the name is what it always was
{
  const withVocabulary = { ...model, hf: { ...model.hf, vocabulary: { repo: "c/d", revision: "4567", tokenizer: "tokenizer.model" } } };
  assert.equal(decodeURIComponent(kept.keptName(model)), "a/b@0123:int8:4096");
  assert.equal(decodeURIComponent(kept.keptName(withVocabulary)), "a/b@0123+c/d@4567:int8:4096");
}
// T136's review: a model whose source changed (the 19 of stage 2) keeps its new conversion under a new name. What it
// was kept as before is replaced: never served, and in the way of the new one where the room is short. Its other
// bits are not, nor another model, nor what ?hf= and folders keep (all "local")
{
  browser();
  const moved = { ...model, hf: { repo: "a/b-GGUF", revision: "89ab", vocabulary: { repo: "a/b", revision: "0123" } } };
  const keepAs = (entry, dtype, id = entry.id) =>
    kept.keep({ ...entry, conversion: { dtype } }, { ...manifest, id }, (a, b) => bytes.slice(a, b), vocabulary);
  await keepAs(model, "int8");
  await keepAs(model, "int6");
  await keepAs({ ...model, hf: { repo: "x/y", revision: "1" } }, "int8", "other");
  await keepAs({ ...model, id: "local", hf: { repo: "p/q", revision: "2" } }, "int8");
  await keepAs(moved, "int6");
  const names = (list) => list.map((one) => decodeURIComponent(one.name)).sort();
  assert.deepEqual(names(await kept.replaced(moved)), ["a/b@0123:int6:4096", "a/b@0123:int8:4096"]);
  assert.deepEqual(names(await kept.replaced({ ...moved, conversion: { dtype: "int8" } })), names(await kept.replaced(moved)),
    "asking for eight bits does not make its own six-bit conversion replaced");
  assert.deepEqual(await kept.replaced({ ...model, id: "local", hf: { repo: "r/s", revision: "3" } }), []);
}
// the review of T203: a Qwen3's two forms share one conversion, kept under the id of the form that converted it. When
// their source changes, opening the other form first replaces it too (it left a 9 GB conversion of the 8B behind)
{
  browser();
  const pair = ["q-thinking", "q"];
  const before = { ...model, id: "q-thinking", hf: { repo: "a/q", revision: "1" } };
  const after = { ...model, id: "q", shares: pair, hf: { repo: "a/q-GGUF", revision: "2", vocabulary: { repo: "a/q", revision: "1" } } };
  await kept.keep({ ...before, conversion: { dtype: "int8" } }, { ...manifest, id: "q-thinking" }, (a, b) => bytes.slice(a, b), vocabulary);
  await kept.keep({ ...model, id: "r", conversion: { dtype: "int8" } }, { ...manifest, id: "r" }, (a, b) => bytes.slice(a, b), vocabulary);
  const names = (list) => list.map((one) => decodeURIComponent(one.name));
  assert.deepEqual(names(await kept.replaced(after)), ["a/q@1:int8:4096"], "the other form's conversion is replaced");
  assert.deepEqual(names(await kept.replaced({ ...after, id: "q-thinking" })), ["a/q@1:int8:4096"]);
  assert.deepEqual(await kept.replaced({ ...after, shares: undefined }), [], "without shares only its own id's");
}
console.log("ok");
