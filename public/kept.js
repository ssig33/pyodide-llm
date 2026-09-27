// kept.js (T99): the models this browser converted from Hugging Face, kept for the next visit. The worker writes
// and reads them; the page lists them and deletes them. Two places, one interface:
//
//   the origin private file system (OPFS), where the browser has it: a folder per model with model.bin (the whole
//   checkpoint in one file), tokenizer.bin and manifest.json. The worker writes and reads with a sync access handle.
//   The Cache API, as before T99 (converted-v1): parts of 8 MiB, the tokenizer and the manifest, under one URL
//   prefix per model. New conversions go there only where there is no OPFS; what a visitor kept there before is read
//   and deleted as it always was, never moved.
//
// The manifest is written last in both: a folder without one is rubbish from a write that did not finish.
// A plain ES module: the worker imports it, and so does the page.

export const CACHE_NAME = "converted-v1";
const FOLDER = "converted-v1";
const PART_BYTES = 8 * 1024 * 1024;

// the name of a model's conversion: its repository, revision, dtype and context, as the Cache API key has it, and
// where the vocabulary and config.json come from another repository than the weights (T136), that one too
export function keptName(model) {
  const { dtype = "int8", max_seq_len = 4096 } = model.conversion ?? {};
  const { vocabulary } = model.hf;
  const from = vocabulary ? `+${vocabulary.repo}@${vocabulary.revision}` : "";
  return encodeURIComponent(`${model.hf.repo}@${model.hf.revision}${from}:${dtype}:${max_seq_len}`);
}
/** T116: the version of what llama2_convert.py writes, the checkpoint's bytes and the options the engine gets.
 * Raise it when either changes: a conversion kept by an older converter is then converted again, and deleted (the
 * options of T106, BOS and specials, stayed wrong in what was kept before). 1: the manifests without it. 3: T127,
 * the templates the converter reads now (selectattr, namespace(), chat_template.jinja, trim_blocks) in the options;
 * 4: and the control pieces of a sentencepiece model as their specials; 5: the nmt and collapse of a sentencepiece
 * model's normalizer (the review of T126: rinna's newlines); 6: a template's strftime_now() as {date:format}, filled
 * when the prompt is sent, and or / and, comments and escapes as Jinja reads them (the review of T127); 7: the
 * rms_norm_eps of a model whose is not 1e-5 (T124: the Qwen2.5 of the list, TinySwallow, DeepSeek-R1 and llm-jp-4, at 1e-6);
 * 8: {prompt:trim} for a template that trims what was typed (T138: Llama 3.1 and 3.2, Swallow 8B). */
export const CONVERTER = 8;
const converterOf = (manifest) => manifest.converter ?? 1;
/** The names a model's conversion may be kept under: its bits, or with none asked for, either of the two the worker
 * may choose (T115) */
export function keptNames(model) {
  const asked = model.conversion?.dtype;
  return (asked ? [asked] : ["int8", "int6"]).map((dtype) => keptName({ ...model, conversion: { ...model.conversion, dtype } }));
}
/** Whether a kept conversion (one of keptModels()) serves a model: kept under one of its names, by this converter */
export const serves = (kept, model) => keptNames(model).includes(kept.name) && converterOf(kept.manifest) === CONVERTER;
/** Whether a kept conversion is of an older converter: not used, and deleted. (One of a newer converter is left
 * alone: a tab of the older page must not delete what the newer one kept.) */
export const outdated = (kept) => converterOf(kept.manifest) < CONVERTER;
/** The conversions kept for a model of the list under a name it has no more, of either bits (T136: its weights come
 * from another repository now, or its vocabulary): never served again, and in the way of the new one (a 7B's 8 GB
 * each). The worker deletes them before it keeps the new one. Not those of ?hf= and of folders, which all have the
 * id "local". shares: the ids of the list that share one conversion (a Qwen3's two forms, T124), kept under the id of
 * whichever converted it (the review of T203: the other form's left its 9 GB behind). */
export async function replaced(model) {
  if (model.id === "local") return [];
  const names = [...keptNames(model), ...keptNames({ ...model, conversion: { ...model.conversion, dtype: undefined } })];
  const ids = model.shares ?? [model.id];
  return (await keptModels()).filter((kept) => ids.includes(kept.manifest.id) && !names.includes(kept.name));
}
const cacheKey = (name, file) => `${self.location.origin}/converted/${name}/${file}`;

async function folders(create = false) {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(FOLDER, { create });
  } catch {
    return undefined;  // no OPFS here (or not yet created)
  }
}
const openCache = () => globalThis.caches?.open(CACHE_NAME).catch(() => undefined) ?? Promise.resolve(undefined);
// OPFS with the synchronous access a worker needs to write a file of gigabytes piece by piece
const opfsWritable = () => typeof FileSystemFileHandle !== "undefined" && "createSyncAccessHandle" in FileSystemFileHandle.prototype;

/** Every kept model: { name, where: "opfs" | "cache", manifest }. */
export async function keptModels() {
  const found = [];
  const directory = await folders();
  if (directory) {
    for await (const [name, handle] of directory.entries()) {
      try {
        const file = await (await handle.getFileHandle("manifest.json")).getFile();
        found.push({ name, where: "opfs", manifest: JSON.parse(await file.text()) });
      } catch {
        // no manifest: not finished
      }
    }
  }
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    const match = request.url.match(/\/converted\/([^/]+)\/manifest\.json$/);
    if (match && !found.some((kept) => kept.name === match[1])) {
      found.push({ name: match[1], where: "cache", manifest: await (await cache.match(request)).json() });
    }
  }
  return found;
}

/** A kept model to read: { manifest, where, parts(), tokenizer() }, or undefined. parts() yields its bytes in order.
 * The first of its names (keptNames) kept by this converter; what an older one kept is deleted on the way (T116). */
export async function openKept(model) {
  for (const name of keptNames(model)) {
    const found = await openNamed(name);
    if (found && converterOf(found.manifest) === CONVERTER) return found;
    if (found && outdated(found)) await forget({ name, where: found.where });
  }
  return undefined;
}
async function openNamed(name) {
  const directory = await folders();
  const folder = await directory?.getDirectoryHandle(name).catch(() => undefined);
  if (folder) {
    try {
      const manifest = JSON.parse(await (await (await folder.getFileHandle("manifest.json")).getFile()).text());
      const file = await (await folder.getFileHandle("model.bin")).getFile();
      if (file.size === manifest.bytes) {
        return {
          manifest,
          where: "opfs",
          async *parts() {
            for (let offset = 0; offset < file.size; offset += PART_BYTES) {
              yield new Uint8Array(await file.slice(offset, offset + PART_BYTES).arrayBuffer());
            }
          },
          tokenizer: async () => new Uint8Array(await (await (await folder.getFileHandle("tokenizer.bin")).getFile()).arrayBuffer()),
        };
      }
    } catch {
      // half there: look in the Cache API, then convert again
    }
  }
  const cache = await openCache();
  const manifest = await (await cache?.match(cacheKey(name, "manifest.json")))?.json();
  if (!manifest) return undefined;
  return {
    manifest,
    where: "cache",
    async *parts() {
      for (let part = 0; part < manifest.parts; part++) {
        const stored = await cache.match(cacheKey(name, `part-${String(part).padStart(3, "0")}`));
        if (!stored) throw new Error("the browser has evicted a part of the kept model");
        yield new Uint8Array(await stored.arrayBuffer());
      }
    },
    tokenizer: async () => new Uint8Array(await (await cache.match(cacheKey(name, "tokenizer.bin"))).arrayBuffer()),
  };
}

/** Keep a conversion: slice(begin, end) gives the checkpoint's bytes (a copy), tokenizer is a Uint8Array.
 * Returns why nothing was kept, or undefined. Whatever was half written is removed again. Nothing asks the browser
 * beforehand how much room there is: its estimate said yes where the write then failed (T60). */
export async function keep(model, kept, slice, tokenizer, signal) {
  const name = keptName(model), manifest = { ...kept, converter: CONVERTER };
  if (opfsWritable()) {
    const directory = await folders(true);
    if (directory) {
      try {
        const folder = await directory.getDirectoryHandle(name, { create: true });
        // a write may come back short instead of throwing when the room runs out: that is a failure too
        const put = (handle, bytes, at) => {
          if (handle.write(bytes, { at }) !== bytes.length) throw new Error("the origin private file system took only part of a write");
        };
        for (const [file, write] of [["model.bin", (handle) => {
          for (let offset = 0; offset < manifest.bytes; offset += PART_BYTES) {
            put(handle, slice(offset, Math.min(offset + PART_BYTES, manifest.bytes)), offset);
          }
        }], ["tokenizer.bin", (handle) => put(handle, tokenizer, 0)],
        ["manifest.json", (handle) => put(handle, new TextEncoder().encode(JSON.stringify(manifest)), 0)]]) {
          const handle = await (await folder.getFileHandle(file, { create: true })).createSyncAccessHandle();
          try {
            handle.truncate(0);
            write(handle);
            handle.flush();
          } finally {
            handle.close();
          }
          signal?.throwIfAborted();
        }
        // read back what the browser says it holds now: a write that did not stay is not a kept model
        const held = await (await folder.getFileHandle("model.bin")).getFile();
        const written = await (await folder.getFileHandle("manifest.json")).getFile();
        if (held.size !== manifest.bytes || !written.size) {
          throw new Error(`the origin private file system holds ${held.size} of ${manifest.bytes} bytes after writing`);
        }
        return undefined;
      } catch (error) {
        await directory.removeEntry(name, { recursive: true }).catch(() => {});
        if (signal?.aborted) throw error;
        return String(error.message ?? error);
      }
    }
  }
  const cache = await openCache();
  if (!cache) return "this browser has neither the origin private file system nor the Cache API here";
  try {
    const parts = Math.ceil(manifest.bytes / PART_BYTES);
    for (let part = 0; part < parts; part++) {
      const copy = slice(part * PART_BYTES, Math.min((part + 1) * PART_BYTES, manifest.bytes));
      await cache.put(cacheKey(name, `part-${String(part).padStart(3, "0")}`), new Response(copy));
      signal?.throwIfAborted();
    }
    await cache.put(cacheKey(name, "tokenizer.bin"), new Response(tokenizer));
    await cache.put(cacheKey(name, "manifest.json"), new Response(JSON.stringify({ ...manifest, parts }), { headers: { "Content-Type": "application/json" } }));
    return undefined;
  } catch (error) {
    await forget({ name, where: "cache" });
    if (signal?.aborted) throw error;
    return String(error.message ?? error);
  }
}

/** T156: a conversion kept as it comes, for a model whose weights go to the GPU alone (nothing holds them whole to
 * copy from afterwards, as keep() does): write(offset, bytes) as they come, in any order; finish(kept, tokenizer) writes
 * the tokenizer and the manifest last (a folder without one is rubbish, as keep()'s) and returns why nothing was kept,
 * or undefined; drop() lets go of it. On OPFS alone (a sync access handle): undefined where there is none. Whatever
 * was kept under the name before is removed first (openKept did not find it serving, or there would be no conversion) */
export async function keeper(model) {
  if (!opfsWritable()) return undefined;
  const directory = await folders(true);
  if (!directory) return undefined;
  const name = keptName(model);
  await directory.removeEntry(name, { recursive: true }).catch(() => {});
  const folder = await directory.getDirectoryHandle(name, { create: true });
  let handle = await (await folder.getFileHandle("model.bin", { create: true })).createSyncAccessHandle();
  let failed = null;
  const put = (target, bytes, at) => {
    if (target.write(bytes, { at }) !== bytes.length) throw new Error("the origin private file system took only part of a write");
  };
  const close = () => {
    handle?.close();
    handle = null;
  };
  const drop = async () => {
    close();
    await directory.removeEntry(name, { recursive: true }).catch(() => {});
  };
  return {
    write(offset, bytes) {
      if (failed || !handle) return;
      try {
        put(handle, bytes, offset);
      } catch (error) {
        failed = String(error.message ?? error);  // (the model goes on; it is not kept)
      }
    },
    async finish(kept, tokenizer) {
      const manifest = { ...kept, converter: CONVERTER };
      try {
        if (failed) throw new Error(failed);
        handle.flush();
        close();
        for (const [file, bytes] of [["tokenizer.bin", tokenizer], ["manifest.json", new TextEncoder().encode(JSON.stringify(manifest))]]) {
          const target = await (await folder.getFileHandle(file, { create: true })).createSyncAccessHandle();
          try {
            target.truncate(0);
            put(target, bytes, 0);
            target.flush();
          } finally {
            target.close();
          }
        }
        const held = await (await folder.getFileHandle("model.bin")).getFile();
        if (held.size !== manifest.bytes) throw new Error(`the origin private file system holds ${held.size} of ${manifest.bytes} bytes after writing`);
        return undefined;
      } catch (error) {
        await drop();
        return String(error.message ?? error);
      }
    },
    drop,
  };
}
/** Delete a kept model (one of keptModels()). */
export async function forget({ name, where }) {
  if (where === "opfs") {
    await (await folders())?.removeEntry(name, { recursive: true }).catch(() => {});
    return;
  }
  const cache = await openCache();
  for (const request of (await cache?.keys()) ?? []) {
    if (request.url.startsWith(cacheKey(name, ""))) await cache.delete(request);
  }
}
