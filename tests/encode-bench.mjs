// T200: the tokenizer's encode() of another commit (main by default) and of the working copy, in one Pyodide, turn by
// turn (AGENTS.md: the same process, alternating), for every kind of tokenizer the engine has:
//   bpe       llama2.c's merges of sentencepiece BPE: the Llama 2 vocabulary (tokenizer.bin of the site)
//   unigram   Viterbi: tiny-lm and llm-jp-3 150M (the site), rinna's spiece.model (nmt, unknown piece)
//   bytebpe   Hugging Face's byte-level BPE: GPT-2 (gpt2), Qwen3 (qwen), Llama 3.2 (llama3), SmolLM2's pattern (gpt2-digits
//             with Pythia's vocabulary would not be one: SmolLM2 comes as a GGUF, so its pattern is run on GPT-2's)
// Before timing, the two must give the same IDs for every text: the timed ones, the prompts and templates of
// src/models.js, and random texts of letters, digits, spaces, line breaks, CJK, emoji and control characters, with
// and without specials. Any difference fails (exit 1). Each side is what a visitor gets from its commit: the old engine
// with the old converter's tokenizer.bin and options and the old list's options for the site's files (T216 put the
// sentencepiece model's map into tokenizer.bin and took the nfkc and nmt options out, so the new options and file
// would leave the old engine without its normalizer). A text the two normalize differently (the map against Python's
// NFKC and the nmt table) is not compared, and counted.
//
// Printed: the constructor's ms (it runs when a model loads), encode()'s ms for a short text (about 64 tokens) and a
// long one (about 1000), old and new, the medians of the rounds after a warm-up round, and where the old encode()
// spends its time (cProfile, tottime a call).
//
//   node tests/encode-bench.mjs [--ref origin/main] [--rounds 7]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { loadPyodide } from "pyodide";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const ref = option("--ref", "origin/main"), rounds = Number(option("--rounds", 7));
if (ref === "origin/main") {
  try { execFileSync("git", ["fetch", "-q", "--depth=1", "origin", "+main:refs/remotes/origin/main"], { cwd: root, stdio: "inherit" }); } catch {}
}
const source = (path) => execFileSync("git", ["show", `${ref}:${path}`], { cwd: root });
const old = source("public/llama2_numpy.py");
// the old converter, on the old engine
const oldConvert = source("public/llama2_convert.py").toString().replace(/^from llama2_numpy import/m, "from old_numpy import");
// the old list's options for the site's tokenizer.bin files (tiny-lm's nfkc, before T216)
fs.mkdirSync(`${root}.tmp/t200/`, { recursive: true });
fs.writeFileSync(`${root}.tmp/t200/models-old.mjs`, source("src/models.js"));
const { MODELS: OLD_MODELS } = await import(`${root}.tmp/t200/models-old.mjs`);

// tokenizer files of Hugging Face at the revisions src/models.js pins, kept in .tmp/t200/
const cache = `${root}.tmp/t200/hf/`;
async function hfFile(id) {
  const entry = MODELS.find((m) => m.id === id);
  const hf = entry.hf.vocabulary ?? entry.hf;
  const name = typeof hf.tokenizer === "string" ? hf.tokenizer : hf.tokenizer[0];
  const target = `${cache}${hf.repo.replace("/", "--")}-${hf.revision}-${name}`;
  if (!fs.existsSync(target)) {
    const response = await fetch(`https://huggingface.co/${hf.repo}/resolve/${hf.revision}/${name}`);
    if (!response.ok) throw new Error(`${id}: ${response.status} for ${name}`);
    fs.mkdirSync(cache, { recursive: true });
    fs.writeFileSync(target, Buffer.from(await response.arrayBuffer()));
  }
  return { name, data: fs.readFileSync(target) };
}
const listed = (models, id) => models.find((m) => m.id === id)?.options ?? {};
const site = (file, id) => ({ name: "tokenizer.bin", data: fs.readFileSync(`${root}public/models/${file}`),
                              options: listed(MODELS, id), oldOptions: listed(OLD_MODELS, id) });
const tokenizers = [
  ["bpe: Llama 2 (stories15M)", site("tokenizer.bin", "stories15M")],
  ["unigram: tiny-lm", site("tiny-lm.tokenizer.bin", "tiny-lm")],
  ["unigram: llm-jp-3 150M", site("llm-jp-3-150m.tokenizer.bin", "llm-jp-3-150m")],
  ["unigram: rinna gpt2 small (spiece.model)", await hfFile("hf-japanese-gpt2-small")],
  ["bpe: Swallow-MS (tokenizer.model, a map)", await hfFile("hf-swallow-ms-7b-instruct")],
  ["bytebpe: GPT-2", await hfFile("hf-gpt2")],
  ["bytebpe: Qwen3", await hfFile("hf-qwen3-0.6b")],
  ["bytebpe: Llama 3.2", await hfFile("hf-llama-3.2-1b-instruct")],
  ["bytebpe: Pythia (runs of spaces added)", await hfFile("hf-pythia-160m")],
];

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
py.FS.writeFile("llama2_numpy.py", fs.readFileSync(`${root}public/llama2_numpy.py`));
py.FS.writeFile("old_numpy.py", old);
py.FS.writeFile("llama2_convert.py", fs.readFileSync(`${root}public/llama2_convert.py`));
py.FS.writeFile("old_convert.py", oldConvert);
// every prompt and template of the list, with a prompt of both languages in it
const templates = [...new Set(MODELS.flatMap((m) => [m.prompt, m.template]).filter((t) => typeof t === "string"))];
py.globals.set("TEMPLATES", py.toPy(templates));
py.globals.set("ROUNDS", rounds);
py.runPython(`
import cProfile, inspect, pstats, io, json, random, statistics, struct, time
import llama2_numpy, old_numpy, llama2_convert as convert, old_convert
clock = time.perf_counter
ENGLISH = ("Lily and Tom went to the park. They saw a big red ball near the old tree, and Tom said, \\"Let's play!\\" "
           "It's 3:45 in the afternoon; the sun was warm, and 12 birds sang in the trees. They'll remember it.\\n\\n")
JAPANESE = ("富士山は、静岡県と山梨県にまたがる活火山である。標高3776.12 mで、日本最高峰の独立峰である。"
            "その美しい姿は、古くから多くの和歌や絵画に描かれてきた。2013年に世界文化遺産に登録された。\\n")
def text_of(tokens_wanted, encode):
    """ENGLISH and JAPANESE in turns until the text has tokens_wanted tokens"""
    text, parts = "", [ENGLISH, JAPANESE]
    while len(encode(text)) < tokens_wanted:
        text += parts[len(text) % 2]
    return text
ALPHABET = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) + list("\\n\\n\\r\\t")
            + list(".,!?;:'\\"()[]{}<>-_=+*/\\\\|@#$%^&~\`") + ["'s", "'LL", "'ve", "'T", " '", "  ", "\\n\\n", " \\n", "\\r\\n"]
            + list("éßÉñçøÅ") + list("ⅫⅣ²³½①０１９") + list("一二三の日本語カタカナひらがなｶﾀｶﾅ漢字ー、。「」")
            + ["\\u3000", "\\u00a0", "\\u2003", "\\u200b", "\\ufeff", "\\u2581", "\\x1c", "\\x01", "\\x7f", "\\x0b", "\\x0c", "\\x85"]
            + ["😀", "👍🏽", "𠮷", "\\U0001F1EF\\U0001F1F5", "\\ud7ff", "\\uffff", "\\U0010ffff", "ç", "e\\u0301", "Ａ", "ｱ", "ﾞ"])
def fuzz(n, seed):
    rng = random.Random(seed)
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randrange(0, 60))) for _ in range(n)]

def made(convert, name, data, options):
    """tokenizer.bin, the options, the specials and the added tokens that are not special of a tokenizer file, as that
    converter makes them (an older one has no map to write, and writes the spaces of a byte-level vocabulary)"""
    maps = hasattr(convert, "tokenizer_json_charsmap")
    if name.endswith(".json"):
        parsed = json.loads(data)
        pieces = list(convert.tokenizer_json_pieces(parsed))
        added = parsed.get("added_tokens", [])
        specials = [t["content"] for t in added if t.get("special")]
        options = convert.tokenizer_json_options(parsed)
        extra = dict(charsmap=convert.tokenizer_json_charsmap(parsed), spaces=options["tokenizer_kind"] != "bytebpe") if maps else {}
        # T143: the converter names the added tokens that are not special as specials, wherever they are written
        named = sorted({t["content"] for t in added if not t.get("special") and t["content"]}, key=lambda t: (-len(t), t)) if maps else []
        return convert.tokenizer_bin(pieces, len(pieces), **extra), options, specials, named
    if name.endswith(".model"):
        pieces = list(convert.sentencepiece_pieces(data))
        extra = dict(charsmap=convert.sentencepiece_charsmap(data)) if maps else {}
        return (convert.tokenizer_bin(pieces, len(pieces), **extra), convert.sentencepiece_options(data),
                convert.sentencepiece_specials(data), [])
    return data, options, ["</s>", "<s>"], []

def build(module, data, options):
    count, offset = 0, 4  # the pieces, counted, up to a sentencepiece model's map (T216)
    while offset < len(data) and bytes(data[offset:offset + 8]) != b"charsmap":
        offset += 8 + struct.unpack_from("<i", data, offset + 4)[0]
        count += 1
    # what each engine takes (T216 took nmt out: the map is in tokenizer.bin, which an engine before it does not read)
    keys = set(inspect.signature(module.Tokenizer.__init__).parameters) - {"self", "data", "vocab_size", "kind"}
    return module.Tokenizer(data, count, kind=options.get("tokenizer_kind", "bpe"), **{k: options[k] for k in keys if k in options})

def bench(label, name, data, options, old_options):
    plain = lambda o: o.to_py() if hasattr(o, 'to_py') else dict(o)
    new_data, new_options, specials, named = made(convert, name, data, plain(options))
    old_data, old_options, _, _ = made(old_convert, name, data, plain(old_options))
    old, new = build(old_numpy, old_data, old_options), build(llama2_numpy, new_data, new_options)
    specials = [s for s in specials if s and s.encode("utf-8") in old.index and s.encode("utf-8") in new.index][:4]
    built = {"old": [], "new": []}
    for r in range(3):
        for which, module, data, options in (("old", old_numpy, old_data, old_options), ("new", llama2_numpy, new_data, new_options)):
            began = clock()
            tokenizer = build(module, data, options)
            built[which].append((clock() - began) * 1000)
    short = text_of(64, old.encode)
    long = text_of(1000, old.encode)
    # the same IDs, or nothing to time
    texts = [short, long, ENGLISH, JAPANESE, *TEMPLATES, *fuzz(400, sum(map(ord, label)))]
    # and with the tokenizer's specials written in them
    with_specials = [text[:len(text) // 2] + specials[i % len(specials)] + text[len(text) // 2:]
                     for i, text in enumerate(texts)] if specials else []
    checked, differently = 0, []
    for text in texts + with_specials:
        if old.normalized(text) != new.normalized(text):
            differently.append(text)  # T216: the model's map against Python's NFKC and the nmt table
            continue
        for these in ((), tuple(specials)):
            a, b = old.encode(text, these), new.encode(text, these)
            if a != b:
                raise AssertionError(f"{label}: encode({text!r}, {these}) is {b} and was {a}")
            checked += 1
    cells = {}
    for r in range(ROUNDS + 1):
        for size, text, n in (("short", short, 20), ("long", long, 2)):
            for which, tokenizer in (("old", old), ("new", new)):
                began = clock()
                for _ in range(n):
                    tokenizer.encode(text)
                if r:
                    cells.setdefault((size, which), []).append((clock() - began) / n * 1000)
    m = {k: statistics.median(v) for k, v in cells.items()}
    lines = []
    for which, tokenizer in (("old", old), ("new", new)):
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(5):
            tokenizer.encode(long)
        profile.disable()
        stats = pstats.Stats(profile).stats
        top = sorted(((v[2] / 5 * 1000, f"{k[2]}") for k, v in stats.items()), reverse=True)[:6]
        lines.append(f"  where the {which} encode() of the long text spends its time (tottime ms a call; cProfile slows it): "
                     + ", ".join(f"{name} {ms:.2f}" for ms, name in top))
    if differently:
        lines.append(f"  {len(differently)} texts normalized differently, not compared (T216), as "
                     + ", ".join(repr(text[:24]) for text in differently[:4]))
    if named:
        # T143: what the new converter names besides (the list's entry may name its own specials over them)
        times = {"without": [], "with": []}
        for r in range(ROUNDS + 1):
            for which, these in (("without", ()), ("with", tuple(named))):
                began = clock()
                for _ in range(20):
                    new.encode(short, these)
                if r:
                    times[which].append((clock() - began) / 20 * 1000)
        lines.append(f"  the new encode() of the short text with the {len(named)} added tokens its converter names as specials: "
                     f"{statistics.median(times['with']):.3f} ms, without them {statistics.median(times['without']):.3f} ms")
    row = (f"| {label} | {len(old.encode(short))} | {m[('short', 'old')]:.3f} | {m[('short', 'new')]:.3f} | "
           f"{m[('short', 'old')] / m[('short', 'new')]:.2f}× | {len(old.encode(long))} | {m[('long', 'old')]:.2f} | "
           f"{m[('long', 'new')]:.2f} | {m[('long', 'old')] / m[('long', 'new')]:.2f}× | "
           f"{statistics.median(built['old']):.0f} / {statistics.median(built['new']):.0f} | {checked} |")
    return row, "\\n".join(lines)
`);
const os = await import("node:os");
console.log(`encode-bench: ${ref} (old) against the working copy (new); ${os.cpus()[0]?.model ?? "?"} (${process.arch}), Node ${process.version}, ` +
  `load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}, ${rounds} rounds`);
const rows = [], notes = [];
for (const [label, file] of tokenizers) {
  py.globals.set("LABEL", label);
  py.globals.set("NAME", file.name);
  py.FS.writeFile("tokenizer.data", file.data);
  py.globals.set("OPTIONS", py.toPy(file.options ?? {}));
  py.globals.set("OLD_OPTIONS", py.toPy(file.oldOptions ?? file.options ?? {}));
  const result = py.runPython("bench(LABEL, NAME, open('tokenizer.data', 'rb').read(), OPTIONS, OLD_OPTIONS)").toJs();
  rows.push(result[0]);
  notes.push(`${label}\n${result[1]}`);
  console.log(result[0]);
}
console.log("encode() in ms: old and new, medians of the rounds; the constructor's ms old / new; how many encodes were compared (all the same)");
console.log("| tokenizer | short (tokens) | old | new | speed-up | long (tokens) | old | new | speed-up | constructor | same IDs |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const row of rows) console.log(row);
for (const note of notes) console.log(note);
console.log(`load after: ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
process.exit(0);
