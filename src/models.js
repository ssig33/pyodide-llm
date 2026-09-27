// The first model is the default: tiny-lm, the lightest one that writes Japanese. A public page should not make a
// phone fetch 171 MB unasked, and it is ready soonest; llm-jp-3 writes far better Japanese and is one choice away.
// Within each group the order is the ones that write Japanese from light to heavy, then the English-only ones from
// light to heavy (T128): MODELS is sorted so at the end of this file, so a model added anywhere takes its place.
// ?model=<id> picks another one. Every file of the first two groups is fetched when the site is built (see the
// Makefile): llm-jp-3 and tiny-lm are converted from their Hugging Face checkpoints by convert_hf.py, and the larger
// models are quantized to int8. bytes is the checkpoint size: it sizes the download buffer and the progress bar.
const JAPANESE = "文章の書き出しを入力（例: 富士山は、）";
const STORY = "Type the beginning of a story (e.g. Lily and Tom went to the park.)";
const unigram = { tokenizer_kind: "unigram" };
// greedy decoding makes small models loop, so the Japanese ones sample, and penalize repetition
// steps: 0 is as many tokens as the context of the model holds. Nothing here holds a model back by default.
const sampled = (repetition_penalty) => ({ steps: 0, temperature: 0.7, topp: 0.9, repetition_penalty });
const greedy = { steps: 0, temperature: 0.0 };

// An instruction-tuned model answers instead of continuing, when its input has the form it was trained on: template
// wraps what the visitor typed ({prompt}), for one turn and no more. The forms are the chat_template of each model.
const LLM_JP_INSTRUCT = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\n\n### 応答:\n";
const ASK_JAPANESE = "質問や指示を入力（例: 日本の首都は？）";
// ChatML. <|im_start|> and <|im_end|> are tokens of their own, so the engine is told to read them as such
const CHATML = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
const chatml = { specials: ["<|im_start|>", "<|im_end|>"], stop_tokens: [0, 2] };
// sarashina2.2's chat_template (and CAT-Translate's, made from it) uses selectattr, which this project's template
// reader does not take (T73): one turn of it, as the real Jinja renders it, is this (T81; the 0.5B Instruct had
// ChatML here until 2026-09-26, whose <|im_start|> its vocabulary does not have). <|user|> (9), <|assistant|> (8)
// and </s> (2) are tokens of their own, and a model that writes the mark of a turn (7 to 9) has ended its answer
const SARASHINA = "<|user|>{prompt}</s><|assistant|>";
const sarashina = { specials: ["<|assistant|>", "<|user|>", "</s>"], stop_tokens: [1, 2, 7, 8, 9] };
const TRANSLATE = "Translate the following Japanese text into English.\n\n{日本語の文} (or English into Japanese)";
// llm-jp-4's chat_template is OpenAI's harmony (T132), with a system message of the model's name, its knowledge cutoff
// and the date ({date}: filled() writes today's). The template ends at "<|start|>assistant" and leaves the channel to
// the model; this one asks for the final channel, the answer, and so skips the analysis a harmony model may write
// first. Its tokenizer.json puts a "▁" before the text after each special token (a normalizer that replaces the start
// of every piece with it), which the engine does not: the space after each special token here makes the same tokens
// (the same IDs as the real Jinja and tokenizers for four prompts, T132). A turn ends with <|return|> (2), <|end|>
// (11) or <|call|> (13), and a new message would start with <|start|> (10)
const HARMONY = "<|start|> system<|message|> You are LLM-jp-4, a large language model trained by LLM-jp.\nKnowledge cutoff: " +
  "2025-12\nCurrent date: {date}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every " +
  "message.<|end|><|start|> user<|message|> {prompt}<|end|><|start|> assistant<|channel|> final<|message|>";
// T125: Mistral's formats, one turn as the real Jinja writes it (the same IDs as the real Jinja and tokenizers for
// four prompts, where the prompt has no space at either end: some templates trim it, the page does not). These
// models come with a sentencepiece tokenizer.model, which the engine reads (their tokenizer.json is a BPE of
// sentencepiece's kind, which it does not). v0.2 writes "<s> [INST]": after the BOS, the engine's dummy prefix is
// that space. v0.3's [INST] and [/INST] are tokens of their own (3 and 4)
const MISTRAL = "[INST] {prompt} [/INST]";
const MISTRAL_V3 = "[INST] {prompt}[/INST]";
// RakutenAI's (2.0 mini and 7B chat): no special tokens, a system sentence and USER / ASSISTANT; the template trims
// what was typed (T138)
const RAKUTEN = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, " +
  "detailed, and polite answers to the user's questions. USER: {prompt:trim} ASSISTANT:";
// T138: Swallow-MS's card always passes this system message, which its template puts in the user's turn; the template
// strips the whole turn, which trims what was typed at its end only (filled() trims both ends: a prompt that begins
// with spaces differs)
const SWALLOW_MS = "[INST] <<SYS>>\nあなたは誠実で優秀な日本人のアシスタントです。\n<</SYS>>\n\n{prompt:trim} [/INST] ";
// zephyr's tokenizer.json puts a "▁" before the text after </s> (a legacy Llama tokenizer), which the engine does
// not: the space after </s> makes the same tokens
const ZEPHYR = "<|user|>\n{prompt}</s> \n<|assistant|>\n";
// T124: Qwen3 thinks before it answers (<think>…</think>, then the answer), which is the form its chat_template writes
// and the converter reads. The same weights answer at once when the answer begins with an empty thought: the form of
// enable_thinking=false. <think> and </think> are tokens of the vocabulary that tokenizer.json does not call special,
// and the converter passes only the special ones: they are named here for every Qwen3, or they would be spelled out
// (in a template, or where a visitor types them: the real tokenizer reads them as the tokens they are)
const QWEN3_AT_ONCE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const qwen3 = { specials: ["<|im_start|>", "<|im_end|>", "</think>", "<think>"] };
// the sampling of Qwen3's model card for either form (its top-k and presence penalty the page's sampler has not)
const thinking = { steps: 0, temperature: 0.6, topp: 0.95, repetition_penalty: 1.0 };
const atOnce = { steps: 0, temperature: 0.7, topp: 0.8, repetition_penalty: 1.0 };
/** A Qwen3 twice (T124, the owner's "両方を別々に用意できないのか"): thinking first, and answering at once. The two
 * share their weights, and so a conversion kept in the browser; only the format differs. */
function thinkingAndNot(id, name, source, download, sizes, chat = {}) {
  const common = { group: "hf", ...source, download, conversion: {}, options: qwen3,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE, ...chat };
  return [
    { ...common, id: `${id}-thinking`, name: `${name} (thinking)`, note: `thinks before it answers · 日本語 / English · ${sizes}`,
      generation: thinking },
    { ...common, id, name: `${name} (no thinking)`, note: `answers at once · 日本語 / English · ${sizes}`,
      generation: chat.generation ?? atOnce, template: QWEN3_AT_ONCE },
  ];
}
/** T203 (T136's fourth stage): a Q8_0 GGUF's weights with the vocabulary and config.json of its original, which
 * tests/gguf_check.py tensors held them to (gguf.yml's candidates) */
const ggufOf = (repo, revision, weights, original, originalRevision, tokenizer = "tokenizer.json") =>
  ({ original, hf: { repo, revision, weights, vocabulary: { repo: original, revision: originalRevision, tokenizer } } });
const harmony = { specials: ["<|channel|>", "<|message|>", "<|start|>", "<|end|>"], stop_tokens: [1, 2, 10, 11, 13] };
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
/** The directives of strftime the chat templates use (%d %b %Y is Llama 3's), for a date */
function strftime(format, date) {
  const two = (n) => String(n).padStart(2, "0");
  const values = { d: two(date.getDate()), m: two(date.getMonth() + 1), Y: String(date.getFullYear()), y: two(date.getFullYear() % 100),
    b: MONTHS[date.getMonth()].slice(0, 3), B: MONTHS[date.getMonth()], a: DAYS[date.getDay()].slice(0, 3), A: DAYS[date.getDay()],
    H: two(date.getHours()), M: two(date.getMinutes()), S: two(date.getSeconds()), "%": "%" };
  return format.replace(/%(.)/g, (directive, letter) => values[letter] ?? directive);
}
// Jinja's trim is Python's str.strip(): the white space of str.isspace() at either end. JavaScript's trim() takes
// another set, U+FEFF too and not U+001C to U+001F nor U+0085 (the review of T138, T144: a prompt with one of them at
// an end differed from the real template's). A loop from either end, not a pattern: [...]+$ tries again from every
// white space in the middle, and a prompt of 100,000 spaces between two words held the page 45 seconds (the review of
// T144). Every one of them is a single UTF-16 unit.
const SPACE = /[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
function strip(text) {
  let start = 0, end = text.length;
  while (start < end && SPACE.test(text[start])) start++;
  while (end > start && SPACE.test(text[end - 1])) end--;
  return text.slice(start, end);
}
/** What the page sends for a prompt in a model's template: {prompt} is what was typed, {prompt:trim} the same
 * without the white space at either end (T138: what a template that pipes the message through Jinja's trim writes;
 * the converter says so), {date} today (YYYY-MM-DD, the visitor's own day), and {date:format} today in strftime's
 * format (what the converter writes for a template's strftime_now(): the review of T127, the day of the conversion
 * was kept with it). What was typed goes in as it is: as a replacement string, its $$, $&, $` and $' were patterns
 * (a typed $' wrote the rest of the template, special tokens and all; the review of T132). */
export function filled(template, prompt, today = new Date()) {
  return template.replace(/\{date(?::([^}]*))?\}/g, (_, format = "%Y-%m-%d") => strftime(format, today))
    .replace(/\{prompt(:trim)?\}/, (_, trim) => (trim ? strip(prompt) : prompt));
}
// Models that huggingface.co serves and this page converts itself (public/llama2_convert.py, the code that builds
// the models above): plain Llama architecture, one safetensors file, a Unigram tokenizer.json or a sentencepiece
// model. revision pins the commit, so that nothing changes under the page. download is the size of model.safetensors.
const hf = (repo, revision, tokenizer = "tokenizer.json") => ({ repo, revision, weights: "model.safetensors", config: "config.json", tokenizer });
const llmJp = { stop_tokens: [1, 2, 7] };

// Where each model comes from, and under which license (T87). A model of this site names its source with
// `source`; one fetched from Hugging Face is its `hf.repo`. The page lists them from here, and
// tests/models-check.mjs fails when a model has no license, so a new model cannot be added without one.
const APACHE = "Apache License 2.0";
const MIT = "MIT License";
const LLAMA_32 = "Llama 3.2 Community License";
// TinySwallow's model card: "derived from Qwen (Apache 2.0) and trained on Gemma data (Gemma Terms, Prohibited Use).
// Use (including commercial) is permitted if you comply with both licenses/policies above."
const APACHE_GEMMA = "Apache License 2.0 (derived from Qwen), and the Gemma Terms of Use and Prohibited Use Policy (trained on Gemma data)";
// T132: Qwen2.5-3B's card names its own license (license_name: qwen-research), which is for research, not commercial
// use. Swallow's card says "META LLAMA 3.1 COMMUNITY LICENSE and Gemma Terms of Use" under License (its metadata says
// llama3.3 and gemma: the words of the card are copied)
const QWEN_RESEARCH = "Qwen Research License Agreement";
const SWALLOW = "Meta Llama 3.1 Community License and Gemma Terms of Use";
export const LICENSES = {
  "sbintuitions/tiny-lm": MIT, "llm-jp/llm-jp-3-150m": APACHE, "karpathy/tinyllamas": MIT, "ellishg/tinyllamas": MIT,
  "llm-jp/llm-jp-3-150m-instruct3": APACHE, "llm-jp/llm-jp-3-440m": APACHE, "llm-jp/llm-jp-3-440m-instruct3": APACHE,
  "llm-jp/llm-jp-3-980m-instruct3": APACHE, "rinna/japanese-gpt2-small": MIT, "rinna/japanese-gpt-neox-small": MIT,
  "Qwen/Qwen2.5-0.5B-Instruct": APACHE, "Qwen/Qwen2.5-Coder-0.5B-Instruct": APACHE, "Qwen/Qwen2.5-1.5B-Instruct": APACHE,
  "sbintuitions/sarashina2.2-0.5b": MIT, "sbintuitions/sarashina2.2-0.5b-instruct-v0.1": MIT,
  "EleutherAI/pythia-70m-deduped": APACHE, "EleutherAI/pythia-160m": APACHE, "EleutherAI/pythia-410m": APACHE,
  "EleutherAI/pythia-1b": APACHE, "EleutherAI/pythia-1.4b": APACHE,
  "HuggingFaceTB/SmolLM2-135M-Instruct": APACHE, "HuggingFaceTB/SmolLM2-360M-Instruct": APACHE,
  "bartowski/SmolLM2-135M-Instruct-GGUF": APACHE,
  "openai-community/gpt2": MIT, "TinyLlama/TinyLlama-1.1B-Chat-v1.0": APACHE,
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MIT,
  "meta-llama/Llama-3.2-1B-Instruct": LLAMA_32, "unsloth/Llama-3.2-1B-Instruct": LLAMA_32,
  // T81 (2026-09-26)
  "SakanaAI/TinySwallow-1.5B-Instruct": APACHE_GEMMA, "llm-jp/llm-jp-3.1-1.8b-instruct4": APACHE,
  "sbintuitions/sarashina2.2-1b-instruct-v0.1": MIT, "cyberagent/CAT-Translate-0.8b": MIT, "cyberagent/CAT-Translate-1.4b": MIT,
  "HuggingFaceTB/SmolLM2-1.7B-Instruct": APACHE,
  // T132 (2026-09-26)
  "Qwen/Qwen2.5-3B-Instruct": QWEN_RESEARCH, "sbintuitions/sarashina2.2-3b-instruct-v0.1": MIT,
  "Qwen/Qwen2.5-7B-Instruct": APACHE, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5": SWALLOW,
  "llm-jp/llm-jp-4-8b-instruct": APACHE,
  // T126
  "rinna/japanese-gpt-1b": MIT,
  // T125
  "Rakuten/RakutenAI-2.0-mini-instruct": APACHE, "Rakuten/RakutenAI-7B-chat": APACHE,
  "tokyotech-llm/Swallow-MS-7b-instruct-v0.1": APACHE, "mistralai/Mistral-7B-Instruct-v0.2": APACHE,
  "mistralai/Mistral-7B-Instruct-v0.3": APACHE, "HuggingFaceH4/zephyr-7b-beta": MIT,
  "meta-llama/Llama-3.2-3B-Instruct": LLAMA_32, "unsloth/Llama-3.2-3B-Instruct": LLAMA_32,
  // T124
  "Qwen/Qwen3-0.6B": APACHE, "Qwen/Qwen3-1.7B": APACHE, "Qwen/Qwen3-4B": APACHE, "Qwen/Qwen3-8B": APACHE,
  "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2": APACHE, "Qwen/Qwen3-4B-Instruct-2507": APACHE,
  "Qwen/Qwen3-4B-Thinking-2507": APACHE,
  // T136 (2026-09-26): the Q8_0 GGUFs the list fetches instead of the safetensors (tests/gguf_check.py tensors passed
  // against the originals), under the license of each card (TinySwallow's GGUF has the words of its original)
  "bartowski/Qwen2.5-0.5B-Instruct-GGUF": APACHE, "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF": APACHE,
  "bartowski/Qwen2.5-1.5B-Instruct-GGUF": APACHE, "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF": APACHE,
  "SakanaAI/TinySwallow-1.5B-Instruct-GGUF": APACHE_GEMMA, "bartowski/SmolLM2-1.7B-Instruct-GGUF": APACHE,
  "bartowski/Qwen2.5-3B-Instruct-GGUF": QWEN_RESEARCH, "bartowski/Qwen2.5-7B-Instruct-GGUF": APACHE,
  // T136's second stage: the GGUF each takes its weights from (the card's license is the original's)
  "mradermacher/sarashina2.2-0.5b-GGUF": MIT, "mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf": MIT,
  "mmnga/sarashina2.2-1b-instruct-v0.1-gguf": MIT, "mmnga/sarashina2.2-3b-instruct-v0.1-gguf": MIT,
  "mmnga-o/CAT-Translate-0.8b-gguf": MIT, "mmnga-o/CAT-Translate-1.4b-gguf": MIT,
  "mmnga/llm-jp-3-980m-instruct3-gguf": APACHE, "mmnga/llm-jp-3.1-1.8b-instruct4-gguf": APACHE,
  "mmnga-o/llm-jp-4-8b-instruct-gguf": APACHE, "mmnga/RakutenAI-2.0-mini-instruct-gguf": APACHE,
  "mradermacher/RakutenAI-7B-chat-GGUF": APACHE, "mmnga/tokyotech-llm-Swallow-MS-7b-instruct-v0.1-gguf": APACHE,
  "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF": APACHE, "TheBloke/Mistral-7B-Instruct-v0.2-GGUF": APACHE,
  "bartowski/Mistral-7B-Instruct-v0.3-GGUF": APACHE, "TheBloke/zephyr-7B-beta-GGUF": MIT,
  "bartowski/Llama-3.2-1B-Instruct-GGUF": LLAMA_32, "bartowski/Llama-3.2-3B-Instruct-GGUF": LLAMA_32,
  "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf": SWALLOW,
  // T136's third stage: GPT-2 and GPT-NeoX (the card's license is the original's)
  "mradermacher/pythia-70m-deduped-GGUF": APACHE, "mradermacher/pythia-160m-GGUF": APACHE,
  "mradermacher/pythia-410m-GGUF": APACHE, "mradermacher/pythia-1b-GGUF": APACHE, "mradermacher/pythia-1.4b-GGUF": APACHE,
  "mradermacher/gpt2-GGUF": MIT,
  // T203 (T136's fourth stage): Qwen3 and DeepSeek-R1 (the card's license is the original's)
  "unsloth/Qwen3-0.6B-GGUF": APACHE, "unsloth/Qwen3-1.7B-GGUF": APACHE, "Qwen/Qwen3-4B-GGUF": APACHE,
  "Qwen/Qwen3-8B-GGUF": APACHE, "mmnga-o/Qwen3-Swallow-8B-RL-v0.2-gguf": APACHE,
  "unsloth/Qwen3-4B-Instruct-2507-GGUF": APACHE, "unsloth/Qwen3-4B-Thinking-2507-GGUF": APACHE,
  "mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF": MIT,
};
/** The Hugging Face repository a model comes from. */
export const sourceOf = (entry) => entry.hf?.repo ?? entry.source;
/** Every source once, in the order of the list, with its license and the names of the models taken from it. A
 * model fetched from a redistribution names both: where it comes from, and whose model it is. The redistribution's
 * line says which it is: "(GGUF)" for a GGUF (T74), "(copy)" for the same safetensors elsewhere (unsloth's Llama;
 * until 2026-09-26 it said "(GGUF)" for those too). A GGUF with a copy's vocabulary (T136: Llama 3.2's) is on all three. */
export function sources(models = MODELS) {
  const bySource = new Map();
  for (const entry of models) {
    // T136: the repository the vocabulary and config.json come from, where it is neither (unsloth's copy of Llama)
    for (const repo of new Set([entry.original, entry.hf?.vocabulary?.repo, sourceOf(entry)].filter(Boolean))) {
      if (!bySource.has(repo)) bySource.set(repo, { repo, license: LICENSES[repo], names: [] });
      const kind = repo !== sourceOf(entry) ? "copy" : entry.hf?.weights?.endsWith(".gguf") ? "GGUF" : "copy";
      bySource.get(repo).names.push(entry.original && repo !== entry.original ? `${entry.name} (${kind})` : entry.name);
    }
  }
  return [...bySource.values()];
}

// group: "site" (built with the site, the default), "original" or "hf"
export const GROUPS = { site: "Models of this site", original: "Unquantized originals", hf: "From Hugging Face, converted in this browser" };

// in the order they were added, more or less; MODELS below is the order of the list
const LISTED = [
  { id: "tiny-lm", name: "tiny-lm 29M", note: "日本語 / English · int8 · 33 MB",
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.bin", bytes: 32891932, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "int8", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "llm-jp-3-150m", name: "llm-jp-3 150M", note: "日本語 / English · int8 · 171 MB",
    source: "llm-jp/llm-jp-3-150m", checkpoint: "llm-jp-3-150m.bin", bytes: 171395100, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "int8", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { id: "stories260K", name: "TinyStories 260K", note: "English · float32 · 1 MB · tiny",
    source: "karpathy/tinyllamas", checkpoint: "stories260K.bin", bytes: 1056540, tokenizer: "tok512.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories3_5M", name: "TinyStories 3.5M", note: "English · float32 · 15 MB · fast",
    source: "ellishg/tinyllamas", checkpoint: "stories3_5M-v4k.bin", bytes: 14887004, tokenizer: "tok4096.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories15M", name: "TinyStories 15M", note: "English · int8 · 17 MB",
    source: "karpathy/tinyllamas", checkpoint: "stories15M.bin", bytes: 17101468, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { id: "stories42M", name: "TinyStories 42M", note: "English · int8 · 47 MB · desktop only",
    source: "karpathy/tinyllamas", checkpoint: "stories42M.bin", bytes: 46925852, tokenizer: "tokenizer.bin", options: { dtype: "int8" },
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // the unquantized originals, to compare with int8
  { group: "original", id: "tiny-lm-f16", name: "tiny-lm 29M (original)", note: "日本語 / English · float16 · 59 MB",
    source: "sbintuitions/tiny-lm", checkpoint: "tiny-lm.f16", bytes: 58724892, tokenizer: "tiny-lm.tokenizer.bin",
    options: { dtype: "float16", ...unigram, nfkc: true, stop_tokens: [1, 2] },
    generation: sampled(1.3), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "llm-jp-3-150m-f16", name: "llm-jp-3 150M (original)", note: "日本語 / English · float16 · 305 MB · desktop only",
    source: "llm-jp/llm-jp-3-150m", checkpoint: "llm-jp-3-150m.f16", bytes: 305161244, tokenizer: "llm-jp-3-150m.tokenizer.bin",
    options: { dtype: "float16", ...unigram, stop_tokens: [1, 2, 7] },
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "original", id: "stories15M-f32", name: "TinyStories 15M (original)", note: "English · float32 · 61 MB",
    source: "karpathy/tinyllamas", checkpoint: "stories15M.f32", bytes: 60816028, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  { group: "original", id: "stories42M-f32", name: "TinyStories 42M (original)", note: "English · float32 · 167 MB · desktop only",
    source: "karpathy/tinyllamas", checkpoint: "stories42M.f32", bytes: 167020572, tokenizer: "tokenizer.bin", options: {},
    generation: greedy, prompt: "Once upon a time", placeholder: STORY },
  // Fetched from huggingface.co and converted in the page. Japanese from light to heavy, then English from light
  // to heavy, as everywhere else. What "fetches" says is the download; int8 is what it becomes here.
  { group: "hf", id: "hf-llm-jp-3-150m-instruct3", name: "llm-jp-3 150M instruct3", note: "answers instructions · 日本語 · fetches 305 MB → int8 171 MB",
    hf: hf("llm-jp/llm-jp-3-150m-instruct3", "5be263e1a3613cd5c163f41ad828c8de6a2aa6ec"), download: 304649360, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-japanese-gpt2-small", name: "japanese-gpt2 small", note: "日本語 · fetches 454 MB → int8 130 MB",
    hf: hf("rinna/japanese-gpt2-small", "f7fdefe2941d9629a7b2894564435e0e035df6a6", "spiece.model"), download: 454274094,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-japanese-gpt-neox-small", name: "japanese-gpt-neox small", note: "日本語 · fetches 663 MB → int8 193 MB",
    hf: hf("rinna/japanese-gpt-neox-small", "84d18c0fa8c9940a61cfc4e25bd9a5686898bac1", "spiece.model"), download: 663470088,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-qwen2.5-0.5b-instruct", name: "Qwen2.5 0.5B Instruct", note: "answers instructions · 日本語 / English · fetches 531 MB (GGUF) → int8 545 MB",
    original: "Qwen/Qwen2.5-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF", revision: "41ba88dbac95fed2528c92514c131d73eb5a174b", weights: "Qwen2.5-0.5B-Instruct-Q8_0.gguf" }, download: 531068480,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m", name: "llm-jp-3 440M", note: "日本語 / English · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m", "0bfbf24efdcc5e4c57327e9c52e8cd832637adc2"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-llm-jp-3-440m-instruct3", name: "llm-jp-3 440M instruct3", note: "answers instructions · 日本語 · fetches 0.9 GB → int8 503 MB",
    hf: hf("llm-jp/llm-jp-3-440m-instruct3", "a308f143bd5824c4033b3a2efaa1c00afbb3aa9e"), download: 894519624, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // sarashina2.2 (T79): the instruct one is published as a single shard with an index (T78)
  { group: "hf", id: "hf-sarashina2.2-0.5b", name: "sarashina2.2 0.5B", note: "日本語 · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "sbintuitions/sarashina2.2-0.5b",
    hf: { repo: "mradermacher/sarashina2.2-0.5b-GGUF", revision: "2aed15b94f8c25b7582369f62d9801f3524e4721", weights: "sarashina2.2-0.5b.Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b", revision: "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d", tokenizer: "tokenizer.model" } },
    download: 845360480, conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りは", placeholder: JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-0.5b-instruct", name: "sarashina2.2 0.5B Instruct", note: "answers instructions · 日本語 · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-0.5b-instruct-v0.1-gguf", revision: "5c71186a7a57b8c0325dec70ba2f295303effcd8", weights: "sarashina2.2-0.5b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-0.5b-instruct-v0.1", revision: "e4b9aacc3f644893d0179847946ef6c58d868f29", tokenizer: "tokenizer.model" } },
    download: 845363072, conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T81: a translator from sarashina2.2, Japanese to English and back, asked in its own words and greedy, as its
  // model card runs it (no generation_config: transformers' defaults; a repetition penalty bends a translation)
  { group: "hf", id: "hf-cat-translate-0.8b", name: "CAT-Translate 0.8B", note: "translates 日本語 ⇄ English · fetches 845 MB (GGUF) → int8 0.9 GB · desktop only",
    original: "cyberagent/CAT-Translate-0.8b",
    hf: { repo: "mmnga-o/CAT-Translate-0.8b-gguf", revision: "c770a1944b55223ccbe766c770882c30c9866445", weights: "CAT-Translate-0.8b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-0.8b", revision: "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9", tokenizer: "tokenizer.model" } }, download: 845363136,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-llm-jp-3-980m-instruct3", name: "llm-jp-3 980M instruct3", note: "answers instructions · 日本語 · fetches 1.1 GB (GGUF) → int8 1.1 GB · desktop only",
    original: "llm-jp/llm-jp-3-980m-instruct3",
    hf: { repo: "mmnga/llm-jp-3-980m-instruct3-gguf", revision: "5966bc9958a7d313da9b8cb4679cfdf6814b6d42", weights: "llm-jp-3-980m-instruct3-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3-980m-instruct3", revision: "c079dbf3f88aa2ab702b9696231fc3336c46b1be", tokenizer: "tokenizer.json" } }, download: 1054638240, conversion: {}, options: llmJp,
    generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T126: a GPT-2 of 1.3B, whose activation is written gelu_fast (the same tanh approximation as gelu_new). Its
  // table of positions holds 1024, which is its context
  { group: "hf", id: "hf-japanese-gpt-1b", name: "japanese-gpt 1B", note: "日本語 · fetches 2.7 GB → int8 1.5 GB · desktop only",
    hf: hf("rinna/japanese-gpt-1b", "33fc2e4b4e97e229d24a2973073a1361157ecef6", "spiece.model"), download: 2655791788,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りは", placeholder: JAPANESE },
  // T81 (2026-09-26): the survey's Japanese models that convert as they are. TinySwallow reads its chat template
  // itself (T73); llm-jp-3.1 has its family's format with the system sentence its model card always passes (its
  // template alone leaves it out)
  { group: "hf", id: "hf-sarashina2.2-1b-instruct", name: "sarashina2.2 1B Instruct", note: "answers instructions · 日本語 · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "sbintuitions/sarashina2.2-1b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-1b-instruct-v0.1-gguf", revision: "cd8b02dcc14f38a5101171f304ab269d514fafd4", weights: "sarashina2.2-1b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-1b-instruct-v0.1", revision: "08cf5a8ae579be0fb5a9f802dda8a26acbc94951", tokenizer: "tokenizer.model" } }, download: 1498333056,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-cat-translate-1.4b", name: "CAT-Translate 1.4B", note: "translates 日本語 ⇄ English · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "cyberagent/CAT-Translate-1.4b",
    hf: { repo: "mmnga-o/CAT-Translate-1.4b-gguf", revision: "fe0ab30f267cd976a4f03d218defa6151af88b87", weights: "CAT-Translate-1.4b-Q8_0.gguf",
          vocabulary: { repo: "cyberagent/CAT-Translate-1.4b", revision: "254120945fd9a61278ac2171ab07c831d56838fa", tokenizer: "tokenizer.model" } }, download: 1498333088,
    conversion: {}, options: sarashina, generation: greedy, template: SARASHINA,
    prompt: "Translate the following Japanese text into English.\n\n富士山は日本でいちばん高い山で、夏には多くの人が登ります。", placeholder: TRANSLATE },
  { group: "hf", id: "hf-tinyswallow-1.5b-instruct", name: "TinySwallow 1.5B Instruct", note: "answers instructions · 日本語 · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "SakanaAI/TinySwallow-1.5B-Instruct",
    hf: { repo: "SakanaAI/TinySwallow-1.5B-Instruct-GGUF", revision: "38c003aaf8be9d17af11dece1fbabeb873c567fa", weights: "tinyswallow-1.5b-instruct-q8_0.gguf" }, download: 1646573920,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: a Mistral (a Llama by another name) of 1.5B, Japanese and English; its sliding window of 8192 is past the
  // context of 4096 the page gives it
  { group: "hf", id: "hf-rakutenai-2.0-mini-instruct", name: "RakutenAI 2.0 mini instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "Rakuten/RakutenAI-2.0-mini-instruct",
    hf: { repo: "mmnga/RakutenAI-2.0-mini-instruct-gguf", revision: "9bd2900dd7bff11c248fb510d4297b3a88817b76", weights: "RakutenAI-2.0-mini-instruct-Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-2.0-mini-instruct", revision: "6d902489587d324b7d5e201299e4e1a169f3a40b", tokenizer: "tokenizer.model" } }, download: 1631976352,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-3.1-1.8b-instruct4", name: "llm-jp-3.1 1.8B instruct4", note: "answers instructions · 日本語 · fetches 2.0 GB (GGUF) → int8 2.1 GB · desktop only",
    original: "llm-jp/llm-jp-3.1-1.8b-instruct4",
    hf: { repo: "mmnga/llm-jp-3.1-1.8b-instruct4-gguf", revision: "14ddbab20c68d8befdced79bd9daa3d7a1a29376", weights: "llm-jp-3.1-1.8b-instruct4-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-3.1-1.8b-instruct4", revision: "f19510db409090bb1737f24f868d17c4bdc86c8e", tokenizer: "tokenizer.json" } }, download: 1987023136,
    conversion: {}, options: llmJp, generation: sampled(1.1), template: LLM_JP_INSTRUCT, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T132: the large ones, in shards (T105). Past 4 GiB with their forward pass they need a 64-bit memory (T101),
  // where the browser has one int8 (T133), else six bits (T98); the 7 to 8B ones do not fit a 32-bit memory even so
  { group: "hf", id: "hf-qwen2.5-3b-instruct", name: "Qwen2.5 3B Instruct", note: "answers instructions · 日本語 / English · fetches 3.3 GB (GGUF) → int8 3.5 GB · desktop only",
    original: "Qwen/Qwen2.5-3B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", revision: "f302c64a2269a69fb27b2f9473b362f5bb8e78d8", weights: "Qwen2.5-3B-Instruct-Q8_0.gguf" }, download: 3285476512,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-sarashina2.2-3b-instruct", name: "sarashina2.2 3B Instruct", note: "answers instructions · 日本語 · fetches 3.6 GB (GGUF) → int8 3.8 GB · desktop only",
    original: "sbintuitions/sarashina2.2-3b-instruct-v0.1",
    hf: { repo: "mmnga/sarashina2.2-3b-instruct-v0.1-gguf", revision: "31d771319b04032f33e0d9d860f3984ea4812154", weights: "sarashina2.2-3b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "sbintuitions/sarashina2.2-3b-instruct-v0.1", revision: "4f3626fb1b64b3e97c908e67f27b2d627ba2a999", tokenizer: "tokenizer.model" } }, download: 3568393312,
    conversion: {}, options: sarashina, generation: sampled(1.1), template: SARASHINA,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-qwen2.5-7b-instruct", name: "Qwen2.5 7B Instruct", note: "answers instructions · 日本語 / English · fetches 8.1 GB (GGUF) → int8 8.6 GB · desktop only · Chrome and Firefox",
    original: "Qwen/Qwen2.5-7B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", revision: "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031", weights: "Qwen2.5-7B-Instruct-Q8_0.gguf" }, download: 8098525888,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // T125: Mistral 7B's of Japanese, whose sliding window of 4096 is the page's context
  { group: "hf", id: "hf-rakutenai-7b-chat", name: "RakutenAI 7B chat", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB · desktop only · Chrome and Firefox",
    original: "Rakuten/RakutenAI-7B-chat",
    hf: { repo: "mradermacher/RakutenAI-7B-chat-GGUF", revision: "d118d842e84cd542a48c6a89464d95b98d37e4e7", weights: "RakutenAI-7B-chat.Q8_0.gguf",
          vocabulary: { repo: "Rakuten/RakutenAI-7B-chat", revision: "7093167c61a0be6161cb68928c939c03fe0ab87d", tokenizer: "tokenizer.model" } }, download: 7835496736,
    conversion: {}, options: {}, generation: sampled(1.1), template: RAKUTEN,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-swallow-ms-7b-instruct", name: "Swallow-MS 7B instruct", note: "answers instructions · 日本語 / English · fetches 7.8 GB (GGUF) → int8 8.3 GB · desktop only · Chrome and Firefox",
    original: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1",
    hf: { repo: "mmnga/tokyotech-llm-Swallow-MS-7b-instruct-v0.1-gguf", revision: "cb9ab1c831cfdb8f56371f0ba94806df0fabd518", weights: "tokyotech-llm-Swallow-MS-7b-instruct-v0.1-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Swallow-MS-7b-instruct-v0.1", revision: "008d006f9065e37e39e31bf117ae8689390953e8", tokenizer: "tokenizer.model" } }, download: 7790109536,
    conversion: {}, options: {}, generation: sampled(1.1), template: SWALLOW_MS,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // Swallow's own chat_template is read (T73): a Japanese system message, and a second BOS before the user's turn,
  // as the real Jinja writes it (the same IDs as the real Jinja and tokenizers, T132)
  { group: "hf", id: "hf-llama-3.1-swallow-8b-instruct", name: "Llama 3.1 Swallow 8B Instruct", note: "answers instructions · 日本語 / English · fetches 8.5 GB (GGUF) → int8 9.0 GB · desktop only · Chrome and Firefox",
    original: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    hf: { repo: "mmnga/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf", revision: "dc00f584312c641eb7af415f7f44bcc4487350cb", weights: "Llama-3.1-Swallow-8B-Instruct-v0.5-Q8_0.gguf",
          vocabulary: { repo: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", revision: "b1f8317099a97e790ec872c1225ca155979b4816", tokenizer: "tokenizer.json" } }, download: 8540772672,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-llm-jp-4-8b-instruct", name: "llm-jp-4 8B instruct", note: "answers instructions · 日本語 / English · fetches 9.1 GB (GGUF) → int8 9.7 GB · desktop only · Chrome and Firefox",
    original: "llm-jp/llm-jp-4-8b-instruct",
    hf: { repo: "mmnga-o/llm-jp-4-8b-instruct-gguf", revision: "7ae4da12cee2f109509cb8e1d01cf8a0f1a5fbc1", weights: "llm-jp-4-8b-instruct-Q8_0.gguf",
          vocabulary: { repo: "llm-jp/llm-jp-4-8b-instruct", revision: "098f2b2cf33021eba19a6d3582aa3d071ccc0aff", tokenizer: "tokenizer.json" } }, download: 9132708384,
    conversion: {}, options: harmony, generation: sampled(1.1), template: HARMONY,
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // English. Pythia is the same design at five sizes: a ladder for measuring (T80)
  { group: "hf", id: "hf-pythia-70m", name: "Pythia 70M", note: "English · fetches 77 MB (GGUF) → int8 96 MB",
    original: "EleutherAI/pythia-70m-deduped",
    hf: { repo: "mradermacher/pythia-70m-deduped-GGUF", revision: "de94e2866e79534ca28e66593069e941988ac217", weights: "pythia-70m-deduped.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-70m-deduped", revision: "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", tokenizer: "tokenizer.json" } },
    download: 76722400,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // T74: from a GGUF (Q8_0): 145 MB instead of the 269 MB of model.safetensors, and the same int8 in the end (99.7%
  // of the most likely tokens and 0.13% of perplexity, tests/gguf_check.py). The GGUF is a redistribution; the
  // model and its license are HuggingFaceTB's (`original`)
  { group: "hf", id: "hf-smollm2-135m-instruct", name: "SmolLM2 135M Instruct", note: "answers instructions · English · fetches 145 MB (GGUF) → int8 145 MB",
    original: "HuggingFaceTB/SmolLM2-135M-Instruct",
    hf: { repo: "bartowski/SmolLM2-135M-Instruct-GGUF", revision: "09816acd5d99df7be770d85ea30822623dab342c",
          weights: "SmolLM2-135M-Instruct-Q8_0.gguf" }, download: 144811360,
    conversion: {}, options: chatml, generation: sampled(1.1), template: CHATML,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-160m", name: "Pythia 160M", note: "English · fetches 175 MB (GGUF) → int8 213 MB",
    original: "EleutherAI/pythia-160m",
    hf: { repo: "mradermacher/pythia-160m-GGUF", revision: "7a79d8f693376ce9f1ba21e2c0586c3bc874e68c", weights: "pythia-160m.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-160m", revision: "50f5173d932e8e61f858120bcb800b97af589f46", tokenizer: "tokenizer.json" } },
    download: 174602272,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-gpt2", name: "GPT-2 124M", note: "English · fetches 178 MB (GGUF) → int8 157 MB",
    original: "openai-community/gpt2",
    hf: { repo: "mradermacher/gpt2-GGUF", revision: "0cda0c2b1459ccd32256c6ddde9d230934112c1c", weights: "gpt2.Q8_0.gguf",
          vocabulary: { repo: "openai-community/gpt2", revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e", tokenizer: "tokenizer.json" } },
    download: 177669376,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-smollm2-360m-instruct", name: "SmolLM2 360M Instruct", note: "answers instructions · English · fetches 386 MB (GGUF) → int8 390 MB",
    original: "HuggingFaceTB/SmolLM2-360M-Instruct",
    hf: { repo: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", revision: "593b5a2e04c8f3e4ee880263f93e0bd2901ad47f", weights: "smollm2-360m-instruct-q8_0.gguf" }, download: 386404992,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-410m", name: "Pythia 410M", note: "English · fetches 433 MB (GGUF) → int8 506 MB",
    original: "EleutherAI/pythia-410m",
    hf: { repo: "mradermacher/pythia-410m-GGUF", revision: "29569dd6a296a653472b3086dbc8ded8479e9707", weights: "pythia-410m.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-410m", revision: "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f", tokenizer: "tokenizer.json" } },
    download: 433397664,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-coder-0.5b-instruct", name: "Qwen2.5 Coder 0.5B Instruct", note: "writes code · English · fetches 531 MB (GGUF) → int8 545 MB",
    original: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF", revision: "69a2c192eed24297fb09a34d8ba948b8624cc3e2", weights: "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf" }, download: 531068576,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "Write a Python function that reverses a string.", placeholder: "Ask for code (e.g. Write a Python function that sorts a list.)" },
  { group: "hf", id: "hf-pythia-1b", name: "Pythia 1B", note: "English · fetches 1.1 GB (GGUF) → int8 1.1 GB · desktop only",
    original: "EleutherAI/pythia-1b",
    hf: { repo: "mradermacher/pythia-1b-GGUF", revision: "56d6c599008603650e8ee607d49b885bfd766b72", weights: "pythia-1b.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-1b", revision: "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2", tokenizer: "tokenizer.json" } },
    download: 1078061728,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  // TinyLlama's template has </s> between the turns: specials makes the tokenizer read it as the token, not as text
  { group: "hf", id: "hf-tinyllama-1.1b-chat", name: "TinyLlama 1.1B Chat", note: "answers instructions · English · fetches 1.2 GB (GGUF) → int8 1.2 GB · desktop only",
    original: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hf: { repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", revision: "52e7645ba7c309695bec7ac98f4f005b139cf465", weights: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
          vocabulary: { repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", revision: "fe8a4ea1ffedaf415f4da2f062534de366a451e6", tokenizer: "tokenizer.model" } }, download: 1170781568,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: "<|user|>\n{prompt}</s>\n<|assistant|>\n",
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T106: Llama 3. The original (meta-llama) is gated, so the same weights come from unsloth's copy (`original`
  // names whose they are). The chat template and its special tokens are read from the model (T73)
  { group: "hf", id: "hf-llama-3.2-1b-instruct", name: "Llama 3.2 1B Instruct", note: "answers instructions · English · fetches 1.3 GB (GGUF) → int8 1.4 GB · desktop only",
    original: "meta-llama/Llama-3.2-1B-Instruct",
    hf: { repo: "bartowski/Llama-3.2-1B-Instruct-GGUF", revision: "067b946cf014b7c697f3654f621d577a3e3afd1c", weights: "Llama-3.2-1B-Instruct-Q8_0.gguf",
          vocabulary: { repo: "unsloth/Llama-3.2-1B-Instruct", revision: "5a8abab4a5d6f164389b1079fb721cfab8d7126c", tokenizer: "tokenizer.json" } }, download: 1321083008,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-pythia-1.4b", name: "Pythia 1.4B", note: "English · fetches 1.5 GB (GGUF) → int8 1.6 GB · desktop only",
    original: "EleutherAI/pythia-1.4b",
    hf: { repo: "mradermacher/pythia-1.4b-GGUF", revision: "24c447e001d1db62d2af39a31084b71de854b24d", weights: "pythia-1.4b.Q8_0.gguf",
          vocabulary: { repo: "EleutherAI/pythia-1.4b", revision: "fedc38a16eea3bd36a96b906d78d11d2ce18ed79", tokenizer: "tokenizer.json" } },
    download: 1506738080,
    conversion: {}, options: {}, generation: sampled(1.1), prompt: "Once upon a time", placeholder: STORY },
  { group: "hf", id: "hf-qwen2.5-1.5b-instruct", name: "Qwen2.5 1.5B Instruct", note: "answers instructions · 日本語 / English · fetches 1.6 GB (GGUF) → int8 1.7 GB · desktop only",
    original: "Qwen/Qwen2.5-1.5B-Instruct",
    hf: { repo: "bartowski/Qwen2.5-1.5B-Instruct-GGUF", revision: "9eadc66189c7641e1ddd226b8267a9119b2ce2d4", weights: "Qwen2.5-1.5B-Instruct-Q8_0.gguf" }, download: 1646573312,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  { group: "hf", id: "hf-smollm2-1.7b-instruct", name: "SmolLM2 1.7B Instruct", note: "answers instructions · English · fetches 1.8 GB (GGUF) → int8 1.9 GB · desktop only",
    original: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    hf: { repo: "bartowski/SmolLM2-1.7B-Instruct-GGUF", revision: "1f03464768bfcc0319fc50da8ff5fb20b6417ba2", weights: "SmolLM2-1.7B-Instruct-Q8_0.gguf" }, download: 1820414944,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-deepseek-r1-qwen-1.5b", name: "DeepSeek-R1 Distill Qwen 1.5B", note: "thinks before it answers · English · fetches 1.9 GB (GGUF) → int8 2.0 GB · desktop only",
    // T203: mradermacher's Q8_0 (its card names the original as its base model, and MIT)
    ...ggufOf("mradermacher/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "dcc15b1cfd0973faf5dc043fd405661fe061e7aa",
      "DeepSeek-R1-Distill-Qwen-1.5B.Q8_0.gguf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"),
    download: 1894532384,
    // the BOS is the tokenizer's <｜begin▁of▁sentence｜> (151646): config.json says 151643, the end of a sentence, and
    // with it perplexity was 2.5 to 2.7 times higher and four answers of five fell apart; the real format opens the
    // thought with <think> (T138's review). The list's options go over what the kept conversion says: no new CONVERTER
    conversion: {}, options: { bos: 151646, specials: ["<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>"], stop_tokens: [151643] },
    // its model card's sampling: temperature 0.6 and top-p 0.95, and no penalty (it names none): Qwen3's thinking (T144)
    generation: thinking, template: "<｜User｜>{prompt}<｜Assistant｜><think>\n",
    prompt: "What is 17 times 24? Think first.", placeholder: "Ask something that needs thinking" },
  // T125: Mistral 7B, and zephyr made from it
  { group: "hf", id: "hf-mistral-7b-instruct-v0.2", name: "Mistral 7B Instruct v0.2", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "mistralai/Mistral-7B-Instruct-v0.2",
    hf: { repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", revision: "3a6fbf4a41a1d52e415a4958cde6856d34b2db93", weights: "mistral-7b-instruct-v0.2.Q8_0.gguf",
          vocabulary: { repo: "mistralai/Mistral-7B-Instruct-v0.2", revision: "63a8b081895390a26e140280378bc85ec8bce07a", tokenizer: "tokenizer.model" } }, download: 7695857952,
    conversion: {}, options: {}, generation: sampled(1.1), template: MISTRAL,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-mistral-7b-instruct-v0.3", name: "Mistral 7B Instruct v0.3", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "mistralai/Mistral-7B-Instruct-v0.3",
    hf: { repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF", revision: "61fd4167fff3ab01ee1cfe0da183fa27a944db48", weights: "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
          vocabulary: { repo: "mistralai/Mistral-7B-Instruct-v0.3", revision: "c170c708c41dac9275d15a8fff4eca08d52bab71", tokenizer: "tokenizer.model" } }, download: 7702565088,
    conversion: {}, options: { specials: ["[/INST]", "[INST]"] }, generation: sampled(1.1), template: MISTRAL_V3,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-zephyr-7b-beta", name: "zephyr 7B beta", note: "answers instructions · English · fetches 7.7 GB (GGUF) → int8 8.2 GB · desktop only · Chrome and Firefox",
    original: "HuggingFaceH4/zephyr-7b-beta",
    hf: { repo: "TheBloke/zephyr-7B-beta-GGUF", revision: "e4714d14e9652aa9658fa937732cceadc63ac42e", weights: "zephyr-7b-beta.Q8_0.gguf",
          vocabulary: { repo: "HuggingFaceH4/zephyr-7b-beta", revision: "892b3d7a7b1cf10c7a701c60881cd93df615734c", tokenizer: "tokenizer.model" } }, download: 7695857344,
    conversion: {}, options: { specials: ["</s>"] }, generation: sampled(1.1), template: ZEPHYR,
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  { group: "hf", id: "hf-llama-3.2-3b-instruct", name: "Llama 3.2 3B Instruct", note: "answers instructions · English · fetches 3.4 GB (GGUF) → int8 3.6 GB · desktop only",
    original: "meta-llama/Llama-3.2-3B-Instruct",
    hf: { repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", revision: "5ab33fa94d1d04e903623ae72c95d1696f09f9e8", weights: "Llama-3.2-3B-Instruct-Q8_0.gguf",
          vocabulary: { repo: "unsloth/Llama-3.2-3B-Instruct", revision: "006f5dcd1393c3add266de40994ba96225e9689d", tokenizer: "tokenizer.json" } }, download: 3421899296,
    conversion: {}, options: {}, generation: sampled(1.1),
    prompt: "What will be popular next? Name three things.", placeholder: "Ask or instruct (e.g. What is the capital of Japan?)" },
  // T124: Qwen3, each twice (thinking and not), and Qwen3's 2507 4B, one of each form. T203: from Q8_0 GGUFs (Qwen's
  // own 0.6B and 1.7B are not the original's weights: tests/gguf_check.py tensors found every matrix 0.9 to 2.2% off
  // its Q8_0, so unsloth's)
  ...thinkingAndNot("hf-qwen3-0.6b", "Qwen3 0.6B",
    ggufOf("unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", "Qwen3-0.6B-Q8_0.gguf",
      "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"), 639447744,
    "fetches 639 MB (GGUF) → int8 671 MB"),
  ...thinkingAndNot("hf-qwen3-1.7b", "Qwen3 1.7B",
    ggufOf("unsloth/Qwen3-1.7B-GGUF", "d7f544eead698dbd1f15126ef60b45a1e1933222", "Qwen3-1.7B-Q8_0.gguf",
      "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"), 1834426944,
    "fetches 1.8 GB (GGUF) → int8 1.9 GB · desktop only"),
  ...thinkingAndNot("hf-qwen3-4b", "Qwen3 4B",
    ggufOf("Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b", "Qwen3-4B-Q8_0.gguf",
      "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"), 4280404704,
    "fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox"),
  ...thinkingAndNot("hf-qwen3-8b", "Qwen3 8B",
    ggufOf("Qwen/Qwen3-8B-GGUF", "7c41481f57cb95916b40956ab2f0b139b296d974", "Qwen3-8B-Q8_0.gguf",
      "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"), 8709518112,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox"),
  // Qwen3 Swallow's card gives one sampling, the thinking one, for both
  ...thinkingAndNot("hf-qwen3-swallow-8b", "Qwen3 Swallow 8B RL",
    ggufOf("mmnga-o/Qwen3-Swallow-8B-RL-v0.2-gguf", "3fc755c6ab3780ebad6671130fc1b630bbd1575b", "Qwen3-Swallow-8B-RL-v0.2-Q8_0.gguf",
      "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "9218f4843b6f93369a0b0999d8f58d61487ea71c"), 8709519968,
    "fetches 8.7 GB (GGUF) → int8 9.2 GB · desktop only · Chrome and Firefox", { generation: thinking }),
  { group: "hf", id: "hf-qwen3-4b-instruct-2507", name: "Qwen3 4B Instruct 2507", note: "answers instructions · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox",
    ...ggufOf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6b655725eafa393f4a9745d460374c9", "Qwen3-4B-Instruct-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"), download: 4280405600,
    conversion: {}, options: qwen3, generation: atOnce, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
  // its chat_template begins the answer with <think> itself
  { group: "hf", id: "hf-qwen3-4b-thinking-2507", name: "Qwen3 4B Thinking 2507", note: "thinks before it answers · 日本語 / English · fetches 4.3 GB (GGUF) → int8 4.5 GB · desktop only · Chrome and Firefox",
    ...ggufOf("unsloth/Qwen3-4B-Thinking-2507-GGUF", "f40adb104d4d44aee52f398b60597c5866a973a3", "Qwen3-4B-Thinking-2507-Q8_0.gguf",
      "Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea"), download: 4280405632,
    conversion: {}, options: qwen3, generation: thinking, prompt: "これからの流行りを3つ挙げてください。", placeholder: ASK_JAPANESE },
];

// T90: memory. A device that runs out of it kills the worker's WebAssembly memory, so the page warns before it
// loads a model that probably does not fit, and says what happened when it did not.
/** What the page takes besides the model: Pyodide, NumPy and the engine (the margin T90 asked for; the heap of
 * llm-jp-3 150M measures 112 MB above its 171 MB of weights, and the tab needs its own). */
export const PAGE_MEMORY = 300e6;
const megabytes = (bytes) => `${Math.round(bytes / 1e6).toLocaleString("en")} MB`;
/** The bytes of a model once loaded: `bytes` of a file of this site, or the "int8 N MB" its note gives for a
 * conversion. undefined when neither says (a file of the visitor's). */
export function modelBytes(entry) {
  // a float16 original is widened to float32 when loaded, next to the file it came from: llm-jp-3 150M's 305 MB
  // file measures about 800 MB of heap (AGENTS.md), so three times the file is the honest estimate
  if (entry.bytes) return entry.options?.dtype === "float16" ? entry.bytes * 3 : entry.bytes;
  const found = /int8 ([\d.]+) (MB|GB)/.exec(entry.note ?? "");
  const int8 = found ? Number(found[1]) * (found[2] === "GB" ? 1e9 : 1e6) : undefined;
  return int8 && entry.conversion?.dtype === "int6" ? int8 * SIX_OF_EIGHT : int8;
}

// T98: a model converted in the page can keep its weights in six bits instead of eight: 24 bytes and a scale per
// group of 32 against 32 and a scale, 7/9 of the size, at +1 to +3.4% of perplexity (measured on eight models), and
// slower on one thread (the groups are widened as they are read). So it is taken where int8 does not fit.
export const SIX_OF_EIGHT = 28 / 36;

/** T128: whether a model writes Japanese (its note says 日本語: Japanese alone, with English, or translating). */
export const writesJapanese = (entry) => (entry.note ?? "").includes("日本語");
/** The models as the list shows them (T128, the owner's order): the groups in the order of GROUPS, and within each
 * the ones that write Japanese from light to heavy, then the English-only ones from light to heavy, by modelBytes()
 * (int8 for a conversion, three times the file for a float16 original). The default, the first, is tiny-lm. */
export const MODELS = LISTED.map((entry) => ({ entry, key: [Object.keys(GROUPS).indexOf(entry.group ?? "site"), writesJapanese(entry) ? 0 : 1, modelBytes(entry)] }))
  .sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || a.key[2] - b.key[2])
  .map(({ entry }) => entry);
// T133: Chromium's navigator.deviceMemory stops at 8: a device that says 8 has 8 GB or more, as many as it likes
export const DEVICE_MEMORY_CAP = 8;
/** The dtype a model of Hugging Face is converted to: the entry's own when it has one (the settings of a visitor's
 * files); else asked is ?bits= (or a setting), "8", "6" or anything else for
 * automatic, which takes int6 where int8 would pass half of what the device says it has (deviceMemory, Chromium
 * only, and below its cap of 8: a device at the cap may have any more), and otherwise leaves the choice to the
 * worker (undefined): it knows the model's header once it converts, and with it what the forward pass needs, and
 * takes int6 where int8 would not fit a 32-bit memory and the browser has no 64-bit one (T115, T133).
 * undefined for a model that is not converted in the page. */
export function weightsFor(entry, asked, deviceMemory) {
  if (!entry.hf) return undefined;
  // a visitor's own files may come with settings that say it ({"conversion": {"dtype": ...}}): they win (T119)
  if (entry.conversion?.dtype) return entry.conversion.dtype;
  if (asked === "6" || asked === "8") return `int${asked}`;
  if (!deviceMemory || deviceMemory >= DEVICE_MEMORY_CAP) return undefined;
  const int8 = modelBytes({ ...entry, conversion: { ...entry.conversion, dtype: "int8" } });
  return int8 && int8 + PAGE_MEMORY > deviceMemory * 2 ** 30 / 2 ? "int6" : undefined;
}
/** A sentence for a device that says it has less memory than twice what the model needs, or "". deviceMemory is
 * navigator.deviceMemory (GB; only Chromium tells, and at most 8): without it nothing is guessed. A device at the cap
 * has 8 GB or more, so it is warned only of a model that needs more than 8 GB (T133). */
export function memoryWarning(entry, deviceMemory) {
  const bytes = modelBytes(entry);
  if (!deviceMemory || !bytes) return "";
  const capped = deviceMemory >= DEVICE_MEMORY_CAP;
  if (bytes + PAGE_MEMORY <= deviceMemory * 2 ** 30 / (capped ? 1 : 2)) return "";
  return `${entry.name} needs about ${megabytes(bytes + PAGE_MEMORY)} of memory, and this device has ` +
    `${capped ? `${DEVICE_MEMORY_CAP} GB or more` : `${deviceMemory} GB`}: it may run out of memory.`;
}
/** What the page says when the worker ran out of memory (heap: the size of its WebAssembly memory then). */
export function memoryFailure(entry, heap, detail) {
  const bytes = modelBytes(entry);
  return `This device ran out of memory for ${entry.name}` +
    (bytes ? ` (it needs about ${megabytes(bytes + PAGE_MEMORY)})` : "") +
    (heap ? `; the page was using ${megabytes(heap)} when it happened` : "") +
    `. A smaller model may fit, or closing other tabs may help.` + (detail ? ` (${detail})` : "");
}
