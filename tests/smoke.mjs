// Smoke test for the deployment: the Python engine under the latest Pyodide (in Node, no browser), with the
// models that `make models` has just produced. A broken engine, a broken conversion or an incompatible Pyodide
// release fails here instead of on the site.
//
//   make models kernels && node tests/smoke.mjs
import fs from "node:fs";
import { version } from "pyodide";
import { pyodideWithEngine } from "./engine.mjs";

const root = new URL("../", import.meta.url).pathname;
// kernel_llama(): the engine with the forward pass of public/forward.js, as the page runs it (T93)
const { pyodide } = await pyodideWithEngine();
for (const file of ["stories260K.bin", "tok512.bin", "stories3_5M-v4k.bin", "tok4096.bin",
                    "stories15M.f32", "tokenizer.bin", "tiny-lm.bin", "tiny-lm.tokenizer.bin"]) {
  pyodide.FS.writeFile(file.split("/").pop(), fs.readFileSync(root + file));
}

const started = Date.now();
const report = pyodide.runPython(`
import sys
import llama2_numpy

def read(name):
    with open(name, "rb") as f:
        return f.read()

# a float32 model with grouped-query attention and llama2.c's tokenizer: greedy text is deterministic
stories = llama2_numpy.Llama(read("stories260K.bin"), read("tok512.bin"))
text = "".join(stories.generate("Once upon a time", steps=40))
expected = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
assert text.startswith(expected), f"stories260K wrote: {text!r}"

# the default model: converted from Hugging Face, quantized to int8, unigram tokenizer, sampled
tiny = llama2_numpy.Llama(read("tiny-lm.bin"), read("tiny-lm.tokenizer.bin"), dtype="int8",
                          tokenizer_kind="unigram", nfkc=True, stop_tokens=(1, 2))
assert tiny.tokenizer.encode("ＡＢＣ") == tiny.tokenizer.encode("ABC"), "NFKC normalization is off"
japanese = "".join(tiny.generate("これからの流行りは", steps=12, temperature=0.7, repetition_penalty=1.3, seed=1))
assert japanese.startswith("これからの流行りは") and len(japanese) > len("これからの流行りは"), f"tiny-lm wrote: {japanese!r}"
assert japanese == "".join(tiny.generate("これからの流行りは", steps=12, temperature=0.7, repetition_penalty=1.3, seed=1)), "a seed must reproduce"

# the SIMD kernels: float32 must write exactly what NumPy writes, int8 computes on the int8 weights
story = "Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine."
numpy15 = llama2_numpy.Llama(read("stories15M.f32"), read("tokenizer.bin"))
simd15 = kernel_llama(read("stories15M.f32"), read("tokenizer.bin"))
assert simd15.backend.startswith("SIMD"), f"the kernels did not load: {simd15.backend}"
reference = "".join(numpy15.generate("Once upon a time", steps=60))
assert reference.startswith(story), f"stories15M wrote: {reference!r}"
assert "".join(simd15.generate("Once upon a time", steps=60)) == reference, "the kernels and NumPy disagree"
fast = kernel_llama(read("tiny-lm.bin"), read("tiny-lm.tokenizer.bin"), dtype="int8",
                          tokenizer_kind="unigram", nfkc=True, stop_tokens=(1, 2))
assert "int8" in fast.backend and len("".join(fast.generate("これからの流行りは", steps=12, temperature=0.7, seed=1))) > 3
# GPT-2 on the kernels (T65): LayerNorm, GELU and the learned positions must write what NumPy writes
import numpy as np
import llama2_convert
dim, hidden, layers, heads, vocab, positions = 32, 64, 2, 4, 320, 16
rng = np.random.default_rng(3)
normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
tensors = {"transformer.wte.weight": normal(vocab, dim), "transformer.wpe.weight": normal(positions, dim),
           "transformer.ln_f.weight": (1.0 + normal(dim) * 0.1).astype(np.float32), "transformer.ln_f.bias": normal(dim)}
for layer in range(layers):
    prefix = f"transformer.h.{layer}."
    for norm in ("ln_1", "ln_2"):
        tensors[prefix + norm + ".weight"] = (1.0 + normal(dim) * 0.1).astype(np.float32)
        tensors[prefix + norm + ".bias"] = normal(dim)
    tensors[prefix + "attn.c_attn.weight"], tensors[prefix + "attn.c_attn.bias"] = normal(dim, 3 * dim), normal(3 * dim)
    tensors[prefix + "attn.c_proj.weight"], tensors[prefix + "attn.c_proj.bias"] = normal(dim, dim), normal(dim)
    tensors[prefix + "mlp.c_fc.weight"], tensors[prefix + "mlp.c_fc.bias"] = normal(dim, hidden), normal(hidden)
    tensors[prefix + "mlp.c_proj.weight"], tensors[prefix + "mlp.c_proj.bias"] = normal(hidden, dim), normal(dim)
gpt2_config = dict(model_type="gpt2", n_embd=dim, n_head=heads, n_layer=layers, n_inner=hidden,
                   n_positions=positions, vocab_size=vocab, activation_function="gelu_new")
source = llama2_convert.Arrays(tensors)
normalized = llama2_convert.normalize(gpt2_config)
header = llama2_convert.checkpoint_header(normalized, source, positions)
gpt2_file = bytearray(llama2_convert.checkpoint_size(header, "float32", {"arch": "gpt2"}))
llama2_convert.convert_weights(source, gpt2_config, "float32", positions, gpt2_file)
gpt2_vocabulary = llama2_convert.tokenizer_bin([("<unk>", 0.0, False)] + [(f"w{i}", -float(i), True) for i in range(vocab - 1)], vocab)
plain = llama2_numpy.Llama(bytes(gpt2_file), gpt2_vocabulary, arch="gpt2")
quick = kernel_llama(bytes(gpt2_file), gpt2_vocabulary, arch="gpt2")
assert quick.backend.startswith("SIMD"), f"the kernels did not load for GPT-2: {quick.backend}"
for pos, token in enumerate([1, 5, 9, 13]):
    wanted, got = plain.forward(token, pos), quick.forward(token, pos)
    assert np.allclose(wanted, got, rtol=1e-4, atol=1e-4), f"GPT-2 kernels differ at {pos}: {np.abs(wanted - got).max()}"

# GPT-NeoX on the kernels (T72): part of every head rotates, and both branches may read the same x
neox_tensors = {"gpt_neox.embed_in.weight": normal(vocab, dim), "embed_out.weight": normal(vocab, dim),
                "gpt_neox.final_layer_norm.weight": (1.0 + normal(dim) * 0.1).astype(np.float32),
                "gpt_neox.final_layer_norm.bias": normal(dim)}
for layer in range(layers):
    prefix = f"gpt_neox.layers.{layer}."
    for norm in ("input_layernorm", "post_attention_layernorm"):
        neox_tensors[prefix + norm + ".weight"] = (1.0 + normal(dim) * 0.1).astype(np.float32)
        neox_tensors[prefix + norm + ".bias"] = normal(dim)
    neox_tensors[prefix + "attention.query_key_value.weight"] = normal(3 * dim, dim)
    neox_tensors[prefix + "attention.query_key_value.bias"] = normal(3 * dim)
    neox_tensors[prefix + "attention.dense.weight"], neox_tensors[prefix + "attention.dense.bias"] = normal(dim, dim), normal(dim)
    neox_tensors[prefix + "mlp.dense_h_to_4h.weight"], neox_tensors[prefix + "mlp.dense_h_to_4h.bias"] = normal(hidden, dim), normal(hidden)
    neox_tensors[prefix + "mlp.dense_4h_to_h.weight"], neox_tensors[prefix + "mlp.dense_4h_to_h.bias"] = normal(dim, hidden), normal(dim)
for rotary_pct, parallel in ((0.25, True), (1.0, False)):
    neox_config = dict(model_type="gpt_neox", hidden_size=dim, num_attention_heads=heads, num_hidden_layers=layers,
                       intermediate_size=hidden, max_position_embeddings=positions, vocab_size=vocab,
                       rotary_pct=rotary_pct, rotary_emb_base=10000.0, use_parallel_residual=parallel,
                       hidden_act="gelu", tie_word_embeddings=False)
    source = llama2_convert.Arrays(neox_tensors)
    header = llama2_convert.checkpoint_header(llama2_convert.normalize(neox_config), source, positions)
    neox_file = bytearray(llama2_convert.checkpoint_size(header, "float32", {"arch": "neox"}))
    llama2_convert.convert_weights(source, neox_config, "float32", positions, neox_file)
    rotary = llama2_convert.rotary_dim(llama2_convert.normalize(neox_config))
    options = dict(arch="neox", rotary=rotary, parallel_residual=parallel)
    plain = llama2_numpy.Llama(bytes(neox_file), gpt2_vocabulary, **options)
    quick = kernel_llama(bytes(neox_file), gpt2_vocabulary, **options)
    assert quick.backend.startswith("SIMD"), f"the kernels did not load for GPT-NeoX: {quick.backend}"
    for pos, token in enumerate([1, 5, 9, 13]):
        wanted, got = plain.forward(token, pos), quick.forward(token, pos)
        assert np.allclose(wanted, got, rtol=1e-4, atol=1e-4), \
            f"GPT-NeoX kernels differ (rotary_pct {rotary_pct}, parallel {parallel}) at {pos}: {np.abs(wanted - got).max()}"

# Qwen3 on the kernels (T124): the norms of every head of q and k must write what NumPy writes, in float32; int8
# computes on its own numbers (the activations are quantized too) and must pick mostly the same tokens
def qwen3_model(dim, heads, kv_heads, head_dim, hidden=96, layers=2, vocab=320, positions=24):
    rng = np.random.default_rng(5)
    normal = lambda *shape: (rng.standard_normal(shape) * 0.3).astype(np.float32)
    near_one = lambda n: (1.0 + normal(n) * 0.3).astype(np.float32)
    tensors = {"model.embed_tokens.weight": normal(vocab, dim), "model.norm.weight": near_one(dim)}
    for layer in range(layers):
        p = f"model.layers.{layer}."
        tensors.update({p + "input_layernorm.weight": near_one(dim), p + "post_attention_layernorm.weight": near_one(dim),
                        p + "self_attn.q_proj.weight": normal(heads * head_dim, dim),
                        p + "self_attn.k_proj.weight": normal(kv_heads * head_dim, dim),
                        p + "self_attn.v_proj.weight": normal(kv_heads * head_dim, dim),
                        p + "self_attn.o_proj.weight": normal(dim, heads * head_dim),
                        p + "self_attn.q_norm.weight": near_one(head_dim), p + "self_attn.k_norm.weight": near_one(head_dim),
                        p + "mlp.gate_proj.weight": normal(hidden, dim), p + "mlp.up_proj.weight": normal(hidden, dim),
                        p + "mlp.down_proj.weight": normal(dim, hidden)})
    config = dict(model_type="qwen3", hidden_size=dim, intermediate_size=hidden, num_hidden_layers=layers,
                  num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, vocab_size=vocab,
                  max_position_embeddings=positions, rope_theta=10000.0, tie_word_embeddings=True, rms_norm_eps=0.25)
    return tensors, config

qwen3_vocabulary = gpt2_vocabulary
# heads of dim / heads, twice as wide as dim (Qwen3 0.6B), and two thirds as wide
for dim, heads, kv_heads, head_dim in ((64, 4, 2, 16), (64, 4, 2, 32), (96, 2, 1, 32)):
    qwen3_tensors, qwen3_config = qwen3_model(dim, heads, kv_heads, head_dim)
    source = llama2_convert.Arrays(qwen3_tensors)
    header = llama2_convert.checkpoint_header(qwen3_config, source, 24)
    for dtype in ("float32", "int8"):
        qwen3_file = bytearray(llama2_convert.checkpoint_size(header, dtype, {"qk_norm": True, "head_dim": head_dim}))
        llama2_convert.convert_weights(source, qwen3_config, dtype, 24, qwen3_file)
        # an epsilon far from 1e-5, so that a kernel that did not take it would differ (T124)
        shape = dict(dtype=dtype, qk_norm=True, head_dim=head_dim, rms_norm_eps=0.25)
        plain = llama2_numpy.Llama(bytes(qwen3_file), qwen3_vocabulary, **shape)
        quick = kernel_llama(bytes(qwen3_file), qwen3_vocabulary, **shape)
        assert quick.backend.startswith("SIMD"), f"the kernels did not load for Qwen3: {quick.backend}"
        # T144: int8 on the int8 kernels, not widened to float32 on the way (which the tokens alone would not show)
        assert (dtype == "int8") == ("int8" in quick.backend), f"Qwen3 {dtype} runs as {quick.backend}"
        same = 0
        for pos, token in enumerate([1, 5, 9, 13, 17, 21, 25, 29]):
            wanted, got = plain.forward(token, pos), quick.forward(token, pos)
            same += int(np.argmax(wanted) == np.argmax(got))
            if dtype == "float32":
                assert np.allclose(wanted, got, rtol=1e-4, atol=1e-4), f"Qwen3 kernels differ ({head_dim}) at {pos}: {np.abs(wanted - got).max()}"
        assert same >= 6, f"Qwen3 int8 on the kernels picks other tokens ({same} of 8 the same)"

# the switches of T52: every one of them must leave a path that still works, and float32 must not change
for disable in ((), ("relaxed",), ("sampler",), ("int8", "relaxed", "sampler"), ("kernels",)):
    switched = kernel_llama(read("stories15M.f32"), read("tokenizer.bin"), disable=disable)
    assert "".join(switched.generate("Once upon a time", steps=40)) == reference[:len("".join(switched.generate("Once upon a time", steps=40)))], \
        f"float32 changed with disable={disable}"
    assert ("without " + ", ".join(disable)) in switched.backend if disable else "without" not in switched.backend
    del switched
try:
    llama2_numpy.Llama(read("stories260K.bin"), read("tok512.bin"), disable=("nonsense",))
    raise AssertionError("a switch that does not exist must be refused")
except ValueError:
    pass

# sampling on the kernels: the token NumPy picks for the same random number, the same penalty, a seed reproduces
import numpy as np
generator = np.random.default_rng(0)
class Fixed:
    def __init__(self, value): self.value = value
    def random(self): return self.value
for spread in (0.5, 2.0, 6.0, 12.0):
    for topp in (0.9, 0.5, 1.0):
        logits = (generator.standard_normal(fast.vocab_size) * spread).astype(np.float32)
        for value in (0.0, generator.random(), 1.0 - 1e-12):
            ours, theirs = fast.sample(logits, 0.7, topp, Fixed(value)), llama2_numpy.Llama.sample(fast, logits, 0.7, topp, Fixed(value))
            # rounding may move the border of the nucleus to a neighbour that is just as probable
            assert ours == theirs or abs(logits[ours] - logits[theirs]) < 1e-3, (spread, topp, value, ours, theirs)
lonely = np.full(fast.vocab_size, -100.0, dtype=np.float32)
lonely[123] = 50.0
assert fast.sample(lonely, 0.7, 0.9, generator) == 123 and 0 <= fast.sample(np.zeros_like(lonely), 0.7, 0.9, generator) < lonely.size
history = [int(token) for token in generator.integers(0, fast.vocab_size, 100)] + [5, 5, 5]
ours, theirs = logits.copy(), logits.copy()
fast.penalize(ours, history, 1.3)
llama2_numpy.Llama.penalize(fast, theirs, history, 1.3)
assert np.allclose(ours, theirs, rtol=1e-6) and not np.array_equal(ours, logits), "the penalty of the kernels is off"
settings = dict(steps=40, temperature=0.7, repetition_penalty=1.3, seed=1)
assert "".join(fast.generate("これからの流行りは", **settings)) == "".join(fast.generate("これからの流行りは", **settings)), "a seed must reproduce on the kernels"
# grouped-query attention, and a head size that is no multiple of 4 (stories3_5M: 26)
# ... and a KV cache that has to grow three times on the way (it starts small and doubles), in both engines
llama2_numpy.KV_START = 8
for checkpoint, vocabulary in [("stories260K.bin", "tok512.bin"), ("stories3_5M-v4k.bin", "tok4096.bin")]:
    plain = llama2_numpy.Llama(read(checkpoint), read(vocabulary))
    grouped = kernel_llama(read(checkpoint), read(vocabulary))
    assert grouped.backend.startswith("SIMD") and grouped.n_kv_heads < grouped.n_heads
    assert "".join(grouped.generate("Once upon a time", steps=60)) == "".join(plain.generate("Once upon a time", steps=60)), checkpoint
# T89: the converter's quantize() on the kernels gives the very bytes of NumPy's, for a whole conversion too
quantize_rows = llama2_numpy.kernel_quantizer("simdkernel.so")
values = (np.random.default_rng(7).standard_normal((96, 256)) * 0.1).astype(np.float32)
values[0, :32] = 0.0
values[3, 9] = 5.0
ours, theirs = quantize_rows(values), llama2_convert.quantize(values)
assert np.array_equal(ours[0].reshape(-1), theirs[0].reshape(-1)) and np.array_equal(ours[1], theirs[1]), "quantize_x is not quantize()"
for tensors_of, config_of, arch in ((tensors, gpt2_config, "gpt2"), (neox_tensors, neox_config, "neox")):
    header = llama2_convert.checkpoint_header(llama2_convert.normalize(config_of), llama2_convert.Arrays(tensors_of), positions)
    numpy_int8, kernel_int8 = (bytearray(llama2_convert.checkpoint_size(header, "int8", {"arch": arch})) for _ in range(2))
    llama2_convert.convert_weights(llama2_convert.Arrays(tensors_of), config_of, "int8", positions, numpy_int8)
    llama2_convert.convert_weights(llama2_convert.Arrays(tensors_of), config_of, "int8", positions, kernel_int8, quantize_rows=quantize_rows)
    assert numpy_int8 == kernel_int8, f"the kernels' quantizer changed the int8 {arch} checkpoint"
# T123: bfloat16 widened on the kernels is NumPy's widening to the bit, every 16-bit pattern (NaNs, infinities,
# subnormals, both zeros), in a length that leaves a tail after the groups of 8
widen = llama2_numpy.kernel_widener("simdkernel.so")
patterns = np.arange(65536 + 5, dtype=np.uint32).astype(np.uint16).tobytes()
assert np.array_equal(widen(patterns).view(np.uint32), llama2_convert.bfloat16(patterns).view(np.uint32)), "widen_bf16 is not bfloat16()"
# T136: GGUF's Q8_0 widened on the kernels is NumPy's q8_0() to the bit: every float16 scale (NaNs, infinities,
# subnormals, both zeros) once, with every int8 (-128 and 127 included) across the blocks, in odd numbers of blocks
q8_0 = llama2_numpy.kernel_q8_0("simdkernel.so")
rng = np.random.default_rng(11)
blocks = np.empty((65536 + 3, 34), dtype=np.uint8)
blocks[:, :2] = np.arange(65536 + 3, dtype=np.uint32).astype(np.uint16).view(np.uint8).reshape(-1, 2)
blocks[:, 2:] = rng.integers(0, 256, (65536 + 3, 32), dtype=np.uint8)
blocks[:256, 2:] = (np.arange(256 * 32) % 256).astype(np.uint8).reshape(256, 32)
for count in (0, 1, 3, 65536 + 3):
    raw = blocks[:count].tobytes()
    assert np.array_equal(q8_0(raw).view(np.uint32), llama2_convert.q8_0(raw).view(np.uint32)), f"widen_q8_0 is not q8_0() ({count} blocks)"
# and a whole conversion of Q8_0 tensors (Stream, fed in odd pieces) writes the same bytes with it as without
q8_tensors, q8_config = qwen3_model(64, 4, 2, 16)
q8_header, q8_file = {}, bytearray()
for name, tensor in q8_tensors.items():
    if tensor.ndim == 2:
        count = tensor.size // 32
        data = np.empty((count, 34), dtype=np.uint8)
        data[:, :2] = (np.abs(rng.standard_normal(count)) * 0.01).astype(np.float16).view(np.uint8).reshape(-1, 2)
        data[:, 2:] = rng.integers(0, 256, (count, 32), dtype=np.uint8)
        data, kind = data.tobytes(), "Q8_0"
    else:
        data, kind = tensor.tobytes(), "F32"
    q8_header[name] = {"dtype": kind, "shape": list(tensor.shape), "data_offsets": [len(q8_file), len(q8_file) + len(data)]}
    q8_file += data
for dtype in ("int8", "float32"):
    outs = []
    for widener in (None, q8_0):
        stream = llama2_convert.Stream(q8_header, 0, q8_config, dtype, 24, q8_0=widener)
        for at in range(0, len(q8_file), 1000):
            stream.feed(bytes(q8_file[at:at + 1000]))
        stream.finish()
        outs.append(bytes(stream.out))
    assert outs[0] == outs[1], f"widen_q8_0 changed the {dtype} checkpoint"
# T98: the six bits too: quantize6_x is quantize6() and pack6() to the byte (a group of zeros, ties that round to
# even, the largest value, and a whole conversion)
ties = values.copy()
ties[5, :32] = np.arange(32, dtype=np.float32) - 15.5  # halves: with the largest 31 the scale is 1, and they all tie
ties[5, 31] = 31.0
assert np.array_equal(np.rint(ties[5, :31]), np.rint(ties[5, :31] / 1.0)) and np.any(np.rint(ties[5, :31]) % 2 == 0)
packed, scales = quantize_rows(values, six=True)
sixes, quarter = llama2_numpy.quantize6(values)
assert np.array_equal(packed.reshape(-1), llama2_numpy.pack6(sixes).reshape(-1)) and np.array_equal(scales, quarter), "quantize6_x is not quantize6()"
packed, scales = quantize_rows(ties, six=True)
sixes, quarter = llama2_numpy.quantize6(ties)
assert np.array_equal(packed.reshape(-1), llama2_numpy.pack6(sixes).reshape(-1)) and np.array_equal(scales, quarter), "quantize6_x rounds otherwise"
for tensors_of, config_of, arch in ((tensors, gpt2_config, "gpt2"), (neox_tensors, neox_config, "neox")):
    header = llama2_convert.checkpoint_header(llama2_convert.normalize(config_of), llama2_convert.Arrays(tensors_of), positions)
    numpy_six, kernel_six = (bytearray(llama2_convert.checkpoint_size(header, "int6", {"arch": arch})) for _ in range(2))
    llama2_convert.convert_weights(llama2_convert.Arrays(tensors_of), config_of, "int6", positions, numpy_six)
    llama2_convert.convert_weights(llama2_convert.Arrays(tensors_of), config_of, "int6", positions, kernel_six, quantize_rows=quantize_rows)
    assert numpy_six == kernel_six, f"the kernels' quantizer changed the int6 {arch} checkpoint"
# T110: float32 to float16 as NumPy rounds it, and attention over a float16 cache the same to the bit as over the
# float32 values it stands for (every head alone, and heads in two ranges)
kernel = llama2_numpy.load_kernels("simdkernel.so")
rng = np.random.default_rng(11)
wide = np.concatenate([rng.standard_normal(4000).astype(np.float32) * 10 ** rng.uniform(-9, 5, 4000).astype(np.float32),
                       np.array([0.0, -0.0, 65504.0, 65520.0, 1e6, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 5.96e-8, -1.5e-5,
                                 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 2.0 ** -14, 2.0 ** -14 * (1 - 2.0 ** -12),
                                 2.0 ** -14 * (1 - 2.0 ** -11)], dtype=np.float32)])  # the last: the subnormal that rounds up into the normals
halves = np.empty(wide.size, dtype=np.float16)
kernel["to_f16"](halves.ctypes.data, wide.ctypes.data, wide.size)
with np.errstate(over="ignore"):
    assert np.array_equal(halves.view(np.uint16), wide.astype(np.float16).view(np.uint16)), "to_f16 is not astype(float16)"
heads, kv_heads, head_size, count = 8, 2, 20, 37  # head_size 20: the loops' remainders too
q = rng.standard_normal(heads * head_size).astype(np.float32)
keys = (rng.standard_normal((count, kv_heads * head_size)) * 3).astype(np.float16)
values = rng.standard_normal((count, kv_heads * head_size)).astype(np.float16)
wide_keys, wide_values = keys.astype(np.float32), values.astype(np.float32)
scores = np.empty(heads * count, dtype=np.float32)
out32, out16 = (np.empty(heads * head_size, dtype=np.float32) for _ in range(2))
kernel["attention"](out32.ctypes.data, q.ctypes.data, wide_keys.ctypes.data, wide_values.ctypes.data, scores.ctypes.data,
                    count - 1, heads, kv_heads, head_size, 0, heads)
kernel["attention_f16"](out16.ctypes.data, q.ctypes.data, keys.ctypes.data, values.ctypes.data, scores.ctypes.data,
                        count - 1, heads, kv_heads, head_size, 0, 3)
kernel["attention_f16"](out16.ctypes.data, q.ctypes.data, keys.ctypes.data, values.ctypes.data, scores.ctypes.data,
                        count - 1, heads, kv_heads, head_size, 3, heads)
assert np.array_equal(out32, out16), "attention_f16 is not attention over the widened cache"
# NumPy takes over when the kernels cannot be loaded
assert llama2_numpy.Llama(read("stories15M.f32"), read("tokenizer.bin"), kernels="missing.so").backend == "NumPy"

f"Python {sys.version.split()[0]}: kernels {simd15.stats['tokens_per_second']:.0f} against NumPy {numpy15.stats['tokens_per_second']:.0f} tok/s, {fast.backend} {fast.stats['tokens_per_second']:.0f} tok/s, stories260K {stories.stats['tokens_per_second']:.0f} tok/s, tiny-lm {tiny.stats['tokens_per_second']:.0f} tok/s, {japanese!r}"
`);
// T144: a file opened from a folder: the worker hands checkpoint_dtype() the model's options as they are, a JavaScript
// object, and it reads the form out of them (a Qwen3 with heads twice as wide as dim / heads)
{
  const header = [64, 96, 2, 4, 2, 320, 24];
  const form = { qk_norm: true, head_dim: 32, tokenizer_kind: "bpe" };
  const size = pyodide.runPython(`import llama2_convert
llama2_convert.checkpoint_size([64, 96, 2, 4, 2, 320, 24], "int8", {"qk_norm": True, "head_dim": 32})`);
  const engine = pyodide.pyimport("llama2_numpy");
  if (engine.checkpoint_dtype(header, size, form) !== "int8") throw new Error("checkpoint_dtype() misread the options");
  let refused = false;
  try {
    engine.checkpoint_dtype(header, size, { qk_norm: true });
  } catch {
    refused = true;
  }
  if (!refused) throw new Error("checkpoint_dtype() took the file without its head_dim");
  engine.destroy();
}
// T151: the CPU's sampling in JavaScript (public/shaders.js's penalizeLikeCpu and sampleLikeCpu: what /benchmark/ holds
// the GPU's SAMPLE to) is the kernel's: the same logits, history and random number, the same penalized logits and the
// same token (or, as above, a neighbour of about the same logit where rounding moves a border: the kernel's exp() is a
// polynomial, JavaScript's is not)
{
  const { penalizeLikeCpu, sampleLikeCpu } = await import("../public/shaders.js");
  pyodide.runPython(`
def kernel_pick(buffer, temperature, topp, value, history, penalty):
    logits = np.frombuffer(buffer.to_bytes(), dtype=np.float32).copy()
    if penalty != 1.0:
        fast.penalize(logits, [int(token) for token in history], penalty)
    return fast.sample(logits, temperature, topp, Fixed(value)), logits.tobytes()
`);
  const pick = pyodide.globals.get("kernel_pick"), vocab = pyodide.globals.get("fast").vocab_size;
  let cases = 0, same = 0;
  for (const spread of [0.5, 2, 6, 12]) {
    for (const topp of [0.9, 0.5, 1]) {
      for (const temperature of [0.7, 1.3, 0]) {
        for (const penalty of [1, 1.3]) {
          for (const value of [0, Math.random(), 1 - 1e-12]) {
            const logits = new Float32Array(vocab).map(() => spread * Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()));
            // the window: 64 of the most likely (a repeated one among them) and older tokens that must not count
            const ranked = [...logits.keys()].sort((a, b) => logits[b] - logits[a]);
            const history = [...ranked.slice(0, 10), ...ranked.slice(10, 70), ranked[12], ranked.at(-1), ranked.at(-2)];
            const result = pick(new Uint8Array(logits.buffer), temperature, topp, value, history, penalty);
            const [theirs, penalized] = result.toJs();
            result.destroy();
            const ours = Float32Array.from(logits);
            penalizeLikeCpu(ours, history, penalty);
            if (!ours.every((v, i) => v === new Float32Array(penalized.buffer, penalized.byteOffset, vocab)[i])) {
              throw new Error(`penalizeLikeCpu is not the kernel's penalize (penalty ${penalty})`);
            }
            const token = sampleLikeCpu(ours, temperature, topp, value);
            if (token !== theirs && !(Math.abs(ours[token] - ours[theirs]) < 1e-3)) {
              throw new Error(`sampleLikeCpu picked ${token}, the kernel ${theirs} (spread ${spread}, top-p ${topp}, T ${temperature}, r ${value})`);
            }
            cases++;
            same += token === theirs;
          }
        }
      }
    }
  }
  pick.destroy();
  if (same < 0.95 * cases) throw new Error(`sampleLikeCpu picked the kernel's token in ${same} of ${cases} cases only`);
  console.log(`T151: sampleLikeCpu picked the kernel's token in ${same} of ${cases} cases (the rest a neighbour of the same logit)`);
}
console.log(`Pyodide ${version}, ${report} (${((Date.now() - started) / 1000).toFixed(1)}s)`);
console.log("ok");
