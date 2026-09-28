# A small Qwen3.5 (T229) of random weights, the same in the tests (NumPy only) and in tests/qwen35_reference.py,
# which runs it in transformers and writes what it computes to tests/fixtures/qwen35-logits.json.
import numpy as np

# two runs of three layers of Gated DeltaNet and one of attention; GQA in both kinds (4 heads of q over 2 of k and v,
# 4 heads of v over 2 of q and k); a head of attention of 32 of which RoPE turns 8
CONFIG = {"model_type": "qwen3_5_text", "vocab_size": 96, "hidden_size": 64, "intermediate_size": 96,
          "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
          "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
          "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
          "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "hidden_act": "silu",
          "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "rope_type": "default"},
          "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
          "bos_token_id": 1, "eos_token_id": 2}
TOKENS = [1, 17, 42, 5, 88, 23, 23, 60, 9, 71, 30, 2, 45, 45, 45, 12]


def weights(config=CONFIG, seed=0):
    """Every tensor of Qwen3_5ForCausalLM, as transformers names them, of random values: the norms away from 0 (their
    weight is 1 + w), A_log and dt_bias of a decay that neither vanishes nor stays put."""
    rng = np.random.default_rng(seed)
    dim, hidden = config["hidden_size"], config["intermediate_size"]
    heads, kv, head = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    k_heads, v_heads = config["linear_num_key_heads"], config["linear_num_value_heads"]
    k_head, v_head, taps = config["linear_key_head_dim"], config["linear_value_head_dim"], config["linear_conv_kernel_dim"]
    k_dim, v_dim = k_heads * k_head, v_heads * v_head
    normal = lambda *shape, scale=0.2: (rng.standard_normal(shape) * scale).astype(np.float32)
    tensors = {"model.embed_tokens.weight": normal(config["vocab_size"], dim, scale=1.0),
               "model.norm.weight": normal(dim), "lm_head.weight": normal(config["vocab_size"], dim)}
    for layer, kind in enumerate(config["layer_types"]):
        at = f"model.layers.{layer}."
        tensors.update({at + "input_layernorm.weight": normal(dim), at + "post_attention_layernorm.weight": normal(dim),
                        at + "mlp.gate_proj.weight": normal(hidden, dim), at + "mlp.up_proj.weight": normal(hidden, dim),
                        at + "mlp.down_proj.weight": normal(dim, hidden)})
        if kind == "full_attention":
            tensors.update({at + "self_attn.q_proj.weight": normal(2 * heads * head, dim),
                            at + "self_attn.k_proj.weight": normal(kv * head, dim),
                            at + "self_attn.v_proj.weight": normal(kv * head, dim),
                            at + "self_attn.o_proj.weight": normal(dim, heads * head),
                            at + "self_attn.q_norm.weight": normal(head), at + "self_attn.k_norm.weight": normal(head)})
        else:
            tensors.update({at + "linear_attn.in_proj_qkv.weight": normal(2 * k_dim + v_dim, dim),
                            at + "linear_attn.in_proj_z.weight": normal(v_dim, dim),
                            at + "linear_attn.in_proj_b.weight": normal(v_heads, dim),
                            at + "linear_attn.in_proj_a.weight": normal(v_heads, dim),
                            at + "linear_attn.conv1d.weight": normal(2 * k_dim + v_dim, 1, taps, scale=0.5),
                            at + "linear_attn.A_log": np.log(rng.uniform(0.5, 4.0, v_heads)).astype(np.float32),
                            at + "linear_attn.dt_bias": normal(v_heads, scale=1.0),
                            at + "linear_attn.norm.weight": (1.0 + normal(v_head)).astype(np.float32),
                            at + "linear_attn.out_proj.weight": normal(dim, v_dim)})
    return tensors
