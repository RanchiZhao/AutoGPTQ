from ._base import BaseGPTQForCausalLM
class MiniCPM3vGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "llm.model.layers"
    outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm", "vpm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]