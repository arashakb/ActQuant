# Export OpenVLA-OFT model to GGUF format
# Outputs: dinov2.gguf, siglip.gguf, proj.gguf, action_head.gguf, proprio_projector.gguf, and language_model folder
#
# Usage:
#   python export_openvla_oft.py
#
# After running, convert LLM with:
#   python convert_hf_to_gguf.py <output_dir>/language_model --outfile <output_dir>/llm.gguf

import gguf
from transformers import AutoModelForVision2Seq, AutoTokenizer
import torch
import numpy as np
import os

# ============== CONFIGURATION ==============
# Model path (local folder with safetensors, config.json, etc.)
model_path = "/path/to/openvla-oft-checkpoints/oft_combined"

# Output directory for all exported files
output_dir = "/path/to/openvla-oft-checkpoints/oft_combined_gguf"

# Action head checkpoint path (usually in the same folder)
action_head_path = os.path.join(model_path, "action_head--300000_checkpoint.pt")

# Proprio projector checkpoint path (usually in the same folder)
proprio_projector_path = os.path.join(model_path, "proprio_projector--300000_checkpoint.pt")

# Precision for saving GGUF files: "bf16", "fp16", or "f32"
OUTPUT_PRECISION = "fp16"

dtype = torch.bfloat16
# ===========================================

os.makedirs(output_dir, exist_ok=True)

# Load VLA model
print(f"Loading model from: {model_path}")
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda")

# Note: LoRA adapter merging is NOT needed for OFT combined checkpoints.
# The base safetensors already contain the fine-tuned weights.
# The lora_adapter/ folder is a training artifact and should not be applied again.
# (Verified: applying LoRA on top of the base weights causes catastrophic performance
# degradation from ~92% to ~8%, confirming the base weights are already merged.)

# Save language model
print("Saving language model...")
llm = vla.language_model.to(dtype)
llm_output_path = os.path.join(output_dir, "language_model")
llm.save_pretrained(llm_output_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(llm_output_path)
print(f"  Saved to: {llm_output_path}")

# Extract vision components
featurizer = vla.vision_backbone.featurizer.to(dtype)       # DINOv2
fused_featurizer = vla.vision_backbone.fused_featurizer.to(dtype)  # SigLIP
projector = vla.projector.to(dtype)


# ============== Helper functions ==============
def add_config(gguf_writer: gguf.GGUFWriter, model_cfg):
    for k, v in model_cfg.items():
        if isinstance(v, bool):
            gguf_writer.add_bool(k, v)
        elif isinstance(v, float):
            gguf_writer.add_float32(k, v)
        elif isinstance(v, int):
            gguf_writer.add_uint32(k, v)
        elif isinstance(v, str):
            gguf_writer.add_string(k, v)
        elif isinstance(v, list):
            gguf_writer.add_array(k, v)
        else:
            raise ValueError(f"Unsupported type: {type(v)}")


def bfloat16_to_uint16_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.cpu().detach().squeeze()
    if tensor.dtype == torch.bfloat16:
        tensor_int16 = tensor.view(torch.int16)
        result = tensor_int16.numpy().view(np.uint16)
    else:
        result = tensor.numpy().view(np.uint16)
    return result


# ============== Export vision models and projector ==============
print("\nExporting vision encoders and projector...")

cfgs = {
    "proj": {},
    "dinov2": {
        "clip.projector_type": "openvla",
        "clip.has_vision_encoder": True,
        "clip.vision.embedding_length": 1024,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4096,
        "clip.vision.block_count": 24,
        "clip.vision.projection_dim": 2560,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_size": 224,
        "clip.vision.patch_size": 14,
        "clip.use_gelu": True,
        "clip.vision.image_mean": [0.484375, 0.455078125, 0.40625],
        "clip.vision.image_std": [0.228515625, 0.2236328125, 0.224609375],
        "clip.vision.feature_layer": [24-2],
    },
    "siglip": {
        "clip.projector_type": "openvla",
        "clip.has_vision_encoder": True,
        "clip.vision.embedding_length": 1152,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4304,
        "clip.vision.block_count": 27,
        "clip.vision.projection_dim": 2560,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_size": 224,
        "clip.vision.patch_size": 14,
        "clip.use_gelu": True,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
        "clip.vision.feature_layer": [27-2],
    }
}

model_params = {
    "cls_token": "v.class_embd",
    "reg_token": "v.reg_embd",
    "patch_embed.proj.weight": "v.patch_embd.weight",
    "patch_embed.proj.bias": "v.patch_embd.bias",
    "pos_embed": "v.position_embd.weight",
    "fc1.weight": "fc1.weight",
    "fc1.bias": "fc1.bias",
    "fc2.weight": "fc2.weight",
    "fc2.bias": "fc2.bias",
    "fc3.weight": "fc3.weight",
    "fc3.bias": "fc3.bias",
}

blk_params = {
    "blocks.%d.attn.qkv.weight": "v.blk.%d.attn_q.weight",
    "blocks.%d.attn.proj.weight": "v.blk.%d.attn_out.weight",
    "blocks.%d.norm1.weight": "v.blk.%d.ln1.weight",
    "blocks.%d.norm2.weight": "v.blk.%d.ln2.weight",
    "blocks.%d.ls1.scale_factor": "v.blk.%d.ls1.weight",
    "blocks.%d.ls2.scale_factor": "v.blk.%d.ls2.weight",
    "blocks.%d.attn.qkv.bias": "v.blk.%d.attn_q.bias",
    "blocks.%d.attn.proj.bias": "v.blk.%d.attn_out.bias",
    "blocks.%d.norm1.bias": "v.blk.%d.ln1.bias",
    "blocks.%d.norm2.bias": "v.blk.%d.ln2.bias",
    "blocks.%d.mlp.fc1.weight": "v.blk.%d.ffn_up.weight",
    "blocks.%d.mlp.fc2.weight": "v.blk.%d.ffn_down.weight",
    "blocks.%d.mlp.fc1.bias": "v.blk.%d.ffn_up.bias",
    "blocks.%d.mlp.fc2.bias": "v.blk.%d.ffn_down.bias",
}

model_names = ["siglip", "dinov2", "proj"]
models = [fused_featurizer, featurizer, projector]

for model_name, model in zip(model_names, models):
    print(f"  Exporting {model_name}...")
    gguf_output_path = os.path.join(output_dir, f"{model_name}.gguf")
    gguf_writer = gguf.GGUFWriter(gguf_output_path, model_name)

    model_cfg = cfgs[model_name]
    add_config(gguf_writer, model_cfg)
    cur_params = {}
    cur_params.update(model_params)

    n_blocks = model_cfg.get("clip.vision.block_count", 0)
    for b_id in range(n_blocks):
        for k, v in blk_params.items():
            new_k = k % b_id
            if new_k in cur_params:
                raise ValueError(f"{new_k} already exists")
            cur_params[new_k] = v % b_id

    for tensor_name, new_tensor_name in cur_params.items():
        if tensor_name not in model.state_dict():
            continue
        param = model.state_dict()[tensor_name]

        target_dtype = None
        raw_dtype = None

        if OUTPUT_PRECISION == "bf16":
            target_dtype = torch.bfloat16
            raw_dtype = gguf.GGMLQuantizationType.BF16
        elif OUTPUT_PRECISION == "fp16":
            target_dtype = torch.float16
            raw_dtype = None
        else:
            target_dtype = torch.float32
            raw_dtype = None

        # 1D tensors (biases, norms, embeddings) must be F32 because CUDA
        # binbcast kernels don't support mixed F32+F16 in ggml_add/ggml_mul,
        # and ggml_concat requires matching types for class/reg embeddings.
        # 2D weight matrices are fine as F16 since ggml_mul_mat handles conversion.
        is_1d_or_embd = param.dim() <= 1 or tensor_name in ("cls_token", "reg_token", "pos_embed")
        if is_1d_or_embd:
            target_dtype = torch.float32
            raw_dtype = None

        if param.dtype != target_dtype:
            param = param.to(target_dtype)

        if "qkv" in tensor_name:
            new_q_name = new_tensor_name
            new_k_name = new_tensor_name.replace("attn_q", "attn_k")
            new_v_name = new_tensor_name.replace("attn_q", "attn_v")
            split_length = param.shape[0] // 3
            q, k, v = torch.split(param, split_length, dim=0)

            if raw_dtype == gguf.GGMLQuantizationType.BF16:
                gguf_writer.add_tensor(new_q_name, bfloat16_to_uint16_numpy(q), raw_dtype=raw_dtype)
                gguf_writer.add_tensor(new_k_name, bfloat16_to_uint16_numpy(k), raw_dtype=raw_dtype)
                gguf_writer.add_tensor(new_v_name, bfloat16_to_uint16_numpy(v), raw_dtype=raw_dtype)
            else:
                gguf_writer.add_tensor(new_q_name, q.squeeze().cpu().numpy(), raw_dtype=raw_dtype)
                gguf_writer.add_tensor(new_k_name, k.squeeze().cpu().numpy(), raw_dtype=raw_dtype)
                gguf_writer.add_tensor(new_v_name, v.squeeze().cpu().numpy(), raw_dtype=raw_dtype)
        else:
            if raw_dtype == gguf.GGMLQuantizationType.BF16:
                gguf_writer.add_tensor(new_tensor_name, bfloat16_to_uint16_numpy(param), raw_dtype=raw_dtype)
            else:
                gguf_writer.add_tensor(new_tensor_name, param.squeeze().cpu().numpy(), raw_dtype=raw_dtype)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"    Saved to: {gguf_output_path}")


# ============== Export OFT action head ==============
print("\nExporting OFT action head...")

if os.path.exists(action_head_path):
    checkpoint = torch.load(action_head_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Detect hyperparameters from checkpoint
    ln1_key = next((k for k in state_dict.keys() if 'layer_norm1.weight' in k), None)
    fc1_key = next((k for k in state_dict.keys() if 'fc1.weight' in k), None)
    fc2_key = next((k for k in state_dict.keys() if 'fc2.weight' in k), None)

    mlp_input_dim = state_dict[ln1_key].shape[0]
    hidden_dim = state_dict[fc1_key].shape[0]
    action_dim = state_dict[fc2_key].shape[0]
    input_dim = mlp_input_dim // action_dim
    num_blocks = sum(1 for k in state_dict.keys() if 'mlp_resnet_blocks' in k and 'ffn.0.weight' in k)

    print(f"  action_dim: {action_dim}, input_dim: {input_dim}, hidden_dim: {hidden_dim}, num_blocks: {num_blocks}")

    # Tensor mapping (strip "module." prefix)
    action_head_mapping = {
        "module.model.layer_norm1.weight": "model.layer_norm1.weight",
        "module.model.layer_norm1.bias": "model.layer_norm1.bias",
        "module.model.fc1.weight": "model.fc1.weight",
        "module.model.fc1.bias": "model.fc1.bias",
        "module.model.layer_norm2.weight": "model.layer_norm2.weight",
        "module.model.layer_norm2.bias": "model.layer_norm2.bias",
        "module.model.fc2.weight": "model.fc2.weight",
        "module.model.fc2.bias": "model.fc2.bias",
    }
    for i in range(num_blocks):
        action_head_mapping[f"module.model.mlp_resnet_blocks.{i}.ffn.0.weight"] = f"model.mlp_resnet_blocks.{i}.ffn.0.weight"
        action_head_mapping[f"module.model.mlp_resnet_blocks.{i}.ffn.0.bias"] = f"model.mlp_resnet_blocks.{i}.ffn.0.bias"
        action_head_mapping[f"module.model.mlp_resnet_blocks.{i}.ffn.1.weight"] = f"model.mlp_resnet_blocks.{i}.ffn.1.weight"
        action_head_mapping[f"module.model.mlp_resnet_blocks.{i}.ffn.1.bias"] = f"model.mlp_resnet_blocks.{i}.ffn.1.bias"

    # Write action head GGUF
    action_head_output = os.path.join(output_dir, "action_head.gguf")
    gguf_writer = gguf.GGUFWriter(action_head_output, "openvla_oft_action_head")

    # Add hyperparameters
    add_config(gguf_writer, {
        "action_dim": action_dim,
        "num_actions_chunk": 8,
        "num_blocks": num_blocks,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    })

    # Export tensors
    for src_name, dst_name in action_head_mapping.items():
        if src_name not in state_dict:
            continue
        param = state_dict[src_name]

        if OUTPUT_PRECISION == "bf16":
            target_dtype = torch.bfloat16
            raw_dtype = gguf.GGMLQuantizationType.BF16
        elif OUTPUT_PRECISION == "fp16":
            target_dtype = torch.float16
            raw_dtype = None
        else:
            target_dtype = torch.float32
            raw_dtype = None

        # 1D tensors (biases, norms) must be F32 for CUDA compatibility
        if param.dim() <= 1:
            target_dtype = torch.float32
            raw_dtype = None

        if param.dtype != target_dtype:
            param = param.to(target_dtype)

        if raw_dtype == gguf.GGMLQuantizationType.BF16:
            gguf_writer.add_tensor(dst_name, bfloat16_to_uint16_numpy(param), raw_dtype=raw_dtype)
        else:
            gguf_writer.add_tensor(dst_name, param.squeeze().cpu().numpy(), raw_dtype=raw_dtype)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"    Saved to: {action_head_output}")
else:
    print(f"  Warning: Action head not found at {action_head_path}")


# ============== Export Proprio Projector ==============
print("\nExporting proprio projector...")

if os.path.exists(proprio_projector_path):
    checkpoint = torch.load(proprio_projector_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Print keys for debugging
    print(f"  Checkpoint keys: {list(state_dict.keys())}")

    # Detect hyperparameters from checkpoint (handles module. prefix)
    fc1_key = next((k for k in state_dict.keys() if k.endswith('fc1.weight')), None)
    fc2_key = next((k for k in state_dict.keys() if k.endswith('fc2.weight')), None)

    if fc1_key and fc2_key:
        proprio_dim = state_dict[fc1_key].shape[1]  # fc1: (llm_dim, proprio_dim)
        llm_dim = state_dict[fc1_key].shape[0]      # fc1: (llm_dim, proprio_dim)

        print(f"  proprio_dim: {proprio_dim}, llm_dim: {llm_dim}")

        # Tensor mapping (strip "module." prefix if present)
        proprio_mapping = {
            "module.fc1.weight": "fc1.weight",
            "module.fc1.bias": "fc1.bias",
            "module.fc2.weight": "fc2.weight",
            "module.fc2.bias": "fc2.bias",
        }

        # Also try without module prefix for compatibility
        if "fc1.weight" in state_dict:
            proprio_mapping = {
                "fc1.weight": "fc1.weight",
                "fc1.bias": "fc1.bias",
                "fc2.weight": "fc2.weight",
                "fc2.bias": "fc2.bias",
            }

        # Write proprio projector GGUF
        proprio_output = os.path.join(output_dir, "proprio_projector.gguf")
        gguf_writer = gguf.GGUFWriter(proprio_output, "openvla_proprio_projector")

        # Add hyperparameters
        add_config(gguf_writer, {
            "proprio_dim": proprio_dim,
            "llm_dim": llm_dim,
        })

        # Export tensors
        for src_name, dst_name in proprio_mapping.items():
            if src_name not in state_dict:
                print(f"  Warning: {src_name} not found in checkpoint")
                continue
            param = state_dict[src_name]

            if OUTPUT_PRECISION == "bf16":
                target_dtype = torch.bfloat16
                raw_dtype = gguf.GGMLQuantizationType.BF16
            elif OUTPUT_PRECISION == "fp16":
                target_dtype = torch.float16
                raw_dtype = None
            else:
                target_dtype = torch.float32
                raw_dtype = None

            # 1D tensors (biases) must be F32 for CUDA compatibility
            if param.dim() <= 1:
                target_dtype = torch.float32
                raw_dtype = None

            if param.dtype != target_dtype:
                param = param.to(target_dtype)

            if raw_dtype == gguf.GGMLQuantizationType.BF16:
                gguf_writer.add_tensor(dst_name, bfloat16_to_uint16_numpy(param), raw_dtype=raw_dtype)
            else:
                gguf_writer.add_tensor(dst_name, param.squeeze().cpu().numpy(), raw_dtype=raw_dtype)

        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()
        print(f"    Saved to: {proprio_output}")
    else:
        print(f"  Warning: Could not find fc1/fc2 weights in proprio_projector checkpoint")
else:
    print(f"  Warning: Proprio projector not found at {proprio_projector_path}")


# ============== Summary ==============
print("\n" + "="*50)
print("Export complete!")
print("="*50)
print(f"\nOutput files in {output_dir}:")
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")
    else:
        print(f"  {f}/ (directory)")

print(f"\nNext step - convert LLM to GGUF:")
print(f"  python convert_hf_to_gguf.py {llm_output_path} --outfile {output_dir}/llm.gguf")
