#!/usr/bin/env python3
"""
export_pi05_llm.py

Exports the PaliGemma LLM portion of a Pi0.5 model as a standalone
llama.cpp-compatible GGUF (BF16). This file is used as input to:
  - llama-imatrix  (collect importance data from calibration embeddings)
  - llama-quantize (apply IQ2_XS, IQ3_S, Q4_K_M, etc.)

The output GGUF uses standard llama.cpp tensor naming (blk.*) and
architecture "llama", so all existing llama.cpp tools work with it.

After quantizing with llama-quantize, use merge_pi05_llm.py to patch
the result back into the full pi05.gguf.

Usage:
    python export_pi05_llm.py \\
        -d /path/to/pi05_libero_base_pytorch \\
        -o /path/to/output_dir

Output:
    <output_dir>/pali_llm_bf16.gguf
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import gguf
from gguf import GGUFReader
from gguf.constants import GGUFValueType
import torch
from safetensors.torch import load_file as torch_load_file


def _copy_kv_field(writer: gguf.GGUFWriter, key: str, field) -> None:
    """Copy a single KV field from GGUFReader to GGUFWriter."""
    if not field.types:
        return
    ftype = field.types[0]
    if ftype == GGUFValueType.UINT8:    writer.add_uint8(key,   field.parts[-1][0])
    elif ftype == GGUFValueType.INT8:   writer.add_int8(key,    field.parts[-1][0])
    elif ftype == GGUFValueType.UINT16: writer.add_uint16(key,  field.parts[-1][0])
    elif ftype == GGUFValueType.INT16:  writer.add_int16(key,   field.parts[-1][0])
    elif ftype == GGUFValueType.UINT32: writer.add_uint32(key,  field.parts[-1][0])
    elif ftype == GGUFValueType.INT32:  writer.add_int32(key,   field.parts[-1][0])
    elif ftype == GGUFValueType.FLOAT32:writer.add_float32(key, field.parts[-1][0])
    elif ftype == GGUFValueType.BOOL:   writer.add_bool(key,    bool(field.parts[-1][0]))
    elif ftype == GGUFValueType.STRING: writer.add_string(key,  bytes(field.parts[-1]).decode("utf-8"))
    elif ftype == GGUFValueType.UINT64: writer.add_uint64(key,  field.parts[-1][0])
    elif ftype == GGUFValueType.INT64:  writer.add_int64(key,   field.parts[-1][0])
    elif ftype == GGUFValueType.FLOAT64:writer.add_float64(key, field.parts[-1][0])
    elif ftype == GGUFValueType.ARRAY:
        if len(field.types) < 2:
            return
        etype = field.types[1]
        data  = field.parts[-1]
        if   etype == GGUFValueType.FLOAT32: writer.add_array(key, list(data.astype(np.float32)))
        elif etype == GGUFValueType.INT32:   writer.add_array(key, list(data.astype(np.int32)))
        elif etype == GGUFValueType.UINT32:  writer.add_array(key, list(data.astype(np.uint32)))
        elif etype == GGUFValueType.STRING:
            strs = []
            for p in field.parts[2:]:
                try: strs.append(bytes(p).decode("utf-8"))
                except Exception: pass
            writer.add_array(key, strs)
        else:
            writer.add_array(key, list(data))


def _add_tokenizer_from_spm(writer: gguf.GGUFWriter, output_dir: str) -> None:
    """Embed SentencePiece tokenizer into the GGUF — required by llama_model_load."""
    import sentencepiece as spm

    tok_path = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(tok_path):
        print(f"  WARNING: {tok_path} not found — no tokenizer embedded (llama-imatrix may fail)")
        return

    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab_size = sp.GetPieceSize()

    # Token type constants (ggml): 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
    def _tok_type(i: int) -> int:
        if sp.IsUnknown(i):  return 2
        if sp.IsControl(i):  return 3
        if sp.IsByte(i):     return 6
        return 1

    tokens = [sp.IdToPiece(i) for i in range(vocab_size)]
    scores = [sp.GetScore(i)  for i in range(vocab_size)]
    types  = [_tok_type(i)    for i in range(vocab_size)]

    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens",     tokens)
    writer.add_array("tokenizer.ggml.scores",     scores)
    writer.add_array("tokenizer.ggml.token_type", types)
    writer.add_uint32("tokenizer.ggml.bos_token_id",     sp.bos_id())
    writer.add_uint32("tokenizer.ggml.eos_token_id",     sp.eos_id())
    writer.add_uint32("tokenizer.ggml.padding_token_id", max(sp.pad_id(), 0))
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)
    print(f"  Embedded SPM tokenizer: {vocab_size} tokens (BOS={sp.bos_id()}, EOS={sp.eos_id()})")


# PaliGemma LLM architecture constants (Gemma 2B / PaliGemma)
N_EMBD   = 2048
N_FF     = 16384
N_HEADS  = 8
N_KV     = 1
N_LAYERS = 18
HEAD_DIM = 256          # N_EMBD // N_HEADS
VOCAB    = 257152
CTX_LEN  = 2048
RMS_EPS  = 1e-6


def detect_prefix(st: dict) -> str:
    keys = list(st.keys())
    return "model." if keys[0].startswith("model.") else ""


def get(st: dict, key: str, prefix: str) -> np.ndarray:
    full = prefix + key
    return st[full].float().numpy()


def write_f16_tensor(writer: gguf.GGUFWriter, name: str, arr: np.ndarray):
    """Write weight matrix as F16."""
    t = torch.from_numpy(arr).to(torch.float16)
    raw = t.numpy()
    writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.F16)


def write_f32_tensor(writer: gguf.GGUFWriter, name: str, arr: np.ndarray):
    """Write norm/bias tensors as F32 — required: op_mul with F32 activations needs F32 src1."""
    raw = arr.astype(np.float32)
    writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType.F32)


def export_llm(st: dict, output_dir: str):
    prefix = detect_prefix(st)
    if prefix:
        print(f"  Detected tensor prefix: '{prefix}'")

    os.makedirs(output_dir, exist_ok=True)
    gguf_path = os.path.join(output_dir, "pali_llm_bf16.gguf")
    writer = gguf.GGUFWriter(gguf_path, "llama")

    # ── Metadata ─────────────────────────────────────────────────────────────
    writer.add_name("pali_llm")
    writer.add_context_length(CTX_LEN)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYERS)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEADS)
    writer.add_head_count_kv(N_KV)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_vocab_size(VOCAB)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    # SwiGLU FFN (gate * up, then down)
    writer.add_uint32("llama.expert_count", 0)
    writer.add_uint32("llama.expert_used_count", 0)

    # ── Tokenizer (required by llama_model_load even for precomputed embeddings) ─
    print("[0/3] Embedding SentencePiece tokenizer...")
    _add_tokenizer_from_spm(writer, output_dir)

    pali_prefix = "paligemma_with_expert.paligemma.model.language_model"

    # ── Token embedding + tied output head ──────────────────────────────────
    print("[1/3] Exporting token embedding...")
    embed_w = get(st, "paligemma_with_expert.paligemma.lm_head.weight", prefix)
    print(f"  embed shape: {embed_w.shape}")
    write_f16_tensor(writer, "token_embd.weight", embed_w)
    write_f16_tensor(writer, "output.weight", embed_w)  # tied

    # ── Output norm (F32, add 1: Gemma delta convention) ─────────────────────
    print("[2/3] Exporting output norm...")
    out_norm_w = get(st, f"{pali_prefix}.norm.weight", prefix)
    write_f32_tensor(writer, "output_norm.weight", 1.0 + out_norm_w)

    # ── Transformer layers ───────────────────────────────────────────────────
    print(f"[3/3] Exporting {N_LAYERS} transformer layers...")
    for i in range(N_LAYERS):
        lp = f"{pali_prefix}.layers.{i}"

        # Pre-attention RMSNorm (F32, add 1: Gemma stores delta w/ formula (1+w), llama uses w directly)
        attn_norm = get(st, f"{lp}.input_layernorm.weight", prefix)
        write_f32_tensor(writer, f"blk.{i}.attn_norm.weight", 1.0 + attn_norm)

        # Attention projections (F16: large matrices)
        q_w = get(st, f"{lp}.self_attn.q_proj.weight", prefix)
        k_w = get(st, f"{lp}.self_attn.k_proj.weight", prefix)
        v_w = get(st, f"{lp}.self_attn.v_proj.weight", prefix)
        o_w = get(st, f"{lp}.self_attn.o_proj.weight", prefix)
        write_f16_tensor(writer, f"blk.{i}.attn_q.weight", q_w)
        write_f16_tensor(writer, f"blk.{i}.attn_k.weight", k_w)
        write_f16_tensor(writer, f"blk.{i}.attn_v.weight", v_w)
        write_f16_tensor(writer, f"blk.{i}.attn_output.weight", o_w)  # llama.cpp name

        # Pre-FFN RMSNorm (F32, add 1: same Gemma delta convention)
        ffn_norm = get(st, f"{lp}.post_attention_layernorm.weight", prefix)
        write_f32_tensor(writer, f"blk.{i}.ffn_norm.weight", 1.0 + ffn_norm)

        # FFN (F16: large matrices)
        gate_w = get(st, f"{lp}.mlp.gate_proj.weight", prefix)
        up_w   = get(st, f"{lp}.mlp.up_proj.weight", prefix)
        down_w = get(st, f"{lp}.mlp.down_proj.weight", prefix)
        write_f16_tensor(writer, f"blk.{i}.ffn_gate.weight", gate_w)
        write_f16_tensor(writer, f"blk.{i}.ffn_up.weight", up_w)
        write_f16_tensor(writer, f"blk.{i}.ffn_down.weight", down_w)

        if (i + 1) % 6 == 0 or i == N_LAYERS - 1:
            print(f"  Exported layers 0–{i}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = os.path.getsize(gguf_path) / 1024**3
    print(f"\nWritten: {gguf_path}  ({size_gb:.2f} GB)")
    print("This file is ready for: llama-imatrix and llama-quantize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PaliGemma LLM from Pi0.5 as standalone llama.cpp BF16 GGUF"
    )
    parser.add_argument("-d", "--dir-model", required=True,
                        help="Path to pi05 model directory (contains model.safetensors)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: same as model dir)")
    args = parser.parse_args()

    model_dir = args.dir_model
    output_dir = args.output_dir or model_dir

    model_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading {model_path}...")
    st = torch_load_file(model_path)
    print("  Loaded (bfloat16 preserved via PyTorch loader)")

    print("\n" + "=" * 60)
    print("Exporting PaliGemma LLM to standalone BF16 GGUF")
    print("=" * 60)
    export_llm(st, output_dir)
    print("\nDone.")
