#!/usr/bin/env python3
"""
merge_pi05_llm.py

Merges a quantized standalone PaliGemma LLM GGUF (blk.* naming from
llama-quantize) back into a full pi05.gguf (pali.blk.* naming).

This is the final step of the imatrix quantization pipeline:
  1. export_pi05_llm.py  → pali_llm_bf16.gguf
  2. llama-imatrix        → imatrix_combined.gguf
  3. llama-quantize       → pali_llm_<QUANT>.gguf
  4. merge_pi05_llm.py   → pi05_<QUANT>.gguf   ← this script

Tensor name mapping (llama-quantize → pi05.gguf):
  blk.{i}.attn_norm.weight   → pali.blk.{i}.attn_norm.weight
  blk.{i}.attn_q.weight      → pali.blk.{i}.attn_q.weight
  blk.{i}.attn_k.weight      → pali.blk.{i}.attn_k.weight
  blk.{i}.attn_v.weight      → pali.blk.{i}.attn_v.weight
  blk.{i}.attn_output.weight → pali.blk.{i}.attn_o.weight   (renamed!)
  blk.{i}.ffn_norm.weight    → pali.blk.{i}.ffn_norm.weight
  blk.{i}.ffn_gate.weight    → pali.blk.{i}.ffn_gate.weight
  blk.{i}.ffn_up.weight      → pali.blk.{i}.ffn_up.weight
  blk.{i}.ffn_down.weight    → pali.blk.{i}.ffn_down.weight
  token_embd.weight / output.weight / output_norm.weight → skipped

Usage:
    python merge_pi05_llm.py \\
        --base  /path/to/pi05_fp16.gguf \\
        --llm   /path/to/pali_llm_iq2xs.gguf \\
        --output /path/to/pi05_iq2xs.gguf \\
        --quant-type IQ2_XS
"""

from __future__ import annotations

import argparse
import os
import struct
import numpy as np
from pathlib import Path
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType


# Tensors in the standalone LLM GGUF that don't belong in pi05.gguf
_LLM_SKIP = {"output.weight", "output_norm.weight"}

# Standalone-LLM tensors that map to top-level pi05 tensor names (not pali.blk.*).
# token_embd → embed lets us route a quantized embedding through merge so the
# output file's embed has a TextEmbed-supported type (Q4_K when llama-quantize
# is called with --token-embedding-type Q4_K). Otherwise the merged file
# inherits the base's embed (often Q2_K), which the C++ TextEmbed rejects.
_LLM_TOPLEVEL_REMAP = {
    "token_embd.weight": "embed.weight",
}

# Pi05 C++ applies (1 + weight) for RMSNorm internally; the standalone LLM GGUF
# has +1 baked in for llama.cpp compatibility. Don't copy these back — keep the
# original raw weights from the base pi05.gguf to avoid double-applying the +1.
_LLM_SKIP_NORMS = {".attn_norm.weight", ".ffn_norm.weight"}


def llm_name_to_pali(name: str) -> str | None:
    """
    Convert a standalone LLM tensor name (blk.*) to pi05.gguf pali.blk.* name.
    Returns None for tensors that should be skipped.
    """
    if name in _LLM_SKIP:
        return None
    if name in _LLM_TOPLEVEL_REMAP:
        return _LLM_TOPLEVEL_REMAP[name]
    if not name.startswith("blk."):
        return None
    # Keep norm tensors from base (Pi05 C++ handles (1+weight) internally)
    if any(name.endswith(s) for s in _LLM_SKIP_NORMS):
        return None
    # attn_output → attn_o (pi05 uses shorter name)
    pali_name = "pali." + name.replace(".attn_output.weight", ".attn_o.weight")
    return pali_name


def pali_name_to_llm(name: str) -> str | None:
    """
    Convert a pi05.gguf pali.blk.* name back to standalone LLM name for lookup.
    """
    if not name.startswith("pali.blk."):
        return None
    llm_name = name[5:]  # strip "pali."
    llm_name = llm_name.replace(".attn_o.weight", ".attn_output.weight")
    return llm_name


def copy_kv_metadata(reader: GGUFReader, writer: GGUFWriter, quant_type: str | None):
    """Copy all key-value metadata from reader to writer, optionally updating quant_llm."""
    for key, field in reader.fields.items():
        # Update the quant_llm metadata to reflect the new quantization
        if key == "pi05.quant_llm" and quant_type:
            writer.add_string(key, quant_type)
            continue

        # Determine the field type and write it
        parts = field.parts
        if not parts:
            continue

        # The field's data is in field.parts[-1] for simple values
        # Use field.types to determine how to re-add
        try:
            _copy_field(writer, key, field)
        except Exception as e:
            print(f"  WARNING: Could not copy KV field '{key}': {e}")


def _copy_field(writer: GGUFWriter, key: str, field):
    """Copy a single KV field from GGUFReader to GGUFWriter."""
    from gguf.constants import GGUFValueType

    if not field.types:
        return

    ftype = field.types[0]

    if ftype == GGUFValueType.UINT8:
        writer.add_uint8(key, field.parts[-1][0])
    elif ftype == GGUFValueType.INT8:
        writer.add_int8(key, field.parts[-1][0])
    elif ftype == GGUFValueType.UINT16:
        writer.add_uint16(key, field.parts[-1][0])
    elif ftype == GGUFValueType.INT16:
        writer.add_int16(key, field.parts[-1][0])
    elif ftype == GGUFValueType.UINT32:
        writer.add_uint32(key, field.parts[-1][0])
    elif ftype == GGUFValueType.INT32:
        writer.add_int32(key, field.parts[-1][0])
    elif ftype == GGUFValueType.FLOAT32:
        writer.add_float32(key, field.parts[-1][0])
    elif ftype == GGUFValueType.BOOL:
        writer.add_bool(key, bool(field.parts[-1][0]))
    elif ftype == GGUFValueType.STRING:
        # String parts: [len_uint64, utf8_bytes]
        val = bytes(field.parts[-1]).decode("utf-8")
        writer.add_string(key, val)
    elif ftype == GGUFValueType.UINT64:
        writer.add_uint64(key, field.parts[-1][0])
    elif ftype == GGUFValueType.INT64:
        writer.add_int64(key, field.parts[-1][0])
    elif ftype == GGUFValueType.FLOAT64:
        writer.add_float64(key, field.parts[-1][0])
    elif ftype == GGUFValueType.ARRAY:
        # Array: field.types = [ARRAY, element_type]
        if len(field.types) < 2:
            return
        etype = field.types[1]
        data = field.parts[-1]
        if etype == GGUFValueType.FLOAT32:
            writer.add_array(key, [float(x) for x in data.astype(np.float32)])
        elif etype == GGUFValueType.INT32:
            writer.add_array(key, [int(x) for x in data.astype(np.int32)])
        elif etype == GGUFValueType.UINT32:
            writer.add_array(key, [int(x) for x in data.astype(np.uint32)])
        elif etype == GGUFValueType.STRING:
            # Array of strings: each part is [len, bytes]
            strs = []
            for p in field.parts[2:]:  # skip array type + count
                try:
                    strs.append(bytes(p).decode("utf-8"))
                except Exception:
                    pass
            writer.add_array(key, strs)
        else:
            writer.add_array(key, list(data))
    else:
        # Fallback: skip unknown types silently
        pass


def merge(base_path: str, llm_path: str, output_path: str, quant_type: str | None):
    print(f"Reading base GGUF: {base_path}")
    base = GGUFReader(base_path, "r")

    print(f"Reading LLM GGUF:  {llm_path}")
    llm = GGUFReader(llm_path, "r")

    # Build lookup: pali.blk.* name → LLM tensor (keyed by llm tensor name)
    llm_tensors: dict[str, object] = {}
    for t in llm.tensors:
        pali_name = llm_name_to_pali(t.name)
        if pali_name is not None:
            llm_tensors[pali_name] = t

    n_llm = len(llm_tensors)
    print(f"  Found {n_llm} LLM tensors to patch into pi05.gguf")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = GGUFWriter(output_path, "pi05")

    # ── Copy KV metadata from base ────────────────────────────────────────────
    print("Copying metadata from base GGUF...")
    copy_kv_metadata(base, writer, quant_type)

    # ── Copy tensors ──────────────────────────────────────────────────────────
    print("Merging tensors...")
    n_patched = 0
    n_copied = 0

    for tensor in base.tensors:
        name = tensor.name

        # Patch from LLM if a matching tensor exists (covers both pali.blk.*
        # and top-level remaps like embed.weight ← token_embd.weight).
        if name in llm_tensors:
            lt = llm_tensors[name]
            if lt.data.dtype == np.uint8:
                writer.add_tensor(name, lt.data, raw_dtype=lt.tensor_type)
            else:
                writer.add_tensor(name, lt.data)
            n_patched += 1
            continue

        if name.startswith("pali.blk."):
            llm_name = pali_name_to_llm(name)
            print(f"  WARNING: '{name}' (llm: '{llm_name}') not found in LLM GGUF — "
                  "keeping original FP16")
            # fall through and copy original

        # Copy non-patched tensors as-is (preserve dtype)
        if tensor.data.dtype == np.uint8:
            writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
        else:
            writer.add_tensor(name, tensor.data)
        n_copied += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = os.path.getsize(output_path) / 1024**3
    print(f"\nMerge complete:")
    print(f"  Patched tensors (quantized LLM):  {n_patched}")
    print(f"  Copied tensors (unchanged):        {n_copied}")
    print(f"  Output: {output_path}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge quantized PaliGemma LLM back into pi05.gguf"
    )
    parser.add_argument("--base", required=True,
                        help="Base pi05_fp16.gguf (all non-LLM tensors)")
    parser.add_argument("--llm", required=True,
                        help="Quantized standalone LLM GGUF (blk.* naming from llama-quantize)")
    parser.add_argument("--output", required=True,
                        help="Output path for merged pi05_<quant>.gguf")
    parser.add_argument("--quant-type", default=None,
                        help="Quant type string to record in pi05.quant_llm metadata (e.g. IQ2_XS)")
    args = parser.parse_args()

    merge(args.base, args.llm, args.output, args.quant_type)
