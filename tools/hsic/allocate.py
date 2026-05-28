#!/usr/bin/env python3
"""
HSIC-based Quantization Allocator (Greedy L2 Layer-Balanced)

Takes per-tensor sensitivity scores and assigns quantization types using
a greedy optimizer that minimizes layer-balanced weighted error.

Objective:  minimize Σ_l (E_l)²
  where     E_l = Σ_{T in layer l} w(T) × noise(type_T)
            noise(type) = 2^(-2 × bpw)
            w(T) = max(score(T), 0)   [negative scores treated as 0]

At each step, upgrades the tensor giving highest improvement-per-bit:
  value = (E_l² - E_l_new²) / Δbits

This naturally concentrates bits on high-sensitivity tensors in high-error
layers (like the heuristic), rather than spreading thinly.

Usage:
    python tools/hsic/allocate.py \
        --scores hsic_tensor_actions_spatial_10_all.json \
        --score-key F_out \
        --target-bpw 2.53 \
        --base-type IQ2_XS
        --max-type Q2_k
"""

import argparse
import heapq
import json
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ─── Quantization Type Registry ───────────────────────────────────────────

@dataclass
class QuantType:
    name: str
    bpw: float
    family: str
    noise: float = 0.0

FAMILY_PREFERENCE = {
    "importance": 0, "non-linear": 1, "microscaling": 2,
    "k-quant": 3, "basic": 4,
}

QUANT_TYPES_ALL = [
    QuantType("IQ1_S",   1.5625, "importance"),
    QuantType("IQ1_M",   1.75,   "importance"),
    QuantType("IQ2_XXS", 2.0625, "importance"),
    QuantType("IQ2_XS",  2.3125, "importance"),
    QuantType("IQ2_S",   2.5625, "importance"),
    QuantType("Q2_K",    2.625,  "k-quant"),
    QuantType("IQ3_XXS", 3.0625, "importance"),
    QuantType("IQ3_S",   3.4375, "importance"),
    QuantType("Q3_K",    3.4375, "k-quant"),
    QuantType("IQ4_XS",  4.25,   "importance"),
    QuantType("Q4_K",    4.50,   "k-quant"),
    QuantType("Q5_K",    5.50,   "k-quant"),
    QuantType("Q6_K",    6.5625, "k-quant"),
    QuantType("Q8_0",    8.50,   "basic"),
]

for qt in QUANT_TYPES_ALL:
    qt.noise = 2.0 ** (-2.0 * qt.bpw)

QUANT_BY_NAME = {qt.name: qt for qt in QUANT_TYPES_ALL}

# llama.cpp preset suffixes (_S/_M/_L) are profiles, not distinct per-tensor types.
# Map them to the underlying k-quant entry so they work as --base-type / --max-type args.
_PRESET_ALIASES: Dict[str, str] = {
    "Q3_K_S": "Q3_K",
    "Q3_K_M": "Q3_K",
    "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K",
    "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K",
    "Q5_K_M": "Q5_K",
}
for _alias, _canonical in _PRESET_ALIASES.items():
    QUANT_BY_NAME[_alias] = QUANT_BY_NAME[_canonical]


def deduplicate_quant_menu(types: List[QuantType], keep: List[str] = None) -> List[QuantType]:
    keep_set = set(keep or [])
    by_bpw: Dict[float, List[QuantType]] = {}
    for qt in types:
        by_bpw.setdefault(qt.bpw, []).append(qt)
    result = []
    for bpw in sorted(by_bpw.keys()):
        candidates = by_bpw[bpw]
        forced = [q for q in candidates if q.name in keep_set]
        if forced:
            result.append(forced[0])
        else:
            best = min(candidates, key=lambda q: FAMILY_PREFERENCE[q.family])
            result.append(best)
    return result


# ─── Tensor Info ──────────────────────────────────────────────────────────

D_MODEL = 4096
D_FFN = 11008
VOCAB_SIZE = 32000

TENSOR_PARAMS = {
    "attn_q": D_MODEL * D_MODEL, "attn_k": D_MODEL * D_MODEL,
    "attn_v": D_MODEL * D_MODEL, "attn_output": D_MODEL * D_MODEL,
    "ffn_gate": D_FFN * D_MODEL, "ffn_up": D_FFN * D_MODEL,
    "ffn_down": D_MODEL * D_FFN,
}

# Non-block tensors (for full-model BPW calculation)
TOKEN_EMBD_PARAMS = VOCAB_SIZE * D_MODEL   # 131.1M
OUTPUT_PARAMS = VOCAB_SIZE * D_MODEL        # 131.1M
NORM_PARAMS = 32 * (D_MODEL + D_MODEL) + D_MODEL  # 0.26M (per-layer norms + final)


def compute_full_model_bpw(
    block_assignments: Dict[str, str],
    token_embd_type: str = "Q2_K",     # IQ2_XS preset default
    output_type: str = "Q6_K",          # our default
    norm_bpw: float = 32.0,            # norms stay at F32
) -> float:
    """Compute full-model BPW including token_embd, output, norms."""
    total_bits = 0.0
    total_params = 0

    total_bits += TOKEN_EMBD_PARAMS * QUANT_BY_NAME[token_embd_type].bpw
    total_params += TOKEN_EMBD_PARAMS

    total_bits += OUTPUT_PARAMS * QUANT_BY_NAME[output_type].bpw
    total_params += OUTPUT_PARAMS

    total_bits += NORM_PARAMS * norm_bpw
    total_params += NORM_PARAMS

    for name, qtype in block_assignments.items():
        n_params = get_n_params(name)
        total_bits += n_params * QUANT_BY_NAME[qtype].bpw
        total_params += n_params

    return total_bits / total_params


def get_n_params(tensor_name: str) -> int:
    ttype = tensor_name.split(".")[2]
    return TENSOR_PARAMS.get(ttype, D_MODEL * D_MODEL)


def get_layer(tensor_name: str) -> int:
    return int(tensor_name.split(".")[1])


# ─── Score Loading ────────────────────────────────────────────────────────

DERIVED_SCORES = {
    "dY": lambda info: info["hsic_y_out"] - info["hsic_y_in"],
    "y_out/x_out": lambda info: info["hsic_y_out"] / info["hsic_x_out"] if info["hsic_x_out"] > 0 else 0,
    "y_out_x_ratio": lambda info: info["hsic_y_out"] * (info["hsic_y_out"] / info["hsic_y_in"]) if info["hsic_y_in"] > 0 else 0,
    "y_out+x_out": lambda info: info["hsic_y_out"] + info["hsic_x_out"],
    # IB with lambda_x=1, lambda_y=N (negated so higher = more sensitive)
    "ib_0.5": lambda info: -(info["hsic_x_out"] - 0.5 * info["hsic_y_out"]),
    "ib_0.3": lambda info: -(info["hsic_x_out"] - 0.3 * info["hsic_y_out"]),
    "ib_0.2": lambda info: -(info["hsic_x_out"] - 0.2 * info["hsic_y_out"]),
    "ib_1.2": lambda info: -(info["hsic_x_out"] - 1.2 * info["hsic_y_out"]),
    "ib_2": lambda info: -(info["hsic_x_out"] - 2.0 * info["hsic_y_out"]),
    "ib_5": lambda info: -(info["hsic_x_out"] - 5.0 * info["hsic_y_out"]),
    "ib_8": lambda info: -(info["hsic_x_out"] - 8.0 * info["hsic_y_out"]),
    "ib_10": lambda info: -(info["hsic_x_out"] - 10.0 * info["hsic_y_out"]),
}


def load_per_tensor_scores(path: str, score_key: str) -> Dict[str, float]:
    with open(path) as f:
        data = json.load(f)

    scores = {}

    if "tensors" in data:
        derive_fn = DERIVED_SCORES.get(score_key)
        for name, info in data["tensors"].items():
            if derive_fn is not None:
                scores[name] = derive_fn(info)
            elif score_key in info:
                scores[name] = info[score_key]
            else:
                scores[name] = 0.0
    elif "layers" in data:
        n_layers = data["n_layers"]
        attn_types = ["attn_q", "attn_k", "attn_v", "attn_output"]
        ffn_types = ["ffn_gate", "ffn_up", "ffn_down"]
        attn_params = sum(TENSOR_PARAMS[t] for t in attn_types)
        ffn_params = sum(TENSOR_PARAMS[t] for t in ffn_types)

        for l in range(n_layers):
            layer_data = data["layers"][str(l)]
            sens_attn = layer_data.get("sens_attn", 0)
            sens_ffn = layer_data.get("sens_ffn", 0)
            for ttype in attn_types:
                scores[f"blk.{l}.{ttype}.weight"] = sens_attn * TENSOR_PARAMS[ttype] / attn_params
            for ttype in ffn_types:
                scores[f"blk.{l}.{ttype}.weight"] = sens_ffn * TENSOR_PARAMS[ttype] / ffn_params
    else:
        print(f"Error: unrecognized JSON format in {path}", file=sys.stderr)
        sys.exit(1)

    return scores


# ─── Greedy L2 Layer-Balanced Optimizer ──────────────────────────────────

def greedy_l2_allocate(
    scores: Dict[str, float],
    allowed_types: List[QuantType],
    target_bpw: float,
    base_type_name: str,
) -> Tuple[Dict[str, str], float, float]:
    """Greedy optimizer minimizing Σ_l (E_l)².

    E_l = Σ_{T in layer l} w(T) × noise(type_T)

    At each step, pick the upgrade with highest:
        value = (E_l² - E_l_new²) / Δbits

    After upgrading, re-push upgrades for all tensors in that layer
    (their improvement values changed because E_l changed).
    """
    sorted_types = sorted(allowed_types, key=lambda qt: qt.bpw)
    n_types = len(sorted_types)
    base_type = next(qt for qt in sorted_types if qt.name == base_type_name)

    tensor_names = sorted(scores.keys())
    total_params = sum(get_n_params(n) for n in tensor_names)
    target_bits = target_bpw * total_params

    # Weights: use max(score, 0) so negative scores → weight 0 (no protection needed)
    weights = {n: max(scores.get(n, 0), 0) for n in tensor_names}

    # Initialize all tensors to base type
    current_type_idx = {n: 0 for n in tensor_names}
    current_bits = sum(base_type.bpw * get_n_params(n) for n in tensor_names)

    if current_bits >= target_bits:
        return {n: base_type.name for n in tensor_names}, current_bits / total_params, 0.0

    # Group tensors by layer
    layer_tensors: Dict[int, List[str]] = {}
    for n in tensor_names:
        l = get_layer(n)
        layer_tensors.setdefault(l, []).append(n)

    # Compute initial per-layer error
    E_l: Dict[int, float] = {}
    for l, names in layer_tensors.items():
        E_l[l] = sum(weights[n] * base_type.noise for n in names)

    # Priority queue with lazy deletion
    counter = 0
    pq = []

    def push_upgrade(name: str, from_idx: int):
        nonlocal counter
        if from_idx >= n_types - 1:
            return
        nxt_idx = from_idx + 1
        l = get_layer(name)
        delta_noise = sorted_types[from_idx].noise - sorted_types[nxt_idx].noise
        delta_bits = (sorted_types[nxt_idx].bpw - sorted_types[from_idx].bpw) * get_n_params(name)
        if delta_bits <= 0:
            return

        el = E_l[l]
        el_new = el - weights[name] * delta_noise
        improvement = el ** 2 - el_new ** 2
        value = improvement / delta_bits

        heapq.heappush(pq, (-value, counter, name, from_idx))
        counter += 1

    # Initialize queue
    for n in tensor_names:
        push_upgrade(n, 0)

    # Greedy loop
    while pq:
        neg_value, _, name, expected_idx = heapq.heappop(pq)

        if current_type_idx[name] != expected_idx:
            continue  # stale entry

        nxt_idx = expected_idx + 1
        delta_bits = (sorted_types[nxt_idx].bpw - sorted_types[expected_idx].bpw) * get_n_params(name)

        if current_bits + delta_bits > target_bits:
            continue

        # Perform upgrade
        l = get_layer(name)
        delta_noise = sorted_types[expected_idx].noise - sorted_types[nxt_idx].noise
        E_l[l] -= weights[name] * delta_noise
        current_type_idx[name] = nxt_idx
        current_bits += delta_bits

        # Push next upgrade for this tensor
        push_upgrade(name, nxt_idx)

        # Re-push for other tensors in same layer (E_l changed)
        for other_name in layer_tensors[l]:
            if other_name != name:
                push_upgrade(other_name, current_type_idx[other_name])

    assignments = {n: sorted_types[current_type_idx[n]].name for n in tensor_names}
    achieved_bpw = current_bits / total_params
    total_obj = sum(E_l[l] ** 2 for l in E_l)

    return assignments, achieved_bpw, total_obj


# ─── Heuristics for Comparison ────────────────────────────────────────────

def iq2s_heuristic(n_layers: int) -> Dict[str, str]:
    """IQ2_S hardcoded: attn_v+attn_output→IQ3_S, ffn_down(L0-3)→IQ3_S, rest→IQ2_XS."""
    a = {}
    for l in range(n_layers):
        for ttype in TENSOR_PARAMS:
            name = f"blk.{l}.{ttype}.weight"
            if ttype in ("attn_v", "attn_output"):
                a[name] = "IQ3_S"
            elif ttype == "ffn_down" and l < n_layers // 8:
                a[name] = "IQ3_S"
            else:
                a[name] = "IQ2_XS"
    return a

# token_embd=IQ3_S, output=Q5_K for IQ2_S
IQ2S_FULL_BPW_ARGS = {"token_embd_type": "IQ3_S", "output_type": "Q5_K"}


def iq2xs_heuristic(n_layers: int) -> Dict[str, str]:
    """IQ2_XS hardcoded: attn_v→Q2_K, ffn_down(L0-3)→Q2_K, rest→IQ2_XS."""
    a = {}
    for l in range(n_layers):
        for ttype in TENSOR_PARAMS:
            name = f"blk.{l}.{ttype}.weight"
            if ttype == "attn_v":
                a[name] = "Q2_K"
            elif ttype == "ffn_down" and l < n_layers // 8:
                a[name] = "Q2_K"
            else:
                a[name] = "IQ2_XS"
    return a

# token_embd=Q2_K, output=Q5_K for IQ2_XS
IQ2XS_FULL_BPW_ARGS = {"token_embd_type": "Q2_K", "output_type": "Q5_K"}

# Our HSIC method: token_embd handled by llama-quantize default, output=Q6_K
HSIC_FULL_BPW_ARGS = {"token_embd_type": "Q2_K", "output_type": "Q6_K"}


# ─── Output ──────────────────────────────────────────────────────────────

def compute_bpw(assignments: Dict[str, str]) -> float:
    total_bits = sum(QUANT_BY_NAME[a].bpw * get_n_params(n) for n, a in assignments.items())
    total_params = sum(get_n_params(n) for n in assignments)
    return total_bits / total_params


def print_comparison_table(
    hsic_assign: Dict[str, str],
    n_layers: int,
    base_type: str,
):
    """Print layer-by-layer comparison of IQ2_XS, IQ2_S, and HSIC allocations."""
    iq2xs = iq2xs_heuristic(n_layers)
    iq2s = iq2s_heuristic(n_layers)

    print()
    print("=" * 140)
    print("LAYER-BY-LAYER TENSOR ASSIGNMENT COMPARISON")
    print("=" * 140)
    xs_full = compute_full_model_bpw(iq2xs, **IQ2XS_FULL_BPW_ARGS)
    s_full = compute_full_model_bpw(iq2s, **IQ2S_FULL_BPW_ARGS)
    h_full = compute_full_model_bpw(hsic_assign, **HSIC_FULL_BPW_ARGS)
    print(f"Block BPW:      IQ2_XS={compute_bpw(iq2xs):.4f}  |  IQ2_S={compute_bpw(iq2s):.4f}  |  HSIC={compute_bpw(hsic_assign):.4f}")
    print(f"Full model BPW: IQ2_XS={xs_full:.4f}  |  IQ2_S={s_full:.4f}  |  HSIC={h_full:.4f}")
    print()

    ttypes = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]

    # Header
    print(f"{'':>3} ", end="")
    for ttype in ttypes:
        print(f"| {ttype:^24} ", end="")
    print("|")

    print(f"{'L':>3} ", end="")
    for _ in ttypes:
        print(f"| {'IQ2_XS':>7} {'IQ2_S':>7} {'HSIC':>7} ", end="")
    print("|")

    print("-" * (4 + 27 * len(ttypes)))

    for l in range(n_layers):
        print(f"{l:>3} ", end="")
        for ttype in ttypes:
            name = f"blk.{l}.{ttype}.weight"
            xs = iq2xs.get(name, base_type)
            s = iq2s.get(name, base_type)
            h = hsic_assign.get(name, base_type)

            # Shorten names for display
            def short(n):
                return n.replace("IQ2_XS", "2XS").replace("IQ2_S", "2S").replace("IQ3_S", "3S") \
                        .replace("IQ3_XXS", "3XX").replace("Q2_K", "2K").replace("Q3_K", "3K") \
                        .replace("IQ4_XS", "4XS").replace("Q4_K", "4K").replace("Q5_K", "5K") \
                        .replace("Q6_K", "6K").replace("Q8_0", "8_0")

            print(f"| {short(xs):>7} {short(s):>7} {short(h):>7} ", end="")
        print("|")

    # Per tensor-type summary
    print()
    print("PER TENSOR-TYPE AVERAGE BPW:")
    print(f"{'Type':<15} {'IQ2_XS':>8} {'IQ2_S':>8} {'HSIC':>8}")
    print("-" * 42)
    for ttype in ttypes:
        xs_bpw = np.mean([QUANT_BY_NAME[iq2xs[f"blk.{l}.{ttype}.weight"]].bpw for l in range(n_layers)])
        s_bpw = np.mean([QUANT_BY_NAME[iq2s[f"blk.{l}.{ttype}.weight"]].bpw for l in range(n_layers)])
        h_bpw = np.mean([QUANT_BY_NAME[hsic_assign[f"blk.{l}.{ttype}.weight"]].bpw for l in range(n_layers)])
        print(f"  {ttype:<13}: {xs_bpw:>8.3f} {s_bpw:>8.3f} {h_bpw:>8.3f}")

    # Type distribution comparison
    print()
    print("TYPE DISTRIBUTION:")
    for label, assign in [("IQ2_XS preset", iq2xs), ("IQ2_S preset", iq2s), ("HSIC", hsic_assign)]:
        counts = {}
        for a in assign.values():
            counts[a] = counts.get(a, 0) + 1
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: QUANT_BY_NAME[x[0]].bpw))
        print(f"  {label:<15}: {dist}")


TASK_DIR_MAP = {
    "spatial": "oft_spatial_gguf",
    "long": "oft_10_gguf",
    "goal": "oft_goal_gguf",
    "object": "oft_object_gguf",
    "combined": "oft_combined_gguf",
}


def infer_task_from_path(scores_path: str) -> str:
    """Infer task name from scores file path."""
    fname = scores_path.lower()
    if "combined" in fname:
        return "combined"
    for task in ["spatial", "object", "goal", "long"]:
        if task in fname:
            return task
    return "combined"  # fallback


def print_quantize_command(assignments: Dict[str, str], scores: Dict[str, float],
                           base_type: str, task: str, score_key: str):
    task_dir = TASK_DIR_MAP.get(task, "oft_combined_gguf")
    out_name = f"llm_{base_type.lower()}_hsic_{score_key.replace('/', '_')}.gguf"

    print()
    print("=" * 110)
    print(f"QUANTIZE COMMAND ({task})")
    print("=" * 110)
    print()

    # Emit --tensor-type for EVERY tensor (including base_type ones) so the
    # resulting llama-quantize command is invariant to whatever heuristic
    # llama_tensor_get_type would otherwise apply for this ftype.
    overrides = [(name, assignments[name])
                 for name in sorted(assignments.keys(),
                                    key=lambda n: scores.get(n, 0), reverse=True)]

    parts = ["./build_oft/bin/llama-quantize \\"]
    parts.append(f"    --imatrix ../openvla-oft-checkpoints/{task_dir}/imatrix_20.gguf \\")
    parts.append(f"    --token-embedding-type {base_type.lower()} \\")
    parts.append("    --output-tensor-type Q5_K \\")
    for tname, ttype in overrides:
        parts.append(f"    --tensor-type {tname}={ttype} \\")
    parts.append(f"    ../openvla-oft-checkpoints/{task_dir}/llm_bf16.gguf \\")
    parts.append(f"    ../openvla-oft-checkpoints/{task_dir}/{out_name} {base_type}")
    print("\n".join(parts))


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Assign quantization types from HSIC sensitivity scores (greedy L2)",
    )
    parser.add_argument("--scores", type=str, required=True)
    parser.add_argument("--score-key", type=str, default="F_out")
    parser.add_argument("--lx", type=float, default=None,
                        help="If set with --ly, override score with -lx*hsic_x_out + ly*hsic_y_out (greedy with custom IB weights)")
    parser.add_argument("--ly", type=float, default=None,
                        help="See --lx")
    parser.add_argument("--target-bpw", type=float, required=True)
    parser.add_argument("--base-type", type=str, default="IQ2_XS")
    parser.add_argument("--max-type", type=str, default="Q8_0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--abs-score", action="store_true",
                        help="Use absolute value of scores (protects both high-positive and high-negative tensors)")
    parser.add_argument("--score-shift", action="store_true",
                        help="Shift scores so min=0 (all tensors participate, no clamping)")
    parser.add_argument("--score-transform", choices=["none", "log", "sqrt", "layer_norm"],
                        default="none",
                        help="Transform scores before allocation: log, sqrt, or layer_norm")

    args = parser.parse_args()

    suffix = ""
    if args.abs_score:
        suffix = " (abs)"
    elif args.score_shift:
        suffix = " (shifted)"
    if args.score_transform != "none":
        suffix += f" (transform={args.score_transform})"
    if args.lx is not None and args.ly is not None:
        # Custom IB weights — re-derive F_out from saved hsic_x_out / hsic_y_out
        with open(args.scores) as f:
            _data = json.load(f)
        scores = {}
        for name, info in _data.get("tensors", {}).items():
            scores[name] = -args.lx * info.get("hsic_x_out", 0) + args.ly * info.get("hsic_y_out", 0)
        suffix += f" (custom lx={args.lx} ly={args.ly})"
        print(f"Loading scores from {args.scores}, key=F_out_custom (lx={args.lx}, ly={args.ly}){suffix}")
    else:
        print(f"Loading scores from {args.scores}, key={args.score_key}{suffix}")
        scores = load_per_tensor_scores(args.scores, args.score_key)
    if args.abs_score:
        scores = {k: abs(v) for k, v in scores.items()}
    elif args.score_shift:
        min_score = min(scores.values())
        scores = {k: v - min_score for k, v in scores.items()}
        print(f"  Shifted by {-min_score:.6f} (original min)")

    if args.score_transform == "log":
        min_pos = min(v for v in scores.values() if v > 0)
        scores = {k: np.log(max(v, min_pos)) - np.log(min_pos) for k, v in scores.items()}
        print(f"  Log transform applied (min_pos={min_pos:.2e})")
    elif args.score_transform == "sqrt":
        scores = {k: np.sqrt(max(v, 0)) for k, v in scores.items()}
        print(f"  Sqrt transform applied")
    elif args.score_transform == "layer_norm":
        layer_means = {}
        for k in scores:
            l = get_layer(k)
            layer_means.setdefault(l, []).append(scores[k])
        layer_means = {l: np.mean(v) for l, v in layer_means.items()}
        scores = {k: scores[k] / layer_means[get_layer(k)] if layer_means[get_layer(k)] > 0 else 0
                  for k in scores}
        print(f"  Layer-norm transform applied")

    print(f"  {len(scores)} tensors loaded")

    base_bpw = QUANT_BY_NAME[args.base_type].bpw
    max_bpw = QUANT_BY_NAME[args.max_type].bpw
    base_canonical = QUANT_BY_NAME[args.base_type].name
    max_canonical = QUANT_BY_NAME[args.max_type].name
    allowed_types = [qt for qt in QUANT_TYPES_ALL if base_bpw <= qt.bpw <= max_bpw]
    allowed_types = deduplicate_quant_menu(allowed_types, keep=[base_canonical, max_canonical])
    print(f"  Allowed types: {[qt.name for qt in allowed_types]}")
    print(f"  Target BPW: {args.target_bpw}")

    assignments, achieved_bpw, total_obj = greedy_l2_allocate(
        scores, allowed_types, args.target_bpw, base_canonical,
    )
    print(f"  Achieved BPW: {achieved_bpw:.4f}")
    print(f"  Total objective: {total_obj:.6e}")

    n_upgraded = sum(1 for a in assignments.values() if a != args.base_type)
    print(f"  {n_upgraded}/{len(assignments)} tensors upgraded from {args.base_type}")

    task = infer_task_from_path(args.scores)
    print_comparison_table(assignments, args.n_layers, args.base_type)
    print_quantize_command(assignments, scores, args.base_type, task, args.score_key)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "assignments": assignments,
                "achieved_bpw": achieved_bpw,
                "total_objective": total_obj,
                "score_key": args.score_key,
                "target_bpw": args.target_bpw,
                "base_type": args.base_type,
            }, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
