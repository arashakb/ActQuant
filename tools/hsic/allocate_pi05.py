#!/usr/bin/env python3
"""
HSIC-based Quantization Allocator for Pi 0.5 (PaliGemma backbone).

Forked from tools/hsic/allocate.py with these substitutions to match Pi 0.5:
    D_MODEL    = 2048    (vs Llama 4096)
    D_FFN      = 16384   (vs Llama 11008)
    VOCAB_SIZE = 257152  (vs Llama 32000)
    N_LAYERS   = 18      (vs Llama 32)
    GQA: 8 Q heads, 1 KV head, head_dim=256 → attn_k/v are (256, 2048) NOT (2048, 2048)

Allocates per-tensor quant types via the same greedy L2 layer-balanced
optimizer:
    minimize Σ_l (E_l)²  with E_l = Σ_T weight(T) × noise(type_T)
"""

import argparse
import heapq
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ─── Quantization Type Registry (same as allocate.py) ────────────────────────

@dataclass
class QuantType:
    name: str
    bpw: float
    family: str
    noise: float = 0.0


FAMILY_PREFERENCE = {"importance": 0, "non-linear": 1, "microscaling": 2, "k-quant": 3, "basic": 4}

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

_PRESET_ALIASES: Dict[str, str] = {
    "Q3_K_S": "Q3_K", "Q3_K_M": "Q3_K", "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K", "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K", "Q5_K_M": "Q5_K",
}
for _alias, _canonical in _PRESET_ALIASES.items():
    QUANT_BY_NAME[_alias] = QUANT_BY_NAME[_canonical]


def deduplicate_quant_menu(types: List[QuantType], keep: List[str] = None) -> List[QuantType]:
    """Remove duplicate-bpw types, keeping the family-preferred one.

    `keep` forces specific names to be retained (e.g., the user's base/max type)
    even when another type at the same bpw would normally win.
    """
    keep_set = set(keep or [])
    by_bpw: Dict[float, List[QuantType]] = {}
    for qt in types:
        by_bpw.setdefault(qt.bpw, []).append(qt)
    out = []
    for bpw in sorted(by_bpw.keys()):
        cands = by_bpw[bpw]
        forced = [q for q in cands if q.name in keep_set]
        if forced:
            out.append(forced[0])
        else:
            out.append(min(cands, key=lambda q: FAMILY_PREFERENCE[q.family]))
    return out


# ─── Pi 0.5 Tensor Dimensions ────────────────────────────────────────────────

D_MODEL    = 2048
D_FFN      = 16384
VOCAB_SIZE = 257152
N_LAYERS   = 18
N_HEADS    = 8
N_KV       = 1
HEAD_DIM   = 256

TENSOR_PARAMS = {
    "attn_q":      D_MODEL * (N_HEADS * HEAD_DIM),   # 2048 × 2048 = 4.19M
    "attn_k":      D_MODEL * (N_KV * HEAD_DIM),      # 2048 × 256 = 524K
    "attn_v":      D_MODEL * (N_KV * HEAD_DIM),      # 2048 × 256 = 524K
    "attn_output": (N_HEADS * HEAD_DIM) * D_MODEL,   # 2048 × 2048 = 4.19M
    "ffn_gate":    D_FFN * D_MODEL,                  # 16384 × 2048 = 33.55M
    "ffn_up":      D_FFN * D_MODEL,                  # 16384 × 2048 = 33.55M
    "ffn_down":    D_MODEL * D_FFN,                  # 2048 × 16384 = 33.55M
}

TOKEN_EMBD_PARAMS = VOCAB_SIZE * D_MODEL    # 526.6M
OUTPUT_PARAMS     = VOCAB_SIZE * D_MODEL    # 526.6M (tied with embed in PaliGemma)
NORM_PARAMS       = N_LAYERS * (D_MODEL + D_MODEL) + D_MODEL  # ~75K


def get_n_params(name: str) -> int:
    ttype = name.split(".")[2]
    return TENSOR_PARAMS.get(ttype, D_MODEL * D_MODEL)


def get_layer(name: str) -> int:
    return int(name.split(".")[1])


def compute_full_model_bpw(
    block_assignments: Dict[str, str],
    token_embd_type: str = "Q4_K",
    output_type: str = "Q6_K",
    norm_bpw: float = 32.0,
) -> float:
    total_bits  = TOKEN_EMBD_PARAMS * QUANT_BY_NAME[token_embd_type].bpw
    total_params = TOKEN_EMBD_PARAMS
    total_bits  += OUTPUT_PARAMS * QUANT_BY_NAME[output_type].bpw
    total_params += OUTPUT_PARAMS
    total_bits  += NORM_PARAMS * norm_bpw
    total_params += NORM_PARAMS
    for n, t in block_assignments.items():
        p = get_n_params(n)
        total_bits  += p * QUANT_BY_NAME[t].bpw
        total_params += p
    return total_bits / total_params


# ─── Score loading ───────────────────────────────────────────────────────────

DERIVED_SCORES = {
    "dY":            lambda info: info["hsic_y_out"] - info["hsic_y_in"],
    "y_out/x_out":   lambda info: info["hsic_y_out"] / info["hsic_x_out"] if info["hsic_x_out"] > 0 else 0,
    "y_out_x_ratio": lambda info: info["hsic_y_out"] * (info["hsic_y_out"] / info["hsic_y_in"]) if info["hsic_y_in"] > 0 else 0,
    "y_out+x_out":   lambda info: info["hsic_y_out"] + info["hsic_x_out"],
    "ib_0.5": lambda info: -(info["hsic_x_out"] - 0.5 * info["hsic_y_out"]),
    "ib_0.3": lambda info: -(info["hsic_x_out"] - 0.3 * info["hsic_y_out"]),
    "ib_0.2": lambda info: -(info["hsic_x_out"] - 0.2 * info["hsic_y_out"]),
    "ib_1.2": lambda info: -(info["hsic_x_out"] - 1.2 * info["hsic_y_out"]),
    "ib_2":   lambda info: -(info["hsic_x_out"] - 2.0 * info["hsic_y_out"]),
    "ib_5":   lambda info: -(info["hsic_x_out"] - 5.0 * info["hsic_y_out"]),
    "ib_8":   lambda info: -(info["hsic_x_out"] - 8.0 * info["hsic_y_out"]),
    "ib_10":  lambda info: -(info["hsic_x_out"] - 10.0 * info["hsic_y_out"]),
    # ─── New score keys for Pi 0.5 ───────────────────────────────────────────
    # Information ADDED by tensor — favors tensors whose output carries more
    # task signal than their input.
    "y_out_minus_x_out": lambda info: info["hsic_y_out"] - info["hsic_x_out"],
    "y_out_minus_y_in":  lambda info: info["hsic_y_out"] - info["hsic_y_in"],
    # Per-output-dim normalized — counter-balances small-d_out tensors (attn_v
    # at d_out=256 in Pi 0.5 GQA) which have smaller absolute HSIC values.
    "y_per_dout":   lambda info: info["hsic_y_out"] / max(info["d_out"], 1),
    "sens_per_dout":lambda info: info["sens"] / max(info["d_out"], 1),
    # Multiplicative — protects tensors that BOTH have high relevance AND have
    # input/output well-aligned with X (the prefix). Captures structural
    # importance more than pure F_out which only weights output relevance.
    "y_out_x_in":   lambda info: info["hsic_y_out"] * info["hsic_x_in"],
    # Marginal information gain RELATIVE to input — fraction of new task
    # information added (saturates near 1 for transformative layers).
    "rel_gain":     lambda info: ((info["hsic_y_out"] - info["hsic_y_in"]) /
                                  max(info["hsic_y_in"], 1e-12)),
    # Layer-position bias to simulate the heuristic's early-layer protection.
    "F_out_early":  lambda info: info["F_out"] * (1.0 + 0.5 / max(info["layer"] + 1, 1)),
}


def load_per_tensor_scores(path: str, score_key: str) -> Dict[str, float]:
    with open(path) as f:
        data = json.load(f)
    if "tensors" not in data:
        print(f"Error: expected 'tensors' in {path}", file=sys.stderr)
        sys.exit(1)
    derive_fn = DERIVED_SCORES.get(score_key)
    out = {}
    for name, info in data["tensors"].items():
        if derive_fn is not None:
            out[name] = derive_fn(info)
        elif score_key in info:
            out[name] = info[score_key]
        else:
            out[name] = 0.0
    return out


# ─── Greedy L2 Allocation (identical to allocate.py) ─────────────────────────

def greedy_l2_allocate(
    scores: Dict[str, float],
    allowed_types: List[QuantType],
    target_bpw: float,
    base_type_name: str,
) -> Tuple[Dict[str, str], float, float]:
    sorted_types = sorted(allowed_types, key=lambda qt: qt.bpw)
    n_types = len(sorted_types)
    base_type = next(qt for qt in sorted_types if qt.name == base_type_name)

    names = sorted(scores.keys())
    total_params = sum(get_n_params(n) for n in names)
    target_bits  = target_bpw * total_params

    weights = {n: max(scores.get(n, 0), 0) for n in names}
    cur_idx = {n: 0 for n in names}
    cur_bits = sum(base_type.bpw * get_n_params(n) for n in names)
    if cur_bits >= target_bits:
        return {n: base_type.name for n in names}, cur_bits / total_params, 0.0

    layer_tensors: Dict[int, List[str]] = {}
    for n in names:
        layer_tensors.setdefault(get_layer(n), []).append(n)

    E_l: Dict[int, float] = {l: sum(weights[n] * base_type.noise for n in lt)
                             for l, lt in layer_tensors.items()}

    counter = 0
    pq = []
    def push(name: str, from_idx: int):
        nonlocal counter
        if from_idx >= n_types - 1: return
        nxt = from_idx + 1
        l = get_layer(name)
        d_noise = sorted_types[from_idx].noise - sorted_types[nxt].noise
        d_bits  = (sorted_types[nxt].bpw - sorted_types[from_idx].bpw) * get_n_params(name)
        if d_bits <= 0: return
        el = E_l[l]
        el_new = el - weights[name] * d_noise
        improvement = el ** 2 - el_new ** 2
        heapq.heappush(pq, (-improvement / d_bits, counter, name, from_idx))
        counter += 1

    for n in names: push(n, 0)

    while pq:
        _, _, name, exp_idx = heapq.heappop(pq)
        if cur_idx[name] != exp_idx: continue
        nxt = exp_idx + 1
        d_bits = (sorted_types[nxt].bpw - sorted_types[exp_idx].bpw) * get_n_params(name)
        if cur_bits + d_bits > target_bits: continue
        l = get_layer(name)
        d_noise = sorted_types[exp_idx].noise - sorted_types[nxt].noise
        E_l[l] -= weights[name] * d_noise
        cur_idx[name] = nxt
        cur_bits += d_bits
        push(name, nxt)
        for o in layer_tensors[l]:
            if o != name: push(o, cur_idx[o])

    assignments = {n: sorted_types[cur_idx[n]].name for n in names}
    achieved_bpw = cur_bits / total_params
    total_obj = sum(E_l[l] ** 2 for l in E_l)
    return assignments, achieved_bpw, total_obj


# ─── Heuristic baseline (Pi 0.5-adapted IQ2_XS) ──────────────────────────────

def iq2xs_heuristic(n_layers: int = N_LAYERS) -> Dict[str, str]:
    """llama-quantize's hardcoded IQ2_XS heuristic, adapted to Pi 0.5's 18 layers:
       attn_v → Q2_K, ffn_down (first n//8 layers) → Q2_K, rest → IQ2_XS.
       For 18 layers, n//8 = 2 → ffn_down upgraded for layers 0-1.
    """
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


def compute_bpw(assignments: Dict[str, str]) -> float:
    total_bits   = sum(QUANT_BY_NAME[a].bpw * get_n_params(n) for n, a in assignments.items())
    total_params = sum(get_n_params(n) for n in assignments)
    return total_bits / total_params


def print_comparison(hsic_assign: Dict[str, str], n_layers: int = N_LAYERS):
    iq2xs = iq2xs_heuristic(n_layers)
    print()
    print("=" * 90)
    print("LAYER-BY-LAYER ASSIGNMENT — IQ2_XS heuristic vs HSIC")
    print("=" * 90)
    print(f"Block BPW:  IQ2_XS_heuristic={compute_bpw(iq2xs):.4f}   HSIC={compute_bpw(hsic_assign):.4f}")

    ttypes = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
    print(f"\n{'L':>3} | " + " | ".join(f"{t:^16}" for t in ttypes) + " |")
    print(f"{'':>3} | " + " | ".join(f"{'IQ2XS':>7} {'HSIC':>7}" for _ in ttypes) + " |")
    print("-" * (5 + 19 * len(ttypes)))

    short = lambda n: (n.replace("IQ2_XS", "2XS").replace("IQ2_S", "2S")
                         .replace("IQ3_S", "3S").replace("IQ3_XXS", "3XX")
                         .replace("Q2_K", "2K").replace("Q3_K", "3K")
                         .replace("IQ4_XS", "4XS").replace("Q4_K", "4K")
                         .replace("Q5_K", "5K").replace("Q6_K", "6K"))
    for l in range(n_layers):
        cells = []
        for t in ttypes:
            name = f"blk.{l}.{t}.weight"
            cells.append(f"{short(iq2xs[name]):>7} {short(hsic_assign[name]):>7}")
        print(f"{l:>3} | " + " | ".join(cells) + " |")

    # Type distribution
    print("\nTYPE DISTRIBUTION:")
    for label, asg in [("IQ2_XS preset", iq2xs), ("HSIC", hsic_assign)]:
        cnt: Dict[str, int] = {}
        for v in asg.values(): cnt[v] = cnt.get(v, 0) + 1
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items(), key=lambda x: QUANT_BY_NAME[x[0]].bpw))
        print(f"  {label:<15}: {dist}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pi 0.5 HSIC quant allocator")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-key", default="F_out")
    ap.add_argument("--lx", type=float, default=None,
                    help="Override IB lx (re-derive F_out=-lx*x_out + ly*y_out)")
    ap.add_argument("--ly", type=float, default=None)
    ap.add_argument("--target-bpw", type=float, required=True)
    ap.add_argument("--base-type", default="IQ2_XS")
    ap.add_argument("--max-type",  default="Q4_K")
    ap.add_argument("--output", default=None)
    ap.add_argument("--abs-score",  action="store_true")
    ap.add_argument("--score-shift", action="store_true")
    ap.add_argument("--score-transform", choices=["none", "log", "sqrt", "layer_norm", "type_norm"],
                    default="none",
                    help="Transform scores before allocation")
    args = ap.parse_args()

    if args.lx is not None and args.ly is not None:
        with open(args.scores) as f:
            data = json.load(f)
        scores = {n: -args.lx * info.get("hsic_x_out", 0) + args.ly * info.get("hsic_y_out", 0)
                  for n, info in data["tensors"].items()}
        print(f"Loaded scores from {args.scores}; key=F_out_custom (lx={args.lx} ly={args.ly})")
    else:
        scores = load_per_tensor_scores(args.scores, args.score_key)
        print(f"Loaded scores from {args.scores}; key={args.score_key}")

    if args.abs_score:
        scores = {k: abs(v) for k, v in scores.items()}
    elif args.score_shift:
        m = min(scores.values())
        scores = {k: v - m for k, v in scores.items()}
        print(f"  Shifted by {-m:.6e}")

    if args.score_transform == "log":
        min_pos = min(v for v in scores.values() if v > 0)
        scores = {k: np.log(max(v, min_pos)) - np.log(min_pos) for k, v in scores.items()}
        print(f"  Log transform applied")
    elif args.score_transform == "sqrt":
        scores = {k: np.sqrt(max(v, 0)) for k, v in scores.items()}
    elif args.score_transform == "layer_norm":
        layer_means: Dict[int, list] = {}
        for k in scores:
            layer_means.setdefault(get_layer(k), []).append(scores[k])
        layer_means = {l: float(np.mean(v)) for l, v in layer_means.items()}
        scores = {k: scores[k] / layer_means[get_layer(k)] if layer_means[get_layer(k)] > 0 else 0
                  for k in scores}
        print(f"  Layer-normalized — each layer's tensors compete on equal footing")
    elif args.score_transform == "type_norm":
        # Normalize within tensor type (so e.g. attn_v tensors all compete with each other,
        # rather than against ffn_* tensors which have larger absolute HSIC values)
        type_means: Dict[str, list] = {}
        for k in scores:
            ttype = k.split(".")[2]
            type_means.setdefault(ttype, []).append(scores[k])
        type_means = {t: float(np.mean(v)) for t, v in type_means.items()}
        scores = {k: scores[k] / type_means[k.split(".")[2]] if type_means[k.split(".")[2]] > 0 else 0
                  for k in scores}
        print(f"  Type-normalized — each tensor type's tensors compete on equal footing")

    print(f"  {len(scores)} tensors")

    base_bpw = QUANT_BY_NAME[args.base_type].bpw
    max_bpw  = QUANT_BY_NAME[args.max_type].bpw
    base_canonical = QUANT_BY_NAME[args.base_type].name
    max_canonical  = QUANT_BY_NAME[args.max_type].name
    allowed = [qt for qt in QUANT_TYPES_ALL if base_bpw <= qt.bpw <= max_bpw]
    allowed = deduplicate_quant_menu(allowed, keep=[base_canonical, max_canonical])
    print(f"  Allowed types: {[qt.name for qt in allowed]}")
    print(f"  Target BPW:    {args.target_bpw}")

    assignments, achieved_bpw, total_obj = greedy_l2_allocate(
        scores, allowed, args.target_bpw, base_canonical,
    )
    print(f"  Achieved BPW:  {achieved_bpw:.4f}")
    print(f"  Objective:     {total_obj:.6e}")

    n_up = sum(1 for v in assignments.values() if v != args.base_type)
    print(f"  {n_up}/{len(assignments)} tensors upgraded from {args.base_type}")

    print_comparison(assignments)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "assignments": assignments,
                "achieved_bpw": achieved_bpw,
                "total_objective": total_obj,
                "score_key": args.score_key,
                "target_bpw": args.target_bpw,
                "base_type": args.base_type,
                "max_type":  args.max_type,
                "n_layers": N_LAYERS,
                "model": "pi05",
            }, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
