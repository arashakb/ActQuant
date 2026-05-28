"""speed_test.py

Latency / throughput speed test for the C++ OpenVLA-OFT pipeline (openvla_oft.so)
under different weight-only quantization combinations.

It compares end-to-end inference latency for the OFT pipeline (DINOv2 + SigLIP +
projector + LLM + action head) for several presets, and reports:
- per-config mean / median / p50 / p95 / p99 / min / max latency
- throughput in actions/s and inferences/s
- model load time
- GPU VRAM delta on load

For clean per-config VRAM accounting and to avoid llama.cpp/ggml backend leakage
across instances, by default each preset is run in its own subprocess
(via --config <name>). The driver collects JSON results and prints a comparison
table.

Usage:
  # run all presets, each in its own subprocess (recommended)
  python speed_test.py --all --iters 20 --warmup 3

  # run a single preset (subprocess worker mode)
  python speed_test.py --config iq2_xs_target --iters 20 --warmup 3 --json-out /tmp/r.json

  # run all presets in the same process (faster, but VRAM numbers will accumulate)
  python speed_test.py --all --no-subprocess
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Ensure the openvla_oft.so built at build_oft/bin is importable.
_BUILD_BIN = "/path/to/ActQuant/build_oft/bin"
if _BUILD_BIN not in sys.path:
    sys.path.insert(0, _BUILD_BIN)

CHECKPOINT_DIR = "/path/to/openvla-oft-checkpoints/oft_combined_gguf"
SAMPLE_OBS_PATH = (
    "/path/to/openvla-oft/experiments/robot/libero/"
    "sample_libero_spatial_observation.pkl"
)
DEFAULT_INSTRUCTION = (
    "pick up the black bowl on the wooden cabinet and place it on the plate"
)
DEFAULT_PROPRIO = [0.1, 0.05, 1.0, 3.0, -0.5, 0.2, 0.02, -0.02]


@dataclass
class Preset:
    name: str
    description: str
    dinov2: str
    siglip: str
    llm: str
    extras: Dict[str, str] = field(default_factory=dict)

    def paths(self) -> Dict[str, str]:
        return {
            "dinov2_model_path": os.path.join(CHECKPOINT_DIR, self.dinov2),
            "siglip_model_path": os.path.join(CHECKPOINT_DIR, self.siglip),
            "proj_model_path": os.path.join(CHECKPOINT_DIR, "proj.gguf"),
            "llm_model_path": os.path.join(CHECKPOINT_DIR, self.llm),
            "action_head_path": os.path.join(CHECKPOINT_DIR, "action_head.gguf"),
            "proprio_proj_path": os.path.join(CHECKPOINT_DIR, "proprio_projector.gguf"),
            "dataset_statistics_path": os.path.join(CHECKPOINT_DIR, "dataset_statistics.json"),
            "tokenizer_path": os.path.join(CHECKPOINT_DIR, "language_model", "tokenizer.json"),
        }


PRESETS: Dict[str, Preset] = {
    "bf16_baseline": Preset(
        name="bf16_baseline",
        description="DINOv2 bf16, SigLIP bf16, LLM bf16 (no quant)",
        dinov2="dinov2.gguf",
        siglip="siglip.gguf",
        llm="llm_bf16.gguf",
    ),
    "q4km_vision_only": Preset(
        name="q4km_vision_only",
        description="DINOv2 q4_k_m, SigLIP q4_k_m, LLM bf16",
        dinov2="dinov2_q4_k_m.gguf",
        siglip="siglip_q4_k_m_padded.gguf",
        llm="llm_bf16.gguf",
    ),
    "q4km_llm_only": Preset(
        name="q4km_llm_only",
        description="DINOv2 bf16, SigLIP bf16, LLM q4_k_m",
        dinov2="dinov2.gguf",
        siglip="siglip.gguf",
        llm="llm_q4_k_m.gguf",
    ),
    "q4km_full": Preset(
        name="q4km_full",
        description="DINOv2 q4_k_m, SigLIP q4_k_m, LLM q4_k_m",
        dinov2="dinov2_q4_k_m.gguf",
        siglip="siglip_q4_k_m_padded.gguf",
        llm="llm_q4_k_m.gguf",
    ),
    "iq2_xs_llm_only": Preset(
        name="iq2_xs_llm_only",
        description="DINOv2 bf16, SigLIP bf16, LLM iq2_xs",
        dinov2="dinov2.gguf",
        siglip="siglip.gguf",
        llm="llm_iq2_xs.gguf",
    ),
    "iq2_xs_target": Preset(
        name="iq2_xs_target",
        description="DINOv2 q4_k_m, SigLIP q4_k_m, LLM iq2_xs (target config)",
        dinov2="dinov2_q4_k_m.gguf",
        siglip="siglip_q4_k_m_padded.gguf",
        llm="llm_iq2_xs.gguf",
    ),
}


def get_gpu_vram_used_mib() -> Optional[int]:
    try:
        gpu_id = 0
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            gpu_id = int(cuda_visible.split(",")[0])
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None
    return None


def load_sample_inputs():
    """Return (full_img: HxWx3 uint8, wrist_img: HxWx3 uint8, proprio: list[float], instruction: str)."""
    if os.path.exists(SAMPLE_OBS_PATH):
        with open(SAMPLE_OBS_PATH, "rb") as f:
            obs = pickle.load(f)
        full_img = np.ascontiguousarray(obs["full_image"], dtype=np.uint8)
        wrist_img = np.ascontiguousarray(obs["wrist_image"], dtype=np.uint8)
        proprio = list(obs["state"].astype(float))
        instruction = obs.get("task_description") or DEFAULT_INSTRUCTION
    else:
        rng = np.random.default_rng(0)
        full_img = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        wrist_img = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        proprio = list(DEFAULT_PROPRIO)
        instruction = DEFAULT_INSTRUCTION
    return full_img, wrist_img, proprio, instruction


def file_size_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return 0.0


def model_weights_gb(preset: Preset) -> float:
    p = preset.paths()
    keys = ["dinov2_model_path", "siglip_model_path", "proj_model_path",
            "llm_model_path", "action_head_path", "proprio_proj_path"]
    return sum(file_size_gb(p[k]) for k in keys)


def percentile(times_ms: np.ndarray, p: float) -> float:
    return float(np.percentile(times_ms, p)) if len(times_ms) else 0.0


def run_one_config(preset: Preset, iters: int, warmup: int,
                   n_ctx: int, max_nodes: int, ngl: int, n_threads: int,
                   task_suite_name: str) -> dict:
    """Load model, run warmup + iters timed iterations, return result dict."""
    import openvla_oft  # imported here so subprocess workers also work

    paths = preset.paths()
    for k, pth in paths.items():
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing GGUF/asset: {k} -> {pth}")

    full_img, wrist_img, proprio, instruction = load_sample_inputs()

    vram_before = get_gpu_vram_used_mib()
    t_load = time.perf_counter()
    model = openvla_oft.OpenvlaOFTPipelineWithProprio(
        dinov2_model_path=paths["dinov2_model_path"],
        siglip_model_path=paths["siglip_model_path"],
        proj_model_path=paths["proj_model_path"],
        llm_model_path=paths["llm_model_path"],
        action_head_path=paths["action_head_path"],
        proprio_proj_path=paths["proprio_proj_path"],
        dataset_statistics_path=paths["dataset_statistics_path"],
        tokenizer_path=paths["tokenizer_path"],
        device_name="CUDA0",
        n_threads=n_threads,
        max_nodes=max_nodes,
        ngl=ngl,
        n_ctx=n_ctx,
        task_suite_name=task_suite_name,
    )
    load_time_s = time.perf_counter() - t_load
    vram_after = get_gpu_vram_used_mib()
    vram_delta_mib = (vram_after - vram_before) if (vram_after is not None and vram_before is not None) else None

    # Warmup
    for _ in range(warmup):
        _ = model.run2(full_img, wrist_img, instruction, proprio)

    # Timed iterations
    times_s: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.run2(full_img, wrist_img, instruction, proprio)
        dt = time.perf_counter() - t0
        times_s.append(dt)
    out = np.asarray(out)
    num_actions = int(out.size // 7)  # OFT outputs 8 chunks * 7 dims = 56

    times_ms = np.asarray(times_s) * 1000.0
    total_s = float(times_ms.sum() / 1000.0)
    result = {
        "preset": preset.name,
        "description": preset.description,
        "files": {
            "dinov2": preset.dinov2,
            "siglip": preset.siglip,
            "llm": preset.llm,
        },
        "model_weights_gb": round(model_weights_gb(preset), 3),
        "load_time_s": round(load_time_s, 3),
        "vram_delta_mib": vram_delta_mib,
        "iters": iters,
        "warmup": warmup,
        "num_actions_per_call": num_actions,
        "latency_ms": {
            "mean": round(float(times_ms.mean()), 3),
            "median": round(float(np.median(times_ms)), 3),
            "p50": round(percentile(times_ms, 50), 3),
            "p95": round(percentile(times_ms, 95), 3),
            "p99": round(percentile(times_ms, 99), 3),
            "min": round(float(times_ms.min()), 3),
            "max": round(float(times_ms.max()), 3),
            "std": round(float(times_ms.std()), 3),
        },
        "throughput": {
            "inferences_per_s": round(iters / total_s, 3) if total_s > 0 else 0.0,
            "actions_per_s": round((iters * num_actions) / total_s, 3) if total_s > 0 else 0.0,
        },
        "raw_times_ms": [round(x, 3) for x in times_ms.tolist()],
    }
    # Drop the model so its destructor runs before VRAM is reread by any caller.
    del model
    return result


def print_result(r: dict, file=sys.stdout) -> None:
    lat = r["latency_ms"]
    tp = r["throughput"]
    vram = r.get("vram_delta_mib")
    vram_str = f"{vram} MiB" if vram is not None else "n/a"
    print("", file=file)
    print(f"=== {r['preset']} ===", file=file)
    print(f"  files: dinov2={r['files']['dinov2']}, siglip={r['files']['siglip']}, llm={r['files']['llm']}", file=file)
    print(f"  weights: {r['model_weights_gb']:.2f} GB | load: {r['load_time_s']:.2f} s | VRAM delta: {vram_str}", file=file)
    print(f"  iters={r['iters']} (warmup={r['warmup']}) | actions/call={r['num_actions_per_call']}", file=file)
    print(f"  latency ms : mean={lat['mean']:.1f} median={lat['median']:.1f} p95={lat['p95']:.1f} "
          f"p99={lat['p99']:.1f} min={lat['min']:.1f} max={lat['max']:.1f} std={lat['std']:.1f}", file=file)
    print(f"  throughput : {tp['inferences_per_s']:.2f} inf/s, {tp['actions_per_s']:.1f} actions/s", file=file)


def print_comparison(results: List[dict], file=sys.stdout) -> None:
    if not results:
        return
    baseline = next((r for r in results if r["preset"] == "bf16_baseline"), results[0])
    base_mean = baseline["latency_ms"]["mean"]
    print("", file=file)
    print("=" * 110, file=file)
    print(f"{'PRESET':<22} {'WEIGHTS_GB':>10} {'LOAD_S':>7} {'VRAM_MIB':>9} "
          f"{'MEAN_MS':>9} {'MED_MS':>9} {'P95_MS':>9} {'INF_S':>8} {'SPEEDUP':>9}", file=file)
    print("-" * 110, file=file)
    for r in results:
        lat = r["latency_ms"]
        tp = r["throughput"]
        vram = r.get("vram_delta_mib")
        vram_str = str(vram) if vram is not None else "n/a"
        speedup = base_mean / lat["mean"] if lat["mean"] > 0 else float("inf")
        print(f"{r['preset']:<22} {r['model_weights_gb']:>10.2f} {r['load_time_s']:>7.2f} "
              f"{vram_str:>9} {lat['mean']:>9.1f} {lat['median']:>9.1f} {lat['p95']:>9.1f} "
              f"{tp['inferences_per_s']:>8.2f} {speedup:>9.2f}x", file=file)
    print("=" * 110, file=file)
    print(f"Speedup is mean-latency relative to '{baseline['preset']}'.", file=file)


def run_all_subprocess(args) -> List[dict]:
    """Run each preset in its own subprocess, collect JSON, print summary."""
    selected = args.presets if args.presets else list(PRESETS.keys())
    out_dir = args.json_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed_test_logs")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results: List[dict] = []
    for name in selected:
        if name not in PRESETS:
            print(f"[skip] unknown preset: {name}")
            continue
        json_path = os.path.join(out_dir, f"{ts}_{name}.json")
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--config", name,
            "--iters", str(args.iters),
            "--warmup", str(args.warmup),
            "--n-ctx", str(args.n_ctx),
            "--max-nodes", str(args.max_nodes),
            "--ngl", str(args.ngl),
            "--n-threads", str(args.n_threads),
            "--task-suite", args.task_suite,
            "--json-out", json_path,
        ]
        print(f"\n>>> running preset '{name}' in subprocess: {json_path}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"[error] preset '{name}' failed (rc={rc}); skipping in summary")
            continue
        with open(json_path) as f:
            r = json.load(f)
        results.append(r)
        print_result(r)
    print_comparison(results)
    summary_path = os.path.join(out_dir, f"{ts}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
    return results


def run_all_inproc(args) -> List[dict]:
    selected = args.presets if args.presets else list(PRESETS.keys())
    results: List[dict] = []
    for name in selected:
        if name not in PRESETS:
            print(f"[skip] unknown preset: {name}")
            continue
        print(f"\n>>> running preset '{name}' (in-process)")
        r = run_one_config(PRESETS[name], args.iters, args.warmup,
                           args.n_ctx, args.max_nodes, args.ngl, args.n_threads,
                           args.task_suite)
        print_result(r)
        results.append(r)
    print_comparison(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(PRESETS.keys()),
                        help="Run only this preset (single-config worker mode).")
    parser.add_argument("--all", action="store_true",
                        help="Run all (or filtered via --presets) presets and print comparison.")
    parser.add_argument("--presets", nargs="+",
                        help="Subset of preset names to run when using --all.")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of timed iterations per preset (default: 20).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations per preset (default: 3).")
    parser.add_argument("--n-ctx", type=int, default=768,
                        help="LLM context size (default: 768, matches LIBERO eval).")
    parser.add_argument("--max-nodes", type=int, default=4096,
                        help="Max graph nodes (default: 4096).")
    parser.add_argument("--ngl", type=int, default=999,
                        help="Number of GPU layers (default: 999, full offload).")
    parser.add_argument("--n-threads", type=int, default=4,
                        help="CPU threads (default: 4).")
    parser.add_argument("--task-suite", default="libero_spatial_no_noops",
                        help="Dataset stats key for combined model "
                             "(default: libero_spatial_no_noops).")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="With --all, run presets in same process (less clean VRAM).")
    parser.add_argument("--json-out",
                        help="If set with --config, write the result JSON here.")
    parser.add_argument("--json-dir",
                        help="Directory for per-preset JSON results when using --all.")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("must specify either --config <name> or --all")

    if args.config:
        preset = PRESETS[args.config]
        r = run_one_config(preset, args.iters, args.warmup,
                           args.n_ctx, args.max_nodes, args.ngl, args.n_threads,
                           args.task_suite)
        print_result(r)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(r, f, indent=2)
            print(f"Saved: {args.json_out}")
        return

    if args.no_subprocess:
        run_all_inproc(args)
    else:
        run_all_subprocess(args)


if __name__ == "__main__":
    main()
