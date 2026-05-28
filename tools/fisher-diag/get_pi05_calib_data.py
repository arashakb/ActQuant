#!/usr/bin/env python3
"""
get_pi05_calib_data.py

Generate raw Pi 0.5 calibration inputs from LIBERO RLDS data, for use by
the Fisher-diagonal computation script (compute_fisher_pi05.py).

Sampling strategy mirrors openvla-oft's get_vla_calib_data.py:
  - 4 task suites: spatial, object, goal, long (= libero_10)
  - Default: 10 / 10 / 10 / 30 episodes (1/1/1/3 per task × 10 tasks)
  - All frames within each selected episode

Outputs (written to --output-dir):
  pixel_values.npy   (N, 3, 224, 224, 3)  uint8   3 cameras × 224x224 RGB
                                                   (right_wrist is zeros for LIBERO)
  state.npy          (N, 8)               float32 raw, un-normalized
  gt_actions.npy     (N, action_horizon, 7) float32 next-H 7-d actions
                                                    (last action repeated past episode end)
  prompts.json       list of N task description strings (lowercase)
  metadata.json      sampling info, suite breakdown, episode info

Usage:
    /path/to/openpi/.venv/bin/python3 \\
        tools/fisher-diag/get_pi05_calib_data.py \\
        --data-dir /path/to/openvla-oft/modified_libero_rlds_data \\
        --output-dir /path/to/openpi/calib_data_raw \\
        --action-horizon 50
"""

from __future__ import annotations

import argparse
import io
import json
import random
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import tqdm
from PIL import Image as PILImage


SUITE_TO_DATASET = {
    "spatial": "libero_spatial_no_noops",
    "object":  "libero_object_no_noops",
    "goal":    "libero_goal_no_noops",
    "long":    "libero_10_no_noops",
}

STATE_DIM = 8
ACTION_DIM = 7
IMAGE_SIZE = 224


# ============================================================================
# Pure-Python TFRecord + tf.train.Example parser
# ============================================================================
# Wire types: 0=varint, 1=64bit, 2=length-delim, 5=32bit
# Feature proto fields: 1=BytesList, 2=FloatList, 3=Int64List

def _vint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _skip(buf: bytes, pos: int, wt: int) -> int:
    if wt == 0:
        while buf[pos] & 0x80:
            pos += 1
        return pos + 1
    if wt == 1: return pos + 8
    if wt == 5: return pos + 4
    if wt == 2:
        n, pos = _vint(buf, pos)
        return pos + n
    raise ValueError(f"Unknown wire type {wt}")


def _parse_bytes_list(data: bytes) -> List[bytes]:
    pos = 0; out = []
    while pos < len(data):
        tag, pos = _vint(data, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 2:
            n, pos = _vint(data, pos)
            out.append(data[pos:pos + n]); pos += n
        else:
            pos = _skip(data, pos, wt)
    return out


def _parse_float_list_packed(data: bytes) -> np.ndarray:
    """Parse a FloatList proto whose `value` field (1) is packed-encoded floats."""
    pos = 0
    floats: List[np.ndarray] = []
    while pos < len(data):
        tag, pos = _vint(data, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 2:
            n, pos = _vint(data, pos)
            chunk = data[pos:pos + n]; pos += n
            floats.append(np.frombuffer(chunk, dtype=np.float32))
        else:
            pos = _skip(data, pos, wt)
    return np.concatenate(floats) if floats else np.zeros(0, dtype=np.float32)


def _parse_feature(data: bytes):
    """Returns ('bytes', list_of_bytes) or ('floats', np.ndarray) or None."""
    pos = 0
    while pos < len(data):
        tag, pos = _vint(data, pos)
        fn, wt = tag >> 3, tag & 7
        if wt != 2:
            pos = _skip(data, pos, wt); continue
        n, pos = _vint(data, pos)
        val = data[pos:pos + n]; pos += n
        if fn == 1: return ("bytes", _parse_bytes_list(val))
        if fn == 2: return ("floats", _parse_float_list_packed(val))
    return None


def _parse_example(raw: bytes, wanted: frozenset) -> Dict:
    pos = 0; features_raw = None
    while pos < len(raw):
        tag, pos = _vint(raw, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 2:
            n, pos = _vint(raw, pos)
            features_raw = raw[pos:pos + n]; pos += n
            break
        pos = _skip(raw, pos, wt)
    if features_raw is None: return {}

    out: Dict = {}
    pos = 0
    while pos < len(features_raw):
        tag, pos = _vint(features_raw, pos)
        fn, wt = tag >> 3, tag & 7
        if fn == 1 and wt == 2:
            n, pos = _vint(features_raw, pos)
            ent = features_raw[pos:pos + n]; pos += n
            ep = 0; key = None; feat_raw = None
            while ep < len(ent):
                etag, ep = _vint(ent, ep)
                efn, ewt = etag >> 3, etag & 7
                if ewt != 2:
                    ep = _skip(ent, ep, ewt); continue
                en, ep = _vint(ent, ep)
                ev = ent[ep:ep + en]; ep += en
                if efn == 1:   key = ev.decode("utf-8")
                elif efn == 2: feat_raw = ev
            if key in wanted and feat_raw is not None:
                out[key] = _parse_feature(feat_raw)
        else:
            pos = _skip(features_raw, pos, wt)
    return out


def _iter_tfrecord(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: return
            length = struct.unpack("<Q", hdr)[0]
            f.read(4)
            data = f.read(length)
            if len(data) < length: return
            f.read(4)
            yield data


# ============================================================================
# Episode iteration
# ============================================================================

WANTED_KEYS = frozenset({
    "steps/observation/image",
    "steps/observation/wrist_image",
    "steps/observation/state",
    "steps/action",
    "steps/language_instruction",
})


def find_builder_dir(suite_root: Path) -> Path:
    if (suite_root / "dataset_info.json").exists():
        return suite_root
    for sub in sorted(suite_root.iterdir()):
        if sub.is_dir() and (sub / "dataset_info.json").exists():
            return sub
    raise FileNotFoundError(f"dataset_info.json not found under {suite_root}")


def list_shards(data_dir: Path, dataset_name: str) -> List[Path]:
    builder = find_builder_dir(data_dir / dataset_name)
    shards = sorted(builder.glob("*.tfrecord*"))
    if not shards:
        raise FileNotFoundError(f"No tfrecord shards in {builder}")
    return shards


def iter_episodes(shards: List[Path]) -> Iterator[Dict]:
    for shard in shards:
        for raw in _iter_tfrecord(shard):
            yield _parse_example(raw, WANTED_KEYS)


def episode_first_lang(feat: Dict) -> str:
    li = feat.get("steps/language_instruction")
    if li is None or li[0] != "bytes" or len(li[1]) == 0:
        return ""
    return li[1][0].decode("utf-8").lower().strip()


def resize_uint8(img_uint8: np.ndarray) -> np.ndarray:
    """HWC uint8 -> 224x224 HWC uint8 via PIL bilinear."""
    if img_uint8.shape[0] == IMAGE_SIZE and img_uint8.shape[1] == IMAGE_SIZE:
        return img_uint8
    pil = PILImage.fromarray(img_uint8).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)
    return np.asarray(pil)


# ============================================================================
# Stratified sampling
# ============================================================================

def index_episodes_by_task(shards: List[Path], desc: str) -> Dict[str, List[int]]:
    """First pass: group episode indices by language_instruction."""
    by_task: Dict[str, List[int]] = {}
    for ep_idx, feat in enumerate(tqdm.tqdm(iter_episodes(shards), desc=f"  Indexing {desc}")):
        lang = episode_first_lang(feat)
        if not lang: continue
        by_task.setdefault(lang, []).append(ep_idx)
    return by_task


def stratified_select(by_task: Dict[str, List[int]], n_episodes: int, seed: int):
    """Pick n_episodes total, balanced across tasks (n_episodes / n_tasks per task)."""
    n_tasks = len(by_task)
    if n_tasks == 0:
        raise ValueError("No tasks with language instructions found.")
    if n_episodes % n_tasks != 0:
        raise ValueError(
            f"--episodes={n_episodes} must be divisible by num_tasks={n_tasks}")
    per_task = n_episodes // n_tasks
    rng = random.Random(seed)
    selected = set()
    ep_to_task: Dict[int, str] = {}
    print(f"  {n_tasks} tasks, {per_task} episode(s) per task = {n_episodes} total:")
    for task in sorted(by_task.keys()):
        avail = by_task[task]
        if per_task > len(avail):
            raise ValueError(
                f"Need {per_task} episodes for task '{task}' but only {len(avail)} available.")
        chosen = rng.sample(avail, per_task)
        selected.update(chosen)
        for ep in chosen: ep_to_task[ep] = task
        short = task[:60] + "..." if len(task) > 60 else task
        print(f"    {short!r:<65} {per_task}/{len(avail)}")
    return selected, ep_to_task


# ============================================================================
# Main collection
# ============================================================================

def collect_suite(
    data_dir: Path,
    suite: str,
    n_episodes: int,
    action_horizon: int,
    seed: int,
):
    dataset_name = SUITE_TO_DATASET[suite]
    shards = list_shards(data_dir, dataset_name)
    print(f"\n{'=' * 70}\nSuite: {suite}  ({dataset_name})  shards: {len(shards)}\n{'=' * 70}")

    # Pass 1: stratified sampling
    by_task = index_episodes_by_task(shards, suite)
    selected, ep_to_task = stratified_select(by_task, n_episodes, seed)

    # Pass 2: extract data for selected episodes
    out = {
        "pixel_values": [],   # list of (3, 224, 224, 3) uint8
        "state":        [],   # list of (8,) float32
        "gt_actions":   [],   # list of (H, 7) float32
        "prompts":      [],   # list of str
    }
    episode_info: List[Dict] = []

    n_done = 0
    pbar = tqdm.tqdm(total=len(selected), desc=f"  Collecting {suite}")
    for ep_idx, feat in enumerate(iter_episodes(shards)):
        if ep_idx not in selected:
            continue

        # Decode per-step bytes/floats
        img_list  = feat.get("steps/observation/image", ("none", []))[1]
        wrist_list = feat.get("steps/observation/wrist_image", ("none", []))[1]
        state_arr  = feat.get("steps/observation/state", ("floats", np.zeros(0, dtype=np.float32)))[1]
        action_arr = feat.get("steps/action", ("floats", np.zeros(0, dtype=np.float32)))[1]

        n_steps = len(img_list)
        if n_steps == 0:
            print(f"  [warn] ep {ep_idx}: no image steps, skipping"); continue
        # Reshape state/actions into per-step rows
        state_per_step = state_arr.reshape(-1, STATE_DIM) if state_arr.size else np.zeros((n_steps, STATE_DIM), np.float32)
        action_per_step = action_arr.reshape(-1, ACTION_DIM) if action_arr.size else np.zeros((n_steps, ACTION_DIM), np.float32)
        if state_per_step.shape[0] != n_steps or action_per_step.shape[0] != n_steps:
            print(f"  [warn] ep {ep_idx}: shape mismatch (img={n_steps} "
                  f"state={state_per_step.shape[0]} action={action_per_step.shape[0]}), skipping"); continue

        task = ep_to_task[ep_idx]

        for i in range(n_steps):
            base_img  = np.asarray(PILImage.open(io.BytesIO(img_list[i])).convert("RGB"))
            wrist_img = (np.asarray(PILImage.open(io.BytesIO(wrist_list[i])).convert("RGB"))
                         if i < len(wrist_list) and wrist_list[i] else np.zeros_like(base_img))

            base_resized  = resize_uint8(base_img)
            wrist_resized = resize_uint8(wrist_img)
            zeros_cam     = np.zeros_like(base_resized)
            cams_3 = np.stack([base_resized, wrist_resized, zeros_cam], axis=0)  # (3, 224, 224, 3) uint8

            # Action chunk: actions[i : i+H], pad-with-last past episode end
            end = min(i + action_horizon, n_steps)
            chunk = action_per_step[i:end]
            if chunk.shape[0] < action_horizon:
                pad = np.tile(chunk[-1:], (action_horizon - chunk.shape[0], 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            assert chunk.shape == (action_horizon, ACTION_DIM)

            out["pixel_values"].append(cams_3)
            out["state"].append(state_per_step[i].astype(np.float32))
            out["gt_actions"].append(chunk.astype(np.float32))
            out["prompts"].append(task)

        episode_info.append({"task": task, "num_frames": n_steps})
        n_done += 1
        pbar.update(1)
        if n_done >= len(selected):
            break
    pbar.close()
    print(f"  Collected {len(out['pixel_values'])} frames from {n_done} episodes")
    return out, episode_info


def main():
    p = argparse.ArgumentParser(description="Generate Pi 0.5 calibration data from LIBERO RLDS")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/path/to/openvla-oft/modified_libero_rlds_data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("/path/to/openpi/calib_data_raw"))
    p.add_argument("--spatial-episodes", type=int, default=10)
    p.add_argument("--object-episodes",  type=int, default=10)
    p.add_argument("--goal-episodes",    type=int, default=10)
    p.add_argument("--long-episodes",    type=int, default=30)
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite_eps = {
        "spatial": args.spatial_episodes,
        "object":  args.object_episodes,
        "goal":    args.goal_episodes,
        "long":    args.long_episodes,
    }
    print("=" * 70)
    print("Pi 0.5 Calibration Data Generator (raw inputs)")
    print("=" * 70)
    print(f"  data_dir:        {args.data_dir}")
    print(f"  output_dir:      {args.output_dir}")
    print(f"  episodes:        {suite_eps}")
    print(f"  action_horizon:  {args.action_horizon}")
    print(f"  seed:            {args.seed}")

    all_pixel_values: List[np.ndarray] = []
    all_state:        List[np.ndarray] = []
    all_actions:      List[np.ndarray] = []
    all_prompts:      List[str] = []
    suite_stats = []

    for suite, n_ep in suite_eps.items():
        out, ep_info = collect_suite(args.data_dir, suite, n_ep, args.action_horizon, args.seed)
        suite_stats.append({
            "suite": suite,
            "dataset": SUITE_TO_DATASET[suite],
            "num_episodes": n_ep,
            "num_frames": len(out["pixel_values"]),
            "episodes": ep_info,
        })
        all_pixel_values.extend(out["pixel_values"])
        all_state.extend(out["state"])
        all_actions.extend(out["gt_actions"])
        all_prompts.extend(out["prompts"])

    N = len(all_pixel_values)
    print(f"\n{'=' * 70}")
    print(f"Total frames: {N}")
    print('=' * 70)

    pv = np.stack(all_pixel_values, axis=0)  # (N, 3, 224, 224, 3) uint8
    np.save(args.output_dir / "pixel_values.npy", pv)
    print(f"  pixel_values.npy : {pv.shape} {pv.dtype}  ({pv.nbytes / 1024**3:.2f} GB)")
    del pv

    sa = np.stack(all_state, axis=0).astype(np.float32)
    np.save(args.output_dir / "state.npy", sa)
    print(f"  state.npy        : {sa.shape} {sa.dtype}  range=[{sa.min():.3f}, {sa.max():.3f}]")

    aa = np.stack(all_actions, axis=0).astype(np.float32)
    np.save(args.output_dir / "gt_actions.npy", aa)
    print(f"  gt_actions.npy   : {aa.shape} {aa.dtype}  range=[{aa.min():.3f}, {aa.max():.3f}]")

    with open(args.output_dir / "prompts.json", "w") as f:
        json.dump(all_prompts, f)
    print(f"  prompts.json     : {len(all_prompts)} task strings")

    metadata = {
        "total_frames":   N,
        "action_horizon": args.action_horizon,
        "image_size":     IMAGE_SIZE,
        "state_dim":      STATE_DIM,
        "action_dim":     ACTION_DIM,
        "n_cameras":      3,
        "camera_order":   ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "right_wrist_is_zeros": True,
        "seed":           args.seed,
        "data_dir":       str(args.data_dir),
        "suites":         suite_stats,
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json    : saved")
    print("\nDone.")


if __name__ == "__main__":
    main()
