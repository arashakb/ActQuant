#!/usr/bin/env python3
"""
Export Continuous OFT Action Targets for AMF

Extracts ground-truth continuous action targets (shape [N, 8, 7]) from an RLDS dataset,
matching the frame ordering of an existing .bin calibration file (produced by
vla-scripts/get_llm_calib_data.py with the same --seed and --num-episodes).

These targets are used by compute_amf.py as a_gt in the Mahalanobis Fisher loss:
    L_amf = ||Σ_task^{-1/2} (a_pred - a_gt)||_F²

Usage:
    python tools/fisher-diag/export_action_targets.py \
        --vla-path ~/arash/openvla-oft-checkpoints/oft_combined \
        --data-root-dir ~/arash/openvla-oft/modified_libero_rlds_data \
        --dataset-name libero_spatial_no_noops \
        --output ~/arash/openvla-oft/calib_data/spatial_10_oft_targets.npy \
        --num-episodes 10 \
        --seed 42

The output .npy has shape (N, 8, 7): N frames, 8-step action chunk, 7-DoF actions.
Actions are in normalized space (approximately [-1, 1], BOUNDS_Q99 normalization).
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tqdm

# Add openvla-oft to path
_OPENVLA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_PATH))


def select_episode_indices_stratified(
    data_root_dir: Path,
    dataset_name: str,
    num_episodes: int,
    image_sizes,
    seed: int,
) -> set:
    """Select episode indices with equal per-task sampling.

    Mirrors get_llm_calib_data.py's select_episode_indices_stratified exactly
    so frame ordering matches the existing .bin calibration file.
    """
    from prismatic.vla.datasets import EpisodicRLDSDataset

    class EpisodeTaskTransform:
        def __call__(self, rlds_batch: Dict) -> Dict:
            lang = rlds_batch["task"]["language_instruction"].decode().lower()
            return {"language_instruction": lang}

    index_dataset = EpisodicRLDSDataset(
        data_root_dir, dataset_name, EpisodeTaskTransform(),
        resize_resolution=image_sizes, shuffle_buffer_size=1, image_aug=False,
    )

    task_to_indices: Dict[str, List[int]] = {}
    num_total = len(index_dataset)
    for ep_idx, episode_frames in enumerate(tqdm.tqdm(
        index_dataset, total=num_total, desc="Indexing episodes by task"
    )):
        if len(episode_frames) == 0:
            continue
        task = episode_frames[0]["language_instruction"]
        task_to_indices.setdefault(task, []).append(ep_idx)

    if num_episodes == -1 or num_episodes >= num_total:
        selected = set(range(num_total))
        print(f"[*] Collecting all episodes: {len(selected)}")
        return selected

    num_tasks = len(task_to_indices)
    if num_tasks == 0:
        raise ValueError("No tasks found while indexing episodes.")
    if num_episodes % num_tasks != 0:
        raise ValueError(
            f"num_episodes={num_episodes} must be divisible by number of tasks={num_tasks}."
        )

    per_task = num_episodes // num_tasks
    selected = set()
    print(f"[*] Stratified sampling: {per_task} episode(s) per task across {num_tasks} tasks")
    for task in sorted(task_to_indices.keys()):
        indices = task_to_indices[task]
        if per_task > len(indices):
            raise ValueError(
                f"Requested {per_task} episodes for task '{task}', "
                f"but only {len(indices)} available."
            )
        chosen = random.sample(indices, per_task)
        selected.update(chosen)
        print(f"    - {task}: selected {len(chosen)} / {len(indices)}")

    print(f"[*] Selected {len(selected)} episodes total (balanced)")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Export continuous OFT action targets for AMF computation"
    )
    parser.add_argument("--vla-path", type=str, required=True,
                        help="HuggingFace Hub ID or local path to OpenVLA-OFT checkpoint")
    parser.add_argument("--data-root-dir", type=Path, required=True,
                        help="Path to RLDS dataset directory")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name (e.g. libero_spatial_no_noops)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .npy file path (shape [N, 8, 7])")
    parser.add_argument("--num-episodes", type=int, default=-1,
                        help="Number of episodes to sample (-1 = all)")
    parser.add_argument("--num-images-in-input", type=int, default=2,
                        help="Number of images (1 or 2, must match .bin export)")
    parser.add_argument("--use-proprio", action="store_true", default=True,
                        help="Use proprioception (must match .bin export)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (must match .bin export)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Register model classes (needed for config loading even without model load)
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load config to get image_sizes (no model load needed)
    print("[*] Loading config...")
    model_config = AutoConfig.from_pretrained(args.vla_path, trust_remote_code=True)
    image_sizes = tuple(model_config.image_sizes)
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)

    # Build dataset with same transforms as get_llm_calib_data.py
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction

    print(f"[*] Loading dataset: {args.dataset_name}")
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(args.num_images_in_input > 1),
        use_proprio=args.use_proprio,
    )

    dataset = EpisodicRLDSDataset(
        args.data_root_dir, args.dataset_name, batch_transform,
        resize_resolution=image_sizes, shuffle_buffer_size=1, image_aug=False,
    )
    print(f"  Total episodes: {len(dataset)}")

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    # Stratified episode selection — identical to get_llm_calib_data.py
    selected = select_episode_indices_stratified(
        args.data_root_dir, args.dataset_name,
        args.num_episodes, image_sizes, args.seed,
    )
    num_total = len(dataset)
    print(f"[*] Extracting actions from {len(selected)} episodes (all frames per episode)")

    actions_list: List[np.ndarray] = []
    episodes_done = 0

    for ep_idx, episode_frames in enumerate(tqdm.tqdm(dataset, total=num_total)):
        if ep_idx not in selected:
            continue

        for i in range(len(episode_frames)):
            batch = collator([episode_frames[i]])
            # batch["actions"]: (B, 8, 7) continuous normalized actions
            # These are the OFT ground-truth targets used in L1 regression training
            actions = batch["actions"].float().numpy()  # (1, 8, 7)
            actions_list.append(actions[0])  # (8, 7)

        episodes_done += 1
        if episodes_done >= len(selected):
            break

    actions_array = np.stack(actions_list, axis=0)  # (N, 8, 7)
    print(f"\n[*] Collected {len(actions_list)} frames from {episodes_done} episodes")
    print(f"  actions_array.shape = {actions_array.shape}")

    # Print per-dimension stats (useful for Σ_task verification)
    a_flat = actions_array.reshape(-1, 7)  # (N*8, 7)
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print(f"\nPer-action-dimension statistics (normalized space):")
    print(f"  {'dim':<10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    print(f"  {'-'*52}")
    for k, name in enumerate(dim_names):
        print(f"  {name:<10} {a_flat[:, k].mean():>10.4f} {a_flat[:, k].std():>10.4f} "
              f"{a_flat[:, k].min():>10.4f} {a_flat[:, k].max():>10.4f}")

    sigma_task = a_flat.std(axis=0) + 1e-8
    sigma_inv_sq = 1.0 / (sigma_task ** 2)
    print(f"\nΣ_task^{{-1}} (Mahalanobis weights per action dim):")
    for k, name in enumerate(dim_names):
        print(f"  {name:<10}: sigma={sigma_task[k]:.4f}, weight={sigma_inv_sq[k]:.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, actions_array)
    print(f"\n[*] Saved to {args.output} (shape={actions_array.shape}, dtype={actions_array.dtype})")
    print("Done!")


if __name__ == "__main__":
    main()
