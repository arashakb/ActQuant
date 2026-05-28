# ActQuant

Official implementation of **"ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models"** ([arXiv:2605.24011](https://arxiv.org/abs/2605.24011)).

ActQuant is a two-stage post-training quantization recipe specialized for
Vision-Language-Action (VLA) models. It produces sub-4-bit checkpoints that
preserve closed-loop task success on LIBERO while shrinking model footprint
by 4–5×.

- **Stage 1 — HSIC-based inter-tensor bit allocation** ([`tools/hsic/`](tools/hsic/)).
  Score every linear weight in the LLM by its Hilbert-Schmidt Independence
  with respect to ground-truth actions, then run a greedy per-layer-L²
  allocator under an average-bits-per-weight budget. Output: one quant
  type per tensor.
- **Stage 2 — Action-Mixed Fisher (AMF) imatrix** ([`tools/fisher-diag/`](tools/fisher-diag/)).
  Compute a per-element Fisher diagonal using an action-loss-weighted blend
  of the LM-head NLL and the action-head L1, stored as a GGUF imatrix.
  Output: a per-weight importance map consumed by `llama-quantize` during
  block-level scale optimization.

Both stages plug into a single `llama-quantize` invocation
(`--tensor-type <per-tensor overrides>` + `--imatrix <AMF.gguf>`), and the
resulting checkpoint is loaded by the C++ runtimes provided here
(`openvla_oft` and `pi05`) for closed-loop LIBERO evaluation.

This repository is a fork of [`llama.cpp`](https://github.com/ggerganov/llama.cpp).
Upstream tooling (`src/`, `common/`, `ggml/`, `tools/quantize/`,
`tools/quantize-vision/`) is preserved unchanged; the ActQuant additions
live under `tools/hsic/`, `tools/fisher-diag/`, `tools/openvla-oft/`,
and `tools/pi0.5/`.

---

## Backbones supported

| Backbone | Path | Built target | Python binding |
|---|---|---|---|
| OpenVLA-OFT | [`tools/openvla-oft/`](tools/openvla-oft/) | `openvla_oft` | `openvla_oft.so` |
| Pi 0.5 (PaliGemma + flow-matching action expert) | [`tools/pi0.5/`](tools/pi0.5/) | `pi05` | `pi05.so` |

---

## Prerequisites

- Ubuntu 22.04 (tested) or any Linux with recent glibc
- NVIDIA CUDA 12.x (tested with 12.2 for OFT, 12.6 for Pi 0.5)
- CMake ≥ 3.14, Ninja or Make
- Conda / Miniconda
- `uv` (for the Pi 0.5 Python env)
- ~50 GB free disk for build outputs and intermediate GGUFs

Two separate Python environments are needed — the two backbones depend on
incompatible upstream stacks:

| Env | Purpose | Source |
|---|---|---|
| `openvla-oft` (Python 3.10, conda) | OFT calibration scripts + LIBERO eval | [openvla-oft](https://github.com/moojink/openvla-oft) |
| `openpi-server` (Python 3.11, uv) | Pi 0.5 calibration scripts + WebSocket policy server | [openpi](https://github.com/Physical-Intelligence/openpi) |
| `openpi-libero` (Python 3.8, conda) | Pi 0.5 LIBERO eval **client** only (`bddl`, `robosuite`) | LIBERO requirements |

[`tools/pi0.5/01_setup_env.sh`](tools/pi0.5/01_setup_env.sh) scripts the
Pi 0.5 environments end-to-end.

---

## Build

ActQuant uses two parallel build trees (one per backbone) because OFT and
Pi 0.5 target different CUDA toolchains.

### Clone & unpack vendored tokenizers

```bash
git clone <this-repo>.git ActQuant
cd ActQuant
unzip vendor/tokenizers-cpp.zip -d vendor/
```

### OpenVLA-OFT build (`build_oft/`)

```bash
mkdir -p build_oft && cd build_oft
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DBUILD_PYTHON=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DPython_EXECUTABLE=$(which python)
cmake --build . -j$(nproc)
cd ..
```

Outputs: `build_oft/bin/openvla_oft` (CLI), `build_oft/bin/openvla_oft.so`
(Python binding), `build_oft/bin/llama-quantize`.

### Pi 0.5 build (`build_openpi/`)

```bash
mkdir -p build_openpi && cd build_openpi
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DBUILD_PI05_PYTHON=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target pi05 pi05_py llama-quantize quantize-vision -j$(nproc)
cd ..
```

Outputs: `build_openpi/bin/pi05` (CLI), `build_openpi/bin/pi05.so`
(Python binding), `build_openpi/bin/llama-quantize`,
`build_openpi/bin/quantize-vision`.

> The `llama-quantize` binary is what both ActQuant stages call. Make sure
> you call the binary from the **same** build tree as the runtime you
> intend to evaluate with — their `libggml*.so` versions may differ.

---

## Checkpoints and calibration data

ActQuant is **not** the source of any model weights or robot data — those
come from already-public releases. Below is what to download and where to
put it; the calibration data is reconstructed locally from public LIBERO
trajectories using scripts that ship with this repo.

### OpenVLA-OFT (combined LIBERO checkpoint)

Download the publicly released checkpoint jointly fine-tuned on all four
LIBERO suites (referred to as `oft_combined` throughout) from the
OpenVLA-OFT model zoo and place it at any path of your choosing
(the run scripts default to `/home/$USER/arash/openvla-oft-checkpoints/oft_combined`;
edit them or pass `--checkpoint <path>` explicitly).

Required files inside the checkpoint dir:

```
oft_combined/
├── config.json
├── *.safetensors
├── action_head--300000_checkpoint.pt
├── proprio_projector--300000_checkpoint.pt
└── dataset_statistics.json
```

### Pi 0.5 LIBERO checkpoint

```bash
huggingface-cli download lerobot/pi05_libero_finetuned_v044 \
    --local-dir /path/to/pi05_libero
huggingface-cli download google/paligemma-3b-pt-224 tokenizer.model \
    --local-dir /path/to/pi05_libero
```

### Calibration data (LIBERO subset, reconstructed locally)

The paper uses a 10/10/10/30-episode split (60 total) drawn from LIBERO
spatial/object/goal/long. The data are **public** (LIBERO ships them); only
the export format is paper-specific. Reconstruct in place:

```bash
# OpenVLA-OFT: flat *_<N>.bin + *_<N>_targets.npy per suite
python tools/fisher-diag/export_oft_actions.py \
    --num-episodes 10 10 10 30 \
    --output-dir /path/to/oft_calib_data

# Pi 0.5: single calib_dir with pixel_values/state/gt_actions/prompts
python tools/fisher-diag/get_pi05_calib_data.py \
    --output-dir /path/to/pi05_calib_data
```

LIBERO itself: `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO && pip install -e .`
plus `apt install libegl1-mesa-dev` and `export MUJOCO_GL=egl` for headless rendering.

---

## Reproducing the paper — end-to-end

The high-level pipeline is identical for both backbones:

```
PyTorch checkpoint
    └── export_*.py            ──►  pi05.gguf / llm_bf16.gguf  (Step 0)
                                          │
   Stage 2 (AMF) ◄────────────────────────┤
   compute_amf_lq.py / compute_fisher_pi05.py
       └── action-mixed Fisher imatrix.gguf
                                          │
   Stage 1 (HSIC) ◄───────────────────────┤
   compute_hsic_*.py  +  allocate*.py
       └── per-tensor type overrides (JSON)
                                          │
   llama-quantize  --imatrix <AMF.gguf>  --tensor-type <overrides>
       └── quantized backbone
                                          │
   merge_pi05_llm.py (Pi 0.5 only) ──►  pi05_<base>_hsic.gguf
                                          │
   LIBERO closed-loop eval through the pybind11 binding (openvla_oft.so / pi05.so)
```

Each backbone has a single driver script that chains everything:

- OFT: [`tools/hsic/run_hsic_fisher_quant.sh`](tools/hsic/run_hsic_fisher_quant.sh)
- Pi 0.5: [`tools/hsic/run_hsic_quant_pi05.sh`](tools/hsic/run_hsic_quant_pi05.sh)

Both expect a few hardcoded paths near the top — set them to match where
you put the checkpoints and calibration data, then:

```bash
bash tools/hsic/run_hsic_fisher_quant.sh \
    --base-type IQ2_XS --max-type Q4_K --score-key sens --num-gpus 8
```

### What the driver does (step by step)

1. **Step 0 — Export to GGUF.** PyTorch → unified `pi05.gguf` (Pi 0.5) or
   `llm_bf16.gguf` + vision GGUFs (OFT). See
   [`tools/openvla-oft/export_openvla_oft.py`](tools/openvla-oft/export_openvla_oft.py)
   and [`tools/pi0.5/export_pi05.py`](tools/pi0.5/export_pi05.py).
2. **Stage 2 — Compute AMF imatrix.** Per-element Fisher diagonal under
   the action-mixed loss (α=0.5 blends LM-head NLL with action-head L1
   for OFT; α=1 for Pi 0.5, no LM-head action vocabulary). Written as a
   GGUF in llama-imatrix per-weight format.
3. **Stage 1 — HSIC scoring + greedy allocation.** Activation-only forward
   pass through the bf16 model to collect per-tensor HSIC sensitivities,
   then a layer-balanced L² greedy allocator picks one quant type per
   tensor under the target BPW budget.
4. **Quantize.** `llama-quantize --imatrix <AMF.gguf> --tensor-type <override-list> <input>.gguf <output>.gguf <base-type>`.
5. **Merge** (Pi 0.5 only). Splice the quantized PaliGemma back into the
   full `pi05.gguf` via [`tools/pi0.5/merge_pi05_llm.py`](tools/pi0.5/merge_pi05_llm.py).
6. **(Optional) Vision tower.** DINOv2 + SigLIP are quantized once to
   Q4_K_M via `quantize-vision` (no HSIC/AMF — it's a fixed feature
   extractor) and reused across all LLM quantizations.

The HSIC activations are cached (`/tmp/hsic_acts_*.npz`) and the AMF GGUF
is reused across HSIC sweeps, so only Stages 4–5 re-run when you sweep
base-type or BPW.

---

## LIBERO evaluation

> **Important:** all closed-loop numbers in the paper are produced by
> routing the LIBERO rollout's `policy.predict_action(...)` through the
> ActQuant Python bindings (`openvla_oft.so` for OFT, `pi05.so` for Pi 0.5).
> The driver scripts that do this are not bundled here — they live in the
> upstream backbone repos and need a small edit to swap in the GGML
> backend. The edit is one-line per backbone; instructions below.

### OpenVLA-OFT

The upstream `openvla-oft` repository ships
`experiments/robot/libero/run_libero_eval.py`. ActQuant's C++ runtime
exposes a drop-in `Policy` via `openvla_oft.so`. To use it:

1. Symlink the binding into your `openvla-oft` conda env so it shadows
   the PyTorch policy:

   ```bash
   SITE=$(conda run -n openvla-oft python -c "import site; print(site.getsitepackages()[0])")
   ln -sf $(pwd)/build_oft/bin/openvla_oft.so $SITE/openvla_oft.so
   for lib in build_oft/bin/libllama.so build_oft/bin/libggml*.so; do
       ln -sf $(pwd)/$lib $(conda info --base)/envs/openvla-oft/lib/$(basename $lib)
   done
   ```

2. Copy or symlink the upstream eval driver as
   `experiments/robot/libero/run_libero_eval_combined_cpp.py` and replace
   the `get_vla(...)` body with the binding:

   ```python
   import openvla_oft  # the .so produced by build_oft/
   vla = openvla_oft.OpenVLA(
       llm_gguf=str(cfg.llm_gguf_path),
       dinov2_gguf=str(cfg.dinov2_gguf_path),
       siglip_gguf=str(cfg.siglip_gguf_path),
       action_head_pt=str(cfg.action_head_path),
       proprio_projector_pt=str(cfg.proprio_projector_path),
   )
   ```

   Everything else (LIBERO env wrapper, rollout loop, per-task tallying)
   stays as upstream. The `_cpp.py` suffix is what the launcher uses to
   pick this driver over the PyTorch one — preserve it.

3. Sanity-check the binding before launching:

   ```bash
   conda activate openvla-oft
   python -c "import openvla_oft, inspect; print(inspect.getfile(openvla_oft))"
   # expected: …/build_oft/bin/openvla_oft.so
   ```

4. Run a suite:

   ```bash
   python experiments/robot/libero/run_libero_eval_combined_cpp.py \
       --multi-gpu --gpus 0,1,2,3,4,5,6,7 \
       --task_suite_name libero_spatial \
       --llm_gguf_name    llm_iq2_xs_hsic_sens_..._fisher.gguf \
       --dinov2_gguf_name dinov2_q4_k_m.gguf \
       --siglip_gguf_name siglip_q4_k_m_padded.gguf
   ```

   Repeat with `libero_object`, `libero_goal`, `libero_10`. Per-task
   success counts land in the driver's log directory.

### Pi 0.5

Pi 0.5 keeps the upstream `openpi/examples/libero/main.py` LIBERO client
intact and talks to a local WebSocket policy server that wraps the
binding. The server lives at
[`tools/pi0.5/serve_policy.py`](tools/pi0.5/serve_policy.py); the multi-GPU
launcher at [`tools/pi0.5/run_libero_eval.sh`](tools/pi0.5/run_libero_eval.sh).

1. Sanity-check the binding:

   ```bash
   PYTHONPATH=$(pwd)/build_openpi/bin python -c "import pi05, inspect; print(inspect.getfile(pi05))"
   # expected: …/build_openpi/bin/pi05.so
   ```

2. Run all four suites against a quantized checkpoint:

   ```bash
   EVAL_DIR=/path/to/pi05_libero_base_gguf/iq2_xs_hsic_eval
   for suite in libero_spatial libero_object libero_goal libero_10; do
       bash tools/pi0.5/run_libero_eval.sh "$suite" 50 5 8 8000 "$EVAL_DIR"
   done
   ```

   `EVAL_DIR` is a directory containing `pi05.gguf` (the quantized
   checkpoint), `tokenizer.model`, and `norm_stats.json`. The driver
   auto-detects GPUs, spawns one `serve_policy.py` per GPU on
   `BASE_PORT + i`, and shards LIBERO tasks across them.

> Never run more than two LIBERO suites in parallel on the same machine —
> the simulator has been observed to deadlock under heavier contention.

---

## Repository layout

```
ActQuant/
├── ggml/                       upstream GGML (CUDA / Metal / Vulkan kernels)
├── src/, common/, include/     upstream llama.cpp core
├── tools/
│   ├── openvla-oft/            OFT C++ runtime + export script + Python binding
│   ├── pi0.5/                  Pi 0.5 C++ runtime + WebSocket server + Python binding
│   ├── hsic/                   ActQuant Stage 1: per-tensor sensitivities + greedy allocator
│   ├── fisher-diag/            ActQuant Stage 2: AMF (Action-Mixed Fisher) imatrix
│   ├── quantize/               llama-quantize entry point (upstream)
│   └── quantize-vision/        DINOv2 / SigLIP quantization (upstream-style)
├── gguf-py/                    Python GGUF library (upstream)
├── vendor/tokenizers-cpp.zip   HuggingFace tokenizers C++ binding (vendored)
├── CMakeLists.txt              top-level build (ActQuant-curated tools list)
└── convert_hf_to_gguf.py       upstream HF→GGUF converter (kept for reference)
```

---

## License & acknowledgements

This repository is released under the **MIT license** (see [`LICENSE`](LICENSE)),
inherited from `llama.cpp`. The ActQuant additions are released under the
same terms.

Built on:

- [llama.cpp](https://github.com/ggerganov/llama.cpp) / [ggml](https://github.com/ggerganov/ggml)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [OpenPI](https://github.com/Physical-Intelligence/openpi) (Pi 0.5)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

---

## Citation

If you use ActQuant, please cite:

```bibtex
@article{actquant2026,
  title        = {ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models},
  author       = {Akbari, Arash and others},
  journal      = {arXiv preprint arXiv:2605.24011},
  year         = {2026}
}
```
