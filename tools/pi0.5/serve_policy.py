#!/usr/bin/env python3
"""
Pi0.5 Policy Server using C++ inference backend with Python WebSocket server.
Compatible with OpenPI's WebSocket protocol for seamless Libero integration.

Usage:
    python serve_policy.py --model-dir /path/to/pi05_libero_pytorch --port 8000
"""

import argparse
import asyncio
import functools
import json
import logging
import time
from pathlib import Path

import msgpack
import numpy as np
import cv2  # for image preprocessing


# ============================================================================
# msgpack_numpy: NumPy array support for msgpack (from OpenPI)
# ============================================================================
def _convert_numpy_recursive(obj):
    """Recursively convert numpy arrays in nested structures to msgpack-serializable format."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),  # Convert tuple to list for msgpack
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    if isinstance(obj, dict):
        return {k: _convert_numpy_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_recursive(v) for v in obj]
    return obj


def _unpack_numpy(obj):
    """Convert msgpack dict back to numpy arrays."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"]))
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class MsgpackNumpyPacker:
    """Packer that handles numpy arrays by converting them before packing."""
    def __init__(self):
        self._packer = msgpack.Packer()

    def pack(self, obj):
        converted = _convert_numpy_recursive(obj)
        return self._packer.pack(converted)


# Create msgpack functions with numpy support
class msgpack_numpy:
    Packer = MsgpackNumpyPacker

    @staticmethod
    def packb(obj):
        return msgpack.packb(_convert_numpy_recursive(obj))

    @staticmethod
    def unpackb(data):
        return msgpack.unpackb(data, object_hook=_unpack_numpy)

import websockets.asyncio.server as ws_server
import websockets.frames

# Import the C++ Pi0.5 module
import pi05

# For loading norm stats from safetensors
try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Image preprocessing: resize_with_pad (same as OpenPI)
# ============================================================================
def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image to target size while preserving aspect ratio, padding with black.

    Args:
        image: Input image as numpy array (H, W, C) in uint8 [0, 255] or float32 [-1, 1]
        height: Target height
        width: Target width

    Returns:
        Resized and padded image with same dtype as input
    """
    is_float = image.dtype == np.float32
    cur_height, cur_width = image.shape[:2]

    # Calculate resize ratio to fit within target size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_h0 = (height - resized_height) // 2
    pad_h1 = height - resized_height - pad_h0
    pad_w0 = (width - resized_width) // 2
    pad_w1 = width - resized_width - pad_w0

    # Pad with black (-1.0 for float, 0 for uint8)
    pad_value = -1.0 if is_float else 0
    padded = np.pad(
        resized,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode='constant',
        constant_values=pad_value
    )

    return padded


# ============================================================================
# Action unnormalization using MEAN_STD normalization
# ============================================================================
def load_norm_stats(model_dir: Path) -> dict:
    """Load normalization statistics from model directory.

    Tries in order:
    1. norm_stats.json (exported by export_pi05.py)
    2. policy_preprocessor safetensors files (original LeRobot format)

    Returns dict with normalization stats. For Pi05, uses quantile normalization (q01/q99).
    For Pi0/LeRobot, uses mean/std normalization.
    """
    norm_stats = {}

    # Try 1: Load from norm_stats.json (preferred, exported by export_pi05.py)
    json_path = model_dir / "norm_stats.json"
    if json_path.exists():
        import json
        with open(json_path, "r") as f:
            data = json.load(f)

        # Check for quantile stats first (Pi05 uses quantile normalization)
        if "action_q01" in data and "action_q99" in data:
            norm_stats["q01"] = np.array(data["action_q01"], dtype=np.float32)
            norm_stats["q99"] = np.array(data["action_q99"], dtype=np.float32)
            norm_stats["use_quantile"] = True
            logger.info(f"Loaded QUANTILE norm stats from {json_path} (action_dim={len(norm_stats['q01'])})")
            return norm_stats

        # Fall back to mean/std (LeRobot format)
        if "action_mean" in data:
            norm_stats["mean"] = np.array(data["action_mean"], dtype=np.float32)
            norm_stats["std"] = np.array(data["action_std"], dtype=np.float32)
            norm_stats["use_quantile"] = False
            logger.info(f"Loaded MEAN_STD norm stats from {json_path} (action_dim={len(norm_stats['mean'])})")
            return norm_stats

    # Try 2: Load from safetensors files (original LeRobot format - uses mean/std)
    if HAS_SAFETENSORS:
        pre_path = model_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"
        if pre_path.exists():
            with safe_open(str(pre_path), framework="numpy") as f:
                if "action.mean" in f.keys():
                    norm_stats["mean"] = f.get_tensor("action.mean").astype(np.float32)
                    norm_stats["std"] = f.get_tensor("action.std").astype(np.float32)
                    norm_stats["use_quantile"] = False
                    logger.info(f"Loaded MEAN_STD norm stats from {pre_path} (action_dim={len(norm_stats['mean'])})")
                    return norm_stats

        post_path = model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        if post_path.exists():
            with safe_open(str(post_path), framework="numpy") as f:
                if "action.mean" in f.keys():
                    norm_stats["mean"] = f.get_tensor("action.mean").astype(np.float32)
                    norm_stats["std"] = f.get_tensor("action.std").astype(np.float32)
                    norm_stats["use_quantile"] = False
                    logger.info(f"Loaded MEAN_STD norm stats from {post_path} (action_dim={len(norm_stats['mean'])})")
                    return norm_stats

    logger.warning("No norm stats found! Actions will NOT be unnormalized.")
    return norm_stats


def unnormalize_actions_mean_std(actions: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Unnormalize actions using MEAN_STD normalization (reverse of normalize).

    LeRobot pi05_libero_finetuned uses MEAN_STD normalization during training:
        normalized = (x - mean) / (std + eps)

    This reverses it:
        unnormalized = x * (std + eps) + mean

    Args:
        actions: Normalized actions, shape (..., action_dim)
        mean: Mean values for each action dimension
        std: Standard deviation values for each action dimension
        eps: Small epsilon for numerical stability

    Returns:
        Unnormalized actions
    """
    action_dim = actions.shape[-1]
    mean = mean[:action_dim]
    std = std[:action_dim]
    return actions * (std + eps) + mean


def unnormalize_actions_quantile(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Unnormalize actions using QUANTILE normalization (reverse of normalize).

    OpenPI Pi0.5 uses QUANTILE normalization during training:
        normalized = (x - q01) / (q99 - q01 + eps) * 2.0 - 1.0

    This reverses it:
        unnormalized = (x + 1.0) / 2.0 * (q99 - q01 + eps) + q01

    Args:
        actions: Normalized actions, shape (..., action_dim), range [-1, 1]
        q01: 1st percentile values for each action dimension
        q99: 99th percentile values for each action dimension
        eps: Small epsilon for numerical stability

    Returns:
        Unnormalized actions
    """
    action_dim = actions.shape[-1]
    q01 = q01[:action_dim]
    q99 = q99[:action_dim]
    return (actions + 1.0) / 2.0 * (q99 - q01 + eps) + q01


class Pi05Policy:
    """Wrapper around C++ Pi05Pipeline for OpenPI-compatible interface."""

    def __init__(self, model_dir: str, device: str = "CPU", n_threads: int = 4,
                 num_flow_steps: int = 10, ode_profile_path: str = ""):
        model_dir = Path(model_dir)
        logger.info(f"Loading Pi0.5 model from {model_dir}")

        # Load normalization statistics
        self.norm_stats = load_norm_stats(model_dir)
        if not self.norm_stats:
            logger.warning("No normalization stats loaded - actions will be raw model output!")

        model_path = model_dir / "pi05.gguf"
        tokenizer_path = model_dir / "tokenizer.model"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.pipeline = pi05.Pi05Pipeline(
            model_path=str(model_path),
            tokenizer_path=str(tokenizer_path),
            device_name=device,
            n_threads=n_threads,
            num_flow_steps=num_flow_steps,
            ode_profile_path=ode_profile_path,
        )

        # Read action dimensions from model config
        self.action_horizon = self.pipeline.action_horizon
        self.action_dim = 7  # Libero uses first 7 dims of model's action_dim

        logger.info(f"Pi0.5 model loaded successfully (action_horizon={self.action_horizon})")

    def infer(self, observation: dict) -> dict:
        """Run inference on observation dict, return action dict."""
        start_time = time.monotonic()

        # Debug: print observation keys
        logger.debug(f"Observation keys: {list(observation.keys())}")

        # Extract images from observation
        base_image = observation.get("observation/image")
        if base_image is None:
            base_image = observation.get("image")

        wrist_image = observation.get("observation/wrist_image")
        if wrist_image is None:
            wrist_image = observation.get("wrist_image")

        prompt = observation.get("prompt", "")

        # Debug: print image info
        if base_image is not None:
            if isinstance(base_image, np.ndarray):
                logger.info(f"Base image: shape={base_image.shape}, dtype={base_image.dtype}")
                # Debug: save first image for inspection
                if not hasattr(self, '_saved_debug_image'):
                    import cv2
                    cv2.imwrite('/tmp/debug_base_image.png', cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
                    logger.info("Saved debug image to /tmp/debug_base_image.png")
                    self._saved_debug_image = True
            else:
                logger.info(f"Base image type: {type(base_image)}")

        logger.info(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")

        if base_image is None:
            logger.error(f"No base image! Available keys: {list(observation.keys())}")
            raise ValueError("No base image provided in observation")

        # Ensure images are numpy arrays with correct dtype (uint8)
        # Note: C++ side handles resize_with_pad preprocessing
        if isinstance(base_image, np.ndarray):
            if base_image.dtype != np.uint8:
                base_image = (base_image * 255).astype(np.uint8) if base_image.max() <= 1 else base_image.astype(np.uint8)

        # Run inference
        try:
            has_wrist = wrist_image is not None and (
                isinstance(wrist_image, np.ndarray) and wrist_image.size > 0
            )
            if has_wrist:
                if wrist_image.dtype != np.uint8:
                    wrist_image = (wrist_image * 255).astype(np.uint8) if wrist_image.max() <= 1 else wrist_image.astype(np.uint8)
                actions_flat = self.pipeline.run_multi(base_image, wrist_image, prompt)
            else:
                actions_flat = self.pipeline.run(base_image, prompt)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Return zero actions on failure
            actions_flat = np.zeros(self.action_horizon * self.action_dim, dtype=np.float32)

        # C++ model outputs [action_horizon, action_dim] elements
        # We need to reshape to [horizon, action_dim] first, then take first 7 dims for Libero
        model_action_dim = self.pipeline.action_dim  # From model config (typically 32)
        expected_size = self.action_horizon * model_action_dim

        if len(actions_flat) >= expected_size:
            # Reshape to [horizon, 32] then take first 7 dims for LIBERO
            actions = actions_flat[:expected_size].reshape(self.action_horizon, model_action_dim)
            actions = actions[:, :self.action_dim]  # Take first 7 dimensions
        else:
            # Fallback: assume output is already in correct shape
            logger.warning(f"Unexpected action size: {len(actions_flat)}, expected {expected_size}")
            actions = actions_flat.reshape(self.action_horizon, -1)
            if actions.shape[1] > self.action_dim:
                actions = actions[:, :self.action_dim]
            elif actions.shape[1] < self.action_dim:
                padded = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
                padded[:, :actions.shape[1]] = actions
                actions = padded

        # Log raw model output BEFORE unnormalization
        logger.info(f"Raw model actions: shape={actions.shape}, range=[{actions.min():.4f}, {actions.max():.4f}], mean={actions.mean():.4f}")
        logger.info(f"First action (raw): {actions[0, :7]}")

        # CRITICAL: Unnormalize actions
        # OpenPI Pi0.5 uses QUANTILE normalization (q01/q99)
        # LeRobot finetuned uses MEAN_STD normalization (mean/std)
        if self.norm_stats:
            if self.norm_stats.get("use_quantile", False):
                # Quantile unnormalization for OpenPI Pi0.5
                actions = unnormalize_actions_quantile(
                    actions,
                    self.norm_stats["q01"],
                    self.norm_stats["q99"]
                )
                logger.info(f"Unnormalized (quantile): range=[{actions.min():.4f}, {actions.max():.4f}]")
            else:
                # Mean/std unnormalization for LeRobot finetuned
                actions = unnormalize_actions_mean_std(
                    actions,
                    self.norm_stats["mean"],
                    self.norm_stats["std"]
                )
                logger.info(f"Unnormalized (mean_std): range=[{actions.min():.4f}, {actions.max():.4f}]")
            logger.info(f"First action (unnorm): {actions[0, :7]}")
        else:
            logger.warning("No norm stats available - returning raw model output!")

        infer_time = time.monotonic() - start_time
        logger.info(f"Inference time: {infer_time*1000:.1f}ms")

        return {
            "actions": actions,
            "server_timing": {
                "infer_ms": infer_time * 1000,
            }
        }


class Pi05WebSocketServer:
    """WebSocket server compatible with OpenPI protocol."""

    def __init__(self, policy: Pi05Policy, host: str = "0.0.0.0", port: int = 8000):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = {
            "server_id": f"pi05_cpp_{int(time.time())}",
            "model": "pi05_libero",
        }

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with ws_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Pi0.5 WebSocket server listening on ws://{self.host}:{self.port}")
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()

        # Send metadata on connection
        await websocket.send(packer.pack(self.metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # Receive observation
                data = await websocket.recv()
                obs = msgpack_numpy.unpackb(data)

                # Run inference
                infer_start = time.monotonic()
                action = self.policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                # Add timing info
                action["server_timing"]["infer_ms"] = infer_time * 1000
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Send response
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                raise

    async def _health_check(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(200, "OK\n")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Policy Server")
    parser.add_argument("--model-dir", "-m", type=str, required=True,
                        help="Model directory (with pi05.gguf and tokenizer.model)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                        help="WebSocket server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--device", "-d", type=str, default="CPU",
                        help="Device (CPU)")
    parser.add_argument("--threads", "-t", type=int, default=4,
                        help="Number of inference threads")
    parser.add_argument("--steps", "-s", type=int, default=10,
                        help="Number of flow matching steps")
    parser.add_argument("--ode-profile", type=str, default="",
                        help="If set, write ODE step distribution profile CSV to this path on exit")

    args = parser.parse_args()

    # Load policy
    policy = Pi05Policy(
        model_dir=args.model_dir,
        device=args.device,
        n_threads=args.threads,
        num_flow_steps=args.steps,
        ode_profile_path=args.ode_profile,
    )

    # Start server
    server = Pi05WebSocketServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
