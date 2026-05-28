import openvla_oft_spatial
import numpy as np
from PIL import Image

# Use OpenvlaOFTPipelineWithProprio for proprio support
model = openvla_oft_spatial.OpenvlaOFTPipelineWithProprio(
    dinov2_model_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/dinov2.gguf",
    siglip_model_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip.gguf",
    proj_model_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/proj.gguf",
    llm_model_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/llm_q4_k_m.gguf",
    action_head_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/action_head.gguf",
    proprio_proj_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/proprio_projector.gguf",
    tokenizer_path="/path/to/openvla-oft-checkpoints/oft_spatial_gguf/language_model/tokenizer.json",
    device_name="CUDA0",
    n_threads=4,
    max_nodes=4096,
    ngl=999,
    n_ctx=768  # Increased for 2-image input (512 + proprio + text + action tokens)
)

# Proprio: 8-dim vector for LIBERO
# Format: [pos_x, pos_y, pos_z, axis_angle_x, axis_angle_y, axis_angle_z, gripper_0, gripper_1]
proprio = [0.1, 0.05, 1.0, 3.0, -0.5, 0.2, 0.02, -0.02]
instruction = "pick up the black bowl on the wooden cabinet and place it on the plate"

# Test images (using same image for both full and wrist for testing purposes)
# In real LIBERO evaluation, these would be different camera views
full_img_path = "/path/to/vote/vote-gguf/2.png"
wrist_img_path = "/path/to/vote/vote-gguf/2.png"  # Same for testing

print(f"Has proprio projector: {model.has_proprio_projector()}")

# Test single-image version with image path (backward compatible)
print("\n=== Test 1: Single image with path ===")
actions = model.run(full_img_path, instruction, proprio)
print(f"Actions shape: {actions.shape}")
print(f"First action chunk (7 dims): {actions[:7]}")

# Test single-image version with numpy array
print("\n=== Test 2: Single image with numpy array ===")
img = Image.open(full_img_path)
img_array = np.array(img)
print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
actions = model.run(img_array, instruction, proprio)
print(f"Actions shape: {actions.shape}")
print(f"First action chunk (7 dims): {actions[:7]}")

# Test two-image version with paths (RECOMMENDED for LIBERO)
print("\n=== Test 3: Two images (full + wrist) with paths ===")
actions = model.run2(full_img_path, wrist_img_path, instruction, proprio)
print(f"Actions shape: {actions.shape}")
print(f"First action chunk (7 dims): {actions[:7]}")

# Test two-image version with numpy arrays
print("\n=== Test 4: Two images (full + wrist) with numpy arrays ===")
full_img = np.array(Image.open(full_img_path))
wrist_img = np.array(Image.open(wrist_img_path))
print(f"Full image shape: {full_img.shape}, Wrist image shape: {wrist_img.shape}")
actions = model.run2(full_img, wrist_img, instruction, proprio)
print(f"Actions shape: {actions.shape}")
print(f"First action chunk (7 dims): {actions[:7]}")

print("\n=== All tests passed! ===")
print("Note: For LIBERO evaluation, use run2() with full_image and wrist_image.")
