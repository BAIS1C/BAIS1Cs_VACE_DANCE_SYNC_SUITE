# BAIS1C_PoseToVideoRenderer.py
import torch
import numpy as np
import cv2

class BAIS1C_PoseToVideoRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_tensor": ("POSE",),
                "audio": ("AUDIO",),  # Optional in pipeline, but ComfyUI requires declaration
            },
            "optional": {
                "width": ("INT", {"default": 460, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 832, "min": 64, "max": 2048}),
                "visualization": (["stickman", "dots", "skeleton"], {"default": "stickman"}),
                "background": (["black", "white", "blue"], {"default": "black"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("stickman_video", "render_info", "audio")
    FUNCTION = "render_video"
    CATEGORY = "BAIS1C VACE Suite/Visualization"

    def __init__(self):
        # Define main skeleton for 23-point body, adjust if your keypoints differ
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),   # Head/neck
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
            (11, 13), (13, 15), (12, 14), (14, 16), # Legs
            (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22) # Feet
        ]

    def render_video(self, pose_tensor, audio, width=460, height=832, visualization="stickman", background="black", debug=False):
        # Convert pose tensor to numpy
        if isinstance(pose_tensor, torch.Tensor):
            poses = pose_tensor.cpu().numpy()
        else:
            poses = np.array(pose_tensor)
        n_frames = poses.shape[0]

        # Background color
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "blue": (20, 20, 40)}
        bg_color = bg_colors.get(background, (0, 0, 0))

        frames = []
        for frame_idx, pose in enumerate(poses):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            keypoints = []
            for x, y in pose[:23]:  # Render only the main body
                px = int(np.clip(x * width, 0, width - 1))
                py = int(np.clip(y * height, 0, height - 1))
                keypoints.append((px, py))
            # Draw skeleton/stickman/dots
            if visualization in ("stickman", "skeleton"):
                for i, j in self.pose_connections:
                    if i < len(keypoints) and j < len(keypoints):
                        cv2.line(frame, keypoints[i], keypoints[j], (255, 255, 255), 2)
                for point in keypoints:
                    cv2.circle(frame, point, 4, (100, 200, 255), -1)
            elif visualization == "dots":
                for point in keypoints:
                    cv2.circle(frame, point, 6, (255, 200, 100), -1)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))
        if debug:
            print(f"[PoseToVideoRenderer] Rendered {len(frames)} stickman frames ({width}x{height})")

        info = f"Stickman video: {n_frames} frames ({width}x{height}), viz={visualization}, bg={background}"
        return torch.stack(frames), info, audio  # Audio passed through unchanged

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": BAIS1C_PoseToVideoRenderer}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": "🦴 BAIS1C Pose To Video Renderer"}

# Self-test (can be run in any script environment for smoke test)
def test_pose_to_video_renderer():
    n_frames = 10
    n_points = 23
    pose_tensor = torch.rand((n_frames, n_points, 2), dtype=torch.float32)
    dummy_audio = {"waveform": np.zeros(44100), "sample_rate": 44100}
    node = BAIS1C_PoseToVideoRenderer()
    images, info, audio = node.render_video(pose_tensor, dummy_audio)
    print(info)
    assert images.shape[0] == n_frames
    print("✅ PoseToVideoRenderer test passed.")

# Uncomment below to run self-test
# test_pose_to_video_renderer()
