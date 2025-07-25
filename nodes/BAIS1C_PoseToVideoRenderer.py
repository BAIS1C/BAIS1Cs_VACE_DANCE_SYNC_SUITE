# BAIS1C_PoseToVideoRenderer.py
# DWPose UCoco23-Only Stickman Renderer (n_frames, 23, 2)

import torch
import numpy as np
import cv2

class BAIS1C_PoseToVideoRenderer:
    """
    Renders pose sequences (n_frames, 23, 2) as stickman/dots/skeleton video.
    Strictly for DWPose UCoco23 outputs.
    Coordinates expected normalized [0,1]; only first 23 points rendered.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_tensor": ("POSE",),
                "audio": ("AUDIO",),  # Audio is passed through
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
        # Skeleton for UCoco23 (see DWPose docs)
        self.pose_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),   # Head/neck
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
            (11, 13), (13, 15), (12, 14), (14, 16), # Legs
            (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22) # Feet
        ]

    def render_video(self, pose_tensor, audio, width=460, height=832,
                     visualization="stickman", background="black", debug=False):

        # --- Shape Check ---
        if isinstance(pose_tensor, torch.Tensor):
            poses = pose_tensor.cpu().numpy()
        else:
            poses = np.array(pose_tensor)
        assert poses.shape[1:] == (23, 2), (
            f"Input pose_tensor must be (n_frames, 23, 2), got {poses.shape}"
        )

        n_frames = poses.shape[0]
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "blue": (20, 20, 40)}
        bg_color = bg_colors.get(background, (0, 0, 0))

        frames = []
        for frame_idx, pose in enumerate(poses):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            # Clip coordinates to [0,1]
            pose = np.clip(pose, 0.0, 1.0)
            keypoints = []
            for x, y in pose:  # 23 points
                px = int(x * width)
                py = int(y * height)
                keypoints.append((px, py))

            # Draw skeleton/stickman/dots
            if visualization in ("stickman", "skeleton"):
                for i, j in self.pose_connections:
                    if i < 23 and j < 23:
                        cv2.line(frame, keypoints[i], keypoints[j], (255, 255, 255), 2)
                for point in keypoints:
                    cv2.circle(frame, point, 4, (100, 200, 255), -1)
            elif visualization == "dots":
                for point in keypoints:
                    cv2.circle(frame, point, 6, (255, 200, 100), -1)
            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))

        if debug:
            print(f"[PoseToVideoRenderer] Rendered {len(frames)} frames ({width}x{height})")

        info = (
            f"Stickman video: {n_frames} frames ({width}x{height}), viz={visualization}, "
            f"bg={background}, skeleton=UCoco23"
        )
        return torch.stack(frames), info, audio

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": BAIS1C_PoseToVideoRenderer}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": "🦴 BAIS1C Pose To Video Renderer (DWPose23)"}

# Self-test
def test_pose_to_video_renderer():
    n_frames = 10
    n_points = 23
    pose_tensor = torch.rand((n_frames, n_points, 2), dtype=torch.float32)
    dummy_audio = {"waveform": np.zeros(44100), "sample_rate": 44100}
    node = BAIS1C_PoseToVideoRenderer()
    images, info, audio = node.render_video(pose_tensor, dummy_audio)
    print(info)
    assert images.shape[0] == n_frames
    assert images.shape[1:] == (832, 460, 3) or images.shape[1:] == (460, 832, 3)
    print("[TEST PASSED] PoseToVideoRenderer.")

# Uncomment to self-test
# test_pose_to_video_renderer()
