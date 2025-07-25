# BAIS1C_PoseToVideoRenderer.py
# DWPose COCO-18 Colorized Stickman Renderer (n_frames, 23, 2) - but only uses first 18 points

import torch
import numpy as np
import cv2

class BAIS1C_PoseToVideoRenderer:
    """
    Renders pose sequences as colorized stickman/dots/skeleton video.
    Input: (n_frames, 23, 2) but only renders first 18 COCO keypoints.
    Coordinates expected normalized [0,1].
    Color-coded body parts for better visual clarity.
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

    # COCO-18 skeleton connections with color groups
    # Format: (point1, point2, color_group)
    pose_connections = [
        # HEAD - Red (255, 100, 100)
        (0, 1, "head"), (0, 2, "head"),     # Nose to eyes
        (1, 3, "head"), (2, 4, "head"),     # Eyes to ears
        (0, 17, "head"),                    # Nose to neck (if available)
        
        # ARMS - Green (100, 255, 100)  
        (5, 7, "arms"), (7, 9, "arms"),     # Left arm
        (6, 8, "arms"), (8, 10, "arms"),    # Right arm
        (5, 17, "arms"), (6, 17, "arms"),   # Shoulders to neck
        
        # TORSO - Blue (100, 150, 255)
        (5, 6, "torso"),                    # Shoulders
        (5, 11, "torso"), (6, 12, "torso"), # Shoulders to hips
        (11, 12, "torso"),                  # Hips
        
        # LEGS - Yellow (255, 255, 100)
        (11, 13, "legs"), (13, 15, "legs"), # Left leg
        (12, 14, "legs"), (14, 16, "legs"), # Right leg
    ]

    # Color definitions (BGR format for OpenCV)
    colors = {
        "head": (100, 100, 255),   # Red
        "arms": (100, 255, 100),   # Green  
        "torso": (255, 150, 100),  # Blue
        "legs": (100, 255, 255),   # Yellow
        "joints": (200, 200, 200), # Light gray for joint dots
    }

    def render_video(self, pose_tensor, audio, width=460, height=832,
                     visualization="stickman", background="black", debug=False):

        # --- Shape Check ---
        if isinstance(pose_tensor, torch.Tensor):
            poses = pose_tensor.cpu().numpy()
        else:
            poses = np.array(pose_tensor)
        
        if debug:
            print(f"[Renderer] Input pose shape: {poses.shape}")
            print(f"[Renderer] First frame non-zero points: {np.count_nonzero(np.any(poses[0], axis=1))}")

        # Accept (n_frames, 23, 2) but only use first 18 points
        assert len(poses.shape) == 3 and poses.shape[2] == 2, (
            f"Input pose_tensor must be (n_frames, num_points, 2), got {poses.shape}"
        )
        
        n_frames = poses.shape[0]
        num_points = min(poses.shape[1], 18)  # Only use first 18 points
        
        # Slice to only use COCO-18 keypoints
        poses = poses[:, :num_points, :]
        
        bg_colors = {"black": (0, 0, 0), "white": (255, 255, 255), "blue": (20, 20, 40)}
        bg_color = bg_colors.get(background, (0, 0, 0))

        frames = []
        for frame_idx, pose in enumerate(poses):
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            
            # Clip coordinates to [0,1]
            pose = np.clip(pose, 0.0, 1.0)
            
            # Convert to pixel coordinates
            keypoints = []
            valid_points = []
            for i, (x, y) in enumerate(pose):
                px = int(x * width)
                py = int(y * height)
                keypoints.append((px, py))
                # Check if point is valid (not at origin from zero-padding)
                valid_points.append(x > 0.001 or y > 0.001)  # Small threshold for floating point

            if debug and frame_idx < 3:
                print(f"[Renderer] Frame {frame_idx}: {sum(valid_points)}/{len(valid_points)} valid points")

            # Draw skeleton/stickman/dots
            if visualization in ("stickman", "skeleton"):
                # Draw colorized bones - only between valid points
                for i, j, color_group in self.pose_connections:
                    if (i < len(keypoints) and j < len(keypoints) and 
                        valid_points[i] and valid_points[j]):
                        color = self.colors[color_group]
                        cv2.line(frame, keypoints[i], keypoints[j], color, 3)  # Slightly thicker lines
                
                # Draw joints - only valid ones, color-coded by body part
                for i, point in enumerate(keypoints):
                    if valid_points[i]:
                        # Determine joint color based on body part
                        if i <= 4 or i == 17:  # Head area
                            joint_color = self.colors["head"]
                        elif 5 <= i <= 10:      # Arms/shoulders
                            joint_color = self.colors["arms"] 
                        elif 11 <= i <= 12:     # Torso/hips
                            joint_color = self.colors["torso"]
                        else:                   # Legs
                            joint_color = self.colors["legs"]
                        
                        cv2.circle(frame, point, 5, joint_color, -1)
                        # Add small white border for visibility
                        cv2.circle(frame, point, 5, (255, 255, 255), 1)
                        
            elif visualization == "dots":
                # Draw colorized dots - only valid ones
                for i, point in enumerate(keypoints):
                    if valid_points[i]:
                        # Color-code the dots by body part
                        if i <= 4 or i == 17:  # Head area
                            dot_color = self.colors["head"]
                        elif 5 <= i <= 10:      # Arms/shoulders
                            dot_color = self.colors["arms"] 
                        elif 11 <= i <= 12:     # Torso/hips
                            dot_color = self.colors["torso"]
                        else:                   # Legs
                            dot_color = self.colors["legs"]
                        
                        cv2.circle(frame, point, 8, dot_color, -1)
                        # White border for visibility
                        cv2.circle(frame, point, 8, (255, 255, 255), 1)

            frames.append(torch.from_numpy(frame.astype(np.float32) / 255.0))

        if debug:
            print(f"[Renderer] Rendered {len(frames)} frames ({width}x{height}) with color-coded skeleton")

        info = (
            f"Colorized stickman video: {n_frames} frames ({width}x{height}), viz={visualization}, "
            f"bg={background}, skeleton=COCO-18 (using {num_points} points), "
            f"colors=Red(head)/Green(arms)/Blue(torso)/Yellow(legs)"
        )
        return torch.stack(frames), info, audio

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": BAIS1C_PoseToVideoRenderer}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseToVideoRenderer": "BAIS1C Pose To Video Renderer (Colorized)"}

# Self-test
def test_pose_to_video_renderer():
    n_frames = 10
    n_points = 23
    pose_tensor = torch.rand((n_frames, n_points, 2), dtype=torch.float32)
    # Zero out the last 5 points to simulate real DWPose output
    pose_tensor[:, 18:, :] = 0.0
    dummy_audio = {"waveform": np.zeros(44100), "sample_rate": 44100}
    node = BAIS1C_PoseToVideoRenderer()
    images, info, audio = node.render_video(pose_tensor, dummy_audio, debug=True)
    print(info)
    assert images.shape[0] == n_frames
    print("[TEST PASSED] Colorized PoseToVideoRenderer.")

# Uncomment to self-test
# test_pose_to_video_renderer()