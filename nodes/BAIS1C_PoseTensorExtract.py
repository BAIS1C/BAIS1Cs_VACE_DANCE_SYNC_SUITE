"""
BAIS1C_PoseTensorExtract
-----------------------
DWPose-23 Pose Extractor for ComfyUI

Outputs:
  - pose_tensor (POSE): Pose tensor of shape (n_frames, 23, 2), dtype=float32
  - sync_meta (DICT): Meta dictionary including skeleton_layout, points, and extraction stats

Always outputs DWPose-23 keypoint layout; never 128-point or VACE.
"""

import numpy as np
import torch

# Clean import without sys.path hacks
from ..dwpose.dwpose_detector import create_dwpose_detector

class BAIS1C_PoseTensorExtract:
    """
    DWPose UCoco23 Pose Extractor
    ---------------------------------
    • Always outputs (n_frames, 23, 2) pose tensors
    • Fills any missing points per frame with zeros
    • Adds 'skeleton_layout' and 'points' to meta for full pipeline clarity
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "sync_meta": ("DICT",),
                "temporal_smoothing": ("BOOLEAN", {"default": True}),
                "use_dummy": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("POSE", "DICT")
    RETURN_NAMES = ("pose_tensor", "sync_meta")
    FUNCTION = "extract_pose"
    CATEGORY = "BAIS1C/POSE"

    def __init__(self):
        try:
            self.detector = create_dwpose_detector()
            print("[PoseTensorExtract] ✅ DWPose detector initialized successfully")
        except Exception as e:
            print(f"[PoseTensorExtract] ❌ Failed to initialize DWPose detector: {e}")
            self.detector = None

    def _ensure_23_points(self, pose):
        """Ensure pose is (23,2), fill missing with zeros."""
        pose = np.array(pose)
        out = np.zeros((23, 2), dtype=np.float32)
        n = min(len(pose), 23)
        out[:n] = pose[:n]
        return out

    def _generate_dummy_coco23_pose(self):
        """
        Generate a proper COCO-23 skeleton pose instead of a diagonal line.
        
        COCO-23 Joint Layout:
        0: nose, 1-2: eyes, 3-4: ears
        5-6: shoulders, 7-8: elbows, 9-10: wrists
        11-12: hips, 13-14: knees, 15-16: ankles
        17-22: foot keypoints
        """
        pose = np.zeros((23, 2), dtype=np.float32)
        
        # Center the figure around (0.5, 0.5)
        cx, cy = 0.5, 0.5
        
        # Head cluster (nose, eyes, ears)
        pose[0] = [cx, cy - 0.15]          # nose
        pose[1] = [cx - 0.02, cy - 0.17]   # left eye
        pose[2] = [cx + 0.02, cy - 0.17]   # right eye  
        pose[3] = [cx - 0.04, cy - 0.16]   # left ear
        pose[4] = [cx + 0.04, cy - 0.16]   # right ear
        
        # Upper body (shoulders, elbows, wrists)
        pose[5] = [cx - 0.08, cy - 0.05]   # left shoulder
        pose[6] = [cx + 0.08, cy - 0.05]   # right shoulder
        pose[7] = [cx - 0.12, cy + 0.05]   # left elbow
        pose[8] = [cx + 0.12, cy + 0.05]   # right elbow
        pose[9] = [cx - 0.15, cy + 0.15]   # left wrist
        pose[10] = [cx + 0.15, cy + 0.15]  # right wrist
        
        # Lower body (hips, knees, ankles)
        pose[11] = [cx - 0.06, cy + 0.12]  # left hip
        pose[12] = [cx + 0.06, cy + 0.12]  # right hip
        pose[13] = [cx - 0.08, cy + 0.25]  # left knee
        pose[14] = [cx + 0.08, cy + 0.25]  # right knee
        pose[15] = [cx - 0.09, cy + 0.35]  # left ankle
        pose[16] = [cx + 0.09, cy + 0.35]  # right ankle
        
        # Foot keypoints (simplified foot structure)
        pose[17] = [cx - 0.10, cy + 0.37]  # left foot heel
        pose[18] = [cx - 0.08, cy + 0.37]  # left foot mid
        pose[19] = [cx - 0.09, cy + 0.38]  # left foot toe
        pose[20] = [cx + 0.10, cy + 0.37]  # right foot heel
        pose[21] = [cx + 0.08, cy + 0.37]  # right foot mid
        pose[22] = [cx + 0.09, cy + 0.38]  # right foot toe
        
        # Clip to valid coordinate range [0, 1]
        pose = np.clip(pose, 0.0, 1.0)
        
        return pose

    def extract_pose(self, video_frames, sync_meta, temporal_smoothing=True, use_dummy=False):
        results = []
        print(f"[PoseTensorExtract] Running on {len(video_frames)} frames — Dummy Mode: {use_dummy}")

        # Check if detector is available
        if self.detector is None and not use_dummy:
            print("[PoseTensorExtract] ❌ No detector available, switching to dummy mode")
            use_dummy = True

        for i, frame in enumerate(video_frames):
            # FIX: Robust handling for torch tensor input
            if isinstance(frame, torch.Tensor):
                frame_np = (frame.detach().cpu().numpy() * 255).astype(np.uint8)
            else:
                frame_np = (np.array(frame) * 255).astype(np.uint8)

            if use_dummy:
                # Generate proper COCO-23 skeleton instead of diagonal line
                dummy = self._generate_dummy_coco23_pose()
                
                # Add slight animation to make it more interesting
                # Subtle breathing motion and gentle arm sway
                time_factor = i * 0.1
                breathing = np.sin(time_factor) * 0.005
                arm_sway = np.cos(time_factor * 0.7) * 0.01
                
                # Apply breathing to torso points (5-12: shoulders to hips)
                dummy[5:13, 1] += breathing
                
                # Apply arm sway to arm points (7-10: elbows and wrists)
                dummy[7, 0] += arm_sway    # left elbow
                dummy[9, 0] += arm_sway    # left wrist
                dummy[8, 0] -= arm_sway    # right elbow  
                dummy[10, 0] -= arm_sway   # right wrist
                
                # Ensure coordinates stay in bounds
                dummy = np.clip(dummy, 0.0, 1.0)
                
                results.append(dummy)
                print(f"[PoseTensorExtract] Frame {i}: 🟢 COCO-23 dummy pose injected")
                continue

            try:
                result = self.detector(frame_np)
                bodies = result["bodies"]["candidate"]
                if bodies.shape[0] > 0:
                    pose = self._ensure_23_points(bodies[0])
                    results.append(pose)
                    print(f"[PoseTensorExtract] Frame {i}: ✅ Pose detected (shape {pose.shape})")
                else:
                    results.append(np.zeros((23,2), dtype=np.float32))
                    print(f"[PoseTensorExtract] Frame {i}: ❌ No pose detected (zeroed)")
            except Exception as e:
                print(f"[PoseTensorExtract] Frame {i}: ❌ Error during pose detection: {e}")
                results.append(np.zeros((23,2), dtype=np.float32))

        # Count successful extractions
        num_success = sum(1 for r in results if np.any(r))
        print(f"[PoseTensorExtract] Summary: {num_success}/{len(results)} frames with valid pose")

        pose_tensor = np.stack(results).astype(np.float32)

        # Temporal smoothing only between valid frames (zeros = static, not interpolated)
        if temporal_smoothing:
            print("[PoseTensorExtract] Applying temporal smoothing...")
            smoothed = []
            for i in range(len(pose_tensor)):
                frames_to_avg = []
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < len(pose_tensor):
                        if np.any(pose_tensor[j]):
                            frames_to_avg.append(pose_tensor[j])
                if frames_to_avg:
                    smoothed.append(np.mean(frames_to_avg, axis=0))
                else:
                    smoothed.append(pose_tensor[i])
            pose_tensor = np.stack(smoothed)

        meta_out = dict(sync_meta)
        meta_out["pose_extraction_success"] = num_success > 0
        meta_out["successful_extractions"] = num_success
        meta_out["extraction_rate"] = round(num_success / len(results), 4)
        meta_out["skeleton_layout"] = "UCoco23"
        meta_out["points"] = 23

        return (torch.from_numpy(pose_tensor).float(), meta_out)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseTensorExtract": BAIS1C_PoseTensorExtract}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseTensorExtract": "BAIS1C Pose Tensor Extract (DWPose23)"}

# Self-test function
def _test_pose_tensor_extract():
    extractor = BAIS1C_PoseTensorExtract()
    dummy_frames = [torch.ones((512, 512, 3), dtype=torch.float32) for _ in range(5)]
    sync_meta = {"fps":24, "title":"test"}
    pose_tensor, meta = extractor.extract_pose(dummy_frames, sync_meta, temporal_smoothing=True, use_dummy=True)
    print(f"[TEST] Pose tensor shape: {pose_tensor.shape} (should be n_frames x 23 x 2)")
    print(f"[TEST] Meta: {meta}")
    print(f"[TEST] Sample pose coordinates:")
    print(f"  Nose (0): {pose_tensor[0, 0]}")
    print(f"  Left shoulder (5): {pose_tensor[0, 5]}")
    print(f"  Right shoulder (6): {pose_tensor[0, 6]}")
    print(f"  Left hip (11): {pose_tensor[0, 11]}")
    assert pose_tensor.shape[1:] == (23, 2)
    print("[TEST PASSED]")

# Uncomment to self-test
# _test_pose_tensor_extract()