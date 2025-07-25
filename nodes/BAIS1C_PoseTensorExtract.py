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
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dwpose.dwpose_detector import create_dwpose_detector

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
        self.detector = create_dwpose_detector()

    def _ensure_23_points(self, pose):
        """Ensure pose is (23,2), fill missing with zeros."""
        pose = np.array(pose)
        out = np.zeros((23, 2), dtype=np.float32)
        n = min(len(pose), 23)
        out[:n] = pose[:n]
        return out

    def extract_pose(self, video_frames, sync_meta, temporal_smoothing=True, use_dummy=False):
        results = []
        print(f"[PoseTensorExtract] Running on {len(video_frames)} frames — Dummy Mode: {use_dummy}")

        for i, frame in enumerate(video_frames):
            # FIX: Robust handling for torch tensor input
            if isinstance(frame, torch.Tensor):
                frame_np = (frame.detach().cpu().numpy() * 255).astype(np.uint8)
            else:
                frame_np = (np.array(frame) * 255).astype(np.uint8)

            if use_dummy:
                dummy = np.zeros((23, 2), dtype=np.float32)
                dummy[:, 0] = np.linspace(0.4, 0.6, 23)
                dummy[:, 1] = np.linspace(0.3, 0.7, 23)
                results.append(dummy)
                print(f"[PoseTensorExtract] Frame {i}: 🟢 Dummy pose injected")
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
    assert pose_tensor.shape[1:] == (23, 2)
    print("[TEST PASSED]")

# Uncomment to self-test
# _test_pose_tensor_extract()
