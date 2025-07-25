import numpy as np
import torch
import sys, os

# Ensure the dwpose module can be found (local import for ComfyUI custom node setup)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dwpose.dwpose_detector import create_dwpose_detector

class BAIS1C_PoseExtractor:
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
        print(f"[PoseExtractor] Running on {len(video_frames)} frames — Dummy Mode: {use_dummy}")

        for i, frame in enumerate(video_frames):
            frame_np = (frame * 255).astype(np.uint8)

            if use_dummy:
                dummy = np.zeros((23, 2), dtype=np.float32)
                dummy[:, 0] = np.linspace(0.4, 0.6, 23)
                dummy[:, 1] = np.linspace(0.3, 0.7, 23)
                results.append(dummy)
                print(f"[PoseExtractor] Frame {i}: 🟢 Dummy pose injected")
                continue

            try:
                result = self.detector(frame_np)
                bodies = result["bodies"]["candidate"]
                if bodies.shape[0] > 0:
                    pose = self._ensure_23_points(bodies[0])
                    results.append(pose)
                    print(f"[PoseExtractor] Frame {i}: ✅ Pose detected ({bodies[0].shape})")
                else:
                    results.append(np.zeros((23,2), dtype=np.float32))
                    print(f"[PoseExtractor] Frame {i}: ❌ No pose detected (zeroed)")
            except Exception as e:
                print(f"[PoseExtractor] Frame {i}: ❌ Error during pose detection: {e}")
                results.append(np.zeros((23,2), dtype=np.float32))

        # Count successful extractions
        num_success = sum(1 for r in results if np.any(r))
        print(f"[PoseExtractor] Summary: {num_success}/{len(results)} frames with valid pose")

        pose_tensor = np.stack(results).astype(np.float32)

        # Temporal smoothing only between valid frames (zeros = static, not interpolated)
        if temporal_smoothing:
            print("[PoseExtractor] Applying temporal smoothing...")
            smoothed = []
            for i in range(len(pose_tensor)):
                frames_to_avg = []
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < len(pose_tensor):
                        # Only smooth if not all zeros
                        if np.any(pose_tensor[j]):
                            frames_to_avg.append(pose_tensor[j])
                if frames_to_avg:
                    smoothed.append(np.mean(frames_to_avg, axis=0))
                else:
                    smoothed.append(pose_tensor[i])
            pose_tensor = np.stack(smoothed)

        # Attach diagnostic metadata — always add skeleton info
        sync_meta = dict(sync_meta)  # Copy to avoid mutating input
        sync_meta["pose_extraction_success"] = num_success > 0
        sync_meta["successful_extractions"] = num_success
        sync_meta["extraction_rate"] = round(num_success / len(results), 4)
        sync_meta["skeleton_layout"] = "UCoco23"
        sync_meta["points"] = 23

        return (torch.from_numpy(pose_tensor).float(), sync_meta)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🦴 BAIS1C Pose Extractor (DWPose23)"}

# Self-test function
def _test_pose_extractor():
    # Test with dummy frames
    extractor = BAIS1C_PoseExtractor()
    dummy_frames = [np.ones((512, 512, 3), dtype=np.float32) for _ in range(5)]
    sync_meta = {"fps":24, "title":"test"}
    pose_tensor, meta = extractor.extract_pose(dummy_frames, sync_meta, temporal_smoothing=True, use_dummy=True)
    print(f"[TEST] Pose tensor shape: {pose_tensor.shape} (should be n_frames x 23 x 2)")
    print(f"[TEST] Meta: {meta}")
    assert pose_tensor.shape[1:] == (23, 2)
    print("[TEST PASSED]")

# Uncomment to self-test
# _test_pose_extractor()
