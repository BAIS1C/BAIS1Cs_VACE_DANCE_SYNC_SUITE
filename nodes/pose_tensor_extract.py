import numpy as np
import torch
import json
import sys, os

# Add the suite root to path for dwpose import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dwpose.dwpose_detector import create_dwpose_detector


class BAIS1C_PoseExtractor:
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
    CATEGORY = "BAS1C/POSE"

    def __init__(self):
        self.detector = create_dwpose_detector()

    def extract_pose(self, video_frames, sync_meta, temporal_smoothing=True, use_dummy=False):
        results = []
        print(f"[PoseExtractor] Running on {len(video_frames)} frames — Dummy Mode: {use_dummy}")

        for i, frame in enumerate(video_frames):
            frame_np = (frame * 255).astype(np.uint8)

            if use_dummy:
                dummy = np.zeros((1, 18, 2), dtype=np.float32)
                dummy[0, :, 0] = np.linspace(0.4, 0.6, 18)
                dummy[0, :, 1] = np.linspace(0.3, 0.7, 18)
                results.append(dummy)
                print(f"[PoseExtractor] Frame {i}: 🟢 Dummy pose injected")
                continue

            try:
                result = self.detector(frame_np)
                bodies = result["bodies"]["candidate"]
                if bodies.shape[0] > 0:
                    results.append(bodies)
                    print(f"[PoseExtractor] Frame {i}: ✅ Pose detected ({bodies.shape})")
                else:
                    results.append(None)
                    print(f"[PoseExtractor] Frame {i}: ❌ No pose detected")
            except Exception as e:
                print(f"[PoseExtractor] Frame {i}: ❌ Error during pose detection: {e}")
                results.append(None)

        # Count successful extractions
        num_success = sum(1 for r in results if r is not None)
        print(f"[PoseExtractor] Summary: {num_success}/{len(results)} frames with valid pose")

        # Fill in blanks
        for i in range(len(results)):
            if results[i] is None:
                results[i] = np.zeros((18, 2), dtype=np.float32)

        pose_tensor = np.stack(results).astype(np.float32)

        # Optional smoothing
        if temporal_smoothing:
            print("[PoseExtractor] Applying temporal smoothing...")
            smoothed = []
            for i in range(len(pose_tensor)):
                frames_to_avg = []
                for offset in [-1, 0, 1]:
                    j = i + offset
                    if 0 <= j < len(pose_tensor):
                        frames_to_avg.append(pose_tensor[j])
                smoothed.append(np.mean(frames_to_avg, axis=0))
            pose_tensor = np.stack(smoothed)

        # Meta diagnostics
        sync_meta["pose_extraction_success"] = num_success > 0
        sync_meta["successful_extractions"] = num_success
        sync_meta["extraction_rate"] = round(num_success / len(results), 4)

        return (pose_tensor, sync_meta)


# ✅ Node registration (was previously missing after rewrite)
NODE_CLASS_MAPPINGS = {
    "BAIS1C_PoseExtractor": BAIS1C_PoseExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_PoseExtractor": "🟢 BAIS1C Pose Extractor",
}
