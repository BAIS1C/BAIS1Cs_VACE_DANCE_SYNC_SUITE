import os
import json
import torch
from pathlib import Path

class BAIS1C_PoseCheckpoint:
    """
    BAIS1C VACE Dance Sync Suite – Pose Checkpoint
    ------------------------------------------------
    • Takes the pose tensor + meta straight from PoseExtractor
    • Persists everything to dance_library/<title>.json
    • Immediately forwards the tensor so the next node can reuse it
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_tensor": ("TENSOR",),      # 128-point tensor
                "meta": ("DICT",),               # full meta dict
            }
        }

    RETURN_TYPES = ("TENSOR", "DICT")
    RETURN_NAMES = ("pose_tensor", "meta")
    FUNCTION = "checkpoint"
    CATEGORY = "BAIS1C VACE Suite/Pose"
    OUTPUT_NODE = False

    def checkpoint(self, pose_tensor: torch.Tensor, meta: dict):
        # 1. Build safe filename
        title = meta.get("title", "untitled_pose")
        safe = "".join(c if c.isalnum() or c in {"-", "_", " "} else "_" for c in str(title))
        lib_dir = Path(__file__).resolve().parent.parent / "dance_library"
        lib_dir.mkdir(parents=True, exist_ok=True)
        file_path = lib_dir / f"{safe}.json"

        # 2. Package payload
        payload = {
            "title": title,
            "meta": meta,
            "pose_tensor": pose_tensor.cpu().tolist(),  # list-of-lists for JSON
            "format": {
                "points": 128,
                "coords": pose_tensor.shape[-1]  # usually 2 or 3
            }
        }

        # 3. Write once, immediately
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # 4. Forward unchanged
        return (pose_tensor, meta)


# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseCheckpoint": BAIS1C_PoseCheckpoint}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseCheckpoint": "📦 BAIS1C Pose Checkpoint"}