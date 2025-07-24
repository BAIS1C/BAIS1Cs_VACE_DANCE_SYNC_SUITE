import os
import json
import torch
import numpy as np
from pathlib import Path

def make_json_safe(obj):
    """
    Recursively convert numpy arrays and torch tensors to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.cpu().tolist()
    else:
        return obj

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

        # 2. Package payload, making everything JSON-safe
        payload = {
            "title": title,
            "meta": make_json_safe(meta),  # <--- Fix for JSON serializability
            "pose_tensor": pose_tensor.cpu().tolist(),  # Always a list-of-lists
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

# ---------------------------
# Self-test (run as __main__)
# ---------------------------
def _test_checkpoint():
    import torch
    import numpy as np

    node = BAIS1C_PoseCheckpoint()
    pose_tensor = torch.zeros((10, 128, 3))
    meta = {
        "foo": np.array([1, 2, 3]),
        "bar": torch.tensor([4, 5, 6]),
        "nested": {"baz": np.ones((2, 2))},
        "title": "TEST JSON_SAFE"
    }
    node.checkpoint(pose_tensor, meta)
    print("Checkpoint JSON save test passed. File written to 'dance_library/TEST_JSON_SAFE.json'.")

if __name__ == "__main__":
    _test_checkpoint()
