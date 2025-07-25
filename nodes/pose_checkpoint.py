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
        return obj.detach().cpu().tolist()  # ✅ FIXED: Added .detach()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Handle numpy scalars
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
                "pose_tensor": ("POSE",),      # ✅ FIXED: Changed from ("TENSOR",) to ("POSE",)
                "meta": ("DICT",),             # full meta dict
            }
        }

    RETURN_TYPES = ("POSE", "DICT")  # ✅ FIXED: Changed from ("TENSOR", "DICT") to ("POSE", "DICT")
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
            "pose_tensor": make_json_safe(pose_tensor),  # ✅ FIXED: Use make_json_safe for consistency
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
    pose_tensor = torch.zeros((10, 128, 3), requires_grad=True)  # Test with requires_grad=True
    meta = {
        "foo": np.array([1, 2, 3]),
        "bar": torch.tensor([4, 5, 6], requires_grad=True),  # Test with requires_grad tensor
        "nested": {"baz": np.ones((2, 2))},
        "title": "TEST JSON_SAFE",
        "fps": 24.0,
        "duration": 5.0
    }
    
    try:
        result_pose, result_meta = node.checkpoint(pose_tensor, meta)
        
        # Verify output types
        assert torch.is_tensor(result_pose), "Output pose should be tensor"
        assert isinstance(result_meta, dict), "Output meta should be dict"
        assert torch.equal(result_pose, pose_tensor), "Pose tensor should pass through unchanged"
        
        print("✅ Checkpoint JSON save test passed. File written to 'dance_library/TEST_JSON_SAFE.json'.")
        print(f"✅ Pose tensor shape: {result_pose.shape}")
        print(f"✅ Meta keys: {list(result_meta.keys())}")
        
        # Verify file was created
        lib_dir = Path(__file__).resolve().parent.parent / "dance_library"
        test_file = lib_dir / "TEST_JSON_SAFE.json"
        if test_file.exists():
            print(f"✅ File successfully created: {test_file}")
            
            # Verify JSON is valid
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            print(f"✅ JSON valid, contains {len(loaded_data['pose_tensor'])} frames")
        else:
            print(f"❌ File was not created at: {test_file}")
            
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    _test_checkpoint()