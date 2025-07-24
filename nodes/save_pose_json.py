# save_pose_json.py
import os, json, torch, numpy as np
from pathlib import Path

class BAIS1C_SavePoseJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_tensor": ("TENSOR",),
                "sync_meta": ("DICT",),
                "filename":  ("STRING", {"default": "my_dance"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "BAIS1C VACE Suite/Utils"
    OUTPUT_NODE = True

    def save(self, pose_tensor, sync_meta, filename):
        lib = Path(__file__).resolve().parent.parent / "dance_library"
        lib.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
        out_file = lib / f"{safe}.json"

        if isinstance(pose_tensor, torch.Tensor):
            pose_np = pose_tensor.cpu().numpy()
        else:
            pose_np = np.array(pose_tensor)

        payload = {
            "title": filename,
            "metadata": {
                "bpm": float(sync_meta.get("bpm", 120)),
                "fps": float(sync_meta.get("fps", 24)),
                "duration": len(pose_np) / float(sync_meta.get("fps", 24)),
                "frame_count": len(pose_np),
                "origin": "extract_only",
                "loop_friendly": True,
            },
            "format_info": {
                "format": "128-point",
                "shape": list(pose_np.shape),
                "coordinate_system": "normalized"
            },
            "pose_tensor": pose_np.tolist()
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[BAIS1C] Saved library pose → {out_file}")
        return ()

NODE_CLASS_MAPPINGS = {"BAIS1C_SavePoseJSON": BAIS1C_SavePoseJSON}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_SavePoseJSON": "💾 Save Pose JSON"}