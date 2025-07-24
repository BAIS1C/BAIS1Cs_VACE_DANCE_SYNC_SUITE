import os
import json
import numpy as np
import torch
import decord

class BAIS1C_PoseExtractor:
    """
    BAIS1C VACE Dance Sync Suite - Pose Extractor (Streamlined)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_obj": ("VIDEO",),
                "meta": ("DICT",),         # Accepts the meta dict output by loader node
                "title": ("STRING", {"default": "untitled_pose"}),
                "save_to_library": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("TENSOR", "DICT")
    RETURN_NAMES = (
        "pose_tensor",
        "meta",
    )
    FUNCTION = "process"
    CATEGORY = "BAIS1C VACE Suite/Pose"

    def process(self, video_obj, meta, title, save_to_library, debug):
        # Extract info from meta dict (all fields provided by loader node)
        video_path = meta.get("video_path", None) or (video_obj.video_path if hasattr(video_obj, "video_path") else video_obj)
        video_fps = float(meta.get("fps", 24))
        sample_stride = int(meta.get("sample_stride", 1))

        # Video properties
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)

        # Adjust stride based on FPS for consistent temporal sampling
        adjusted_stride = max(1, sample_stride * max(1, int(video_fps / 24)))
        frame_indices = list(range(0, total_frames, adjusted_stride))

        # Dummy pose extraction (for demo/testing)
        pose_tensor = []
        for idx in frame_indices:
            pose = np.zeros((128, 3), dtype=np.float32)
            pose_tensor.append(pose)

        pose_tensor = np.stack(pose_tensor)
        torch_pose_tensor = torch.from_numpy(pose_tensor)

        # Update meta for outputs
        meta_out = dict(meta)
        meta_out.update({
            "title": title,
            "adjusted_stride": adjusted_stride,
            "processed_frames": len(frame_indices),
        })

        # Save to library if toggled
        if save_to_library:
            library_dir = self._find_dance_library_directory()
            os.makedirs(library_dir, exist_ok=True)
            safe_title = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in title]).strip()
            filename = os.path.join(library_dir, f"{safe_title or 'untitled_pose'}.json")
            export = {
                "title": meta_out["title"],
                "meta": meta_out,
                "tensor": pose_tensor.tolist(),
            }
            with open(filename, "w") as f:
                json.dump(export, f, indent=2)
            if debug:
                print(f"[BAIS1C] Saved pose to {filename}")

        return (torch_pose_tensor, meta_out)

    def _find_dance_library_directory(self):
        suite_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        library_dir = os.path.join(suite_root, "dance_library")
        return library_dir

NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt)"}
