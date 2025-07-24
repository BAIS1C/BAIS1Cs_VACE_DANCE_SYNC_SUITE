# File: pose_tensor_extract.py
# BAIS1C VACE Dance Sync Suite – Pose Extractor (streamlined)

import os
import numpy as np
import torch
import decord

class BAIS1C_PoseExtractor:
    """
    BAIS1C VACE Dance Sync Suite – Pose Extractor (128-point)
    Receives a VIDEO and the sync_meta dict from Source Video Loader,
    extracts a 128-point pose tensor, and immediately passes both
    tensor and sync_meta downstream.  No persistence logic here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_obj": ("VIDEO",),
                "sync_meta": ("DICT",),      # sync_meta from Source Video Loader
                "title":    ("STRING", {"default": "untitled_pose"}),
                "debug":    ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("TENSOR", "DICT")
    RETURN_NAMES  = ("pose_tensor", "sync_meta")
    FUNCTION      = "extract"
    CATEGORY      = "BAIS1C VACE Suite/Pose"
    OUTPUT_NODE   = False

    def extract(self, video_obj, sync_meta, title, debug):
        # ------------------------------------------------------------------
        # 1. Resolve video path
        # ------------------------------------------------------------------
        video_path = (
            sync_meta.get("video_path")
            or (video_obj.video_path if hasattr(video_obj, "video_path") else str(video_obj))
        )
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ------------------------------------------------------------------
        # 2. Video properties
        # ------------------------------------------------------------------
        vr           = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps    = float(sync_meta.get("fps", vr.get_avg_fps()))

        # Consistent temporal sampling (optional stride tweak)
        sample_stride = int(sync_meta.get("sample_stride", 1))
        adjusted_stride = max(1, sample_stride * max(1, int(video_fps / 24)))
        frame_indices = list(range(0, total_frames, adjusted_stride))

        if debug:
            print(f"[BAIS1C PoseExtractor] {len(frame_indices)} frames selected "
                  f"(stride {adjusted_stride}) from {total_frames} total.")

        # ------------------------------------------------------------------
        # 3. Dummy 128-point pose extraction (replace with real model)
        # ------------------------------------------------------------------
        pose_list = []
        for idx in frame_indices:
            pose = np.zeros((128, 3), dtype=np.float32)  # (x, y, confidence)
            pose_list.append(pose)

        pose_tensor = torch.from_numpy(np.stack(pose_list))

        # ------------------------------------------------------------------
        # 4. Augment sync_meta for downstream nodes
        # ------------------------------------------------------------------
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "title":            title,
            "adjusted_stride":  adjusted_stride,
            "processed_frames": len(frame_indices),
        })

        return (pose_tensor, sync_meta_out)


# --------------------------------------------------------------------------
# Node registration
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS        = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt)"}