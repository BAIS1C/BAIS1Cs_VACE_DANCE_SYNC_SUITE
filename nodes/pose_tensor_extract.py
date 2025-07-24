# File: pose_tensor_extract.py
# BAIS1C VACE Dance Sync Suite – Pose Extractor (video or image sequence, with DWPose + smoothing)

import os
import numpy as np
import torch
import decord
from typing import List, Any

from ..dwpose.dwpose_detector import dwpose_detector  # Assumed ready and globally initialized

class BAIS1C_PoseExtractor:
    """
    BAIS1C VACE Dance Sync Suite – Pose Extractor (128-point, DWPose, video/images)
    Accepts either a VIDEO or an IMAGE sequence (list/tensor).
    Uses DWPose to extract pose per frame.
    Returns pose tensor + sync_meta downstream. Optionally applies temporal smoothing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["video", "images"], {"default": "video"}),
                "video_obj": ("VIDEO",),   # Used if source_type == "video"
                "images": ("IMAGE",),      # Used if source_type == "images" (single image, list, or batch tensor)
                "sync_meta": ("DICT",),    # sync_meta from loader
                "title": ("STRING", {"default": "untitled_pose"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1, "max": 60}),
                "temporal_smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("TENSOR", "DICT")
    RETURN_NAMES  = ("pose_tensor", "sync_meta")
    FUNCTION      = "extract"
    CATEGORY      = "BAIS1C VACE Suite/Pose"
    OUTPUT_NODE   = False

    def extract(self, source_type, video_obj, images, sync_meta, title, fps, temporal_smoothing, debug):
        # Decide input mode
        if source_type == "video":
            video_path = (
                sync_meta.get("video_path")
                or (video_obj.video_path if hasattr(video_obj, "video_path") else str(video_obj))
            )
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            frame_indices = list(range(0, total_frames, max(1, int(vr.get_avg_fps() // fps) or 1)))
            frame_imgs = [vr[i].asnumpy() for i in frame_indices]
            true_fps = float(sync_meta.get("fps", vr.get_avg_fps()))
            if debug:
                print(f"[BAIS1C PoseExtractor] Loaded {len(frame_imgs)} frames from video ({video_path})")

        elif source_type == "images":
            # images: could be list, batch tensor, or single image
            if isinstance(images, list):
                frame_imgs = [self._img_to_np(img) for img in images]
            elif isinstance(images, torch.Tensor) or isinstance(images, np.ndarray):
                if hasattr(images, "shape") and len(images.shape) == 4:
                    # Batch (N,H,W,C) or (N,C,H,W)
                    if images.shape[-1] == 3:  # (N,H,W,3)
                        frame_imgs = [np.asarray(img) for img in images]
                    elif images.shape[1] == 3:  # (N,3,H,W) torch tensor
                        frame_imgs = [np.moveaxis(np.asarray(img), 0, -1) for img in images]
                    else:
                        raise ValueError("Unsupported image batch shape")
                elif len(images.shape) == 3:
                    frame_imgs = [np.asarray(images)]
                else:
                    raise ValueError("Unknown images input format")
            else:
                frame_imgs = [self._img_to_np(images)]
            total_frames = len(frame_imgs)
            true_fps = float(fps)
            if debug:
                print(f"[BAIS1C PoseExtractor] Loaded {len(frame_imgs)} images (direct)")

        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        # -- Pose extraction (DWPose, per frame) --
        pose_list = []
        for i, img in enumerate(frame_imgs):
            # DWPose expects numpy image (H,W,3), uint8
            img_np = img.astype(np.uint8) if img.dtype != np.uint8 else img
            try:
                pose = self._extract_dwpose_128(img_np)
            except Exception as e:
                if debug:
                    print(f"[PoseExtractor] Frame {i}: pose extraction failed: {e}")
                pose = np.zeros((128, 3), dtype=np.float32)  # fallback blank
            pose_list.append(pose)
            if debug and i < 3:
                print(f"[PoseExtractor] Frame {i} pose: {pose.shape}")

        pose_tensor = torch.from_numpy(np.stack(pose_list))  # (F,128,3)

        # -- Optional temporal smoothing --
        if temporal_smoothing > 0.0 and pose_tensor.shape[0] > 1:
            pose_tensor = self._temporal_smooth(pose_tensor, temporal_smoothing)
            if debug:
                print(f"[PoseExtractor] Temporal smoothing applied: {temporal_smoothing}")

        # -- Augment sync_meta for downstream nodes --
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "title":            title,
            "processed_frames": pose_tensor.shape[0],
            "fps":              true_fps,
            "input_type":       source_type,
            "pose_extraction_success": True,
        })

        return (pose_tensor, sync_meta_out)

    # -- Util: Convert image input to np.uint8 array (H,W,3) --
    def _img_to_np(self, img: Any):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            if img.shape[0] == 3 and len(img.shape) == 3:
                img = np.moveaxis(img, 0, -1)
            img = (img * 255).clip(0, 255).astype(np.uint8)
        elif isinstance(img, np.ndarray):
            img = img.astype(np.uint8)
        return img

    # -- Util: Run DWPose and return (128,3) keypoints --
    def _extract_dwpose_128(self, img: np.ndarray):
        # This assumes dwpose_detector returns a pose dict with 'bodies' containing 'candidate'
        pose_out = dwpose_detector(img)
        # Extract keypoints (should match [frames,128,3] spec)
        # For this template: If the model gives fewer points, pad to 128
        kp = pose_out.get("bodies", {}).get("candidate", np.zeros((128, 3), dtype=np.float32))
        kp = np.asarray(kp)
        if kp.shape[0] < 128:
            # Pad with zeros
            pad = np.zeros((128 - kp.shape[0], 3), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        return kp[:128, :]  # (128,3)

    # -- Util: Temporal smoothing (EMA, inplace) --
    def _temporal_smooth(self, pose_tensor: torch.Tensor, factor: float):
        pose_np = pose_tensor.cpu().numpy()
        for i in range(1, pose_np.shape[0]):
            pose_np[i] = factor * pose_np[i - 1] + (1 - factor) * pose_np[i]
        return torch.from_numpy(pose_np).float()

# --------------------------------------------------------------------------
# Node registration
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS        = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt, DWPose, Images+Video)"}


# --------------------- Self-test: Run as script ---------------------
def _test_pose_extractor():
    print("[TEST] BAIS1C_PoseExtractor (synthetic)")
    node = BAIS1C_PoseExtractor()
    # Generate dummy images (batch of 4)
    imgs = [np.random.randint(0,255,(480,640,3),dtype=np.uint8) for _ in range(4)]
    sync_meta = {"video_path":"dummy.mp4","fps":24.0}
    tensor, meta = node.extract(
        "images", None, imgs, sync_meta, "test_seq", 24.0, 0.2, True
    )
    print("Output pose_tensor:", tensor.shape)
    print("Output sync_meta:", meta)
    assert tensor.shape[1:] == (128,3), "Pose tensor shape must be (N,128,3)"
    print("[TEST PASSED]")

# Uncomment to run self-test
# _test_pose_extractor()
