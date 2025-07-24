import numpy as np
import torch
from ..dwpose.dwpose_detector import dwpose_detector

class BAIS1C_PoseExtractor:
    """
    Pose Extractor (images → pose_tensor, sync_meta)
    - Takes images (required) and sync_meta (DICT)
    - Extracts pose per frame using DWPose
    - Outputs pose_tensor (POSE) and sync_meta (DICT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sync_meta": ("DICT",),
                "title": ("STRING", {"default": "untitled_pose"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1, "max": 60}),
                "temporal_smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES  = ("POSE", "DICT")
    RETURN_NAMES  = ("pose_tensor", "sync_meta")
    FUNCTION      = "extract"
    CATEGORY      = "BAIS1C VACE Suite/Pose"
    OUTPUT_NODE   = False

    def extract(self, images, sync_meta, title, fps, temporal_smoothing, debug):
        # Image normalization and shape fix
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.shape[-1] != 3:
            raise ValueError("Images input must have shape (N,H,W,3)")

        total_frames = images.shape[0]
        pose_list = []
        for i in range(total_frames):
            img = images[i].astype(np.uint8)
            try:
                pose = self._extract_dwpose_128(img)
            except Exception as e:
                if debug:
                    print(f"[PoseExtractor] Frame {i}: pose extraction failed: {e}")
                pose = np.zeros((128, 3), dtype=np.float32)
            pose_list.append(pose)

        pose_tensor = torch.from_numpy(np.stack(pose_list))  # (F,128,3)

        # Optional temporal smoothing
        if temporal_smoothing > 0.0 and pose_tensor.shape[0] > 1:
            pose_tensor = self._temporal_smooth(pose_tensor, temporal_smoothing)
            if debug:
                print(f"[PoseExtractor] Temporal smoothing applied: {temporal_smoothing}")

        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "title": title,
            "processed_frames": pose_tensor.shape[0],
            "fps": fps,
            "pose_extraction_success": True,
        })

        return pose_tensor, sync_meta_out

    def _extract_dwpose_128(self, img):
        pose_out = dwpose_detector(img)
        kp = pose_out.get("bodies", {}).get("candidate", np.zeros((128, 3), dtype=np.float32))
        kp = np.asarray(kp)
        if kp.shape[0] < 128:
            pad = np.zeros((128 - kp.shape[0], 3), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        return kp[:128, :]  # (128,3)

    def _temporal_smooth(self, pose_tensor, factor):
        pose_np = pose_tensor.cpu().numpy()
        for i in range(1, pose_np.shape[0]):
            pose_np[i] = factor * pose_np[i - 1] + (1 - factor) * pose_np[i]
        return torch.from_numpy(pose_np).float()

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt, DWPose, Images Only)"}
