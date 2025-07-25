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

    def __init__(self):
        """Check DWPose availability at initialization"""
        self.dwpose_available = dwpose_detector is not None
        if not self.dwpose_available:
            print("🚨 [BAIS1C_PoseExtractor] WARNING: DWPose detector not loaded!")
            print("   Pose extraction will use dummy data.")
            print("   Check DWPose model files in ComfyUI/models/dwpose/:")
            print("   - yolox_l.onnx")
            print("   - dw-ll_ucoco_384.onnx")
        else:
            print("✅ [BAIS1C_PoseExtractor] DWPose detector ready")

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
        """Extract poses with proper error handling and DWPose validation"""
        
        # Early warning if DWPose unavailable
        if not self.dwpose_available and debug:
            print(f"[PoseExtractor] DWPose unavailable - will generate dummy poses")
        
        # Image normalization and shape fix
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if images.shape[-1] != 3:
            raise ValueError(f"Images input must have shape (N,H,W,3), got {images.shape}")

        total_frames = images.shape[0]
        pose_list = []
        successful_extractions = 0
        
        for i in range(total_frames):
            img = images[i].astype(np.uint8)
            try:
                if self.dwpose_available:
                    pose = self._extract_dwpose_128(img)
                    successful_extractions += 1
                else:
                    pose = self._create_dummy_pose(i, total_frames)
                    if debug and i == 0:
                        print(f"[PoseExtractor] Using dummy poses (DWPose not available)")
            except Exception as e:
                if debug:
                    print(f"[PoseExtractor] Frame {i}: Pose extraction failed: {e}")
                pose = self._create_dummy_pose(i, total_frames)
            
            pose_list.append(pose)

        pose_tensor = torch.from_numpy(np.stack(pose_list))  # (F,128,2)

        # Optional temporal smoothing
        if temporal_smoothing > 0.0 and pose_tensor.shape[0] > 1:
            pose_tensor = self._temporal_smooth(pose_tensor, temporal_smoothing)
            if debug:
                print(f"[PoseExtractor] Temporal smoothing applied: {temporal_smoothing}")

        # Always update sync_meta consistently
        sync_meta_out = dict(sync_meta)
        sync_meta_out.update({
            "title": title,
            "processed_frames": pose_tensor.shape[0],
            "fps": fps,
            "pose_extraction_success": self.dwpose_available and successful_extractions > 0,
            "dwpose_available": self.dwpose_available,
            "successful_extractions": successful_extractions,
            "extraction_rate": successful_extractions / total_frames if total_frames > 0 else 0.0
        })

        if debug:
            print(f"[PoseExtractor] Completed: {successful_extractions}/{total_frames} frames")
            print(f"[PoseExtractor] Success rate: {sync_meta_out['extraction_rate']:.1%}")

        return pose_tensor, sync_meta_out

    def _extract_dwpose_128(self, img):
        """Safe DWPose extraction with proper error handling"""
        if not self.dwpose_available:
            raise RuntimeError("DWPose detector not available")
        
        try:
            pose_out = dwpose_detector(img)
            kp = pose_out.get("bodies", {}).get("candidate", np.zeros((128, 3), dtype=np.float32))
            kp = np.asarray(kp)
            
            if kp.shape[0] < 128:
                pad = np.zeros((128 - kp.shape[0], 3), dtype=np.float32)
                kp = np.concatenate([kp, pad], axis=0)
            
            # Return 2D coordinates (x, y) only
            return kp[:128, :2].astype(np.float32)  # (128, 2)
            
        except Exception as e:
            raise RuntimeError(f"DWPose extraction failed: {e}")

    def _create_dummy_pose(self, frame_idx, total_frames):
        """Create realistic dummy pose when DWPose fails"""
        # Create a basic T-pose with slight animation
        pose = np.zeros((128, 2), dtype=np.float32)
        
        # Basic body keypoints (first 23 points)
        t = frame_idx / max(1, total_frames - 1)  # 0 to 1
        sway = 0.02 * np.sin(t * 2 * np.pi)  # Gentle sway
        
        body_pose = np.array([
            [0.5, 0.12],           # 0: nose
            [0.5, 0.20],           # 1: neck  
            [0.48, 0.12],          # 2: right eye
            [0.52, 0.12],          # 3: left eye
            [0.46, 0.14],          # 4: right ear
            [0.54, 0.14],          # 5: left ear
            [0.40 + sway, 0.28],   # 6: left shoulder
            [0.60 + sway, 0.28],   # 7: right shoulder
            [0.32 + sway, 0.48],   # 8: left elbow
            [0.68 + sway, 0.48],   # 9: right elbow
            [0.23 + sway, 0.68],   # 10: left wrist
            [0.77 + sway, 0.68],   # 11: right wrist
            [0.44, 0.62],          # 12: left hip
            [0.56, 0.62],          # 13: right hip
            [0.39, 0.84],          # 14: left knee
            [0.61, 0.84],          # 15: right knee
            [0.37, 1.00],          # 16: left ankle
            [0.63, 1.00],          # 17: right ankle
        ], dtype=np.float32)
        
        # Fill in basic body pose
        pose[:len(body_pose)] = body_pose
        
        return np.clip(pose, 0.0, 1.0)

    def _temporal_smooth(self, pose_tensor, factor):
        """Apply temporal smoothing to reduce jitter"""
        pose_np = pose_tensor.cpu().numpy()
        for i in range(1, pose_np.shape[0]):
            pose_np[i] = factor * pose_np[i - 1] + (1 - factor) * pose_np[i]
        return torch.from_numpy(pose_np).float()

# Node registration
NODE_CLASS_MAPPINGS = {"BAIS1C_PoseExtractor": BAIS1C_PoseExtractor}
NODE_DISPLAY_NAME_MAPPINGS = {"BAIS1C_PoseExtractor": "🎯 BAIS1C Pose Extractor (128pt, DWPose, Images Only)"}

# Self-test function
def _test_pose_extractor():
    """Test pose extractor with and without DWPose"""
    extractor = BAIS1C_PoseExtractor()
    
    # Test with dummy data
    test_images = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
    test_sync_meta = {"fps": 24.0, "duration": 0.2}
    
    try:
        pose_tensor, sync_meta_out = extractor.extract(
            test_images, test_sync_meta, "test_pose", 24.0, 0.1, True
        )
        
        print(f"✅ Pose extraction test passed")
        print(f"   Output shape: {pose_tensor.shape}")
        print(f"   DWPose available: {sync_meta_out['dwpose_available']}")
        print(f"   Success rate: {sync_meta_out['extraction_rate']:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Pose extraction test failed: {e}")
        return False

if __name__ == "__main__":
    _test_pose_extractor()