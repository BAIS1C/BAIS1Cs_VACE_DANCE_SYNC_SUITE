import os
import numpy as np
import torch

from .wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """
    def __init__(self, model_det, model_pose, device='cpu'):
        # Validate model files exist before initialization
        if not os.path.exists(model_det):
            raise FileNotFoundError(f"Detection model not found: {model_det}")
        if not os.path.exists(model_pose):
            raise FileNotFoundError(f"Pose model not found: {model_pose}")
            
        self.pose_estimation = Wholebody(model_det=model_det, model_pose=model_pose, device=device)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            # un_visible = subset < 0.3
            # candidate[un_visible] = -1

            # foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose


def create_dwpose_detector():
    """Factory function to create DWpose detector with proper error handling"""
    # Set the default path to comfyui\models\dwpose relative to this file.
    DEFAULT_DWPOSE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "dwpose"))
    dwpose_dir = os.environ.get("dwpose", DEFAULT_DWPOSE_PATH)
    
    model_det_path = os.path.join(dwpose_dir, "yolox_l.onnx")
    model_pose_path = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")
    
    # Check if directory exists
    if not os.path.exists(dwpose_dir):
        raise FileNotFoundError(
            f"DWPose model directory not found: {dwpose_dir}\n"
            f"Please create the directory and place the required .onnx files:\n"
            f"- yolox_l.onnx\n"
            f"- dw-ll_ucoco_384.onnx"
        )
    
    # Check individual model files
    if not os.path.exists(model_det_path):
        raise FileNotFoundError(
            f"Detection model not found: {model_det_path}\n"
            f"Please download yolox_l.onnx to {dwpose_dir}"
        )
    if not os.path.exists(model_pose_path):
        raise FileNotFoundError(
            f"Pose model not found: {model_pose_path}\n"
            f"Please download dw-ll_ucoco_384.onnx to {dwpose_dir}"
        )
    
    print(f"[BAIS1C VACE Suite] Loading DWPose models from: {dwpose_dir}")
    return DWposeDetector(model_det=model_det_path, model_pose=model_pose_path, device=device)

# Create the detector instance with error handling
try:
    dwpose_detector = create_dwpose_detector()
    print(f"[BAIS1C VACE Suite] DWPose detector initialized successfully on {device}")
except Exception as e:
    print(f"[BAIS1C VACE Suite] WARNING: DWPose detector initialization failed: {e}")
    dwpose_detector = None

# Self-test function
def test_dwpose_detector():
    """Basic test to verify detector functionality"""
    if dwpose_detector is None:
        print("❌ DWPose detector not initialized")
        return False
        
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        result = dwpose_detector(test_image)
        required_keys = ['bodies', 'hands', 'hands_score', 'faces', 'faces_score']
        
        if all(key in result for key in required_keys):
            print("✅ DWPose detector test passed")
            return True
        else:
            print(f"❌ DWPose detector missing keys: {set(required_keys) - set(result.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ DWPose detector test failed: {e}")
        return False

# Uncomment to run self-test
# test_dwpose_detector()