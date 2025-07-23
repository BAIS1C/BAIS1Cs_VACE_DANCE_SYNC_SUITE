# BAIS1C VACE Dance Sync Suite - DWPose Module
# Pose detection using DWPose (Densepose + Whole-body pose estimation)

import os

# Import core DWPose components
try:
    from .dwpose_detector import DWposeDetector, dwpose_detector, create_dwpose_detector
    from .wholebody import Wholebody
    from .onnxdet import nms, multiclass_nms, inference_detector
    from .onnxpose import inference_pose
    from .util import draw_pose, draw_bodypose, draw_handpose, draw_facepose
    
    print(f"[BAIS1C VACE Suite] DWPose module loaded successfully")
    
    # Check if detector is initialized
    if dwpose_detector is not None:
        print(f"[BAIS1C VACE Suite] ✅ DWPose detector ready for pose extraction")
    else:
        print(f"[BAIS1C VACE Suite] ⚠️ DWPose detector not initialized - check model files")
        print(f"[BAIS1C VACE Suite] Required models in models/dwpose/:")
        print(f"  - yolox_l.onnx (person detection)")
        print(f"  - dw-ll_ucoco_384.onnx (pose estimation)")
    
    __all__ = [
        "DWposeDetector",
        "dwpose_detector", 
        "create_dwpose_detector",
        "Wholebody",
        "nms",
        "multiclass_nms", 
        "inference_detector",
        "inference_pose",
        "draw_pose",
        "draw_bodypose",
        "draw_handpose", 
        "draw_facepose"
    ]
    
except ImportError as e:
    print(f"[BAIS1C VACE Suite] ❌ Failed to load DWPose components: {e}")
    print(f"[BAIS1C VACE Suite] DWPose directory: {os.path.dirname(__file__)}")
    print(f"[BAIS1C VACE Suite] Check that all DWPose files are present:")
    print(f"  - dwpose_detector.py")
    print(f"  - wholebody.py") 
    print(f"  - onnxdet.py")
    print(f"  - onnxpose.py")
    print(f"  - util.py")
    
    # Set None values to prevent import errors
    DWposeDetector = None
    dwpose_detector = None
    create_dwpose_detector = None
    Wholebody = None
    
    __all__ = []
    
    # Re-raise the error to alert user
    raise

# Module metadata
__version__ = "1.0.0"
__description__ = "DWPose integration for BAIS1C VACE Dance Sync Suite"