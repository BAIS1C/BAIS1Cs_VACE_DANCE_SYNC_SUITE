# BAIS1C VACE Dance Sync Suite
# Main initialization file for ComfyUI custom node package
#
# This suite provides:
# - Video source loading with BPM analysis
# - Pose tensor extraction using DWPose
# - Music-synchronized dance animation (Music Control Net)
# - Professional-grade BPM/FPS synchronization

import os
import sys

# Add the suite directory to Python path for internal imports
suite_dir = os.path.dirname(os.path.abspath(__file__))
if suite_dir not in sys.path:
    sys.path.insert(0, suite_dir)

# Import node registrations from nodes package
try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    print(f"[BAIS1C VACE Suite] Successfully loaded custom node suite from: {suite_dir}")
    print(f"[BAIS1C VACE Suite] Available components:")
    print(f"  - Source Video Loader (video + audio + BPM analysis)")  
    print(f"  - Pose Tensor Extractor (DWPose → 128-point JSON)")
    print(f"  - Music Control Net (pose-to-music synchronization)")
    print(f"  - Dance library management")
    print(f"  - Professional BPM/FPS sync algorithms")
    
    # Check for required directories
    dance_library_dir = os.path.join(suite_dir, "dance_library")
    dwpose_dir = os.path.join(suite_dir, "dwpose")
    
    if os.path.exists(dance_library_dir):
        json_count = len([f for f in os.listdir(dance_library_dir) if f.endswith('.json')])
        print(f"[BAIS1C VACE Suite] Dance library: {json_count} poses available")
    else:
        print(f"[BAIS1C VACE Suite] ⚠️ Dance library directory not found: {dance_library_dir}")
    
    if os.path.exists(dwpose_dir):
        onnx_files = [f for f in os.listdir(dwpose_dir) if f.endswith('.onnx')]
        if len(onnx_files) >= 2:
            print(f"[BAIS1C VACE Suite] ✅ DWPose models found: {len(onnx_files)} ONNX files")
        else:
            print(f"[BAIS1C VACE Suite] ⚠️ DWPose models incomplete: {len(onnx_files)}/2 ONNX files")
            print(f"[BAIS1C VACE Suite] Required: yolox_l.onnx, dw-ll_ucoco_384.onnx")
    else:
        print(f"[BAIS1C VACE Suite] ⚠️ DWPose directory not found: {dwpose_dir}")
    
    # Export for ComfyUI
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    
except ImportError as e:
    print(f"[BAIS1C VACE Suite] ❌ Failed to load nodes: {e}")
    print(f"[BAIS1C VACE Suite] Suite directory: {suite_dir}")
    print(f"[BAIS1C VACE Suite] Check that all required files are present:")
    print(f"  - nodes/__init__.py")
    print(f"  - nodes/BAIS1C_SourceVideoLoader.py") 
    print(f"  - nodes/pose_tensor_extract.py")
    print(f"  - nodes/music_control_net.py")
    print(f"  - dwpose/ directory with ONNX models")
    
    # Provide empty mappings to prevent ComfyUI errors
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    raise

# Suite metadata
__version__ = "1.0.0"
__author__ = "BAIS1C"
__description__ = "Professional dance synchronization suite for ComfyUI"
