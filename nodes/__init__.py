from .pose_tensor_extract import BAIS1C_PoseExtractor

NODE_CLASS_MAPPINGS = {
    "BAIS1C_PoseExtractor": BAIS1C_PoseExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAIS1C_PoseExtractor": "🎯 Extract Pose Tensors (128pts)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]