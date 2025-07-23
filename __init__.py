from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export for ComfyUI registration
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Suite metadata
__version__ = "1.0.0"
__author__ = "BAIS1C"
__description__ = "VACE Dance Sync Suite - Pose extraction and music synchronization"

print(f"[BAIS1C VACE Dance Sync Suite] v{__version__} loaded! 🕺")