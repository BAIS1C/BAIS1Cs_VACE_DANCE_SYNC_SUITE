# BAIS1C VACE Dance Sync Suite - Node Registration
# Registers all ComfyUI nodes for the suite

from .source_video_loader import NODE_CLASS_MAPPINGS as LOADER_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY
from .pose_tensor_extract import NODE_CLASS_MAPPINGS as EXTRACT_NODES, NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY
from .music_control_net import NODE_CLASS_MAPPINGS as CONTROL_NODES, NODE_DISPLAY_NAME_MAPPINGS as CONTROL_DISPLAY
from .simple_dance_poser import NODE_CLASS_MAPPINGS as SIMPLE_NODES, NODE_DISPLAY_NAME_MAPPINGS as SIMPLE_DISPLAY
from .pose_checkpoint import NODE_CLASS_MAPPINGS as CHECKPOINT_NODES, NODE_DISPLAY_NAME_MAPPINGS as CHECKPOINT_DISPLAY
from .save_pose_json import NODE_CLASS_MAPPINGS as SAVE_NODES, NODE_DISPLAY_NAME_MAPPINGS as SAVE_DISPLAY

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register Source Video Loader
NODE_CLASS_MAPPINGS.update(LOADER_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY)

# Register Pose Tensor Extractor  
NODE_CLASS_MAPPINGS.update(EXTRACT_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(EXTRACT_DISPLAY)

# Register Music Control Net (the magic sync node)
NODE_CLASS_MAPPINGS.update(CONTROL_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROL_DISPLAY)

# Register Simple Dance Poser (creative experimentation)
NODE_CLASS_MAPPINGS.update(SIMPLE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SIMPLE_DISPLAY)

#Register Pose Checkpoint
NODE_CLASS_MAPPINGS.update(CHECKPOINT_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(CHECKPOINT_DISPLAY)

#Register Save Pose JSON
NODE_CLASS_MAPPINGS.update(SAVE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(SAVE_DISPLAY)

# Debug: Print registered nodes
print(f"[BAIS1C VACE Suite] Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")

# Validation: Ensure all nodes have proper registration
def validate_node_registration():
    """Validate that all nodes are properly registered"""
    issues = []
    
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        # Check if node has display name
        if node_name not in NODE_DISPLAY_NAME_MAPPINGS:
            issues.append(f"Missing display name for {node_name}")
        
        # Check if node class has required methods
        if not hasattr(node_class, 'INPUT_TYPES'):
            issues.append(f"{node_name} missing INPUT_TYPES classmethod")
        
        if not hasattr(node_class, 'RETURN_TYPES'):
            issues.append(f"{node_name} missing RETURN_TYPES")
        
        if not hasattr(node_class, 'FUNCTION'):
            issues.append(f"{node_name} missing FUNCTION")
        
        if not hasattr(node_class, 'CATEGORY'):
            issues.append(f"{node_name} missing CATEGORY")
    
    if issues:
        print(f"[BAIS1C VACE Suite] ⚠️ Registration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"[BAIS1C VACE Suite] ✅ All nodes registered correctly")
        return True

# Run validation on import
validate_node_registration()
