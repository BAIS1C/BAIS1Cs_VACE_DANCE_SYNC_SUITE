from .nodes.BAIS1C_SourceVideoLoader import NODE_CLASS_MAPPINGS as LOADER_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY
from .nodes.BAIS1C_PoseTensorExtract import NODE_CLASS_MAPPINGS as EXTRACT_NODES, NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY
from .nodes.BAIS1C_MusicControlNet import NODE_CLASS_MAPPINGS as CONTROL_NODES, NODE_DISPLAY_NAME_MAPPINGS as CONTROL_DISPLAY
from .nodes.BAIS1C_PoseToVideoRenderer import NODE_CLASS_MAPPINGS as RENDERER_NODES, NODE_DISPLAY_NAME_MAPPINGS as RENDERER_DISPLAY

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register core nodes
NODE_CLASS_MAPPINGS.update(LOADER_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY)
NODE_CLASS_MAPPINGS.update(EXTRACT_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(EXTRACT_DISPLAY)
NODE_CLASS_MAPPINGS.update(CONTROL_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROL_DISPLAY)
NODE_CLASS_MAPPINGS.update(RENDERER_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(RENDERER_DISPLAY)

# Debug: Print registered nodes
print(f"[BAIS1C VACE Suite] Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  - {display_name} ({node_name})")

# Validation (unchanged)
def validate_node_registration():
    issues = []
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        if node_name not in NODE_DISPLAY_NAME_MAPPINGS:
            issues.append(f"Missing display name for {node_name}")
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

validate_node_registration()
