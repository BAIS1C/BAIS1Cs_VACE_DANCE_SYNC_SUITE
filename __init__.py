from .nodes.BAIS1C_SourceVideoLoader import NODE_CLASS_MAPPINGS as LOADER_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY
from .nodes.BAIS1C_PoseTensorExtract import NODE_CLASS_MAPPINGS as EXTRACT_NODES, NODE_DISPLAY_NAME_MAPPINGS as EXTRACT_DISPLAY
from .nodes.BAIS1C_MusicControlNet import NODE_CLASS_MAPPINGS as CONTROL_NODES, NODE_DISPLAY_NAME_MAPPINGS as CONTROL_DISPLAY
from .nodes.BAIS1C_PoseToVideoRenderer import NODE_CLASS_MAPPINGS as RENDERER_NODES, NODE_DISPLAY_NAME_MAPPINGS as RENDERER_DISPLAY

# ANSI color codes for styling
class Colors:
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# ASCII Art Banner
def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
██████╗  █████╗ ██╗███████╗ ██╗ ██████╗
██╔══██╗██╔══██╗██║██╔════╝███║██╔════╝
██████╔╝███████║██║███████╗╚██║██║     
██╔══██╗██╔══██║██║╚════██║ ██║██║     
██████╔╝██║  ██║██║███████║ ██║╚██████╗
╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝ ╚═╝ ╚═════╝
{Colors.END}
{Colors.MAGENTA}🕺 VACE DANCE SYNC SUITE 🎵{Colors.END}
{Colors.YELLOW}═══════════════════════════════════════{Colors.END}
"""
    
    dancing_stick_man = f"""
{Colors.GREEN}     ○     {Colors.CYAN}♪ ♫ ♪{Colors.END}
{Colors.GREEN}    /|\\    {Colors.CYAN}♫ ♪ ♫{Colors.END}
{Colors.GREEN}    / \\    {Colors.CYAN}♪ ♫ ♪{Colors.END}
{Colors.WHITE}  Dancing with the beat!{Colors.END}
"""
    
    print(banner)
    print(dancing_stick_man)

# Node icons and descriptions
NODE_ICONS = {
    "BAIS1C_SourceVideoLoader": f"{Colors.BLUE}📺{Colors.END}",
    "BAIS1C_PoseTensorExtract": f"{Colors.GREEN}🕺{Colors.END}",
    "BAIS1C_MusicControlNet": f"{Colors.MAGENTA}🎵{Colors.END}",
    "BAIS1C_PoseToVideoRenderer": f"{Colors.CYAN}🎬{Colors.END}"
}

NODE_DESCRIPTIONS = {
    "BAIS1C_SourceVideoLoader": f"{Colors.BLUE}Video & Audio Prep + BPM Detection{Colors.END}",
    "BAIS1C_PoseTensorExtract": f"{Colors.GREEN}DWPose-23 Extraction & Smoothing{Colors.END}",
    "BAIS1C_MusicControlNet": f"{Colors.MAGENTA}Beat-Sync Dance Retargeting Engine{Colors.END}",
    "BAIS1C_PoseToVideoRenderer": f"{Colors.CYAN}Stickman Video Visualization{Colors.END}"
}

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

# Validation with style
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
        print(f"{Colors.RED}{Colors.BOLD}⚠️  Registration Issues Found:{Colors.END}")
        for issue in issues:
            print(f"  {Colors.RED}❌ {issue}{Colors.END}")
        return False
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ All nodes validated successfully{Colors.END}")
        return True

# Main initialization
def initialize_suite():
    print_banner()
    
    print(f"{Colors.YELLOW}{Colors.BOLD}🚀 Loading BAIS1C VACE Dance Sync Suite...{Colors.END}")
    print()
    
    # Node registration status
    print(f"{Colors.WHITE}{Colors.BOLD}📦 Registered {len(NODE_CLASS_MAPPINGS)} nodes:{Colors.END}")
    
    for node_name in NODE_CLASS_MAPPINGS.keys():
        icon = NODE_ICONS.get(node_name, "🔧")
        description = NODE_DESCRIPTIONS.get(node_name, "Unknown Node")
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        
        print(f"  {icon} {Colors.BOLD}{display_name}{Colors.END}")
        print(f"    {description}")
        print(f"    {Colors.WHITE}Class:{Colors.END} {Colors.CYAN}{node_name}{Colors.END}")
        print()
    
    # Validation
    validation_success = validate_node_registration()
    
    if validation_success:
        print(f"""
{Colors.GREEN}{Colors.BOLD}🎉 BAIS1C VACE Suite Ready!{Colors.END}
{Colors.CYAN}╔══════════════════════════════════════╗{Colors.END}
{Colors.CYAN}║{Colors.END} {Colors.WHITE}✨ Dance + Music + AI = Magic ✨{Colors.END}     {Colors.CYAN}║{Colors.END}
{Colors.CYAN}║{Colors.END} {Colors.YELLOW}🎵 BPM Detection & Beat Sync{Colors.END}         {Colors.CYAN}║{Colors.END}
{Colors.CYAN}║{Colors.END} {Colors.GREEN}🕺 23-Point Pose Extraction{Colors.END}          {Colors.CYAN}║{Colors.END}
{Colors.CYAN}║{Colors.END} {Colors.MAGENTA}🎬 VACE-Ready Animation Pipeline{Colors.END}     {Colors.CYAN}║{Colors.END}
{Colors.CYAN}╚══════════════════════════════════════╝{Colors.END}
""")
    else:
        print(f"{Colors.RED}{Colors.BOLD}💥 Suite initialization failed - check errors above{Colors.END}")
    
    return validation_success

# Run initialization
try:
    initialize_suite()
except Exception as e:
    print(f"{Colors.RED}{Colors.BOLD}💥 Critical Error During Suite Loading:{Colors.END}")
    print(f"{Colors.RED}❌ {str(e)}{Colors.END}")
    print(f"{Colors.YELLOW}🔧 Check your installation and dependencies{Colors.END}")