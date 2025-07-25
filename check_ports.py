#!/usr/bin/env python3
"""
Port Type Validator - Run this to check all your node files for consistency
"""

import os
import re
from pathlib import Path

def validate_port_types(repo_path):
    """Validate all node files have consistent POSE typing"""
    issues = []
    node_files = []
    
    # Find all Python files in nodes directory
    nodes_dir = Path(repo_path) / "nodes"
    if nodes_dir.exists():
        node_files.extend(nodes_dir.glob("*.py"))
    
    # Also check root level Python files
    node_files.extend(Path(repo_path).glob("BAIS1C_*.py"))
    
    for file_path in node_files:
        if file_path.name.startswith("__"):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for TENSOR instead of POSE
            tensor_matches = re.findall(r'"pose[^"]*":\s*\("TENSOR"', content, re.IGNORECASE)
            if tensor_matches:
                issues.append(f"❌ {file_path.name}: Found TENSOR instead of POSE: {tensor_matches}")
            
            # Check for inconsistent return types
            return_type_matches = re.findall(r'RETURN_TYPES\s*=\s*\([^)]*"TENSOR"[^)]*\)', content)
            if return_type_matches:
                issues.append(f"❌ {file_path.name}: RETURN_TYPES uses TENSOR: {return_type_matches}")
            
            # Check for missing .detach() on tensor serialization
            if 'torch.is_tensor' in content and '.cpu().tolist()' in content:
                if '.detach().cpu().tolist()' not in content:
                    issues.append(f"⚠️  {file_path.name}: Possible missing .detach() in tensor serialization")
            
            # Check for sync_meta consistency
            sync_meta_variants = re.findall(r'"[^"]*meta[^"]*":\s*\("DICT"', content)
            for variant in sync_meta_variants:
                if 'sync_meta' not in variant:
                    issues.append(f"⚠️  {file_path.name}: Non-standard meta naming: {variant}")
                    
        except Exception as e:
            issues.append(f"❌ {file_path.name}: Failed to read file: {e}")
    
    return issues

def check_workflow_compatibility(repo_path):
    """Check workflow JSON files for port type mismatches"""
    issues = []
    json_files = list(Path(repo_path).glob("*.json"))
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for TENSOR connections to pose nodes
            if '"TENSOR"' in content and any(node in content for node in ['BAIS1C_Pose', 'PoseExtractor', 'PoseCheckpoint']):
                issues.append(f"⚠️  {json_path.name}: Workflow may have TENSOR/POSE type mismatches")
                
        except Exception as e:
            issues.append(f"❌ {json_path.name}: Failed to read workflow: {e}")
    
    return issues

def main():
    """Run validation on current directory"""
    repo_path = Path.cwd()
    print(f"🔍 Validating port types in: {repo_path}")
    print("=" * 60)
    
    # Check node files
    node_issues = validate_port_types(repo_path)
    if node_issues:
        print("📁 NODE FILE ISSUES:")
        for issue in node_issues:
            print(f"  {issue}")
    else:
        print("✅ NODE FILES: All port types look consistent")
    
    print()
    
    # Check workflows
    workflow_issues = check_workflow_compatibility(repo_path)
    if workflow_issues:
        print("📋 WORKFLOW ISSUES:")
        for issue in workflow_issues:
            print(f"  {issue}")
    else:
        print("✅ WORKFLOWS: No obvious type mismatches found")
    
    print()
    print("=" * 60)
    
    if node_issues or workflow_issues:
        print("❌ VALIDATION FAILED - Fix issues above before testing")
        return False
    else:
        print("✅ VALIDATION PASSED - Ready for pipeline testing")
        return True

if __name__ == "__main__":
    main()