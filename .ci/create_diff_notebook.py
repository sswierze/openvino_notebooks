#!/usr/bin/env python3
"""
Create a diff notebook showing what was patched in LLM notebooks.
This script compares original notebooks with their patched versions and creates
a comprehensive diff report in notebook format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import difflib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def load_notebook(notebook_path: Path) -> Dict:
    """Load a notebook file and return its content."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {notebook_path}: {e}")
        return {}


def get_cell_content(cell: Dict) -> str:
    """Extract content from a notebook cell."""
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            return ''.join(source)
        return source
    elif cell.get('cell_type') == 'markdown':
        source = cell.get('source', [])
        if isinstance(source, list):
            return ''.join(source)
        return source
    return ""


def find_differences(original_nb: Dict, patched_nb: Dict) -> List[Tuple[int, str, str, str]]:
    """Find differences between original and patched notebooks."""
    differences = []
    
    orig_cells = original_nb.get('cells', [])
    patched_cells = patched_nb.get('cells', [])
    
    max_cells = max(len(orig_cells), len(patched_cells))
    
    for i in range(max_cells):
        orig_cell = orig_cells[i] if i < len(orig_cells) else {}
        patched_cell = patched_cells[i] if i < len(patched_cells) else {}
        
        orig_content = get_cell_content(orig_cell)
        patched_content = get_cell_content(patched_cell)
        
        if orig_content != patched_content:
            cell_type = orig_cell.get('cell_type', patched_cell.get('cell_type', 'unknown'))
            differences.append((i + 1, cell_type, orig_content, patched_content))
    
    return differences


def create_diff_html(original: str, patched: str, cell_num: int, cell_type: str) -> str:
    """Create HTML diff view for cell content."""
    differ = difflib.HtmlDiff(tabsize=4)
    diff_html = differ.make_table(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        f"Original Cell {cell_num} ({cell_type})",
        f"Patched Cell {cell_num} ({cell_type})",
        context=True,
        numlines=3
    )
    return diff_html


def create_unified_diff(original: str, patched: str, cell_num: int) -> str:
    """Create unified diff format for cell content."""
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"Original Cell {cell_num}",
        tofile=f"Patched Cell {cell_num}",
        lineterm=''
    ))
    return ''.join(diff_lines)


def analyze_patch_types(differences: List[Tuple[int, str, str, str]]) -> Dict[str, List[str]]:
    """Analyze the types of patches made."""
    patch_analysis = {
        'pip_installs_removed': [],
        'model_changes': [],
        'device_changes': [],
        'value_attribute_fixes': [],
        'other_changes': []
    }
    
    for cell_num, cell_type, original, patched in differences:
        classified = False
        
        # Check for pip install removal (more specific check)
        if ('%pip install' in original or '!pip install' in original) and ('Package installation disabled' in patched or 'pip install' not in patched):
            patch_analysis['pip_installs_removed'].append(f"Cell {cell_num}: Removed pip install commands")
            classified = True
        
        # Check for configuration fetching replacement  
        if ('config_shared_path' in original or 'llm_config.py' in original) and 'Package installation disabled' in patched:
            patch_analysis['other_changes'].append(f"Cell {cell_num}: Configuration fetching replaced (cell alignment issue)")
            classified = True
        
        # Check for model changes
        if ('model_id' in original or 'model_configuration' in original) and ('microsoft/Phi-3.5-mini-instruct' in patched or 'LLM Testing Override' in patched):
            patch_analysis['model_changes'].append(f"Cell {cell_num}: Model configuration updated")
            classified = True
        
        # Check for value attribute fixes
        if '.value' in patched and '.value' not in original:
            patch_analysis['value_attribute_fixes'].append(f"Cell {cell_num}: Added .value attribute fixes")
            classified = True
        
        # Check for device changes
        if ('device' in original.lower() and 'device' in patched.lower() and 
            ('Device Testing Override' in patched or 'CPU' in patched or 'GPU' in patched or 'NPU' in patched)):
            patch_analysis['device_changes'].append(f"Cell {cell_num}: Device configuration changed")
            classified = True
        
        # Only mark as "other" if it's a significant change that wasn't classified
        if not classified and len(original.strip()) > 10 and len(patched.strip()) > 10:
            # Check if it's a substantial content change (not just whitespace)
            orig_lines = [l.strip() for l in original.split('\n') if l.strip()]
            patch_lines = [l.strip() for l in patched.split('\n') if l.strip()]
            if orig_lines != patch_lines:
                patch_analysis['other_changes'].append(f"Cell {cell_num}: Other modifications detected")
    
    return patch_analysis


def create_diff_notebook(notebooks_dir: Path, topology: str, device: str) -> Path:
    """Create a comprehensive diff notebook."""
    
    # Define topology to notebook mapping
    topologies = {
        "phi-3.5": ["llm-chatbot/llm-chatbot.ipynb"],
        "llama-3.2": ["llm-chatbot/llm-chatbot-generate-api.ipynb"], 
        "qwen-7b": ["llm-chatbot/llm-chatbot.ipynb"]
    }
    
    if topology not in topologies:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Create new notebook
    nb = new_notebook()
    
    # Add title
    title_cell = new_markdown_cell(f"""# LLM Notebook Diff Report
    
**Topology:** {topology}  
**Device:** {device.upper()}  
**Generated:** {Path(__file__).name}

This notebook shows the differences between original notebooks and their patched versions used for LLM testing.
""")
    nb.cells.append(title_cell)
    
    # Process each notebook for this topology
    total_changes = 0
    
    for notebook_rel_path in topologies[topology]:
        notebook_path = notebooks_dir / "notebooks" / notebook_rel_path
        patched_path = notebook_path.parent / f"test_{notebook_path.name}"
        
        # Add section header
        section_cell = new_markdown_cell(f"""## 📓 {notebook_rel_path}

**Original:** `{notebook_path}`  
**Patched:** `{patched_path}`
""")
        nb.cells.append(section_cell)
        
        if not notebook_path.exists():
            error_cell = new_markdown_cell(f"❌ **Error:** Original notebook not found at `{notebook_path}`")
            nb.cells.append(error_cell)
            continue
            
        if not patched_path.exists():
            error_cell = new_markdown_cell(f"❌ **Error:** Patched notebook not found at `{patched_path}`")
            nb.cells.append(error_cell)
            continue
        
        # Load notebooks
        original_nb = load_notebook(notebook_path)
        patched_nb = load_notebook(patched_path)
        
        if not original_nb or not patched_nb:
            error_cell = new_markdown_cell("❌ **Error:** Failed to load one or both notebooks")
            nb.cells.append(error_cell)
            continue
        
        # Find differences
        differences = find_differences(original_nb, patched_nb)
        total_changes += len(differences)
        
        if not differences:
            no_diff_cell = new_markdown_cell("✅ **No differences found** - notebooks are identical")
            nb.cells.append(no_diff_cell)
            continue
        
        # Analyze patch types
        patch_analysis = analyze_patch_types(differences)
        
        # Add summary
        summary_lines = [f"### 📊 Patch Summary ({len(differences)} changes found)"]
        
        for category, changes in patch_analysis.items():
            if changes:
                category_name = category.replace('_', ' ').title()
                summary_lines.append(f"\n**{category_name}:**")
                for change in changes:
                    summary_lines.append(f"- {change}")
        
        summary_cell = new_markdown_cell('\n'.join(summary_lines))
        nb.cells.append(summary_cell)
        
        # Add detailed diffs
        details_cell = new_markdown_cell("### 🔍 Detailed Cell-by-Cell Differences")
        nb.cells.append(details_cell)
        
        for cell_num, cell_type, original, patched in differences:
            # Cell difference header
            diff_header = new_markdown_cell(f"""#### Cell {cell_num} ({cell_type})
            
**Change Type:** {cell_type} cell modification
""")
            nb.cells.append(diff_header)
            
            # Unified diff
            unified_diff = create_unified_diff(original, patched, cell_num)
            if unified_diff:
                diff_cell = new_code_cell(unified_diff)
                diff_cell.metadata = {"tags": ["diff"]}
                nb.cells.append(diff_cell)
            
            # Show original content
            if original.strip():
                orig_cell = new_markdown_cell("**Original Content:**")
                nb.cells.append(orig_cell)
                orig_code_cell = new_code_cell(original)
                orig_code_cell.metadata = {"tags": ["original"]}
                nb.cells.append(orig_code_cell)
            
            # Show patched content
            if patched.strip():
                patched_cell = new_markdown_cell("**Patched Content:**")
                nb.cells.append(patched_cell)
                patched_code_cell = new_code_cell(patched)
                patched_code_cell.metadata = {"tags": ["patched"]}
                nb.cells.append(patched_code_cell)
            
            # Add separator
            separator_cell = new_markdown_cell("---")
            nb.cells.append(separator_cell)
    
    # Add final summary
    final_summary = new_markdown_cell(f"""## 📋 Final Summary

**Total Changes Found:** {total_changes}  
**Topology:** {topology}  
**Device:** {device.upper()}

### Key Patch Operations:
1. **Environment Protection:** Removed pip install commands to prevent conflicts
2. **Model Compatibility:** Updated model IDs to use supported models
3. **Widget Fixes:** Added .value attributes for proper widget handling
4. **Device Configuration:** Ensured proper device targeting

This diff report helps you understand exactly what modifications were made to the original notebooks for testing purposes.
""")
    nb.cells.append(final_summary)
    
    # Save the notebook in working directory to keep repository clean
    working_dir = Path("llm_test_working_dir")
    working_dir.mkdir(exist_ok=True)
    output_path = working_dir / f"diff_report_{topology}_{device}.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    return output_path


def main():
    """Main function to create diff notebook."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create LLM notebook diff report")
    parser.add_argument("--topology", choices=["phi-3.5", "llama-3.2", "qwen-7b"], 
                       required=True, help="Topology to analyze")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "npu"], 
                       help="Device used for patching")
    parser.add_argument("--notebooks-dir", default="..", 
                       help="Path to notebooks directory")
    
    args = parser.parse_args()
    
    notebooks_dir = Path(args.notebooks_dir)
    
    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    try:
        output_path = create_diff_notebook(notebooks_dir, args.topology, args.device)
        print(f"✅ Diff report created: {output_path}")
        print(f"📂 Open in Jupyter: jupyter notebook {output_path}")
    except Exception as e:
        print(f"❌ Error creating diff report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()