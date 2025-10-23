#!/usr/bin/env python3
"""
LLM-specific notebook patcher for OpenVINO testing.
"""

import argparse
import re
from pathlib import Path
import nbformat

# LLM topologies and their notebooks
LLM_NOTEBOOKS = {
    "phi-3.5": ["notebooks/llm-chatbot/llm-chatbot.ipynb"],
    "llama-3.2": ["notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb"], 
    "qwen-7b": ["notebooks/llm-chatbot/llm-chatbot.ipynb"]
}

# Model mappings for each topology
MODEL_MAPPINGS = {
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",  # Keep original phi model
    "llama-3.2": "Qwen/Qwen2.5-0.5B-Instruct", 
    "qwen-7b": "Qwen/Qwen2.5-0.5B-Instruct"
}

def prevent_pip_installs(nb, notebook_path):
    """Prevent notebooks from installing packages that could conflict with our environment."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Check for any pip install commands
            if "%pip install" in source or "!pip install" in source:
                found = True
                
                # Replace the entire cell with a simple environment info message
                cell.source = '''# Package installation disabled for testing
# Original cell contained pip install commands that have been disabled
# to prevent conflicts with the test environment.

print("Package installation cell disabled - using pre-configured environment")
print("Environment packages should already be available from setup")'''
                
                print(f"Disabled pip install cell in {notebook_path}")
    
    if found:
        print(f"Prevented pip installs in {notebook_path}")
    
    return nb

def force_tqdm_console_mode(nb, notebook_path):
    """Force tqdm to use console mode instead of notebook widgets."""
    # Add a cell at the beginning that sets environment variables for console mode
    tqdm_patch_cell = nbformat.v4.new_code_cell(source='''# Force console mode for progress bars to avoid widget issues
import os

# Disable all notebook widgets and force text-based output
os.environ["TQDM_DISABLE"] = "0"  # Keep tqdm enabled but force text mode  
os.environ["TQDM_NCOLS"] = "80"   # Set fixed width
os.environ["TQDM_MININTERVAL"] = "1"  # Update every second
os.environ["TQDM_NOTEBOOK"] = "false"  # Force console mode
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Keep progress but force console
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # Disable Jupyter platform detection

# Simple patch to avoid notebook widget detection
try:
    import tqdm.auto
    # Just set the auto-detection to use std tqdm
    if hasattr(tqdm.auto, 'tqdm'):
        original_tqdm = tqdm.auto.tqdm
        # Replace with standard tqdm if it tries to use notebook version
        import tqdm.std
        tqdm.auto.tqdm = tqdm.std.tqdm
    print("✅ Set tqdm to use console mode")
except ImportError:
    print("⚠️  tqdm not available")
''')
    
    # Insert at the beginning of the notebook
    nb.cells.insert(0, tqdm_patch_cell)
    print(f"Added simplified tqdm console mode patch to {notebook_path}")
    
    return nb

def patch_model_selection(nb, notebook_path, topology=None):
    """Patch model selection dropdown/text field with specified topology."""
    if not topology or topology not in MODEL_MAPPINGS:
        return nb
    
    model_id = MODEL_MAPPINGS[topology]
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for model dropdown widget patterns
            patterns = [
                r'model_id\s*=\s*text\(',
                r'model_id\s*=\s*Dropdown\(',
                r'model_id\.value',
                r'model_language\s*=\s*Dropdown\(',
                r'model_language\.value',
                r'model_ids\s*=\s*list\(SUPPORTED_LLM_MODELS\[model_language\]\)',
                r'get_llm_selection_widget\(',
                r'model_id_widget\.value',
                r'model_id_widget\.label'
            ]
            
            for pattern in patterns:
                if re.search(pattern, source):
                    found = True
                    # Add model override, XetHub disable, and working directory setup
                    override_code = f'''# LLM Testing Override
import os
import sys
from pathlib import Path

# Disable XetHub to avoid 500 errors
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Setup working directory to avoid cluttering repo
working_dir = Path.cwd().parent.parent / "llm_test_working_dir"
working_dir.mkdir(exist_ok=True)

# Redirect all cache directories to working directory
os.environ["HF_HOME"] = str(working_dir / "huggingface")
os.environ["HF_HUB_CACHE"] = str(working_dir / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(working_dir / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(working_dir / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(working_dir / "transformers")

# Add virtual environment bin to PATH for optimum-cli access
venv_bin = Path(sys.executable).parent
current_path = os.environ.get("PATH", "")
if str(venv_bin) not in current_path:
    os.environ["PATH"] = f"{{str(venv_bin)}}:{{current_path}}"
    print(f"Added venv bin to PATH: {{str(venv_bin)}}")

model_language = "English"
model_id = "{model_id}"
print(f"Using model: {{model_id}}")
print(f"Working directory: {{working_dir}}")

'''
                    cell.source = override_code + source
                    break
    
    # Second pass: patch model_configuration lookup to avoid KeyError
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Also patch model_configuration lookup to avoid KeyError (handle both .value and non-.value versions)
            config_pattern = r'model_configuration\s*=\s*SUPPORTED_LLM_MODELS\[(model_language(?:\.value)?)\]\[(model_id(?:\.value)?)\]'
            if re.search(config_pattern, source):
                found = True
                print(f"Found model_configuration lookup in {notebook_path}")
                # Replace the lookup with a default configuration that includes required fields
                # Use a clean model name for local paths (replace slashes with dashes)
                clean_model_name = model_id.replace('/', '-')
                default_config = f'{{"model_id": "{model_id}", "clean_model_name": "{clean_model_name}", "remote_code": False, "start_message": "", "history_template": "{{history}}", "current_message_template": "{{current_message}}"}}'
                cell.source = re.sub(
                    config_pattern,
                    f'# LLM Testing Override - using default config\nmodel_configuration = {default_config}',
                    source
                )
                print(f"Patched model_configuration lookup in {notebook_path}")
    
    if found:
        print(f"Patched model selection in {notebook_path} with {topology} -> {model_id}")
    
    return nb

def redirect_model_directories(nb, notebook_path):
    """Redirect model directories to working directory to avoid repo clutter."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for model directory definitions
            patterns = [
                r'fp16_model_dir\s*=\s*Path\(',
                r'int8_model_dir\s*=\s*Path\(',
                r'int4_model_dir\s*=\s*Path\(',
                r'_model_dir\s*=\s*Path\('
            ]
            
            for pattern in patterns:
                if re.search(pattern, source):
                    found = True
                    # Add working directory redirection for model directories
                    redirect_code = '''# Model Directory Override - redirect to working directory
working_dir = Path.cwd().parent.parent / "llm_test_working_dir"
working_dir.mkdir(exist_ok=True)

# Redirect model directories to working directory
model_base_dir = working_dir / "models"
model_base_dir.mkdir(exist_ok=True)

'''
                    # Replace model directory paths to use working directory
                    source = re.sub(
                        r'(fp16_model_dir\s*=\s*)Path\(([^)]+)\)',
                        r'\1model_base_dir / Path(\2).name',
                        source
                    )
                    source = re.sub(
                        r'(int8_model_dir\s*=\s*)Path\(([^)]+)\)',
                        r'\1model_base_dir / Path(\2).name',
                        source
                    )
                    source = re.sub(
                        r'(int4_model_dir\s*=\s*)Path\(([^)]+)\)',
                        r'\1model_base_dir / Path(\2).name',
                        source
                    )
                    
                    cell.source = redirect_code + source
                    break
    
    if found:
        print(f"Redirected model directories in {notebook_path} to working directory")
    
    return nb

def patch_device_selection(nb, notebook_path, test_device="CPU"):
    """Patch device selection dropdown with specified device."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for device dropdown patterns
            patterns = [
                r'device\s*=\s*Dropdown\(',
                r'device\.value',
                r'core_device\s*=\s*Dropdown\(',
                r'core_device\.value',
                r'if\s+["\']GPU["\'].*in\s+device',
                r'device\s+in\s+\[.*GPU.*\]'
            ]
            
            for pattern in patterns:
                if re.search(pattern, source):
                    found = True
                    # Add device override, XetHub disable, and working directory setup
                    override_code = f'''# Device Testing Override
import os
from pathlib import Path

# Disable XetHub to avoid 500 errors
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Setup working directory to avoid cluttering repo
working_dir = Path.cwd().parent.parent.parent / "llm_test_working_dir"
working_dir.mkdir(exist_ok=True)

# Redirect all cache directories to working directory
os.environ["HF_HOME"] = str(working_dir / "huggingface")
os.environ["HF_HUB_CACHE"] = str(working_dir / "huggingface" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(working_dir / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(working_dir / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(working_dir / "transformers")

device = "{test_device}"
print(f"Using device: {{device}}")
print(f"Working directory: {{working_dir}}")

'''
                    cell.source = override_code + source
                    break
    
    if found:
        print(f"Patched device selection in {notebook_path} to use {test_device}")
    
    return nb

def disable_gradio_debug(nb, notebook_path):
    """Disable Gradio debug mode to prevent hanging."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for Gradio launch patterns
            if "demo.launch" in source or "demo.queue().launch" in source:
                found = True
                # Replace debug=True with debug=False
                source = re.sub(r'debug\s*=\s*True', 'debug=False', source)
                # Add share=False to prevent sharing
                if "share=True" in source:
                    source = source.replace("share=True", "share=False")
                
                cell.source = source
    
    if found:
        print(f"Disabled Gradio debug mode in {notebook_path}")
    
    return nb

def add_environment_info(nb, notebook_path):
    """Add environment information cell for debugging."""
    # Create new cell using nbformat
    env_cell = nbformat.v4.new_code_cell(source="""# Environment Information
import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import openvino as ov
    print(f"OpenVINO version: {ov.__version__}")
except ImportError:
    print("OpenVINO not found")

try:
    import importlib.metadata
    optimum_version = importlib.metadata.version('optimum-intel')
    print(f"Optimum-intel version: {optimum_version}")
except Exception:
    try:
        import optimum
        if hasattr(optimum, '__version__'):
            print(f"Optimum version: {optimum.__version__}")
        else:
            print("Optimum found but version unknown")
    except ImportError:
        print("Optimum not found")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    print("PyTorch not found")
""")
    
    # Insert at the beginning after any markdown headers
    insert_index = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            insert_index = i
            break
    
    nb.cells.insert(insert_index, env_cell)
    print(f"Added environment info cell to {notebook_path}")
    return nb

def fix_value_attributes(nb, notebook_path):
    """Fix .value attribute references that cause AttributeError."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Common .value patterns that need fixing
            replacements = [
                ('model_id.value', 'model_id'),
                ('device.value', 'device'), 
                ('core_device.value', 'core_device'),
                ('model_language.value', 'model_language'),
                ('compression_mode.value', 'compression_mode'),
                ('precision.value', 'precision'),
                ('quantization_mode.value', 'quantization_mode')
            ]
            
            modified = False
            for old, new in replacements:
                if old in source:
                    source = source.replace(old, new)
                    modified = True
                    found = True
            
            if modified:
                cell.source = source
    
    if found:
        print(f"Fixed .value attribute references in {notebook_path}")
    
    return found

def fix_model_paths(nb, notebook_path):
    """Fix model directory paths to avoid invalid repo IDs with multiple slashes."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Fix model directory construction that causes HF validation errors
            patterns = [
                (r'fp16_model_dir\s*=\s*Path\(model_id\)\s*/\s*["\']FP16["\']', 
                 'fp16_model_dir = Path(model_id.replace("/", "-")) / "FP16"'),
                (r'int8_model_dir\s*=\s*Path\(model_id\)\s*/\s*["\']INT8_compressed_weights["\']', 
                 'int8_model_dir = Path(model_id.replace("/", "-")) / "INT8_compressed_weights"'),
                (r'int4_model_dir\s*=\s*Path\(model_id\)\s*/\s*["\']INT4_compressed_weights["\']', 
                 'int4_model_dir = Path(model_id.replace("/", "-")) / "INT4_compressed_weights"')
            ]
            
            modified = False
            for pattern, replacement in patterns:
                if re.search(pattern, source):
                    source = re.sub(pattern, replacement, source)
                    modified = True
                    found = True
            
            # Fix tokenizer loading to use original model_id instead of local model_dir
            if 'AutoTokenizer.from_pretrained(model_dir' in source:
                source = source.replace(
                    'AutoTokenizer.from_pretrained(model_dir',
                    'AutoTokenizer.from_pretrained(model_id'
                )
                found = True
                
            # Fix OVModel loading to use model_dir when it exists, otherwise model_id
            if 'OVModelForCausalLM.from_pretrained(' in source and 'model_dir' in source:
                # Find the complete if/elif/else structure for model_dir assignment
                lines = source.split('\n')
                if_start = -1
                else_end = -1
                
                for i, line in enumerate(lines):
                    if 'if model_to_run.value ==' in line and 'INT4' in line:
                        if_start = i
                    if if_start != -1 and (line.strip().startswith('model_dir = fp16_model_dir') or 
                                          (line.strip().startswith('else:') and i < len(lines) - 1 and 
                                           'model_dir = fp16_model_dir' in lines[i+1])):
                        # Find the line after model_dir = fp16_model_dir
                        for j in range(i, min(i + 3, len(lines))):
                            if 'model_dir = fp16_model_dir' in lines[j]:
                                else_end = j
                                break
                        break
                
                if if_start != -1 and else_end != -1:
                    # Insert the check right after the complete if/elif/else structure
                    model_load_fix = '''# LLM Testing Override - use model_dir if exists, otherwise model_id
if not model_dir.exists():
    print(f"Model directory {model_dir} does not exist, using original model_id: {model_id}")
    actual_model_path = model_id
else:
    actual_model_path = model_dir'''
                    
                    lines.insert(else_end + 1, model_load_fix)
                    source = '\n'.join(lines)
                    
                    # Replace model_dir with actual_model_path in the OVModel loading
                    source = source.replace(
                        'OVModelForCausalLM.from_pretrained(\n    model_dir,',
                        'OVModelForCausalLM.from_pretrained(\n    actual_model_path,'
                    )
                    # Also fix the config parameter to use model_id instead of model_dir
                    source = source.replace(
                        'config=AutoConfig.from_pretrained(model_dir,',
                        'config=AutoConfig.from_pretrained(model_id,'
                    )
                    # Force export=True to convert from PyTorch when OpenVINO model doesn't exist
                    if 'export=True' not in source:
                        source = source.replace(
                            'ov_model = OVModelForCausalLM.from_pretrained(\n    actual_model_path,',
                            'ov_model = OVModelForCausalLM.from_pretrained(\n    actual_model_path,\n    export=True,'
                        )
                    found = True
            
            if modified or ('AutoTokenizer.from_pretrained(model_id' in source) or ('actual_model_path' in source):
                cell.source = source
    
    if found:
        print(f"Fixed model paths and loading in {notebook_path}")
    
    return found

def fix_model_availability_check(nb, notebook_path):
    """Fix model availability dropdown to handle case when no models exist yet."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for model availability check that can cause IndexError
            if ('available_models = []' in source and 'available_models[0]' in source):
                found = True
                # Replace the problematic dropdown creation
                source = source.replace(
                    'value=available_models[0],',
                    'value=available_models[0] if available_models else "FP16",'
                )
                # Also ensure FP16 is in the list if nothing is available
                source = source.replace(
                    'available_models = []',
                    'available_models = []\n# LLM Testing Override - ensure at least FP16 option\nif not available_models:\n    available_models.append("FP16")'
                )
                cell.source = source
    
    if found:
        print(f"Fixed model availability check in {notebook_path}")
    
    return found

def patch_optimum_cli_path(nb, notebook_path):
    """Patch optimum-cli commands to use full path from virtual environment."""
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == "code":
            source = cell.source
            
            # Look for imports of llm_config
            if 'from llm_config import' in source or 'import llm_config' in source:
                found = True
                # Add optimum-cli path fix at the beginning
                patch_code = '''# Fix optimum-cli path for virtual environment
import sys
from pathlib import Path

# Patch optimum-cli command to use full path
def patch_optimum_cli():
    try:
        import llm_config
        original_get_optimum_cli_command = llm_config.get_optimum_cli_command
        
        def patched_get_optimum_cli_command(*args, **kwargs):
            command = original_get_optimum_cli_command(*args, **kwargs)
            # Replace optimum-cli with full path
            venv_bin = Path(sys.executable).parent
            optimum_cli_path = venv_bin / "optimum-cli"
            return command.replace("optimum-cli", str(optimum_cli_path), 1)
        
        llm_config.get_optimum_cli_command = patched_get_optimum_cli_command
        print(f"✅ Patched optimum-cli to use full path: {venv_bin / 'optimum-cli'}")
    except Exception as e:
        print(f"⚠️  Could not patch optimum-cli path: {e}")

patch_optimum_cli()

'''
                cell.source = patch_code + source
                break
    
    if found:
        print(f"Added optimum-cli path fix to {notebook_path}")
    
    return nb

def patch_notebook(notebook_path, test_device="CPU", topology=None):
    """Patch a single notebook for LLM testing."""
    print(f"Patching {notebook_path} for topology={topology}, device={test_device}")
    
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Apply patches
    nb = force_tqdm_console_mode(nb, notebook_path)
    nb = prevent_pip_installs(nb, notebook_path)
    nb = patch_optimum_cli_path(nb, notebook_path)
    nb = patch_model_selection(nb, notebook_path, topology)
    nb = redirect_model_directories(nb, notebook_path)
    nb = patch_device_selection(nb, notebook_path, test_device)
    nb = disable_gradio_debug(nb, notebook_path)
    nb = add_environment_info(nb, notebook_path)
    fix_value_attributes(nb, notebook_path)
    fix_model_paths(nb, notebook_path)
    fix_model_availability_check(nb, notebook_path)
    
    # Create test notebook with "test_" prefix (required by validation system)
    notebook_dir = notebook_path.parent
    notebook_name = notebook_path.name
    test_notebook_path = notebook_dir / f"test_{notebook_name}"
    
    # Write the patched notebook as test notebook
    with open(test_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    print(f"[OK] Successfully created test notebook: {test_notebook_path}")
    return test_notebook_path

def main():
    """Main function for LLM notebook patching."""
    parser = argparse.ArgumentParser(description="LLM Notebook Patcher")
    parser.add_argument("notebooks_dir", help="Directory containing notebooks")
    parser.add_argument("-td", "--test_device", default="CPU", help="Test device (CPU/GPU/NPU)")
    parser.add_argument("-t", "--topology", help="LLM topology (phi-3.5/llama-3.2/qwen-7b)")
    parser.add_argument("--notebooks", nargs="+", help="Specific notebooks to patch")
    
    args = parser.parse_args()
    
    notebooks_dir = Path(args.notebooks_dir)
    
    if not notebooks_dir.exists():
        print(f"Error: Directory {notebooks_dir} does not exist")
        return 1
    
    # Determine which notebooks to patch
    if args.notebooks:
        # Patch specific notebooks
        notebooks_to_patch = [notebooks_dir / nb for nb in args.notebooks]
    elif args.topology and args.topology in LLM_NOTEBOOKS:
        # Patch notebooks for specific topology
        notebooks_to_patch = [notebooks_dir / nb for nb in LLM_NOTEBOOKS[args.topology]]
    else:
        # Patch all LLM notebooks
        notebooks_to_patch = []
        for topology_notebooks in LLM_NOTEBOOKS.values():
            notebooks_to_patch.extend([notebooks_dir / nb for nb in topology_notebooks])
    
    # Filter to existing notebooks
    existing_notebooks = [nb for nb in notebooks_to_patch if nb.exists()]
    
    if not existing_notebooks:
        print("No notebooks found to patch!")
        return 1
    
    print(f"Found {len(existing_notebooks)} notebooks to patch:")
    for nb in existing_notebooks:
        print(f"  - {nb}")
    
    # Patch each notebook
    patched_notebooks = []
    for notebook_path in existing_notebooks:
        try:
            test_notebook = patch_notebook(notebook_path, args.test_device, args.topology)
            patched_notebooks.append(test_notebook)
        except Exception as e:
            print(f"[ERROR] Failed to patch {notebook_path}: {e}")
            continue
    
    print(f"\n[OK] Successfully patched {len(patched_notebooks)} notebooks")
    for nb in patched_notebooks:
        print(f"  - {nb}")
    
    return 0

if __name__ == "__main__":
    exit(main())
