#!/usr/bin/env python3
"""
Simple LLM Notebook Test Runner

Uses the existing OpenVINO notebook validation infrastructure to test
the 3 LLM topologies you specified.

Usage examples:
  # Test all topologies on CPU
  python simple_llm_test.py --device cpu
  
  # Test specific topology on GPU  
  python simple_llm_test.py --topology qwen-7b --device gpu
  
  # Test with specific OpenVINO version
  python simple_llm_test.py --openvino-version 2024.4.0
  
  # Create diff reports
  python simple_llm_test.py --topology phi-3.5 --create-diff
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Global variable to store venv python path
venv_python_path = None


def print_library_versions():
    """Print versions of all OpenVINO-related libraries."""
    libraries = [
        ("openvino", "OpenVINO"),
        ("openvino_tokenizers", "OpenVINO Tokenizers"),
        ("openvino_genai", "OpenVINO GenAI"),
        ("optimum.intel", "Optimum Intel"),
        ("optimum", "Optimum"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("onnx", "ONNX"),
        ("gradio", "Gradio"),
        ("ipywidgets", "IPyWidgets"),
        ("treon", "Treon")
    ]
    
    print("\n📦 Installed Library Versions:")
    print("=" * 50)
    
    # Use venv python if available, otherwise use system python
    python_exec = venv_python_path if venv_python_path else sys.executable
    
    for package_name, display_name in libraries:
        try:
            # Try different common ways to get version
            version_checks = [
                f"import {package_name}; print({package_name}.__version__)",
                f"import {package_name}; print(getattr({package_name}, '__version__', 'unknown'))",
                f"import importlib.metadata; print(importlib.metadata.version('{package_name.replace('_', '-')}'))",
                f"import pkg_resources; print(pkg_resources.get_distribution('{package_name.replace('_', '-')}').version)"
            ]
            
            version = None
            for check in version_checks:
                try:
                    result = subprocess.run(
                        [python_exec, "-c", check],
                        capture_output=True, text=True, check=True
                    )
                    version = result.stdout.strip()
                    if version and version != 'unknown':
                        break
                except:
                    continue
            
            if version and version != 'unknown':
                print(f"✅ {display_name:<20}: {version}")
            else:
                print(f"❌ {display_name:<20}: Not installed")
        except (subprocess.CalledProcessError, Exception):
            print(f"❌ {display_name:<20}: Not installed")
    
    print("=" * 50)


def setup_environment(openvino_version="nightly", optimum_intel_version="latest", reuse_venv=False):
    """Setup environment with OpenVINO + optimum-intel using virtual environment."""
    global venv_python_path
    
    venv_path = Path("llm_test_venv")
    
    # Check if we should reuse existing venv
    if reuse_venv and venv_path.exists():
        print(f"🔁 Reusing existing {venv_path} (skipping package installation)")
        
        # Get venv python executable
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"
        
        # Update global python executable
        venv_python_path = str(venv_python)
        
        # Print library versions and return
        print_library_versions()
        return True
    
    print(f"🔧 Setting up environment with OpenVINO {openvino_version} + optimum-intel {optimum_intel_version}")
    
    # Set environment variables to disable interactive widgets when running headless
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    os.environ["IPYWIDGETS_DEFAULT_BACKEND"] = "null"
    
    # Force tqdm to use text output instead of notebook widgets
    os.environ["TQDM_DISABLE"] = "0"  # Don't completely disable, just force text mode
    os.environ["TQDM_NCOLS"] = "80"   # Set fixed width for text progress bars
    os.environ["TQDM_MININTERVAL"] = "1"  # Update every second
    
    # Force huggingface_hub to use text progress bars
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Keep progress but force text
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Disable Jupyter context detection in libraries
    os.environ["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib
    os.environ["JUPYTER_RUNTIME_DIR"] = "/tmp"
    
    # Force IPython to not detect notebook environment
    os.environ["JPY_PARENT_PID"] = ""
    
    # Prevent tqdm from detecting notebook environment
    os.environ["TQDM_NOTEBOOK"] = "false"
    
    # Force HuggingFace libraries to use console progress bars
    os.environ["_TQDM_CONSOLE"] = "1"
    
    # Make sure we're not in a Jupyter context
    os.environ.pop("JUPYTER_RUNTIME_DIR", None)
    os.environ.pop("JUPYTER_CONFIG_DIR", None)
    
    # Create virtual environment (venv_path already defined above)
    if venv_path.exists():
        print("🔄 Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    print("🏗️  Creating virtual environment...")
    result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                          capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"❌ Failed to create virtual environment: {result.stderr}")
        return False
    
    # Get venv python executable
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
        venv_pip = venv_path / "Scripts" / "pip.exe"
    else:
        venv_python = venv_path / "bin" / "python"
        venv_pip = venv_path / "bin" / "pip"
    
    # Update global python executable for subsequent operations
    venv_python_path = str(venv_python)
    
    # Upgrade pip first
    print("⬆️  Upgrading pip in virtual environment...")
    result = subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], 
                          capture_output=False, text=True, check=False)
    if result.returncode != 0:
        print(f"⚠️  Warning: pip upgrade had issues")
    
    # Install packages
    if openvino_version == "nightly":
        ov_package = "openvino-nightly"
    else:
        ov_package = f"openvino=={openvino_version}"
    
    # We'll install optimum-intel from git (like the notebook) instead of using version parameter
    
        # Core packages needed for notebook execution - matching notebook requirements exactly
    packages = [
        ov_package, 
        "openvino-tokenizers[transformers]", "openvino-genai",
        "torch>=2.1", "transformers==4.53.3", "gradio>=4.19", "ipywidgets", 
        "nncf==2.15.0", "datasets<4.0.0", "accelerate", "huggingface-hub>=0.26.5",
        "einops", "transformers_stream_generator", "tiktoken", "bitsandbytes",
        "treon", "nbformat", "notebook", "jupyter", "ipykernel"
    ]
    
    # Install optimum-intel from git (like the notebook does)
    git_packages = ["git+https://github.com/huggingface/optimum-intel.git"]
    
    # First uninstall any existing optimum packages (like the notebook does)
    print("Uninstalling existing optimum packages...")
    result = subprocess.run([str(venv_python), "-m", "pip", "uninstall", "-y", "optimum", "optimum-intel"], 
                          capture_output=False, text=True, check=False)
    print(f"   Uninstall completed with code: {result.returncode}")
    
    # Install packages with extra index for nightly builds
    print("Installing core packages...")
    print(f"📦 Installing {len(packages)} packages: {', '.join(packages[:5])}{'...' if len(packages) > 5 else ''}")
    install_cmd = [str(venv_python), "-m", "pip", "install", "--pre"] + packages + [
        "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu"
    ]
    result = subprocess.run(install_cmd, capture_output=False, text=True, check=False)
    print(f"   Core packages installation completed with code: {result.returncode}")
    if result.returncode != 0:
        print(f"⚠️  Warning: Core packages installation had issues")
    
    # Install git packages separately
    for package in git_packages:
        print(f"Installing {package}...")
        result = subprocess.run([str(venv_python), "-m", "pip", "install", package], 
                              capture_output=False, text=True, check=False)
        print(f"   {package} installation completed with code: {result.returncode}")
        if result.returncode != 0:
            print(f"⚠️  Warning: {package} installation had issues")
    
    # Verify installation and print comprehensive version info
    print_library_versions()
    return True



def create_diff_report(topology, device, notebooks_dir):
    """Create a diff report notebook showing what was patched."""
    try:
        # Import the diff creation functionality
        import create_diff_notebook
        
        print(f"\n📊 Creating diff report for {topology} on {device}...")
        
        notebooks_path = Path(notebooks_dir)
        output_path = create_diff_notebook.create_diff_notebook(notebooks_path, topology, device)
        
        print(f"✅ Diff report created: {output_path}")
        print(f"📖 Open with: jupyter notebook {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create diff report: {e}")
        return False


def run_llm_test(topology, device="cpu", notebooks_dir=".", debug=False):
    """Run test for specific LLM topology."""
    
    # Define topology to notebook mapping
    topologies = {
        "phi-3.5": ["notebooks/llm-chatbot/llm-chatbot.ipynb"],
        "llama-3.2": ["notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb"], 
        "qwen-7b": ["notebooks/llm-chatbot/llm-chatbot.ipynb"]
    }
    
    if topology not in topologies:
        print(f"❌ Unknown topology: {topology}")
        return False
    
    notebooks_path = Path(notebooks_dir)
    validation_script = Path("validate_notebooks.py")  # We're already in .ci directory
    patch_script = Path("patch_notebooks_llm.py")      # We're already in .ci directory
    
    if not validation_script.exists():
        print(f"❌ Validation script not found: {validation_script}")
        return False
    
    print(f"🚀 Testing {topology} on {device.upper()}")
    
    # Step 1: Patch notebooks for device and topology
    print(f"📝 Patching notebooks for {topology} on {device.upper()}...")
    
    # Use venv python if available
    python_exec = venv_python_path if venv_python_path else str(Path(sys.executable).absolute())
    
    patch_cmd = [
        python_exec, str(patch_script.absolute()),
        str(notebooks_path),
        "-td", device.upper(),
        "-t", topology
    ]
    
    result = subprocess.run(patch_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"❌ Patching failed: {result.stderr}")
        return False
    
    print("✅ Patching successful")
    
    # Step 2: Run validation on original notebooks (not test notebooks)
    print("🔍 Running validation...")
    original_notebooks = topologies[topology]
    
    # Use original notebook paths for test list
    test_notebook_paths = original_notebooks
    
    # Create temporary test list file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for notebook in test_notebook_paths:
            f.write(notebook + '\n')
        test_list_file = f.name
    
    print("📋 Test notebooks to validate:")
    for notebook in test_notebook_paths:
        print(f"  - {notebook}")
    
    try:
        validate_cmd = [
            venv_python_path if venv_python_path else str(Path(sys.executable).absolute()), 
            str(validation_script.absolute()),
            "--device", device,
            "--os", "ubuntu-22.04",  # Add OS specification
            "--python", "3.11",      # Add Python version
            "--timeout", "1800",  # 30 minutes
            "--test_list", test_list_file
        ]
        
        print(f"\n🚀 Running validation command:")
        print(f"   {' '.join(validate_cmd)}")
        print(f"📁 Working directory: {os.getcwd()}")
        print("📊 Watch for detailed cell execution progress below...\n")
        
        # Run validation framework with direct output (no capture for treon visibility)
        print(f"\n🚀 Running validation:")
        print(f"Command: {' '.join(validate_cmd)}")
        print(f"Working directory: {os.getcwd()}")
        print("=" * 60)
        
        result = subprocess.run(validate_cmd, check=False)
        exit_code = result.returncode
        
    finally:
        # Clean up temp file
        os.unlink(test_list_file)
    
    if exit_code == 0:
        print(f"✅ {topology} on {device}: PASSED")
        return True
    else:
        print(f"❌ {topology} on {device}: FAILED")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple LLM Topology Tester")
    parser.add_argument("--topology", choices=["phi-3.5", "llama-3.2", "qwen-7b"],
                       help="Topology to test (if not specified, tests all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "npu"],
                       help="Device to test on")
    parser.add_argument("--openvino-version", default="nightly",
                       help="OpenVINO version")
    parser.add_argument("--optimum-intel-version", default="latest",
                       help="Optimum Intel version (default: latest)")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup environment")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with verbose cell execution logging")
    parser.add_argument("--notebooks-dir", default="..",
                       help="Path to notebooks directory")
    parser.add_argument("--create-diff", action="store_true",
                       help="Create diff report showing what was patched")
    parser.add_argument("--reuse-venv", action="store_true",
                       help="Reuse existing virtual environment without reinstalling packages")
    
    args = parser.parse_args()
    
    # Setup environment
    success = setup_environment(args.openvino_version, args.optimum_intel_version, args.reuse_venv)
    if not success:
        print("❌ Environment setup failed")
        sys.exit(1)
    
    if args.setup_only:
        print("✅ Environment setup complete")
        return
    
    # Run tests
    topologies_to_test = [args.topology] if args.topology else ["phi-3.5", "llama-3.2", "qwen-7b"]
    
    results = []
    for topology in topologies_to_test:
        success = run_llm_test(topology, args.device, args.notebooks_dir, args.debug)
        results.append((topology, success))
        
        # Create diff report if requested and test was successful
        if args.create_diff and success:
            create_diff_report(topology, args.device, args.notebooks_dir)
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for topology, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {topology}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()