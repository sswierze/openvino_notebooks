# LLM Testing Infrastructure

This directory contains automated testing tools for Large Language Model (LLM) notebooks in OpenVINO.

## Quick Start

### Test a specific LLM topology:
```bash
cd .ci
python3 simple_llm_test.py --topology phi-3.5 --device cpu
```

### Test all LLM topologies:
```bash
cd .ci
python3 simple_llm_test.py --device cpu
```

### Create diff reports showing what was patched:
```bash
cd .ci
python3 simple_llm_test.py --topology qwen-7b --device cpu --create-diff
```

## Supported Topologies

- **phi-3.5**: Microsoft Phi-3.5-mini-instruct model
- **llama-3.2**: Llama 3.2 model (uses generate-api notebook)
- **qwen-7b**: Qwen 7B model

## Supported Devices

- **cpu**: CPU inference (default)
- **gpu**: GPU inference 
- **npu**: NPU inference

## What It Does

1. **Environment Setup**: Creates isolated virtual environment with OpenVINO nightly + optimum-intel
2. **Notebook Patching**: Modifies notebooks for testing (disables pip installs, sets model overrides, device configs)
3. **Validation**: Runs notebooks using OpenVINO validation framework
4. **Diff Reports**: Optionally creates visual reports showing what was changed

## Key Features

- ✅ **Repository Clean**: All downloads/cache go to `llm_test_working_dir/` outside repo
- ✅ **Environment Isolated**: Uses dedicated virtual environment
- ✅ **Comprehensive Logging**: Verbose output for debugging
- ✅ **Model Flexibility**: Automatic model override for topology testing
- ✅ **Device Support**: CPU/GPU/NPU device testing

## Output

- Test results printed to console
- Working files in `../../llm_test_working_dir/`
- Diff reports in `../../llm_test_working_dir/diff_report_*.ipynb`

## Advanced Usage

```bash
# Setup environment only
python3 simple_llm_test.py --setup-only

# Specific OpenVINO version
python3 simple_llm_test.py --openvino-version 2024.4.0

# Custom notebooks directory
python3 simple_llm_test.py --notebooks-dir /path/to/notebooks

# Enable debug mode
python3 simple_llm_test.py --debug
```

## Files

- `simple_llm_test.py` - Main test orchestrator
- `patch_notebooks_llm.py` - LLM-specific notebook patcher  
- `create_diff_notebook.py` - Diff report generator
- `validate_notebooks.py` - Extended validation framework

## Example Output

```
🔧 Setting up environment with OpenVINO nightly + optimum-intel latest
📦 Installing 20 packages: openvino-nightly, openvino-tokenizers[transformers]...
✅ OpenVINO               : 2025.4.0-20223-9e7c1b77635
✅ Optimum Intel         : 1.26.0.dev0+95269b4
🚀 Testing phi-3.5 on CPU
📝 Patching notebooks for phi-3.5 on CPU...
✅ Patching successful
🔍 Running validation...
✅ phi-3.5 on CPU: PASSED (60.42s)
📊 Creating diff report for phi-3.5 on cpu...
✅ Diff report created: llm_test_working_dir/diff_report_phi-3.5_cpu.ipynb
```

## Troubleshooting

- **Permission errors**: Ensure write access to parent directory for `llm_test_working_dir/`
- **Network issues**: Check internet connection for model downloads
- **Memory issues**: LLM models require significant RAM (8GB+ recommended)
- **GPU issues**: Ensure proper GPU drivers for GPU testing