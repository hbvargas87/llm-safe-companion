"""Check adapter tensors for NaN and print summary."""
import torch
import os
import json
from safetensors.torch import load_file as load_safetensors

ADAPTER_PATH = r"F:\LLM\Proyectos\Stardusts\outputs\qwen2_7b_sft\final\adapter_model.safetensors"
CONFIG_PATH = ADAPTER_PATH.replace("adapter_model.safetensors", "adapter_config.json")

print("=== ADAPTER VERIFICATION ===\n")

# File exists?
print(f"File exists: {os.path.exists(ADAPTER_PATH)}")
if os.path.exists(ADAPTER_PATH):
    size_mb = os.path.getsize(ADAPTER_PATH) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
else:
    print("Adapter file not found!")
    exit(1)

# Load
print("\nLoading state dict...")
state_dict = load_safetensors(ADAPTER_PATH)
print(f"Total tensors: {len(state_dict)}")

# NaN check
nan_tensors = []
for name, tensor in state_dict.items():
    if torch.isnan(tensor).any():
        nan_tensors.append(name)

print(f"\nTensors with NaN: {len(nan_tensors)}/{len(state_dict)}")

if len(nan_tensors) == 0:
    print("RESULT: ADAPTER IS CLEAN - NO NaN VALUES!\n")
else:
    print(f"PROBLEM: {len(nan_tensors)} tensors contain NaN:\n")
    for t in nan_tensors[:15]:
        print(f"  - {t}")
    print()

# Sample tensors
print("--- Sample tensor stats ---")
for i, (name, tensor) in enumerate(state_dict.items()):
    if i >= 5:
        break
    print(f"\n  {name}")
    print(f"    dtype: {tensor.dtype}, shape: {tensor.shape}")
    print(f"    min: {tensor.min().item():.6f}")
    print(f"    max: {tensor.max().item():.6f}")
    print(f"    mean: {tensor.mean().item():.6f}")
    print(f"    std: {tensor.std().item():.6f}")

# Config info
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    print(f"\n--- Adapter Config ---")
    print(f"  base_model: {config.get('base_model_name_or_path', 'N/A')}")
    print(f"  quant_method: {config.get('quant_method', 'N/A')}")
    print(f"  r: {config.get('r', 'N/A')}")
    print(f"  target_modules: {config.get('target_modules', 'N/A')}")
    print(f"  inference_mode: {config.get('inference_mode', 'N/A')}")

