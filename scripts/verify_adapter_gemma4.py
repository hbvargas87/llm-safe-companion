"""Check Gemma 4 adapter tensors for NaN and print summary.

Usage:
    conda activate kid-ai
    python scripts/verify_adapter_gemma4.py

Or with custom path:
    python scripts/verify_adapter_gemma4.py --path outputs/gemma4_4b_sft/final/
"""

import torch
import os
import json
import argparse
from pathlib import Path
from safetensors.torch import load_file as load_safetensors


def verify_adapter(adapter_path: str):
    """Verify adapter for NaN values and print summary."""
    adapter_path = Path(adapter_path)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"

    print("=== GEMMA 4 ADAPTER VERIFICATION ===\n")

    # File exists?
    print(f"File exists: {safetensors_path.exists()}")
    if safetensors_path.exists():
        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
    else:
        print("Adapter file not found!")
        print(f"  Expected: {safetensors_path}")
        exit(1)

    # Load
    print("\nLoading state dict...")
    state_dict = load_safetensors(str(safetensors_path))
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
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n--- Adapter Config ---")
        print(f"  base_model: {config.get('base_model_name_or_path', 'N/A')}")
        print(f"  quant_method: {config.get('quant_method', 'N/A')}")
        print(f"  r: {config.get('r', 'N/A')}")
        print(f"  lora_alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"  target_modules: {config.get('target_modules', 'N/A')}")
        print(f"  inference_mode: {config.get('inference_mode', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Gemma 4 adapter for NaN values"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="outputs/gemma4_4b_sft/final/",
        help="Path to adapter directory (default: outputs/gemma4_4b_sft/final/)",
    )
    args = parser.parse_args()
    verify_adapter(args.path)


if __name__ == "__main__":
    main()
