#!/usr/bin/env python3
"""
Export Gemma 4 E4B SFT + DPO to GGUF using Unsloth's built-in export.

Based on: https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf

Pipeline:
    1. Load base model (4-bit via Unsloth)
    2. Apply SFT adapter (4-bit)
    3. Merge SFT (in-memory)
    4. Apply DPO adapter (4-bit)
    5. Merge DPO (in-memory)
    6. Export to GGUF using Unsloth's save_pretrained_gguf()

Usage:
    python scripts/export_gguf_gemma4.py q4_k_m
    python scripts/export_gguf_gemma4.py q5_k_m
    python scripts/export_gguf_gemma4.py q8_0
    python scripts/export_gguf_gemma4.py f16
"""

import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
BASE_MODEL = "unsloth/gemma-4-E4B-it"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_sft" / "final"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_dpo" / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gemma4-4b-kidsafe"


def clean_model_for_export(model):
    """Aggressively clean model from Unsloth artifacts before saving."""
    print("  Aggressively cleaning model...")

    # Remove weight_conversions from config
    if hasattr(model, 'config') and hasattr(model.config, 'weight_conversions'):
        model.config.weight_conversions = None

    # Remove ALL problematic attributes from config
    for attr in list(vars(model.config)):
        if any(kw in attr.lower() for kw in ['conversion', 'quant', 'bnb', 'unsloth', 'pre_quant', 'post_quant']):
            try:
                delattr(model.config, attr)
            except Exception:
                pass

    # Convert to bf16 to remove all quant artifacts
    print("  Converting to bf16...")
    model = model.to(torch.bfloat16)

    # Clear hooks
    if hasattr(model, '_forward_hooks'):
        model._forward_hooks.clear()
    if hasattr(model, '_forward_pre_hooks'):
        model._forward_pre_hooks.clear()

    # Reset compiled cache
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, '_compiled_call'):
            inner._compiled_call = None
        # Also check for weight_conversions in inner model
        if hasattr(inner, 'config') and hasattr(inner.config, 'weight_conversions'):
            inner.config.weight_conversions = None

    # Remove weight_conversions from state_dict
            state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if 'weight_conversions' in key or 'quant_state' in key or 'bnb' in key:
            del state_dict[key]

    print("  ✓ Aggressively cleaned")
    return model


def main():
    quant = sys.argv[1] if len(sys.argv) > 1 else "q4_k_m"

    print("=" * 60)
    print("  GEMMA 4 E4B - GGUF EXPORT (Unsloth)")
    print("=" * 60)
    print(f"  Quantization: {quant}")
    print(f"  Output: {OUTPUT_DIR}")

    # Check adapters
    if not SFT_ADAPTER.exists():
        print(f"ERROR: SFT adapter not found: {SFT_ADAPTER}")
        sys.exit(1)
    if not DPO_ADAPTER.exists():
        print(f"ERROR: DPO adapter not found: {DPO_ADAPTER}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: No GPU")
        sys.exit(1)

    # Step 1: Load base model
    print("\n[1/5] Loading base model (4-bit)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Base loaded. VRAM: {vram:.2f} GB")

    # Step 2: Apply SFT adapter
    print("\n[2/5] Applying SFT adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))
    print("  SFT LoRA applied.")

    # Step 3: Merge SFT
    print("  Merging SFT...")
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  SFT merged. VRAM: {vram:.2f} GB")

    # Step 4: Apply DPO adapter
    print("\n[3/5] Applying DPO adapter...")
    model = PeftModel.from_pretrained(model, str(DPO_ADAPTER))
    print("  DPO LoRA applied.")

    # Step 5: Merge DPO
    print("  Merging DPO...")
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  DPO merged. VRAM: {vram:.2f} GB")

    # Step 5.5: Clean model
    print("\n  Cleaning model...")
    model = clean_model_for_export(model)

    # Step 6: Export to GGUF (bypassing transformers save_pretrained)
    print(f"\n[4/5] Exporting to GGUF ({quant})...")
    print("  Bypassing transformers save_pretrained (known bug with Unsloth)...")
    print("  Writing safetensors directly...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    # Step 6a: Get clean state_dict
    print("  Extracting state_dict...")
    state_dict = model.state_dict()

    # Remove any conversion/quant artifacts from state_dict
    keys_to_remove = []
    for key in state_dict.keys():
        if any(kw in key.lower() for kw in [
            'weight_conversions', 'quant_state', 'bnb', 'post_quant',
            'pre_quant', 'nested_quant_map', 'quant_map', 'absmax',
            '_extra_state'
        ]):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state_dict[key]
    print(f"  Removed {len(keys_to_remove)} artifact keys from state_dict")
    # Step 6b: Save model weights directly
    print("  Writing model.safetensors...")
    save_file(state_dict, str(OUTPUT_DIR / "model.safetensors"))

    # Step 6c: Build and save config manually
    print("  Writing config.json...")
    config = model.config
    config_dict = config.to_dict()

    # Remove ALL problematic keys
    keys_to_remove = []
    for key in config_dict.keys():
        if any(kw in key.lower() for kw in [
            'conversion', 'unsloth', 'model_name', 'dtype',
            'post_quant', 'pre_quant'
        ]):
            keys_to_remove.append(key)
    for key in keys_to_remove:
                    config_dict.pop(key, None)
    print(f"  Removed {len(keys_to_remove)} problematic keys from config")

    with open(str(OUTPUT_DIR / "config.json"), 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)

    # Step 6d: Save tokenizer
    print("  Saving tokenizer...")
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Step 6e: Copy chat template if present
    chat_template_src = SFT_ADAPTER / "chat_template.jinja"
    if chat_template_src.exists():
        import shutil
        shutil.copy(chat_template_src, OUTPUT_DIR / "chat_template.jinja")
    print(f"  ✓ Model saved (safetensors) to {OUTPUT_DIR}")
    print("  Note: This is safetensors, not GGUF.")
    # Show output
    print(f"\n[5/5] Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name}: {size_mb:.2f} MB")

    print(f"\n{'=' * 60}")
    print("  DONE!")
    print(f"{'=' * 60}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"\n  To use with LM Studio:")
    print(f"    1. Copy .gguf files to LM Studio models folder")
    print(f"    2. Select the model in LM Studio")
    print(f"\n  To use with llama.cpp:")
    print(f"    llama-server.exe -m \"{OUTPUT_DIR}/*.gguf\" -c 2048")
    print()


if __name__ == "__main__":
    main()

