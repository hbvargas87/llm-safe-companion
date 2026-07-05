"""Merge BOTH SFT + DPO adapters with base model and export to GGUF.

Flow:
    1. Load base Qwen2-7B-Instruct
    2. Merge SFT adapter → model with child-friendly tone
    3. Merge DPO adapter → model with safety alignment
    4. Convert merged model to GGUF fp16
    5. Quantize to Q4_K_M

Usage:
    conda activate kid-ai
    python scripts/merge_and_export.py --step all
"""

import os
import sys
import subprocess
import argparse
import torch
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SFT_ADAPTER = PROJECT_ROOT / "outputs" / "qwen2_7b_sft" / "final"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "qwen2_7b_dpo" / "final"
BASE_MODEL = "Qwen/Qwen2-7B-Instruct"
MERGED_DIR = PROJECT_ROOT / "outputs" / "qwen2_7b_merged"

# GGUF output files
GGUF_FP16_OUTPUT = PROJECT_ROOT / "outputs" / "qwen2_7b_merged" / "qwen2-7b-kidsafe.gguf"
GGUF_Q4_OUTPUT = PROJECT_ROOT / "outputs" / "qwen2_7b_merged" / "qwen2-7b-kidsafe.Q4_K_M.gguf"

# llama.cpp source (has convert_hf_to_gguf.py)
LLAMA_CPP_SRC_DIR = PROJECT_ROOT.parent / "llama-cpp-src"

# llama.cpp pre-compiled binaries (has llama-quantize.exe, llama-server.exe, etc.)
LLAMA_CPP_BIN_DIR = PROJECT_ROOT.parent / "llama-cpp"

QUANT_OPTIONS = {
    "q4_k_m": {"desc": "~4.9 GB, good balance", "vram": "~7 GB"},
    "q5_k_m": {"desc": "~5.5 GB, better quality", "vram": "~7.5 GB"},
    "q8_0":   {"desc": "~7.7 GB, near fp16 quality", "vram": "~9 GB"},
}


def merge_adapters():
    """Merge BOTH SFT and DPO adapters in sequence.

    Flow:
        1. Load base Qwen2-7B-Instruct
        2. Merge SFT adapter → model with child-friendly tone
        3. Merge DPO adapter → model with safety alignment
        4. Save final merged model
    """
    print("=" * 60)
    print("  STEP 1: Merge SFT + DPO Adapters")
    print("=" * 60)

    # Verify both adapters exist
    if not SFT_ADAPTER.exists():
        print(f"ERROR: SFT adapter not found: {SFT_ADAPTER}")
        sys.exit(1)
    if not DPO_ADAPTER.exists():
        print(f"ERROR: DPO adapter not found: {DPO_ADAPTER}")
        sys.exit(1)

    print(f"  Base:       {BASE_MODEL}")
    print(f"  SFT Adapter: {SFT_ADAPTER}")
    print(f"  DPO Adapter: {DPO_ADAPTER}")
    print(f"  Output:      {MERGED_DIR}")

    # Check VRAM
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    else:
        print("  WARNING: No GPU detected - CPU will be very slow")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Step 1: Load base model
    print("\n  [1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("  Base model loaded.")

    # Step 2: Merge SFT adapter
    print("\n  [2/4] Merging SFT adapter (child-friendly tone)...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    model = model.merge_and_unload()
    print("  SFT adapter merged successfully.")

    # Step 3: Merge DPO adapter
    print("\n  [3/4] Merging DPO adapter (safety alignment)...")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER)
    model = model.merge_and_unload()
    print("  DPO adapter merged successfully.")

    # Step 4: Save final model
    print(f"\n  [4/4] Saving final merged model to {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    # Copy chat template from DPO adapter (last one wins)
    for adapter_path in [SFT_ADAPTER, DPO_ADAPTER]:
        chat_template_src = adapter_path / "chat_template.jinja"
        if chat_template_src.exists():
            import shutil
            shutil.copy(chat_template_src, MERGED_DIR / "chat_template.jinja")
            print(f"  Copied chat template from {adapter_path.name}")
            break

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_size = sum(f.stat().st_size for f in MERGED_DIR.rglob('*') if f.is_file())
    print(f"\n  Final model size: {total_size / (1024**3):.2f} GB")
    print("  Merge complete!")
    return MERGED_DIR


def convert_to_gguf_fp16(merged_dir: Path):
    """Convert merged HF model to GGUF fp16 using convert_hf_to_gguf.py."""
    print("\n" + "=" * 60)
    print("  STEP 2: Convert to GGUF fp16")
    print("=" * 60)

    if not merged_dir.exists():
        print(f"ERROR: Merged model not found: {merged_dir}")
        sys.exit(1)

    if not LLAMA_CPP_SRC_DIR.exists():
        print(f"ERROR: llama.cpp source not found: {LLAMA_CPP_SRC_DIR}")
        print(f"  Expected: {LLAMA_CPP_SRC_DIR / 'convert_hf_to_gguf.py'}")
        sys.exit(1)

    convert_script = LLAMA_CPP_SRC_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: convert_hf_to_gguf.py not found at: {convert_script}")
        sys.exit(1)

    print(f"  Source: {LLAMA_CPP_SRC_DIR}")
    print(f"  Input:  {merged_dir}")
    print(f"  Output: {GGUF_FP16_OUTPUT}")

    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(GGUF_FP16_OUTPUT),
        "--outtype", "f16",
    ]

    print(f"\n  Running conversion (this may take 5-10 min)...")
    result = subprocess.run(cmd, cwd=str(LLAMA_CPP_SRC_DIR))

    if result.returncode != 0:
        print(f"  ERROR: Conversion failed (exit code {result.returncode})")
        sys.exit(1)

    if GGUF_FP16_OUTPUT.exists():
        size_gb = GGUF_FP16_OUTPUT.stat().st_size / (1024**3)
        print(f"  Done. Size: {size_gb:.2f} GB")
    else:
        print(f"  ERROR: Output not created at {GGUF_FP16_OUTPUT}")
        sys.exit(1)


def quantize_to_q4():
    """Quantize fp16 GGUF to Q4_K_M using llama-quantize.exe."""
    print("\n" + "=" * 60)
    print("  STEP 3: Quantize to Q4_K_M")
    print("=" * 60)

    if not LLAMA_CPP_BIN_DIR.exists():
        print(f"ERROR: llama.cpp binaries not found: {LLAMA_CPP_BIN_DIR}")
        sys.exit(1)

    quantize_exe = LLAMA_CPP_BIN_DIR / "llama-quantize.exe"
    if not quantize_exe.exists():
        print(f"ERROR: llama-quantize.exe not found at: {quantize_exe}")
        sys.exit(1)

    if not GGUF_FP16_OUTPUT.exists():
        print(f"ERROR: fp16 GGUF not found at {GGUF_FP16_OUTPUT}")
        print("  Run STEP 2 first.")
        sys.exit(1)

    print(f"  Binary:  {quantize_exe}")
    print(f"  Input:   {GGUF_FP16_OUTPUT}")
    print(f"  Output:  {GGUF_Q4_OUTPUT}")

    cmd = [
        str(quantize_exe),
        str(GGUF_FP16_OUTPUT),
        str(GGUF_Q4_OUTPUT),
        "Q4_K_M",
    ]

    print(f"\n  Quantizing (this may take 2-5 min)...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  ERROR: Quantization failed (exit code {result.returncode})")
        sys.exit(1)

    if GGUF_Q4_OUTPUT.exists():
        size_gb = GGUF_Q4_OUTPUT.stat().st_size / (1024**3)
        print(f"  Done. Size: {size_gb:.2f} GB  |  VRAM needed: ~7 GB")
    else:
        print(f"  ERROR: Output not created at {GGUF_Q4_OUTPUT}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter and export to GGUF (Q4_K_M)"
    )
    parser.add_argument(
        "--step",
        choices=["merge", "convert", "quantize", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Kid-Safe LLM - Model Export Tool")
    print("=" * 60)
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Source:  {LLAMA_CPP_SRC_DIR}")
    print(f"  Bin:     {LLAMA_CPP_BIN_DIR}")
    print()

    if args.step in ("merge", "all"):
        merged_dir = merge_adapters()
    else:
        merged_dir = MERGED_DIR

    if args.step in ("convert", "all"):
        convert_to_gguf_fp16(merged_dir)

    if args.step in ("quantize", "all"):
        quantize_to_q4()

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print(f"\n  GGUF file: {GGUF_Q4_OUTPUT}")
    print(f"\n  LM Studio: copy the .gguf to your models folder")
    print(f"  llama.cpp: llama-server.exe -m \"{GGUF_Q4_OUTPUT}\" -c 2048")
if __name__ == "__main__":
    main()

