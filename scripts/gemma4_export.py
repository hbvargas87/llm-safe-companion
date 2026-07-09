"""Merge Gemma 4 E4B SFT adapter with base model and export to GGUF.

Flow:
    1. Load base gemma-4-E4B-it
    2. Merge SFT adapter → model with child-friendly tone
    3. Convert merged model to GGUF fp16
    4. Quantize to Q4_K_M / Q5_K_M / Q8_0

Usage:
    conda activate kid-ai
    python scripts/gemma4_export.py --step all --quant q4_k_m
"""

import os
import sys
import subprocess
import argparse
import torch
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "unsloth/gemma-4-E4B-it"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_sft" / "final"
MERGED_DIR = PROJECT_ROOT / "outputs" / "gemma4_4b_merged"

# GGUF output files
GGUF_FP16_OUTPUT = MERGED_DIR / "gemma4-4b-kidsafe.gguf"
GGUF_Q4_OUTPUT = MERGED_DIR / "gemma4-4b-kidsafe.Q4_K_M.gguf"
GGUF_Q5_OUTPUT = MERGED_DIR / "gemma4-4b-kidsafe.Q5_K_M.gguf"
GGUF_Q8_OUTPUT = MERGED_DIR / "gemma4-4b-kidsafe.Q8_0.gguf"

# llama.cpp source (has convert_hf_to_gguf.py)
LLAMA_CPP_SRC_DIR = PROJECT_ROOT.parent / "llama-cpp-src"

# llama.cpp pre-compiled binaries (has llama-quantize.exe, llama-server.exe, etc.)
LLAMA_CPP_BIN_DIR = PROJECT_ROOT.parent / "llama-cpp"

QUANT_OPTIONS = {
    "q4_k_m": {"output": GGUF_Q4_OUTPUT, "desc": "~3.5 GB, good balance", "vram": "~5 GB"},
    "q5_k_m": {"output": GGUF_Q5_OUTPUT, "desc": "~4.0 GB, better quality", "vram": "~5.5 GB"},
    "q8_0":   {"output": GGUF_Q8_OUTPUT, "desc": "~5.5 GB, near fp16 quality", "vram": "~7 GB"},
}


def merge_adapter():
    """Merge SFT adapter with base Gemma 4 model.

    Flow:
        1. Load base gemma-4-E4B-it
        2. Merge SFT adapter → model with child-friendly tone
        3. Save final merged model
    """
    print("=" * 60)
    print("  STEP 1: Merge SFT Adapter with Gemma 4 E4B")
    print("=" * 60)

    # Verify adapter exists
    if not SFT_ADAPTER.exists():
        print(f"ERROR: SFT adapter not found: {SFT_ADAPTER}")
        print(f"  Make sure you ran gemma4_train.py first.")
        sys.exit(1)

    print(f"  Base:        {BASE_MODEL}")
    print(f"  SFT Adapter: {SFT_ADAPTER}")
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
    print("\n  [1/3] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("  Base model loaded.")

    # Step 2: Merge SFT adapter
    print("\n  [2/3] Merging SFT adapter (child-friendly tone)...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    model = model.merge_and_unload()
    print("  SFT adapter merged successfully.")

    # Step 3: Save final model
    print(f"\n  [3/3] Saving merged model to {MERGED_DIR}...")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    # Copy chat template from adapter
    chat_template_src = SFT_ADAPTER / "chat_template.jinja"
    if chat_template_src.exists():
        import shutil
        shutil.copy(chat_template_src, MERGED_DIR / "chat_template.jinja")
        print(f"  Copied chat template from adapter")
    else:
        print(f"  WARNING: No chat_template.jinja found in adapter")

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
    _quantize("Q4_K_M", GGUF_Q4_OUTPUT)


def quantize_to_q5():
    """Quantize fp16 GGUF to Q5_K_M using llama-quantize.exe."""
    _quantize("Q5_K_M", GGUF_Q5_OUTPUT)


def quantize_to_q8():
    """Quantize fp16 GGUF to Q8_0 using llama-quantize.exe."""
    _quantize("Q8_0", GGUF_Q8_OUTPUT)


def _quantize(quant_name: str, output_path: Path):
    """Run quantization with given quantization type."""
    print(f"\n  Quantizing to {quant_name}...")

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

    cmd = [
        str(quantize_exe),
        str(GGUF_FP16_OUTPUT),
        str(output_path),
        quant_name,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  ERROR: Quantization failed (exit code {result.returncode})")
        sys.exit(1)

    if output_path.exists():
        size_gb = output_path.stat().st_size / (1024**3)
        opt = QUANT_OPTIONS.get(quant_name.lower(), {})
        print(f"  Done. Size: {size_gb:.2f} GB  |  VRAM needed: {opt.get('vram', 'N/A')}")
    else:
        print(f"  ERROR: Output not created at {output_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Gemma 4 SFT adapter and export to GGUF"
    )
    parser.add_argument(
        "--step",
        choices=["merge", "convert", "quantize", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--quant",
        choices=["q4_k_m", "q5_k_m", "q8_0", "all"],
        default="q4_k_m",
        help="Quantization type (default: q4_k_m)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Gemma 4 E4B - Model Export Tool")
    print("=" * 60)
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Source:  {LLAMA_CPP_SRC_DIR}")
    print(f"  Bin:     {LLAMA_CPP_BIN_DIR}")
    print()

    if args.step in ("merge", "all"):
        merged_dir = merge_adapter()
    else:
        merged_dir = MERGED_DIR

    if args.step in ("convert", "all"):
        convert_to_gguf_fp16(merged_dir)

    if args.step in ("quantize", "all"):
        if args.quant in ("q4_k_m", "all"):
            quantize_to_q4()
        if args.quant in ("q5_k_m", "all"):
            quantize_to_q5()
        if args.quant in ("q8_0", "all"):
            quantize_to_q8()

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("=" * 60)

    # Show available outputs
    print(f"\n  Available GGUF files:")
    for name, opt in QUANT_OPTIONS.items():
        path = opt["output"]
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"    {name}: {path.name} ({size_gb:.2f} GB) — {opt['desc']}")
        else:
            print(f"    {name}: {opt['desc']} — NOT YET GENERATED")

    print(f"\n  LM Studio: copy the .gguf to your models folder")
    print(f"  llama.cpp: llama-server.exe -m \"{GGUF_Q4_OUTPUT}\" -c 2048")


if __name__ == "__main__":
    main()
