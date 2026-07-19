"""Merge Gemma 4 E4B SFT + DPO adapters and export to GGUF.

Flow:
    1. Load base gemma-4-E4B-it
    2. Apply SFT adapter (child-friendly tone) as LoRA
    3. Merge SFT LoRA into base (in-memory, no save)
    4. Apply DPO adapter (safety alignment) as LoRA
    5. Merge DPO LoRA into base (in-memory, no save)
    6. Convert directly to GGUF fp16 (no intermediate save)
    7. Quantize to Q4_K_M / Q5_K_M / Q8_0

IMPORTANT: Do NOT save merged model to disk — 4-bit merge corrupts safetensors.

Usage:
    python scripts/gemma4_export.py --step all --quant q4_k_m
    python scripts/gemma4_export.py --step all --quant q5_k_m
    python scripts/gemma4_export.py --step all --quant q8_0
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
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_dpo" / "final"

# GGUF output files (directly in outputs, no temp merged dir)
GGUF_FP16_OUTPUT = PROJECT_ROOT / "outputs" / "gemma4-4b-kidsafe.gguf"
GGUF_Q4_OUTPUT = PROJECT_ROOT / "outputs" / "gemma4-4b-kidsafe.Q4_K_M.gguf"
GGUF_Q5_OUTPUT = PROJECT_ROOT / "outputs" / "gemma4-4b-kidsafe.Q5_K_M.gguf"
GGUF_Q8_OUTPUT = PROJECT_ROOT / "outputs" / "gemma4-4b-kidsafe.Q8_0.gguf"

# llama.cpp source (has convert_hf_to_gguf.py)
LLAMA_CPP_SRC_DIR = PROJECT_ROOT.parent / "llama-cpp-src"

# llama.cpp pre-compiled binaries (has llama-quantize.exe, llama-server.exe, etc.)
LLAMA_CPP_BIN_DIR = PROJECT_ROOT.parent / "llama-cpp"

QUANT_OPTIONS = {
    "q4_k_m": {"output": GGUF_Q4_OUTPUT, "desc": "~3.5 GB, good balance", "vram": "~5 GB"},
    "q5_k_m": {"output": GGUF_Q5_OUTPUT, "desc": "~4.0 GB, better quality", "vram": "~5.5 GB"},
    "q8_0":   {"output": GGUF_Q8_OUTPUT, "desc": "~5.5 GB, near fp16 quality", "vram": "~7 GB"},
}


def build_merged_model():
    """Build fully merged model in memory: base + SFT + DPO.

    CRITICAL: Do NOT save to disk. 4-bit merge corrupts safetensors.
    Returns (model, tokenizer) ready for GGUF conversion.
    """
    print("=" * 60)
    print("  STEP 1: Build merged model (base + SFT + DPO)")
    print("=" * 60)

    # Verify adapters exist
    if not SFT_ADAPTER.exists():
        print(f"ERROR: SFT adapter not found: {SFT_ADAPTER}")
        print(f"  Run gemma4_train.py first.")
        sys.exit(1)
    if not DPO_ADAPTER.exists():
        print(f"ERROR: DPO adapter not found: {DPO_ADAPTER}")
        print(f"  Run gemma4_train_dpo.py first.")
        sys.exit(1)

    print(f"  Base:        {BASE_MODEL}")
    print(f"  SFT Adapter: {SFT_ADAPTER}")
    print(f"  DPO Adapter: {DPO_ADAPTER}")

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Step 1: Load base model
    print("\n  [1/4] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  Base loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Step 2: Apply SFT adapter as LoRA
    print("\n  [2/4] Applying SFT adapter (child-friendly tone)...")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    print("  SFT LoRA applied.")

    # Step 3: Merge SFT LoRA into base (in-memory only)
    print("  Merging SFT LoRA into base weights...")
    model = model.merge_and_unload()
    print("  SFT merged. VRAM: {0:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))

    # Step 4: Apply DPO adapter as LoRA
    print("\n  [3/4] Applying DPO adapter (safety alignment)...")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER)
    print("  DPO LoRA applied.")

    # Step 5: Merge DPO LoRA into base (in-memory only)
    print("  Merging DPO LoRA into base weights...")
    model = model.merge_and_unload()
    print("  DPO merged. VRAM: {0:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))

    # Get inner model for GGUF conversion
    inner_model = model.model if hasattr(model, 'model') else model

    print("\n  [4/4] Model ready for GGUF conversion (in memory only)")
    print("  NOTE: Not saving to disk — 4-bit merge corrupts safetensors")
    return inner_model, tokenizer


def convert_to_gguf_fp16(model, tokenizer):
    """Convert in-memory model to GGUF fp16 using convert_hf_to_gguf.py."""
    print("\n" + "=" * 60)
    print("  STEP 2: Convert to GGUF fp16")
    print("=" * 60)

    if not LLAMA_CPP_SRC_DIR.exists():
        print(f"ERROR: llama.cpp source not found: {LLAMA_CPP_SRC_DIR}")
        sys.exit(1)

    convert_script = LLAMA_CPP_SRC_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: convert_hf_to_gguf.py not found at: {convert_script}")
        sys.exit(1)

    # Save model to temp dir for conversion (convert needs filesystem access)
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="gguf_convert_"))
    print(f"  Temp dir: {temp_dir}")

    # Save model and tokenizer to temp
    model.save_pretrained(str(temp_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(temp_dir))

    # Fix config.json if needed (Gemma 4 dimensions in text_config)
    import json
    config_path = temp_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'text_config' in cfg and 'hidden_size' not in cfg:
            tc = cfg['text_config']
            for key in ['hidden_size', 'intermediate_size', 'num_hidden_layers',
                        'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                        'max_position_embeddings', 'head_dim', 'rms_norm_eps',
                        'hidden_activation', 'initializer_range', 'tie_word_embeddings',
                        'bos_token_id', 'eos_token_id', 'pad_token_id',
                        'sliding_window', 'use_cache', 'attention_dropout',
                        'attention_bias', 'rope_parameters', 'final_logit_softcapping']:
                if key in tc:
                    cfg[key] = tc[key]
            del cfg['text_config']
            for key in ['unsloth_fixed', 'unsloth_version', 'model_name', 'dtype']:
                cfg.pop(key, None)
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print("  Fixed config.json dimensions")

    print(f"  Input:  {temp_dir}")
    print(f"  Output: {GGUF_FP16_OUTPUT}")

    cmd = [
        sys.executable, str(convert_script),
        str(temp_dir),
        "--outfile", str(GGUF_FP16_OUTPUT),
        "--outtype", "f16",
    ]

    print(f"\n  Running conversion (this may take 5-10 min)...")
    result = subprocess.run(cmd, cwd=str(LLAMA_CPP_SRC_DIR))

    # Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

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
        description="Merge Gemma 4 SFT + DPO adapters and export to GGUF"
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
    print("  Gemma 4 E4B - SFT + DPO to GGUF Export")
    print("=" * 60)
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Source:  {LLAMA_CPP_SRC_DIR}")
    print(f"  Bin:     {LLAMA_CPP_BIN_DIR}")
    print()

    if args.step in ("merge", "all"):
        model, tokenizer = build_merged_model()
    else:
        print("ERROR: --step merge is required before convert")
        sys.exit(1)

    if args.step in ("convert", "all"):
        convert_to_gguf_fp16(model, tokenizer)

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

