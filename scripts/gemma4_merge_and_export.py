#!/usr/bin/env python3
"""
Merge SFT + DPO adapters for Gemma 4 E4B and export to GGUF.

Uses Unsloth's native save_pretrained_gguf() method instead of llama.cpp,
which avoids compatibility issues with Gemma 4 E4B architecture.

Pipeline:
    1. Load base gemma-4-E4B-it in BF16 (not 4-bit — avoids corruption)
    2. Apply SFT adapter (child-friendly tone) as LoRA
    3. Merge SFT LoRA into base (in-memory)
    4. Apply DPO adapter (safety alignment) as LoRA
    5. Merge DPO LoRA into base (in-memory)
    6. Export directly to GGUF using Unsloth's native method

Why this approach:
    - llama.cpp's convert_hf_to_gguf.py doesn't support Gemma 4 E4B architecture
    - Unsloth's save_pretrained_gguf() handles the conversion natively
    - Single quantization step (cleaner than 4-bit -> bf16 -> quantize)

Usage:
    python scripts/gemma4_merge_and_export.py --out merged/gemma4-E4B-kidsafe --gguf --quant q4_k_m
    python scripts/gemma4_merge_and_export.py --out merged/gemma4-E4B-kidsafe --gguf --quant all
    python scripts/gemma4_merge_and_export.py --out merged/gemma4-E4B-kidsafe          # merge only, no GGUF
"""

import argparse
import json
import shutil
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/gemma-4-E4B-it"
SFT_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_sft" / "final"
DPO_ADAPTER = PROJECT_ROOT / "outputs" / "gemma4_4b_dpo" / "final"


def print_vram(stage: str):
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [{stage}] VRAM allocated: {allocated:.2f} GB / reserved: {reserved:.2f} GB / total: {total:.2f} GB")


def force_cpu_load():
    """
    Fallback: load base model on CPU using transformers directly.
    Slower but uses system RAM (128 GB available).
    Returns (model, tokenizer).
    """
    print("\n  Loading base model on CPU (transformers)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Downloading/loading {BASE_MODEL} to CPU (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("  Base model loaded on CPU.")
    return model, tokenizer


def load_tokenizer():
    """Load tokenizer from base model."""
    print(f"  Loading tokenizer from {BASE_MODEL}...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        return processor.tokenizer
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


def build_merged_model(out_dir: Path, use_cpu: bool = False):
    """
    Build fully merged model in BF16: base + SFT + DPO.

    Args:
        out_dir: Where to save the merged HF checkpoint.
        use_cpu: If True, skip GPU loading and use CPU directly.

    Returns:
        (model, tokenizer) ready for GGUF export.
    """
    print("=" * 60)
    print("  STEP 1: Build merged model (base + SFT + DPO) in BF16")
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

    if torch.cuda.is_available() and not use_cpu:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        print("  Mode: CPU fallback (128 GB DDR5 available)")

    from peft import PeftModel

    if use_cpu:
        model, tokenizer = force_cpu_load()
    else:
        # ── GPU path: Unsloth FastLanguageModel in BF16 ──
        from unsloth import FastLanguageModel

        print("\n  [1/5] Loading base model in BF16 (GPU)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=1024,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        print_vram("Base loaded")

    # ── Apply and merge SFT adapter ──
    print("\n  [2/5] Applying and merging SFT adapter...")
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER))
    model = model.merge_and_unload()
    if not use_cpu:
        torch.cuda.empty_cache()
        print_vram("SFT merged")

    # ── Apply and merge DPO adapter ──
    print("\n  [3/5] Applying and merging DPO adapter...")
    model = PeftModel.from_pretrained(model, str(DPO_ADAPTER))
    model = model.merge_and_unload()
    if not use_cpu:
        torch.cuda.empty_cache()
        print_vram("DPO merged")

    # ── Free VRAM before saving ──
    if not use_cpu:
        torch.cuda.empty_cache()

    # ── Save merged checkpoint (HF format) ──
    print(f"\n  [4/5] Saving merged checkpoint to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(str(out_dir), safe_serialization=True)
    print("  Model saved (safe_serialization=True).")

    # Save tokenizer
    tokenizer.save_pretrained(str(out_dir))
    print("  Tokenizer saved.")

    # Copy chat_template.jinja and processor_config.json from DPO checkpoint
    for fname in ("chat_template.jinja", "processor_config.json"):
        src = DPO_ADAPTER / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
            print(f"  Copied {fname} from DPO checkpoint (source of truth)")

    # ── Verify checkpoint ──
    print(f"\n  Verifying merged checkpoint...")
    config_path = out_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'text_config' in cfg:
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
            print("  Fixed config.json: promoted text_config dimensions to top-level")
        else:
            print("  config.json looks correct (no text_config nesting).")

    print(f"\n  Merged checkpoint ready at: {out_dir}")
    print(f"  Checkpoint size: {sum(f.stat().st_size for f in out_dir.rglob('*') if f.is_file()) / 1e9:.2f} GB")

    return model, tokenizer


def export_to_gguf(model, tokenizer, out_dir: Path, quant: str):
    """
    Export merged model to GGUF using Unsloth's native save_pretrained_gguf().

    This bypasses llama.cpp entirely, avoiding compatibility issues with
    Gemma 4 E4B architecture.

    Args:
        model: The merged model (bf16).
        tokenizer: The tokenizer.
        out_dir: Base directory for GGUF output.
        quant: Quantization method (q4_k_m, q5_k_m, q8_0, f16).
    """
    print("\n" + "=" * 60)
    print("  STEP 2: Export to GGUF (Unsloth native)")
    print("=" * 60)

    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    if quant == "all":
        quant_methods = ["q4_k_m", "q5_k_m", "q8_0"]
    else:
        quant_methods = [quant]

    for qm in quant_methods:
        print(f"\n  Exporting to {qm}...")
        gguf_path = gguf_dir / f"gemma4-E4B-kidsafe.{qm}.gguf"

        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method=qm,
            )
            if gguf_path.exists():
                size_gb = gguf_path.stat().st_size / (1024**3)
                print(f"  Done. {gguf_path.name}: {size_gb:.2f} GB")
            else:
                print(f"  ERROR: Output not created at {gguf_path}")
        except Exception as e:
            print(f"  ERROR exporting {qm}: {e}")

    # List all generated GGUF files
    print(f"\n  Generated GGUF files:")
    for f in sorted(gguf_dir.glob("*.gguf")):
        size_gb = f.stat().st_size / (1024**3)
        print(f"    {f.name}: {size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Gemma 4 SFT + DPO adapters (BF16) and export to GGUF"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="merged/gemma4-E4B-kidsafe",
        help="Output directory for merged HF checkpoint (relative to project root)",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Also export to GGUF using Unsloth's native method",
    )
    parser.add_argument(
        "--quant",
        choices=["q4_k_m", "q5_k_m", "q8_0", "all"],
        default="q4_k_m",
        help="Quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU loading (uses 128 GB DDR5 RAM instead of GPU VRAM)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Gemma 4 E4B — SFT + DPO Merge → GGUF Export")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Output dir:   {args.out}")
    print(f"  GGUF:         {'YES' if args.gguf else 'NO'}")
    if args.gguf:
        print(f"  Quantize:     {args.quant}")
    print(f"  CPU mode:     {'YES' if args.cpu else 'NO (GPU)'}")
    print()

    # Step 1: Build merged model
    out_dir = PROJECT_ROOT / args.out
    model, tokenizer = build_merged_model(out_dir, use_cpu=args.cpu)

    # Free VRAM after merge
    if not args.cpu:
        torch.cuda.empty_cache()

    # Step 2: Export to GGUF (optional)
    if args.gguf:
        export_to_gguf(model, tokenizer, PROJECT_ROOT / "outputs", args.quant)

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)

    # Show results
    print(f"\n  Merged checkpoint: {out_dir}")
    print(f"  Checkpoint files:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    if args.gguf:
        gguf_dir = PROJECT_ROOT / "outputs" / "merged" / "gemma4-E4B-kidsafe" / "gguf"
        if gguf_dir.exists():
            print(f"\n  GGUF files:")
            for f in sorted(gguf_dir.glob("*.gguf")):
                size_gb = f.stat().st_size / (1024**3)
                print(f"    {f.name}: {size_gb:.2f} GB")

            print(f"\n  LM Studio: copy any .gguf to your models folder")
            print(f"  llama.cpp: llama-server.exe -m \"{gguf_dir / 'gemma4-E4B-kidsafe.Q4_K_M.gguf'}\" -c 2048")


if __name__ == "__main__":
    main()
