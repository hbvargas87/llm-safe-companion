#!/usr/bin/env python3
"""
Merge SFT adapter into base model for Gemma 4 E4B.

Usage:
    python scripts/merge_sft_gemma4.py

Creates: ./outputs/gemma4_4b_sft/merged/
    - Modelo fusionado (base + SFT LoRA weights merged)
    - Listo para DPO training
"""

import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def merge_sft():
    base_model = "unsloth/gemma-4-E4B-it"
    sft_adapter = "./outputs/gemma4_4b_sft/final"
    merged_dir = "./outputs/gemma4_4b_sft/merged"

    print('=' * 60)
    print('  GEMMA 4 E4B - Merge SFT Adapter into Base Model')
    print('=' * 60)

    # Step 1: Load base model
    print('\nStep 1: Loading base model...')
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    print(f'  Base loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    # Step 2: Load SFT adapter weights
    print(f'\nStep 2: Loading SFT adapter from: {sft_adapter}')
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sft_adapter)
    print('  SFT adapter weights applied.')

    # Step 3: Convert to trainable model (required for merging)
    print('\nStep 3: Converting to trainable model...')
    model = model.merge_and_unload()
    print('  SFT LoRA merged into base weights.')

    # Step 4: Save merged model with correct config
    print(f'\nStep 4: Saving merged model to: {merged_dir}')
    os.makedirs(merged_dir, exist_ok=True)

    # Get the inner model (Unsloth wraps in FastLanguageModel)
    inner_model = model.model if hasattr(model, 'model') else model

    # Strip Unsloth weight conversions that block save_pretrained
    if hasattr(inner_model, 'hf_model_wrapped_with_model_parallelism'):
        inner_model = inner_model.hf_model_wrapped_with_model_parallelism
    if hasattr(inner_model, 'model'):
        inner_model = inner_model.model

    # Remove weight_conversions from config to allow saving
    if hasattr(inner_model.config, 'weight_conversions'):
        inner_model.config.weight_conversions = None
    for attr in list(vars(inner_model.config)):
        if 'conversion' in attr.lower() or 'quant' in attr.lower():
            try:
                delattr(inner_model.config, attr)
            except Exception:
                pass

    # Fix Gemma 4 config: dimensions are nested in text_config but need to be at top level
    # This is a known Unsloth/transformers mismatch
    import json
    config_path = os.path.join(merged_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'text_config' in cfg:
            tc = cfg['text_config']
            # Copy critical dimensions to top level
            for key in ['hidden_size', 'intermediate_size', 'num_hidden_layers',
                        'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                        'max_position_embeddings', 'head_dim', 'rms_norm_eps',
                        'hidden_activation', 'initializer_range', 'tie_word_embeddings',
                        'bos_token_id', 'eos_token_id', 'pad_token_id',
                        'sliding_window', 'use_cache', 'attention_dropout',
                        'attention_bias', 'rope_parameters', 'final_logit_softcapping']:
                if key in tc:
                    cfg[key] = tc[key]
            # Remove the nested text_config to avoid confusion
            del cfg['text_config']
            # Remove Unsloth-specific keys
            for key in ['unsloth_fixed', 'unsloth_version', 'model_name', 'dtype']:
                cfg.pop(key, None)
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)
            print('  Fixed config.json: dimensions moved to top level')

    inner_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print('  Done!')

    # Verify
    print(f'\n  Files saved:')
    for f in sorted(Path(merged_dir).iterdir()):
        size_gb = f.stat().st_size / (1024**3)
        print(f'    {f.name}: {size_gb:.2f} GB')

    print(f'\n{"=" * 60}')
    print('  MERGE COMPLETED!')
    print(f'  Merged model: {merged_dir}')
    print(f'  Ready for DPO training.')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    merge_sft()

