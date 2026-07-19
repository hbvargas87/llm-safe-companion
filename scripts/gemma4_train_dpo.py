#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training for Gemma 4 E4B

Based on: Unsloth's official Zephyr DPO notebook (zephyr_(7b)_dpo.py)
https://github.com/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb

Usage:
    python scripts/gemma4_train_dpo.py --config configs/gemma4_4b_dpo.yaml
    python scripts/gemma4_train_dpo.py --config configs/gemma4_4b_dpo.yaml --resume

Pipeline:
    SFT (gemma4_train.py) -> Modelo SFT -> DPO (este script) -> Modelo DPO -> GGUF export

IMPORTANT: DPO se entrena sobre el modelo SFT (HuggingFace), NO sobre GGUF.
"""

import os
import sys
import yaml
import re
import argparse
from pathlib import Path

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f'Config loaded from: {config_path}')
    return config


def apply_chat_template(example, tokenizer, task="dpo",
                        assistant_prefix="<|assistant|>\n"):
    """Format DPO data following Unsloth's Zephyr notebook pattern.
    
    Handles both formats:
    1. Text format: chosen/rejected are plain strings (with system/prompt columns)
    2. Messages format: chosen/rejected are lists of {role, content} dicts
    """
    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task == "dpo":
        if not all(k in example.keys() for k in ("chosen", "rejected")):
            raise ValueError(
            f"DPO task requires [chosen, rejected] keys but found {list(example.keys())}"
        )

        # --- Text format: chosen/rejected are plain strings ---
        if isinstance(example["chosen"], str):
            system = example.get("system", "")
            prompt_text = example["prompt"]
            chosen_text = example["chosen"]
            rejected_text = example["rejected"]

            # Build prompt messages (system + user)
            prompt_messages = []
            if system:
                prompt_messages.append({"role": "system", "content": system})
            prompt_messages.append({"role": "user", "content": prompt_text})

            # Build chosen/rejected messages (assistant only)
            chosen_messages = [{"role": "assistant", "content": chosen_text}]
            rejected_messages = [{"role": "assistant", "content": rejected_text}]

        # --- Messages format: chosen/rejected are lists of dicts ---
        elif isinstance(example["chosen"], list):
            # Extract prompt messages (system + user)
            prompt_messages = [
                msg for msg in example["chosen"]
                if msg["role"] == "user"
        ]
            if not prompt_messages:
                prompt_messages = [{"role": "user", "content": ""}]

            # Insert system message if missing
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])

            # Extract chosen/rejected responses (skip system message)
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
        else:
            raise ValueError(
                f"chosen must be str or list, got {type(example['chosen'])}"
            )

        # Apply chat template to all three
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
    )
        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False
    )
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False
    )

        # Strip assistant prefix from chosen/rejected
        example["text_chosen"] = _strip_prefix(
            example["text_chosen"], assistant_prefix
        )
        example["text_rejected"] = _strip_prefix(
            example["text_rejected"], assistant_prefix
        )
    else:
        raise ValueError(
            f"DPO task requires [chosen, rejected] keys but found {list(example.keys())}"
        )
    return example


def load_and_format_dpo_dataset(config: dict, tokenizer):
    """Load DPO dataset and format following Unsloth's Zephyr notebook."""
    dataset_path = PROJECT_ROOT / 'data' / 'dpo' / 'dpo_dataset.jsonl'

    print(f'Loading DPO dataset: {dataset_path}')
    dataset = load_dataset(
        'json',
        data_files=str(dataset_path),
        split='train',
    )
    print(f'   Total pairs: {len(dataset)}')

    # Format following Unsloth's Zephyr notebook pattern
    formatted = dataset.map(
        lambda ex: apply_chat_template(ex, tokenizer, task="dpo"),
        desc="Formatting DPO samples",
    )

    # Remove original columns (prompt, system, chosen, rejected) to avoid duplicates
    # after rename, then rename text_* columns to TRL expected names
    original_cols = [c for c in formatted.column_names if c not in ("text_prompt", "text_chosen", "text_rejected")]
    formatted = formatted.remove_columns(original_cols)

    # Rename to TRL expected column names
    formatted = formatted.rename_columns({
        "text_prompt": "prompt",
        "text_chosen": "chosen",
        "text_rejected": "rejected",
    })

    print(f'   Dataset formatted: {len(formatted)} samples')
    print(f'   Columns: {formatted.column_names}')
    return formatted


def train_dpo(config_path: str, resume: bool = False):
    """
    Main DPO training function for Gemma 4 E4B.
    
    Based on Unsloth's official Zephyr DPO notebook pattern.
    """
    print('\n' + '=' * 70)
    print('  GEMMA 4 E4B - DPO Safety Alignment Training')
    print('  Model: unsloth/gemma-4-E4B-it (SFT fine-tuned)')
    print('  Hardware: NVIDIA RTX 5080 (16GB VRAM)')
    print('  Based on: Unsloth Zephyr DPO notebook')
    print('=' * 70 + '\n')

    config = load_config(config_path)

    training_config = config['training']
    sft_adapter_path = "./outputs/gemma4_4b_sft/final"
    max_seq_length = training_config.get('max_seq_length', 1024)

    # Step 1: Load BASE model, apply SFT adapter as LoRA, then merge in memory
    # NOTE: We do NOT save the merged model to disk because 4-bit merge corrupts safetensors
    print('\nStep 1: Loading base model and applying SFT adapter...')
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f'  Base model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    # Apply SFT adapter as LoRA on top of base
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    print(f'  SFT adapter (LoRA) applied.')

    # Merge in memory (do NOT save to disk - 4-bit merge corrupts safetensors)
    print('  Merging SFT LoRA into base weights (in memory only)...')
    model = model.merge_and_unload()
    print('  SFT LoRA merged. Ready for DPO.')

    # Step 2: Load and Format DPO Dataset
    print('\nStep 2: Preparing DPO dataset...')
    dpo_dataset = load_and_format_dpo_dataset(config, tokenizer)

    # Step 3: Add new LoRA adapters for DPO
    print('\nStep 3: Configuring DPO LoRA adapters...')
    lora_cfg = training_config['lora']

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=0,  # Unsloth only supports dropout=0
        target_modules=lora_cfg['target_modules'],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM!
        random_state=training_config.get('seed', 3407),
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # Step 4: Patch DPOTrainer (REQUIRED for Unsloth optimizations)
    print('\nStep 4: Patching DPOTrainer...')
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    print('  DPOTrainer patched!')

    # Step 5: Create DPO Trainer
    print('\nStep 5: Creating DPO Trainer...')
    from trl import DPOTrainer, DPOConfig

    dpo_args = DPOConfig(
        output_dir=training_config['output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        warmup_ratio=training_config['warmup_ratio'],
        num_train_epochs=training_config['num_train_epochs'],
        learning_rate=training_config['learning_rate'],
        logging_steps=training_config.get('logging_steps', 1),
        optim="adamw_8bit",  # 8-bit optimizer for less VRAM
        weight_decay=training_config.get('weight_decay', 0.0),
        lr_scheduler_type=training_config['lr_scheduler_type'],
        seed=training_config.get('seed', 3407),
        report_to='tensorboard',
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL creates reference from SFT model
        args=dpo_args,
        beta=training_config.get('dpo_beta', 0.1),
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=training_config.get('max_prompt_length', 512),
    )

    # Step 6: Train
    print('\nStep 6: Starting DPO training...')
    print(f'   Output dir: {training_config["output_dir"]}')
    print(f'   Epochs: {training_config["num_train_epochs"]}')
    print(f'   Batch size: {training_config["per_device_train_batch_size"]} x {training_config["gradient_accumulation_steps"]} (accumulated)')
    print(f'   Learning rate: {training_config["learning_rate"]}')
    print(f'   DPO beta: {training_config.get("dpo_beta", 0.1)}')
    print(f'   Max length: {max_seq_length}')
    print(f'   Max prompt length: {training_config.get("max_prompt_length", 512)}')
    print()

    try:
        # Memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        total_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name} | Total: {total_mem} GB | Reserved: {start_mem} GB")

        train_result = trainer.train(resume_from_checkpoint=resume)

        used_mem = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        lora_mem = round(used_mem - start_mem, 3)
        print(f"\nTraining time: {train_result.metrics['train_runtime']}s "
              f"({round(train_result.metrics['train_runtime']/60, 2)} min)")
        print(f"Peak VRAM: {used_mem} GB (training used {lora_mem} GB)")

        # Post-training: save model
        print('\nStep 7: Saving DPO adapter...')
        final_output_dir = os.path.join(training_config['output_dir'], 'final')
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        metrics = train_result.metrics
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        print(f'\n{"=" * 70}')
        print('  DPO TRAINING COMPLETED!')
        print(f'  Adapter saved to: {final_output_dir}')
        if 'train_loss' in metrics:
            print(f'  Final loss: {metrics["train_loss"]:.4f}')
        print(f'{"=" * 70}\n')

        # Quick inference test
        print('Quick inference test...')
        test_prompt = 'What is photosynthesis?'
        system_message = (
            'You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. '
            'You respond in an educational, empathetic, and age-appropriate manner. '
            'You always maintain a positive and reassuring tone. '
            'You are not a psychologist, but you always listen with understanding. '
            'You respond in the same language the child uses.'
        )
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': test_prompt},
        ]

        # Disable gradient checkpointing for inference
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt',
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                use_cache=True,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        prompt_len = len(inputs['input_ids'][0])
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        print(f'\n   Prompt: {test_prompt}')
        print(f'   Response: {response[:500]}')

    except Exception as e:
        print(f'\n{"=" * 70}')
        print(f'  ERROR DURING DPO TRAINING')
        print(f'  {str(e)}')
        print(f'{"=" * 70}\n')
        raise


def main():
    parser = argparse.ArgumentParser(
        description='DPO Training for Gemma 4 E4B (based on Unsloth Zephyr notebook)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gemma4_4b_dpo.yaml',
        help='Path to YAML config file',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from previous checkpoint',
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Config not found: {config_path}')
        print(f'   Expected path: {config_path.resolve()}')
        sys.exit(1)

    train_dpo(config_path=args.config, resume=args.resume)


if __name__ == '__main__':
    main()

