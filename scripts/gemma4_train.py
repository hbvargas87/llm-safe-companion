#!/usr/bin/env python3
"""
Gemma 4 E4B - Fine-Tuning Script with Unsloth + QLoRA
Trains gemma-4-E4B-it with QLoRA using Unsloth + TRL

Usage:
    python scripts/gemma4_train.py --config configs/gemma4_4b_qlora.yaml
    python scripts/gemma4_train.py --config configs/gemma4_4b_qlora.yaml --resume

Key differences from Qwen2 training:
- Uses Unsloth FastModel (60% less VRAM, 1.5x faster)
- Chat template: gemma-4 (no thinking mode)
- use_cache=True is CRITICAL (fixes garbage logits bug)
- Loss 13-15 is NORMAL for Gemma 4 (not model collapse)
- LoRA: r=8, alpha=8 (Unsloth recommendation)
- Dataset format: conversations list with role/content dicts
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Hugging Face imports
import torch
from datasets import load_dataset, Dataset

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Config loaded from: {config_path}')
    return config


def load_model_and_tokenizer(config: dict):
    """Load Gemma 4 E4B with Unsloth FastModel (4-bit QLoRA)."""
    from unsloth import FastModel

    model_name = config['model']['name_or_path']
    max_seq_length = config['training']['max_seq_length']

    print(f'  Loading {model_name} with Unsloth (4-bit QLoRA)...')
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,  # Auto-detect (bf16 on RTX 5080)
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    print(f'  Model loaded. VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    # Apply chat template (gemma-4, no thinking)
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",  # No thinking mode for kids
    )
    print(f'  Chat template applied: gemma-4 (no thinking)')

    return model, tokenizer


def setup_lora(config: dict):
    """Configure LoRA for Gemma 4 E4B (Unsloth recommended: r=8, alpha=8)."""
    from unsloth import FastModel

    model = config.get('_model', None)  # Model is passed via config after loading
    lora_config = config['training']['lora']

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,   # Text only (no vision fine-tuning)
        finetune_language_layers=True,   # Language layers
        finetune_attention_modules=True, # Attention layers
        finetune_mlp_modules=True,       # MLP layers (always on)
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        random_state=config['training'].get('seed', 42),
    )
    model.print_trainable_parameters()
    return model


def format_dataset_for_gemma4(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    """
    Format dataset for Gemma 4 training.

    Gemma 4 expects conversations as lists of role/content dicts.
    Converts from our kidsafe_tone_dataset.jsonl format.

    Dataset format in kidsafe_tone_dataset.jsonl:
    - 'messages': list of {role, content} dicts (already in chat format)

    Gemma 4 chat template output (no thinking):
    <bos><|turn>user
    Hello<turn|>
    <|turn>model
    Hi!<turn|>
    """
    system_message = (
        'You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. '
        'You respond in an educational, empathetic, and age-appropriate manner. '
        'You always maintain a positive and reassuring tone. '
        'You are not a psychologist, but you always listen with understanding. '
        'You respond in the same language the child uses.'
    )

    def convert_to_gemma4_format(sample):
        """Convert messages format to Gemma 4 conversation format."""
        messages = sample['messages']

        # Ensure system message is present
        has_system = any(m['role'] == 'system' for m in messages)
        if not has_system:
            messages = [{'role': 'system', 'content': system_message}] + messages

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Remove leading <bos> token (tokenizer adds it during training)
        text = text.removeprefix('<bos>')

        # Truncate if too long
        if len(text) > max_seq_length:
            text = text[:max_seq_length]

        return {'text': text}

    formatted = dataset.map(
        convert_to_gemma4_format,
        remove_columns=['messages'],
        desc='Formatting for Gemma 4',
    )

    print(f'  Formatted: {len(formatted)} samples')
    return formatted


def load_and_prepare_dataset(config: dict, tokenizer, max_seq_length: int):
    """Load and prepare dataset for Gemma 4 training."""
    from unsloth.chat_templates import standardize_data_formats

    dataset_path = PROJECT_ROOT / 'data' / 'filtered_datasets'
    jsonl_path = dataset_path / 'kidsafe_tone_dataset.jsonl'

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f'Dataset not found: {jsonl_path}\n'
            'Expected: kidsafe_tone_dataset.jsonl in data/filtered_datasets/'
        )

    print(f'Loading dataset: {jsonl_path.name}')
    dataset = load_dataset('json', data_files=str(jsonl_path), split='train')
    print(f'  Total samples: {len(dataset)}')

    # Standardize data formats (Unsloth utility)
    dataset = standardize_data_formats(dataset)

    # Format for Gemma 4
    formatted = format_dataset_for_gemma4(dataset, tokenizer, max_seq_length)

    return formatted


def train(config_path: str, resume: bool = False):
    """
    Main training function for Gemma 4 E4B with Unsloth + QLoRA.

    CRITICAL NOTES:
    - Loss 13-15 is NORMAL for Gemma 4 (multimodal quirk, not collapse)
    - use_cache=True is REQUIRED (fixes garbage logits bug in E2B/E4B)
    - Unsloth handles gradient accumulation fix internally
    """
    from trl import SFTTrainer, SFTConfig

    print('\n' + '='*70)
    print('  GEMMA 4 E4B - Fine-Tuning with Unsloth + QLoRA')
    print('  Model: unsloth/gemma-4-E4B-it')
    print('  Hardware: NVIDIA RTX 5080 (16GB VRAM)')
    print('  Technique: QLoRA 4-bit (Unsloth optimized)')
    print('='*70 + '\n')

    config = load_config(config_path)
    torch.manual_seed(config['training'].get('seed', 42))

    # Step 1: Load Model and Tokenizer
    print('\nStep 1: Loading model and tokenizer...')
    model, tokenizer = load_model_and_tokenizer(config)

    # Step 2: Setup LoRA
    print('\nStep 2: Configuring LoRA adapters...')
    config['_model'] = model
    model = setup_lora(config)

    # Step 3: Load Dataset
    print('\nStep 3: Preparing dataset...')
    max_seq_length = config['training']['max_seq_length']
    dataset = load_and_prepare_dataset(config, tokenizer, max_seq_length)

    # Step 4: Configure Training
    print('\nStep 4: Configuring training...')

    training_config = config['training']

    training_args = SFTConfig(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        bf16=True,
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        seed=training_config.get('seed', 42),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        max_seq_length=max_seq_length,
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_steps=training_config.get('eval_steps', 100),
        eval_strategy='no',
        report_to='tensorboard',
    )

    # Step 5: Create Trainer
    print('\nStep 5: Creating Trainer...')

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Step 6: Train
    print('\nStep 6: Starting training...')
    print(f'   Output dir: {training_config["output_dir"]}')
    print(f'   Epochs: {training_config["num_train_epochs"]}')
    print(f'   Batch size: {training_config["per_device_train_batch_size"]} x {training_config["gradient_accumulation_steps"]} (accumulated)')
    print(f'   Learning rate: {training_config["learning_rate"]}')
    print(f'   Max seq length: {max_seq_length}')
    print()

    try:
        train_result = trainer.train(resume_from_checkpoint=resume)

        # --- Post-training check ---
        # NOTE: Loss 13-15 is NORMAL for Gemma 4 multimodal models!
        # This is a quirk, NOT model collapse.
        final_loss = train_result.metrics.get('train_loss', None)
        is_nan = final_loss is not None and torch.isnan(torch.tensor(final_loss))
        # We do NOT check for "too high" loss — 13-15 is expected
        model_collapsed = is_nan

        # Step 7: Save Model
        print('\nStep 7: Saving model...')

        final_output_dir = os.path.join(training_config['output_dir'], 'final')
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        metrics = train_result.metrics
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

        print(f'\n{"="*70}')
        print('  TRAINING COMPLETED!')
        print(f'  Model saved to: {final_output_dir}')
        print(f'  Final loss: {metrics.get("train_loss", "N/A")}')
        print(f'  NOTE: Loss 13-15 is NORMAL for Gemma 4 (multimodal quirk)')
        if model_collapsed:
            print(f'\n  WARNING: Model may have NaN loss! Retrain recommended.')
        print(f'{"="*70}\n')

        # Step 8: Inference Test (only if model looks healthy)
        if not model_collapsed:
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

            # CRITICAL: use_cache=True for Gemma 4 E2B/E4B
            from transformers import TextStreamer
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    use_cache=True,  # CRITICAL for E2B/E4B!
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            prompt_len = len(inputs['input_ids'][0])
            generated_ids = outputs[0][prompt_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            print(f'\n   Prompt: {test_prompt}')
            print(f'   Response: {response_text[:500]}')
        else:
            print('\nSkipping inference test (model may have NaN loss)')

    except Exception as e:
        print(f'\n{"="*70}')
        print(f'  ERROR DURING TRAINING')
        print(f'  {str(e)}')
        print(f'{"="*70}\n')
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Fine-Tuning Gemma 4 E4B with Unsloth + QLoRA'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/gemma4_4b_qlora.yaml',
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

    train(config_path=args.config, resume=args.resume)


if __name__ == '__main__':
    main()
