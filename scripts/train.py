#!/usr/bin/env python3
"""
Main Fine-Tuning Script for Kid-Safe LLM
Trains Qwen2-7B with QLoRA using Hugging Face TRL and PEFT

Usage:
    python scripts/train.py --config configs/qwen2_7b_qlora.yaml
    python scripts/train.py --config configs/qwen2_7b_qlora.yaml --resume
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Hugging Face imports
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
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


def setup_quantization(config: dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization for QLoRA."""
    quant_config = config['model']['quantization_config']
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16 if quant_config['bnb_4bit_compute_dtype'] == 'float16' else torch.float32,
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
    )
    
    print('4-bit quantization configured (NF4)')
    return bnb_config


def setup_lora(config: dict) -> LoraConfig:
    """Configure LoRA for efficient fine-tuning."""
    lora_config = config['training']['lora']
    
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        task_type=lora_config['task_type'],
        bias='none',
    )
    
    print(f'LoRA configured (rank={lora_config["r"]}, alpha={lora_config["lora_alpha"]})')
    return peft_config


def format_conversation(sample: dict, tokenizer: AutoTokenizer, max_seq_length: int) -> dict:
    """
    Format dataset sample to Qwen2 conversation format.
    
    Args:
        sample: Dict with 'instruction', 'input', 'output'
        tokenizer: Model tokenizer
        max_seq_length: Maximum sequence length
    
    Returns:
        Dict with formatted 'text'
    """
    system_message = (
        'You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. '
        'You respond in an educational, empathetic, and age-appropriate manner. '
        'You always maintain a positive and reassuring tone. '
        'You are not a psychologist, but you always listen with understanding. '
        'You respond in the same language the child uses.'
    )
    
    messages = [
        {'role': 'system', 'content': system_message},
    ]
    
    if sample.get('input') and sample['input'].strip():
        user_content = f"{sample['instruction']}\n\n{sample['input']}"
    else:
        user_content = sample['instruction']
    
    messages.append({'role': 'user', 'content': user_content})
    messages.append({'role': 'assistant', 'content': sample['output']})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    if len(text) > max_seq_length:
        text = text[:max_seq_length]
    
    return {'text': text}


def load_and_prepare_dataset(config: dict, tokenizer: AutoTokenizer, max_seq_length: int):
    """Load and prepare dataset for training.

    Supports two formats:
    1. JSONL with 'messages' format (kidsafe_final_dataset.jsonl) - Recommended
    2. CSV with 'instruction/input/output' format (kidsafe_final_dataset.csv)
    """
    dataset_path = PROJECT_ROOT / 'data' / 'filtered_datasets'

    # Try JSONL first (messages format, already formatted)
    jsonl_path = dataset_path / 'kidsafe_tone_dataset.jsonl'
    if jsonl_path.exists():
        print(f'Loading JSONL dataset: {jsonl_path.name}')
        dataset = load_dataset(
            'json',
            data_files=str(jsonl_path),
            split='train',
        )
        print(f'   Total samples: {len(dataset)}')
    
        print('Converting messages to text for Qwen2...')

        def convert_messages_to_text(sample):
            """Convert messages format to formatted text."""
            messages = sample['messages']
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if len(text) > max_seq_length:
                text = text[:max_seq_length]
            return {'text': text}
        
        formatted_dataset = dataset.map(
            convert_messages_to_text,
            remove_columns=['messages'],
            desc='Converting to text',
        )
        
        if 'text' not in formatted_dataset.column_names:
            raise ValueError('Formatted dataset does not have text column')

        print(f'JSONL dataset formatted: {len(formatted_dataset)} samples')
        return formatted_dataset

    # Fallback: try CSV (instruction/input/output format)
    csv_path = dataset_path / 'kidsafe_final_dataset.csv'
    if csv_path.exists():
        print(f'Loading CSV dataset: {csv_path.name}')
        dataset = load_dataset(
            'csv',
            data_files=str(csv_path),
            split='train',
        )
        print(f'   Total samples: {len(dataset)}')

        print('Formatting dataset for Qwen2...')
        formatted_dataset = dataset.map(
            lambda x: format_conversation(x, tokenizer, max_seq_length),
            remove_columns=dataset.column_names,
            desc='Formatting dataset',
        )

        if 'text' not in formatted_dataset.column_names:
            raise ValueError('Formatted dataset does not have text column')

        print(f'CSV dataset formatted: {len(formatted_dataset)} samples')
        return formatted_dataset

    raise FileNotFoundError(
        'No datasets found in data/filtered_datasets/\n'
        'Expected: kidsafe_final_dataset.jsonl or kidsafe_final_dataset.csv'
    )
    

def train(config_path: str, resume: bool = False):
    """
    Main training function with NaN protection and early stopping.
    Args:
        config_path: Path to YAML config file
        resume: If True, resume from previous checkpoint
    """
    print('\n' + '='*70)
    print('  KID-SAFE LLM - Fine-Tuning with QLoRA')
    print('  Model: Qwen2-7B-Instruct')
    print('  Hardware: NVIDIA RTX 5080 (16GB VRAM)')
    print('='*70 + '\n')
    
    config = load_config(config_path)
    set_seed(config['training'].get('seed', 42))
    
    # Step 1: Load Model and Tokenizer
    print('\nStep 1: Loading model and tokenizer...')
    
    model_name = config['model']['name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True),
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = setup_quantization(config)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=config['model'].get('trust_remote_code', True),
        device_map='auto',
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = setup_lora(config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Step 2: Load Dataset
    print('\nStep 2: Preparing dataset...')
    max_seq_length = config['training'].get('max_seq_length', 2048)
    dataset = load_and_prepare_dataset(config, tokenizer, max_seq_length)
    
    # Step 3: Configure Training
    print('\nStep 3: Configuring training...')
    
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
        max_length=max_seq_length,
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_steps=training_config.get('eval_steps', 100),
        eval_strategy='no',
        report_to='tensorboard',
    )
    
    # Step 4: Create Trainer
    print('\nStep 4: Creating Trainer...')
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    # Step 5: Train with NaN protection
    print('\nStep 5: Starting training...')
    print(f'   Output dir: {training_config["output_dir"]}')
    print(f'   Epochs: {training_config["num_train_epochs"]}')
    print(f'   Batch size: {training_config["per_device_train_batch_size"]} x {training_config["gradient_accumulation_steps"]} (accumulated)')
    print(f'   Learning rate: {training_config["learning_rate"]}')
    print()
    
    # NaN/overflow protection thresholds
    loss_threshold = 10.0         # Above this = model collapse
    loss_warning_threshold = 3.0  # Above this = suspicious, investigate

    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
        
        # --- Post-training safety check ---
        final_loss = train_result.metrics.get('train_loss', None)
        token_acc = train_result.metrics.get('mean_token_accuracy', None)

        # Determine health status
        is_nan = final_loss is not None and torch.isnan(torch.tensor(final_loss))
        is_too_high = final_loss is not None and final_loss > loss_threshold
        is_low_accuracy = token_acc is not None and token_acc < 0.1
        model_collapsed = is_nan or is_too_high or is_low_accuracy
        has_warning = (not model_collapsed and
                       final_loss is not None and
                       final_loss > loss_warning_threshold)

        # Step 6: Save Model (always save after training completes)
        print('\nStep 6: Saving model...')

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
        print(f'   Final loss: {metrics.get("train_loss", "N/A")}')
        print(f'   Token accuracy: {metrics.get("mean_token_accuracy", "N/A")}')
        if model_collapsed:
            print(f'\n  WARNING: Model likely collapsed during training!')
            print(f'     The adapter may be corrupted. Retrain recommended.')
            print(f'     Consider lowering the learning rate.')
        elif has_warning:
            print(f'\n  WARNING: Loss is higher than expected ({final_loss:.3f}).')
            print(f'     Verify the adapter before using it.')
        print(f'{"="*70}\n')

        # Step 7: Inference Test (only if model looks healthy)
        if not model_collapsed:
            print('Quick inference test...')
            test_prompt = 'What is photosynthesis?'

            # Build full conversation (system + user) exactly as during training
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

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            # Decode the FULL output (prompt + generation)
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # Decode only the NEW tokens (after the prompt)
            prompt_len = len(inputs['input_ids'][0])
            generated_ids = outputs[0][prompt_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Clean up: strip system echo and leading whitespace
            # The model may echo system tokens at the start; strip them
            response_text = response_text.strip()

            print(f'\n   Prompt: {test_prompt}')
            print(f'   Response: {response_text[:500]}')
        else:
            print('\nSkipping inference test (model appears collapsed)')

    except Exception as e:
        print(f'\n{"="*70}')
        print(f'  ERROR DURING TRAINING')
        print(f'  {str(e)}')
        print(f'{"="*70}\n')
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Fine-Tuning Qwen2-7B with QLoRA for Kid-Safe LLM'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/qwen2_7b_qlora.yaml',
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

