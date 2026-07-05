#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training for Kid-Safe LLM — TRL 1.6.0

Trains Qwen2-7B with QLoRA using DPO to reinforce safe behavior.

Usage:
    python scripts/train_dpo.py --config configs/qwen2_7b_dpo.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Hugging Face imports
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Config loaded from: {config_path}')
    return config


def load_dpo_dataset(config: dict):
    """Load DPO dataset from JSONL file in conversational format."""
    from datasets import load_dataset
    
    dataset_path = PROJECT_ROOT / 'data' / 'dpo' / 'dpo_dataset.jsonl'
    
    print(f'Loading DPO dataset: {dataset_path}')
    dataset = load_dataset(
        'json',
        data_files=str(dataset_path),
        split='train',
    )
    
    print(f'   Total pairs: {len(dataset)}')
    
    # TRL 1.6.0 expects conversational format:
    #   prompt: list of messages (system + user)
    #   chosen: list with assistant message
    #   rejected: list with assistant message
    def format_dpo_sample(sample):
        """Format DPO sample for TRL 1.6.0 conversational format."""
        system = sample.get('system', '')
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        return {
            'prompt': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt},
            ],
            'chosen': [{'role': 'assistant', 'content': chosen}],
            'rejected': [{'role': 'assistant', 'content': rejected}],
        }
    
    formatted_dataset = dataset.map(
        format_dpo_sample,
        remove_columns=dataset.column_names,
        desc='Formatting DPO samples',
    )
    
    print(f'   Dataset formatted: {len(formatted_dataset)} samples')
    return formatted_dataset


def train_dpo(config_path: str, resume: bool = False):
    """
    Main DPO training function for TRL 1.6.0.
    
    TRL 1.6.0 DPOTrainer loads the model internally when given a string name.
    It handles quantization (bnb_config) and PEFT (peft_config) automatically.
    """
    print('\n' + '='*70)
    print('  KID-SAFE LLM - DPO Safety Alignment Training (TRL 1.6.0)')
    print('  Model: Qwen2-7B-Instruct + LoRA (4-bit NF4)')
    print('  Hardware: NVIDIA RTX 5080 (16GB VRAM)')
    print('='*70 + '\n')
    
    config = load_config(config_path)

    # Step 1: Load DPO Dataset
    print('\nStep 1: Preparing DPO dataset...')
    dpo_dataset = load_dpo_dataset(config)
    
    # Step 2: Configure DPO Training
    print('\nStep 2: Configuring DPO training...')
    
    training_config = config['training']
    model_name = config['model']['name_or_path']
    
    # Load tokenizer for processing_class
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config['model'].get('trust_remote_code', True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # DPOConfig for TRL 1.6.0
    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['quantization_config']['load_in_4bit'],
        bnb_4bit_quant_type=config['model']['quantization_config']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16 if config['model']['quantization_config']['bnb_4bit_compute_dtype'] == 'float16' else torch.float32,
        bnb_4bit_use_double_quant=config['model']['quantization_config']['bnb_4bit_use_double_quant'],
    )
    
    dpo_args = DPOConfig(
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
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 50),
        save_total_limit=training_config.get('save_total_limit', 3),
        eval_strategy='no',
        report_to='tensorboard',
        # DPO-specific
        beta=training_config.get('dpo_beta', 0.1),
        # Pass quantization via model_init_kwargs (TRL 1.6.0 API)
        model_init_kwargs={"quantization_config": bnb_config},
    )
    
    # LoRA config for TRL 1.6.0
    lora_cfg = training_config['lora']
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        task_type=lora_cfg['task_type'],
        bias='none',
    )

    # Step 3: Create DPO Trainer
    print('\nStep 3: Creating DPO Trainer...')

    trainer = DPOTrainer(
        model=model_name,  # String name — TRL loads + quantizes internally
        ref_model=None,    # TRL auto-creates reference model
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Step 4: Train
    print('\nStep 4: Starting DPO training...')
    print(f'   Output dir: {training_config["output_dir"]}')
    print(f'   Epochs: {training_config["num_train_epochs"]}')
    print(f'   Batch size: {training_config["per_device_train_batch_size"]} x {training_config["gradient_accumulation_steps"]} (accumulated)')
    print(f'   Learning rate: {training_config["learning_rate"]}')
    print(f'   DPO beta: {training_config.get("dpo_beta", 0.1)}')
    print()
    
    try:
        train_result = trainer.train(resume_from_checkpoint=resume)
        
        # Post-training: save model
        print('\nStep 5: Saving DPO adapter...')
        final_output_dir = os.path.join(training_config['output_dir'], 'final')
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        
        print(f'\n{"="*70}')
        print('  DPO TRAINING COMPLETED!')
        print(f'  Adapter saved to: {final_output_dir}')
        if 'train_loss' in metrics:
            print(f'  Final loss: {metrics["train_loss"]:.4f}')
        print(f'{"="*70}\n')
        
    except Exception as e:
        print(f'\n{"="*70}')
        print(f'  ERROR DURING DPO TRAINING')
        print(f'  {str(e)}')
        print(f'{"="*70}\n')
        raise


def main():
    parser = argparse.ArgumentParser(
        description='DPO Training for Kid-Safe LLM (TRL 1.6.0)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/qwen2_7b_dpo.yaml',
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

