#!/usr/bin/env python3
"""
Gemma 4 E4B - Fine-Tuning Script with QLoRA + TRL
Trains gemma-4-E4B-it with QLoRA using HuggingFace + TRL

Usage:
    python scripts/gemma4_train.py --config configs/gemma4_4b_qlora.yaml
    python scripts/gemma4_train.py --config configs/gemma4_4b_qlora.yaml --resume

Key differences from Qwen2 training:
- Gemma 4 E4B is multimodal (vision+text+audio) but we fine-tune text only
- use_cache=True is CRITICAL for E2B/E4B (fixes garbage logits bug)
- Loss 13-15 is NORMAL for Gemma 4 (not model collapse)
- LoRA: r=16, alpha=16 (compromise between r=8 and r=32)
- Dataset format: conversations list with role/content dicts
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, Dataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f'Config loaded from: {config_path}')
    return config


def load_model_and_tokenizer(config: dict):
    """Load Gemma 4 E4B with 4-bit QLoRA via HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = config['model']['name_or_path']
    print(f'  Loading {model_name} with 4-bit QLoRA...')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

    # Apply chat template
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    print(f'  Chat template: gemma-4 (no thinking)')

    return model, tokenizer


def setup_lora(model, config: dict):
    """Configure LoRA adapters using PEFT."""
    from peft import LoraConfig, get_peft_model

    lora_config = config['training']['lora']
    target = lora_config.get('target_modules', "all-linear")
    if isinstance(target, str):
        if target == "all-linear":
            target = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config.get('lora_dropout', 0),
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def format_dataset_for_gemma4(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    """Format dataset for Gemma 4 chat template."""
    system_message = (
        'You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. '
        'You respond in an educational, empathetic, and age-appropriate manner. '
        'You always maintain a positive and reassuring tone. '
        'You are not a psychologist, but you always listen with understanding. '
        'You respond in the same language the child uses.'
    )

    def convert(sample):
        messages = sample['messages']
        has_system = any(m['role'] == 'system' for m in messages)
        if not has_system:
            messages = [{'role': 'system', 'content': system_message}] + messages

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        text = text.removeprefix('<bos>')
        if len(text) > max_seq_length:
            text = text[:max_seq_length]
        return {'text': text}

    formatted = dataset.map(convert, remove_columns=['messages'], desc='Formatting')
    print(f'  Formatted: {len(formatted)} samples')
    return formatted


def load_and_prepare_dataset(config: dict, tokenizer, max_seq_length: int):
    """Load and prepare dataset."""
    from unsloth.chat_templates import standardize_data_formats

    jsonl_path = PROJECT_ROOT / 'data' / 'filtered_datasets' / 'kidsafe_tone_dataset.jsonl'
    if not jsonl_path.exists():
        raise FileNotFoundError(f'Dataset not found: {jsonl_path}')

    print(f'Loading dataset: {jsonl_path.name}')
    dataset = load_dataset('json', data_files=str(jsonl_path), split='train')
    print(f'  Total samples: {len(dataset)}')

    dataset = standardize_data_formats(dataset)
    formatted = format_dataset_for_gemma4(dataset, tokenizer, max_seq_length)
    return formatted


def train(config_path: str, resume: bool = False):
    """Main training function."""
    from trl import SFTTrainer, SFTConfig

    print('\n' + '=' * 70)
    print('  GEMMA 4 E4B - Fine-Tuning with QLoRA + TRL')
    print('  Model: unsloth/gemma-4-E4B-it')
    print('  Hardware: NVIDIA RTX 5080 (16GB VRAM)')
    print('=' * 70 + '\n')

    config = load_config(config_path)
    torch.manual_seed(config['training'].get('seed', 42))

    # Step 1: Load Model
    print('\nStep 1: Loading model and tokenizer...')
    model, tokenizer = load_model_and_tokenizer(config)

    # Step 2: Setup LoRA
    print('\nStep 2: Configuring LoRA adapters...')
    model = setup_lora(model, config)

    # Step 3: Load Dataset
    print('\nStep 3: Preparing dataset...')
    max_seq_length = config['training']['max_seq_length']
    dataset = load_and_prepare_dataset(config, tokenizer, max_seq_length)

    # Step 4: Tokenize
    print('\nStep 4: Tokenizing dataset...')

    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, desc='Tokenizing')
    tokenized = tokenized.remove_columns(['text'])
    print(f'  Tokenized: {len(tokenized)} samples | Columns: {tokenized.column_names}')

    # Step 5: Training Config
    print('\nStep 5: Configuring training...')
    tc = config['training']

    training_args = SFTConfig(
        output_dir=tc['output_dir'],
        num_train_epochs=tc['num_train_epochs'],
        per_device_train_batch_size=tc['per_device_train_batch_size'],
        gradient_accumulation_steps=tc['gradient_accumulation_steps'],
        learning_rate=tc['learning_rate'],
        lr_scheduler_type=tc['lr_scheduler_type'],
        warmup_ratio=tc['warmup_ratio'],
        weight_decay=tc['weight_decay'],
        max_grad_norm=tc['max_grad_norm'],
        bf16=True,
        dataloader_num_workers=0,
        seed=tc.get('seed', 42),
        gradient_checkpointing=True,
        max_length=max_seq_length,
        logging_steps=tc.get('logging_steps', 10),
        save_steps=tc.get('save_steps', 100),
        save_total_limit=tc.get('save_total_limit', 3),
        eval_strategy='no',
        report_to='tensorboard',
        remove_unused_columns=False,
    )

    # Step 6: Create Trainer
    print('\nStep 6: Creating Trainer...')
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        processing_class=tokenizer,
        args=training_args,
    )

    # Step 7: Train
    print('\nStep 7: Starting training...')
    print(f'   Output dir: {tc["output_dir"]}')
    print(f'   Epochs: {tc["num_train_epochs"]}')
    print(f'   Batch size: {tc["per_device_train_batch_size"]} x {tc["gradient_accumulation_steps"]} (accum)')
    print(f'   Learning rate: {tc["learning_rate"]}')
    print(f'   Max length: {max_seq_length}')
    print(f'   LoRA: r={tc["lora"]["r"]}, alpha={tc["lora"]["lora_alpha"]}')
    print()

    try:
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

        # Post-training check
        final_loss = train_result.metrics.get('train_loss', None)
        is_nan = final_loss is not None and torch.isnan(torch.tensor(final_loss))
        model_collapsed = is_nan

        # Save
        print('\nStep 8: Saving model...')
        final_dir = os.path.join(tc['output_dir'], 'final')
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()

        print(f'\n{"=" * 70}')
        print('  TRAINING COMPLETED!')
        print(f'  Saved to: {final_dir}')
        print(f'  Final loss: {final_loss}')
        print(f'  NOTE: Loss 13-15 is NORMAL for Gemma 4 (multimodal quirk)')
        if model_collapsed:
            print(f'\n  WARNING: NaN loss detected! Retrain recommended.')
        print(f'{"=" * 70}\n')

        # Inference test
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

            # CRITICAL: Disable gradient checkpointing so use_cache=True works
            # Gemma 4 requires use_cache=True to avoid garbage generation
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            print('  Disabled gradient checkpointing, enabled use_cache=True')

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
        else:
            print('\nSkipping inference test (NaN loss)')

    except Exception as e:
        print(f'\n{"=" * 70}')
        print(f'  ERROR DURING TRAINING')
        print(f'  {str(e)}')
        print(f'{"=" * 70}\n')
        raise


def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning Gemma 4 E4B with QLoRA + TRL')
    parser.add_argument('--config', type=str, default='configs/gemma4_4b_qlora.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Config not found: {config_path}')
        sys.exit(1)

    train(config_path=args.config, resume=args.resume)


if __name__ == '__main__':
    main()

