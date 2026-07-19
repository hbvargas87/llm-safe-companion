#!/usr/bin/env python3
"""
Inference script for Gemma 4 E4B (SFT + DPO merged).

Loads base model, applies SFT adapter (child-friendly tone),
then applies DPO adapter (safety alignment), and merges both.

Usage:
    python scripts/inference_gemma4.py
    python scripts/inference_gemma4.py --prompt "Hola, ¿qué es la fotosíntesis?"
    python scripts/inference_gemma4.py --interactive
"""

import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_fully_merged_model():
    """Load base + SFT + DPO merged model (in memory)."""
    print("=" * 60)
    print("  GEMMA 4 E4B - Kid-Safe Model Loader")
    print("=" * 60)

    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Step 1: Load base model
    print("\n[1/4] Loading base model (unsloth/gemma-4-E4B-it)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Base loaded. VRAM: {vram:.2f} GB")

    # Step 2: Apply SFT adapter (child-friendly tone)
    print("\n[2/4] Applying SFT adapter (child-friendly tone)...")
    sft_path = PROJECT_ROOT / "outputs" / "gemma4_4b_sft" / "final"
    model = PeftModel.from_pretrained(model, str(sft_path))
    print("  SFT LoRA applied.")

    # Step 3: Merge SFT
    print("  Merging SFT LoRA into base...")
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  SFT merged. VRAM: {vram:.2f} GB")

    # Step 4: Apply DPO adapter (safety alignment)
    print("\n[3/4] Applying DPO adapter (safety alignment)...")
    dpo_path = PROJECT_ROOT / "outputs" / "gemma4_4b_dpo" / "final"
    model = PeftModel.from_pretrained(model, str(dpo_path))
    print("  DPO LoRA applied.")

    # Step 5: Merge DPO
    print("  Merging DPO LoRA into base...")
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  DPO merged. VRAM: {vram:.2f} GB")

    # Prepare for inference
    print("\n[4/4] Preparing for inference...")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    print("  Gradient checkpointing disabled, use_cache=True enabled")

    print(f"\n  Model ready! VRAM used: {vram:.2f} GB")
    print(f"  SFT: {sft_path}")
    print(f"  DPO: {dpo_path}")
    return model, tokenizer


SYSTEM_MESSAGE = (
    "You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. "
    "You respond in an educational, empathetic, and age-appropriate manner. "
    "You always maintain a positive and reassuring tone. "
    "You are not a psychologist, but you always listen with understanding. "
    "You respond in the same language the child uses."
)


def generate(model, tokenizer, user_message: str, max_new_tokens: int = 200,
             temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True):
    """Generate a response to a user message."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    prompt_len = len(inputs["input_ids"][0])
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response


def interactive_mode(model, tokenizer):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'reset' to clear conversation history.")
    print("  Type 'params T TP' to change temperature and top_p")
    print("=" * 60)

    conversation = []
    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 200

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            conversation = []
            print("  Conversation history cleared.")
            continue

        if user_input.lower().startswith("params "):
            parts = user_input.split()
            if len(parts) >= 3:
                try:
                    temperature = float(parts[1])
                    top_p = float(parts[2])
                    print(f"  Parameters updated: temp={temperature}, top_p={top_p}")
                except ValueError:
                    print("  Usage: params <temperature> <top_p>")
            continue

        if not user_input:
            continue

        # Build conversation messages
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_input})

        print("Kid-Safe: ", end="", flush=True)

        try:
            response = generate(
                model, tokenizer, user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(response)
            print()

            # Update conversation history (keep last 6 turns for context)
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response})
            if len(conversation) > 12:
                conversation = conversation[-12:]

        except Exception as e:
            print(f"\n  ERROR: {e}")


def single_prompt(model, tokenizer, prompt: str):
    """Generate a single response."""
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    response = generate(model, tokenizer, prompt)
    print(f"Response: {response}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Gemma 4 E4B (SFT + DPO merged)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate a response for",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9)",
    )
    args = parser.parse_args()

    model, tokenizer = load_fully_merged_model()

    if args.prompt:
        single_prompt(model, tokenizer, args.prompt)
    elif args.interactive:
        interactive_mode(model, tokenizer)
    else:
        # Default: run a few test prompts
        print("\nRunning test prompts...\n")

        test_prompts = [
            "What is photosynthesis?",
            "¿Qué es la fotosíntesis?",
            "I'm feeling sad today, what should I do?",
            "¿Qué es un agujero negro?",
            "Can you help me with my homework about animals?",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i}/{len(test_prompts)} ---")
            print(f"Prompt: {prompt}")
            print("-" * 40)
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"Response: {response}")
            print()


if __name__ == "__main__":
    main()
