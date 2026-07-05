"""
Test inference with the trained adapter.
Loads Qwen2-7B (4-bit) + LoRA adapter and runs a quick generation test.
"""
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = r"F:\LLM\Proyectos\Stardusts\outputs\qwen2_7b_sft\final"

SYSTEM_PROMPT = (
    "You are Kid-Safe, a friendly and safe assistant for children aged 6 to 12. "
    "You respond in an educational, empathetic, and age-appropriate manner. "
    "You always maintain a positive and reassuring tone. "
    "You are not a psychologist, but you always listen with understanding. "
    "You respond in the same language the child uses."
)

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

PROMPTS = [
    "What is photosynthesis?",
    "¿Qué es la fotosíntesis?",
    "Why is the sky blue?",
    "¿Por qué el cielo es azul?",
    "Tell me a fun fact about animals.",
    "Cuéntame un dato divertido sobre los animales.",
]


def main():
    print("=" * 70)
    print("  KID-SAFE LLM — Inference Test")
    print("=" * 70)
    print()

    # Load base model (4-bit)
    print("Loading base model (Qwen2-7B-Instruct, 4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load adapter
    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print()

    # Test prompts
    for prompt in PROMPTS:
        print("-" * 70)
        print(f"Prompt: {prompt}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the NEW tokens (after the prompt)
        prompt_len = len(inputs["input_ids"][0])
        generated_ids = outputs[0][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up: strip any system echo
        response = response.strip()
        if response.startswith("system"):
            # Model echoed system tokens; try to find actual response
            for tag in ["assistant\n", "assistant:", "## Assistant"]:
                if tag in response:
                    response = response.split(tag, 1)[-1].strip()
                    break

        print(f"Response: {response[:500]}")
        if len(response) > 500:
            print(f"  ... ({len(response)} chars total)")
        print()

    print("=" * 70)
    print("  Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

