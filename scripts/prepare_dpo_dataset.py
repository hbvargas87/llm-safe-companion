"""
Prepare DPO dataset by downloading from Hugging Face and mixing with kid-safe safety pairs.

This script:
1. Downloads latam-gpt/tulu-3-dpo-spanish from Hugging Face
2. Converts it to our JSONL format with safety categories
3. Mixes with our 30 kid-safe safety pairs
4. Outputs final dataset for DPO training
"""

import json
import os
import random
from datasets import load_dataset

# Configuration
HUGGINGFACE_DATASET = "latam-gpt/tulu-3-dpo-spanish"
OUTPUT_FILE = "data/filtered_datasets/dpo_kidsafe.jsonl"
MAX_HF_PAIRS = 500  # Take 500 pairs from HF dataset
KIDSAFE_PAIRS_FILE = "data/filtered_datasets/dpo_kidsafe_kids.jsonl"  # Our kid-safe pairs

def load_kidsafe_pairs():
    """Load our kid-safe safety pairs."""
    pairs = []
    if os.path.exists(KIDSAFE_PAIRS_FILE):
        with open(KIDSAFE_PAIRS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line))
        print(f"✓ Loaded {len(pairs)} kid-safe safety pairs from {KIDSAFE_PAIRS_FILE}")
    else:
        print(f"⚠ Warning: {KIDSAFE_PAIRS_FILE} not found. Will only use HF dataset.")
    return pairs

def convert_hf_sample(sample):
    """Convert Hugging Face dataset sample to our JSONL format."""
    # Extract chosen and rejected content
    chosen_content = sample["chosen"]["content"] if isinstance(sample["chosen"], dict) else sample["chosen"]
    rejected_content = sample["rejected"]["content"] if isinstance(sample["rejected"], dict) else sample["rejected"]
    
    # Create a generic category based on source if available
    category = sample.get("source", "general_safety")
    
    return {
        "system": "Eres Kid, un asistente amigable y seguro para niños. Siempre respondes de forma positiva, educativa y apropiada para la edad. Si alguien te hace una pregunta peligrosa o inapropiada, rechazas amablemente la pregunta y rediriges a un tema seguro y educativo.",
        "prompt": sample["translated_prompt"] if "translated_prompt" in sample and sample["translated_prompt"] else sample["prompt"],
        "chosen": chosen_content,
        "rejected": rejected_content,
        "category": category
    }

def download_and_prepare_hf_dataset(max_pairs=500):
    """Download HF dataset and convert to our format."""
    print(f"Downloading {HUGGINGFACE_DATASET} from Hugging Face...")
    dataset = load_dataset(HUGGINGFACE_DATASET, split="train")
    
    print(f"✓ Downloaded {len(dataset)} samples. Converting to JSONL format...")
    
    # Shuffle and take subset
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(max_pairs, len(dataset))))
    
    converted_pairs = []
    for sample in dataset:
        try:
            converted = convert_hf_sample(sample)
            converted_pairs.append(converted)
        except Exception as e:
            print(f"⚠ Skipping sample due to error: {e}")
            continue
    
    print(f"✓ Converted {len(converted_pairs)} samples from HF dataset")
    return converted_pairs

def save_dataset(pairs, output_file):
    """Save pairs to JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {len(pairs)} pairs to {output_file}")

def main():
    print("=" * 60)
    print("Preparing DPO Dataset: Kid-Safe + General Safety")
    print("=" * 60)
    
    # Load kid-safe pairs
    kidsafe_pairs = load_kidsafe_pairs()
    
    # Download and convert HF dataset
    hf_pairs = download_and_prepare_hf_dataset(max_pairs=MAX_HF_PAIRS)
    
    # Mix both datasets
    all_pairs = kidsafe_pairs + hf_pairs
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)
    
    # Save
    save_dataset(all_pairs, OUTPUT_FILE)
    
    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    print(f"Total pairs: {len(all_pairs)}")
    print(f"  - Kid-safe safety pairs: {len(kidsafe_pairs)}")
    print(f"  - General safety pairs (HF): {len(hf_pairs)}")
    
    # Category distribution
    categories = {}
    for pair in all_pairs:
        cat = pair["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print("\n✓ Dataset preparation complete!")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Ready for DPO training with: train_dpo.py")

if __name__ == "__main__":
    main()