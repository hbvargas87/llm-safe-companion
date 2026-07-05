"""
Descarga datasets públicos seguros para entrenamiento infantil.
Datasets seleccionados:
1. dolly (instrucciones benignas)
2. self_instruct (variado, filtraremos)
3. kindergarten_math (matemáticas básicas)
"""

from datasets import load_dataset
import os

OUTPUT_DIR = "data/raw_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DESCARGANDO DATASETS PARA ENTRENAMIENTO INFANTIL")
print("=" * 60)

# 1. Dolly 15K - instrucciones benignas
print("\n📦 Descargando Dolly 15K...")
try:
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly.to_csv(os.path.join(OUTPUT_DIR, "dolly_train.csv"), index=False)
    print(f"   ✅ Dolly: {len(dolly)} ejemplos")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. OpenOrca (versión simplificada, filtraremos después)
print("\n📦 Descargando OpenOrca (mini)...")
try:
    orca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    orca_list = list(orca.take(5000))  # Tomamos 5k para empezar
    import pandas as pd
    pd.DataFrame(orca_list).to_csv(os.path.join(OUTPUT_DIR, "openorca_sample.csv"), index=False)
    print(f"   ✅ OpenOrca: {len(orca_list)} ejemplos (muestra)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Datasets educativos bilingües
print("\n📦 Descargando datos educativos bilingües...")
try:
    # OPUS100 para pares es/en (podemos usar para traducción educativa)
    opus = load_dataset("opus100/opus100", language_pair="en-es", split="train")
    opus_sample = opus.shuffle(seed=42).select(range(min(3000, len(opus))))
    opus_sample.to_csv(os.path.join(OUTPUT_DIR, "opus_es_en.csv"), index=False)
    print(f"   ✅ OPUS100 es-en: {len(opus_sample)} pares")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ Descarga completada. Revisa la carpeta data/raw_datasets/")
print("=" * 60)
