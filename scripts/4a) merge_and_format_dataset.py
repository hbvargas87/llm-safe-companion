# F:\LLM\Proyectos\Stardusts\scripts\merge_and_format_dataset.py
import pandas as pd
import json
import os

# 1. Cargar datasets
print("Cargando datasets...")
dolly_path = "data/filtered_datasets/dolly_filtered.csv"
openorca_path = "data/filtered_datasets/openorca_filtered.csv"
kids_path = "data/filtered_datasets/kids_bilingual_data.csv"

dfs = []
for path in [dolly_path, openorca_path, kids_path]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ {os.path.basename(path)}: {len(df)} ejemplos")
        dfs.append(df)
    else:
        print(f"  ⚠️  {os.path.basename(path)} no encontrado, se omitirá.")

if not dfs:
    raise FileNotFoundError("No se encontraron datasets para unir.")

# Unir todos
df_combined = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Total combinado: {len(df_combined)} ejemplos")

# 2. Formatear para Qwen (ChatML / Messages format compatible with TRL)
def format_for_qwen(row):
    instruction = str(row.get('instruction', ''))
    context = str(row.get('input', ''))
    output = str(row.get('output', ''))
    
    # Construir mensaje del usuario
    if context and context.strip():
        user_msg = f"{instruction}\n\n{context}"
    else:
        user_msg = instruction
        
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output}
        ],
        "category": row.get('category', 'general'),
        "language": row.get('language', 'en')
    }

print("Formateando para Qwen/TRL...")
formatted_data = df_combined.apply(format_for_qwen, axis=1).tolist()

# 3. Guardar en JSONL (formato estándar para TRL SFTTrainer)
output_path = "data/final_dataset/kids_qwen_sft.jsonl"
os.makedirs("data/final_dataset", exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ Dataset final guardado: {output_path}")
print(f"   Total de ejemplos: {len(formatted_data)}")
print(f"   Formato: JSONL con estructura 'messages' para TRL SFTTrainer")

# Estadísticas rápidas
langs = pd.Series([item['language'] for item in formatted_data]).value_counts()
cats = pd.Series([item['category'] for item in formatted_data]).value_counts()
print(f"\n🌍 Distribución por idioma:\n{langs}")
print(f"\n📚 Top 5 categorías:\n{cats.head()}")

