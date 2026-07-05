# scripts/filter_datasets.py
"""
Filtra datasets para contenido apropiado para niños 6-12 años.
- Elimina contenido violento, sexual, político, etc.
- Mantiene educación, ciencia, matemáticas, creatividad
- Traduce/adapta al español donde sea posible
"""

import pandas as pd
import re
import os

INPUT_DIR = "data/raw_datasets"
OUTPUT_DIR = "data/filtered_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Palabras/frases que indican contenido inapropiado
BLOCKLIST = [
    # Violencia
    r'\b(kill|murder|death|die|dead|violence|attack|weapon|gun|blood|pain|hurt|die|suicide)\b',
    r'\b(matar|muerte|matar|violencia|ataque|arma|sangre|dolor|suicidio)\b',
    
    # Sexual
    r'\b(sex|nude|porn|adult|erotic|lust|naked|breast|genital)\b',
    r'\b(sexo|desnudo|porno|erótico|desnuda|pecho|genital)\b',
    
    # Político/controversial
    r'\b(politics|vote|election|biden|trump|obama|war|terrorism)\b',
    r'\b(política|votar|elección|guerra|terrorismo)\b',
    
    # Sustancias
    r'\b(drug|alcohol|marijuana|weed|cocaine|smoking)\b',
    r'\b(droga|alcohol|marihuana|cocaína|fumar)\b',
    
    # Religioso controversial
    r'\b(religion|god|jesus|allah|islam|christian|atheist)\b',
    r'\b(religión|dios|islám|cristiano|ateo)\b',
    
    # Otros
    r'\b(racism|hate|discriminat|abuse|bully)\b',
    r'\b(racismo|odio|discriminación|abuso|acoso)\b',
]

blocklist_pattern = re.compile('|'.join(BLOCKLIST), re.IGNORECASE)

# Categorías educativas que SÍ queremos
EDUCATIONAL_KEYWORDS = [
    r'\b(math|science|learn|teach|explain|what is|how does|why|count|number|animal|plant)\b',
    r'\b(matemática|ciencia|aprender|enseñar|explicar|qué es|cómo|por qué|contar|número|animal|planta)\b',
    r'\b(story|create|draw|imagine|fun|game|play|friend)\b',
    r'\b(historia|crear|dibujar|imaginar|divertido|juego|jugar|amigo)\b',
]

educational_pattern = re.compile('|'.join(EDUCATIONAL_KEYWORDS), re.IGNORECASE)

def is_safe(text):
    """Verifica que el texto no contenga contenido inapropiado"""
    return not bool(blocklist_pattern.search(text))

def is_educational(text):
    """Verifica que el texto tenga contenido educativo"""
    return bool(educational_pattern.search(text))

def filter_dolly():
    print("\n🔍 Filtrando Dolly 15K...")
    df = pd.read_csv(os.path.join(INPUT_DIR, "dolly_train.csv"))
    print(f"   Original: {len(df)} ejemplos")
    
    # Filtrar por seguridad
    safe_mask = df.apply(
        lambda row: is_safe(str(row['instruction'])) and is_safe(str(row.get('context', ''))), 
        axis=1
    )
    df_safe = df[safe_mask]
    print(f"   Después de filtro de seguridad: {len(df_safe)}")
    
    # Priorizar educativo (pero no excluir todo lo demás benigno)
    edu_mask = df_safe.apply(lambda row: is_educational(str(row['instruction'])), axis=1)
    df_edu = df_safe[edu_mask]
    
    # Mezclar: 80% educativo, 20% benigno general
    df_other = df_safe[~edu_mask].sample(frac=0.3, random_state=42)
    df_final = pd.concat([df_edu, df_other], ignore_index=True)
    
    # Preparar formato instruct
    records = []
    for _, row in df_final.iterrows():
        records.append({
            'instruction': row['instruction'],
            'input': row.get('context', ''),
            'output': row['response'],
            'category': row.get('category', 'general'),
            'language': 'en'
        })
    
    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "dolly_filtered.csv"), index=False)
    print(f"   ✅ Final: {len(df_out)} ejemplos")
    return df_out

def filter_openorca():
    print("\n🔍 Filtrando OpenOrca...")
    df = pd.read_csv(os.path.join(INPUT_DIR, "openorca_sample.csv"))
    print(f"   Original: {len(df)} ejemplos")
    
    safe_mask = df.apply(
        lambda row: is_safe(str(row['question'])) and is_safe(str(row.get('response', ''))), 
        axis=1
    )
    df_safe = df[safe_mask]
    print(f"   Después de filtro de seguridad: {len(df_safe)}")
    
    # Tomar máximo 2000 ejemplos seguros
    df_final = df_safe.head(2000)
    
    records = []
    for _, row in df_final.iterrows():
        records.append({
            'instruction': row['question'],
            'input': '',
            'output': row['response'],
            'category': 'general',
            'language': 'en'
        })
    
    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "openorca_filtered.csv"), index=False)
    print(f"   ✅ Final: {len(df_out)} ejemplos")
    return df_out

print("=" * 60)
print("FILTRANDO DATASETS PARA CONTENIDO INFANTIL")
print("=" * 60)

dolly_filtered = filter_dolly()
openorca_filtered = filter_openorca()

# Combinar todo
combined = pd.concat([dolly_filtered, openorca_filtered], ignore_index=True)
combined.to_csv(os.path.join(OUTPUT_DIR, "combined_en_filtered.csv"), index=False)

print("\n" + "=" * 60)
print(f"✅ Combinado total: {len(combined)} ejemplos en inglés")
print(f"📁 Guardado en: {OUTPUT_DIR}/")
print("=" * 60)