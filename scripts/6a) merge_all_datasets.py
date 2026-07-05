#!/usr/bin/env python3
"""
Script de Merge Final para Kid-Safe LLM.

Combina todos los datasets disponibles en un solo archivo CSV listo para entrenamiento.

Entradas:
- data/filtered_datasets/no_robots_filtered_es.csv
- data/filtered_datasets/cosmopedia_filtered_es.csv
- data/filtered_datasets/synthetic_kids_data.csv
- data/filtered_datasets/kids_bilingual_data.csv
- data/filtered_datasets/dolly_filtered.csv (inglés, como respaldo)

Salida:
- data/filtered_datasets/kidsafe_final_dataset.csv
- data/filtered_datasets/kidsafe_final_dataset.jsonl (formato HuggingFace)
- data/filtered_datasets/merge_report.json (estadísticas detalladas)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

OUTPUT_DIR = "data/filtered_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Archivos de entrada (orden de prioridad)
DATASET_FILES = [
    {
        "name": "oasst2_filtered",
        "path": os.path.join(OUTPUT_DIR, "oasst2_filtered.csv"),
        "priority": 3,
        "max_samples": 24000,
    },
    {
        "name": "kids_bilingual",
        "path": os.path.join(OUTPUT_DIR, "kids_bilingual_data.csv"),
        "priority": 4,
        "max_samples": 500,
    },
    {
        "name": "dolly_en",
        "path": os.path.join(OUTPUT_DIR, "dolly_filtered.csv"),
        "priority": 5,
        "max_samples": 10000,
    },
]

# =============================================================================
# FUNCIONES DE NORMALIZACIÓN
# =============================================================================

def normalize_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Normaliza un DataFrame al formato estándar del proyecto.
    
    Formato esperado:
    - instruction: str
    - input: str (opcional)
    - output: str
    - category: str
    - language: str ('es' o 'en')
    """
    print(f"      Normalizando {source_name}...")
    
    # Mapeo de columnas comunes
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'instruction':
            col_map[col] = 'instruction'
        elif col_lower in ['response', 'output', 'answer', 'text']:
            col_map[col] = 'output'
        elif col_lower in ['input', 'context', 'question', 'prompt']:
            col_map[col] = 'input'
        elif col_lower in ['category', 'type', 'label']:
            col_map[col] = 'category'
        elif col_lower in ['language', 'lang', 'locale']:
            col_map[col] = 'language'
    
    df = df.rename(columns=col_map)
    
    # Asegurar columnas requeridas
    required_cols = ['instruction', 'output']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"      ⚠️  Columnas faltantes: {missing_cols}")
        return None
    
    # Rellenar columnas opcionales
    if 'input' not in df.columns:
        df['input'] = ''
    if 'category' not in df.columns:
        df['category'] = source_name
    if 'language' not in df.columns:
        # Detectar idioma automáticamente
        df['language'] = detect_language(df['instruction'])
    
    # Limpiar texto
    df['instruction'] = df['instruction'].astype(str).str.strip()
    df['input'] = df['input'].astype(str).str.strip()
    df['output'] = df['output'].astype(str).str.strip()
    
    # Eliminar filas vacías
    df = df[df['instruction'].apply(len) > 0]
    df = df[df['output'].apply(len) > 0]
    
    # Limitar a max_samples
    if len(df) > 100000:  # Límite de seguridad
        df = df.sample(n=100000, random_state=42)
    
    print(f"      ✅ {len(df)} muestras normalizadas")
    
    return df[['instruction', 'input', 'output', 'category', 'language']]


def detect_language(text: str) -> str:
    """
    Detecta si el texto está en español o inglés.
    Heurística simple basada en palabras clave y caracteres.
    """
    spanish_markers = ['el', 'la', 'los', 'las', 'un', 'una', 'es', 'son', 'de', 'del',
                       'que', 'en', 'por', 'con', 'para', 'como', 'más', 'tiene',
                       'qué', 'cuál', 'cuánto', 'dónde', 'cuándo', 'cómo', 'pero',
                       'porque', 'este', 'esta', 'ese', 'esa', 'su', 'sus', 'muy']
    
    english_markers = ['the', 'is', 'are', 'was', 'were', 'a', 'an', 'of', 'to', 'in',
                       'for', 'with', 'on', 'at', 'by', 'from', 'or', 'and', 'but',
                       'what', 'where', 'when', 'how', 'who', 'which', 'this', 'that']
    
    spanish_chars = set('áéíóúüñÁÉÍÓÚÜÑ')
    
    text_lower = text.lower()
    
    spanish_count = sum(1 for word in text_lower.split() if word.strip('.,;:!?()[]{}"\'') in spanish_markers)
    english_count = sum(1 for word in text_lower.split() if word.strip('.,;:!?()[]{}"\'') in english_markers)
    spanish_char_count = sum(1 for c in text if c in spanish_chars)
    
    # Si tiene caracteres acentuados españoles o más palabras en español
    if spanish_char_count > 2 or spanish_count > english_count:
        return 'es'
    else:
        return 'en'


def filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados basados en instruction + output."""
    before = len(df)
    df = df.drop_duplicates(subset=['instruction', 'output'])
    after = len(df)
    removed = before - after
    
    if removed > 0:
        print(f"      🗑️  {removed} duplicados eliminados")
    
    return df


def filter_too_similar(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Filtra muestras demasiado similares usando similitud de texto simple.
    (Versión ligera sin embeddings pesados)
    """
    if len(df) < 1000:
        return df  # Saltar para datasets pequeños
    
    print(f"      🔍 Filtrando similitud (> {threshold*100:.0f}%)...")
    
    # Usar similitud de palabras clave simples
    import re
    
    def get_keywords(text):
        """Extrae palabras clave únicas de un texto."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Eliminar stop words simples
        stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'es', 'son', 'de', 'del',
                      'que', 'en', 'por', 'con', 'para', 'como', 'a', 'the', 'is', 'are',
                      'to', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'or', 'and'}
        return set(w for w in words if w not in stop_words and len(w) > 2)
    
    keywords_list = df['instruction'].apply(get_keywords).tolist()
    
    unique_indices = []
    for i in range(len(df)):
        is_duplicate = False
        for j in unique_indices:
            # Jaccard similarity
            intersection = len(keywords_list[i] & keywords_list[j])
            union = len(keywords_list[i] | keywords_list[j])
            if union > 0 and (intersection / union) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
    
    df = df.iloc[unique_indices].reset_index(drop=True)
    print(f"      ✅ {len(df)} muestras únicas (similitud filtrada)")
    
    return df


# =============================================================================
# MERGE PRINCIPAL
# =============================================================================

def merge_all_datasets():
    """Combina todos los datasets disponibles."""
    print("\n" + "=" * 70)
    print("  MERGE FINAL DE DATASETS - KID-SAFE LLM")
    print("  Combinando todos los datasets en un solo archivo")
    print("=" * 70)
    
    all_datasets = []
    dataset_stats = {}
    
    # 1. Cargar y normalizar cada dataset
    print("\n📥 Paso 1: Cargando y normalizando datasets...")
    
    for dataset_info in DATASET_FILES:
        path = dataset_info["path"]
        name = dataset_info["name"]
        
        if not os.path.exists(path):
            print(f"   ⚠️  {name}: Archivo no encontrado, saltando...")
            print(f"      Ruta: {path}")
            dataset_stats[name] = {"status": "not_found", "samples": 0}
            continue
        
        try:
            df = pd.read_csv(path)
            print(f"   📄 {name}: {len(df)} muestras cargadas")
            
            normalized = normalize_dataset(df, name)
            
            if normalized is not None:
                all_datasets.append(normalized)
                dataset_stats[name] = {
                    "status": "loaded",
                    "samples": len(normalized),
                    "language_dist": normalized['language'].value_counts().to_dict(),
                }
            else:
                dataset_stats[name] = {"status": "failed", "samples": 0}
                
        except Exception as e:
            print(f"   ❌ {name}: Error al cargar - {str(e)}")
            dataset_stats[name] = {"status": "error", "error": str(e), "samples": 0}
    
    # 2. Combinar todos los datasets
    print("\n🔗 Paso 2: Combinando datasets...")
    
    if not all_datasets:
        print("   ❌ No hay datasets para combinar.")
        return None
    
    combined = pd.concat(all_datasets, ignore_index=True)
    print(f"   Total combinado: {len(combined)} muestras")
    
    # 3. Eliminar duplicados
    print("\n🧹 Paso 3: Eliminando duplicados...")
    combined = filter_duplicates(combined)
    
    # 4. Filtrar similitud
    print("\n🔍 Paso 4: Filtrando similitud...")
    combined = filter_too_similar(combined)
    
    # 5. Mezclar aleatoriamente
    print("\n🎲 Paso 5: Mezclando dataset...")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. Estadísticas finales
    print("\n📊 Paso 6: Generando estadísticas...")
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(combined),
        "by_language": combined['language'].value_counts().to_dict(),
        "by_category": combined['category'].value_counts().head(20).to_dict(),
        "datasets_used": {k: v for k, v in dataset_stats.items() if v.get('status') == 'loaded'},
        "avg_instruction_length": int(combined['instruction'].str.len().mean()),
        "avg_output_length": int(combined['output'].str.len().mean()),
        "min_instruction_length": int(combined['instruction'].str.len().min()),
        "max_instruction_length": int(combined['instruction'].str.len().max()),
    }
    
    # 7. Guardar outputs
    print("\n💾 Paso 7: Guardando archivos...")
    
    # CSV principal
    final_csv = os.path.join(OUTPUT_DIR, "kidsafe_final_dataset.csv")
    combined.to_csv(final_csv, index=False, encoding='utf-8')
    print(f"   ✅ CSV: {final_csv}")
    print(f"      Tamaño: {len(combined)} muestras")
    
    # JSONL (formato HuggingFace)
    final_jsonl = os.path.join(OUTPUT_DIR, "kidsafe_final_dataset.jsonl")
    with open(final_jsonl, 'w', encoding='utf-8') as f:
        for _, row in combined.iterrows():
            # Formato conversacional para Qwen2
            messages = []
            
            # System message
            messages.append({
                "role": "system",
                "content": "Eres Kid-Safe, un asistente amigable y seguro para niños de 6 a 12 años. Respondes de manera educativa, empática y apropiada para la edad."
            })
            
            # User message
            if row['input'] and str(row['input']).strip():
                user_content = f"{row['instruction']}\n\n{row['input']}"
            else:
                user_content = row['instruction']
            
            messages.append({"role": "user", "content": user_content})
            
            # Assistant message
            messages.append({"role": "assistant", "content": row['output']})
            
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
    
    print(f"   ✅ JSONL: {final_jsonl}")
    print(f"      Formato: HuggingFace conversational")
    
    # Reporte de merge
    report_path = os.path.join(OUTPUT_DIR, "merge_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Reporte: {report_path}")
    
    # Mostrar estadísticas
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    print(f"   Total de muestras: {stats['total_samples']}")
    print(f"   Español: {stats['by_language'].get('es', 0)}")
    print(f"   Inglés: {stats['by_language'].get('en', 0)}")
    print(f"\n   Top categorías:")
    for cat, count in list(stats['by_category'].items())[:10]:
        print(f"      {cat}: {count}")
    print(f"\n   Longitud promedio:")
    print(f"      Instruction: {stats['avg_instruction_length']} chars")
    print(f"      Output: {stats['avg_output_length']} chars")
    print(f"\n   📁 Ubicación: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")
    
    return combined


def main():
    df = merge_all_datasets()
    
    if df is not None:
        print("✅ MERGE COMPLETADO")
        print("\n  Siguiente paso: Entrenamiento")
        print("  1. Validar entorno: python scripts/validate_env.py")
        print("  2. Entrenar: python scripts/train.py --config configs/qwen2_7b_qlora.yaml")
        print()
    else:
        print("❌ MERGE FALLIDO. Revisa los mensajes de arriba.")
        print("   Asegúrate de haber ejecutado:")
        print("   - scripts/4a) download_spanish_datasets.py")
        print("   - scripts/5a) generate_synthetic_kids_data.py")
        print()


if __name__ == "__main__":
    main()
