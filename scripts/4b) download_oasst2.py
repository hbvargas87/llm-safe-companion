#!/usr/bin/env python3
"""
Descarga y filtrado de OpenAssistant/oasst2 para Kid-Safe LLM.

Estrategia mejorada:
1. Descargar oasst2 completo en memoria (es ligero: 200MB)
2. Filtrar por español, seguridad, review
3. Agrupar por message_tree_id (árboles de conversación)
4. Extraer pares instruction-output de cada árbol
5. Formatear para Qwen2

Salida: data/filtered_datasets/oasst2_filtered.csv
"""

import os
import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import json

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

OUTPUT_DIR = "data/filtered_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SAMPLES = 50000  # Objetivo de muestras

# =============================================================================
# FILTROS DE SEGURIDAD
# =============================================================================

BLOCKLIST = [
    # Violencia
    r'\b(kill|murder|death|die|dead|violence|attack|weapon|gun|blood|pain|hurt|suicide)\b',
    r'\b(matar|muerte|muerto|violencia|ataque|arma|sangre|dolor|suicidio)\b',
    
    # Sexual
    r'\b(sex|nude|porn|adult|erotic|lust|naked|breast|genital)\b',
    r'\b(sexo|desnudo|porno|erótico|desnuda|pecho|genital)\b',
    
    # Político/controversial
    r'\b(politics|vote|election|biden|trump|war|terrorism)\b',
    r'\b(política|votar|elección|guerra|terrorismo)\b',
    
    # Sustancias
    r'\b(drug|alcohol|marijuana|cocaine|smoking)\b',
    r'\b(droga|alcohol|marihuana|cocaína|fumar)\b',
    
    # Odio
    r'\b(racism|hate|discriminat|abuse|bully)\b',
    r'\b(racismo|odio|discriminación|abuso|acoso)\b',
    
    # Auto-lesión
    r'\b(self.harm|self.injuri)\b',
]

blocklist_pattern = re.compile('|'.join(BLOCKLIST), re.IGNORECASE)


def is_safe(text: str) -> bool:
    """Verifica que el texto no contenga contenido inapropiado."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return not bool(blocklist_pattern.search(text_lower))


# =============================================================================
# FUNCIONES DE EXTRACCIÓN
# =============================================================================

def extract_conversation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae pares instruction-output de OASST2 usando la estructura de árbol.
    Estrategia: Para cada tree, emparejar prompts con sus hijos assistants directos.
    """
    pairs = []

    # Agrupar por árbol de mensaje
    for tree_id, group in df.groupby('message_tree_id'):
        if len(group) < 2:
            continue

        # Identificar prompts en este árbol
        prompts = group[group['role'] == 'prompter']

        # Para cada prompt, encontrar sus hijos directos (assistant)
        for _, prompt_row in prompts.iterrows():
            prompt_id = prompt_row['message_id']
            prompt_text = prompt_row['text']

            # Buscar hijos directos de este prompt
            children = group[group['parent_id'] == prompt_id]
            assistants = children[children['role'] == 'assistant']

            if assistants.empty:
                continue

            # Ordenar por rank (manejo de NaN con na_position)
            # En OASST2, rank NaN suele indicar el primer mensaje o el prompt mismo
            # Aquí ordenamos los assistants por rank para priorizar las mejores respuestas
            assistants_sorted = assistants.sort_values(by='rank', na_position='last')

            # Tomar la mejor respuesta (rank más bajo, o el primero si hay NaN)
            # O tomar todas si se desea más variedad, pero para calidad, tomamos la top 1 o top 2
            for _, child_row in assistants_sorted.head(2).iterrows():
                pairs.append({
                        'instruction': prompt_text,
                        'output': child_row['text'],
                    'category': 'conversation'
                })

    return pd.DataFrame(pairs)


        # =============================================================================
# DESCARGA Y FILTRADO
        # =============================================================================

def download_oasst2():
    """
    Descarga y filtra OpenAssistant/oasst2.

    Estrategia:
    - Descargar dataset completo (es ligero: 200MB)
    - Filtrar por lang == 'es'
    - Filtrar por seguridad (detoxify + blocklist)
    - Filtrar por review_result == True y deleted == False
    - Agrupar por message_tree_id
    - Extraer pares user/assistant de cada árbol
    """
    print("\n" + "=" * 70)
    print("  DESCARGANDO: OpenAssistant/oasst2")
    print("=" * 70)
        
    output_path = os.path.join(OUTPUT_DIR, "oasst2_filtered.csv")

    try:
        print("   Cargando oasst2 (modo no streaming para agrupar)...")
        dataset = load_dataset(
            "OpenAssistant/oasst2",
            split="train",
        )

        print(f"   Dataset cargado: {len(dataset)} mensajes totales")

        # Convertir a DataFrame para filtrado rápido
        df = pd.DataFrame(dataset)
        print(f"   Convertido a DataFrame: {len(df)} filas")

        # Filtro 1: Solo español
        print("   Filtro 1: Idioma español...")
        df_es = df[df['lang'] == 'es'].copy()
        print(f"      Español: {len(df_es)} / {len(df)} ({len(df_es)/len(df)*100:.1f}%)")

        # Filtro 2: No eliminado
        print("   Filtro 2: No eliminado...")
        df_not_deleted = df_es[~df_es['deleted']].copy()
        print(f"      No eliminados: {len(df_not_deleted)} ({len(df_not_deleted)/len(df_es)*100:.1f}%)")

        # Filtro 3: Revisado y aprobado
        print("   Filtro 3: Revisado y aprobado...")
        df_reviewed = df_not_deleted[df_not_deleted['review_result'] == True].copy()
        print(f"      Aprobados: {len(df_reviewed)} ({len(df_reviewed)/len(df_not_deleted)*100:.1f}%)")

        # Filtro 4: Seguridad por detoxify
        print("   Filtro 4: Seguridad por detoxify...")

        def check_detoxify(row):
            detoxify = row.get('detoxify', {})
            if detoxify is None:
                return True  # Si no tiene detoxify, lo aceptamos temporalmente
            toxicity = detoxify.get('toxicity', 0)
            severe_toxicity = detoxify.get('severe_toxicity', 0)
            obscene = detoxify.get('obscene', 0)
            # Umbral conservador
            return toxicity <= 0.3 and severe_toxicity <= 0.1 and obscene <= 0.3

        df_safe_detox = df_reviewed[df_reviewed.apply(check_detoxify, axis=1)].copy()
        print(f"      Seguros (detoxify): {len(df_safe_detox)} ({len(df_safe_detox)/len(df_reviewed)*100:.1f}%)")

        # Filtro 5: Seguridad por blocklist
        print("   Filtro 5: Seguridad por blocklist...")
        df_safe_blocklist = df_safe_detox[df_safe_detox['text'].apply(is_safe)].copy()
        print(f"      Seguros (blocklist): {len(df_safe_blocklist)} ({len(df_safe_blocklist)/len(df_safe_detox)*100:.1f}%)")

        # Filtro 6: Longitud mínima
        print("   Filtro 6: Longitud mínima...")
        df_safe_blocklist = df_safe_blocklist[df_safe_blocklist['text'].str.len() >= 20].copy()
        print(f"      Con longitud válida: {len(df_safe_blocklist)} ({len(df_safe_blocklist)/len(df_safe_detox)*100:.1f}%)")

        print(f"\n   Total mensajes seguros en español: {len(df_safe_blocklist)}")

        if len(df_safe_blocklist) == 0:
            print("   ❌ No se encontraron mensajes seguros en español.")
            return None
        
# =============================================================================
        # EXTRAER PARES INSTRUCTION-OUTPUT DE ÁRBOLES
# =============================================================================
        print("\n" + "-" * 70)
        print("  EXTRAYENDO PARES DE CONVERSACIONES")
        print("- " * 70)

        # Usar la nueva función de extracción
        extracted_pairs = extract_conversation_pairs(df_safe_blocklist)

        print(f"   Pares instruction-output extraídos: {len(extracted_pairs)}")

        if len(extracted_pairs) == 0:
            print("   No se pudieron extraer pares válidos con la estructura nativa.")
            print("   Intentando estrategia alternativa...")

            # ESTRATEGIA ALTERNATIVA: Usar mensajes individuales
            # Cada mensaje de prompter como instruction, cada assistant como output
            prompter_messages = df_safe_blocklist[df_safe_blocklist['role'] == 'prompter']
            assistant_messages = df_safe_blocklist[df_safe_blocklist['role'] == 'assistant']

            print(f"   Mensajes de prompter: {len(prompter_messages)}")
            print(f"   Mensajes de assistant: {len(assistant_messages)}")

            # Emparejar por índice (asumiendo que vienen en orden)
            min_len = min(len(prompter_messages), len(assistant_messages))
            samples = []
            for i in range(min_len):
                samples.append({
                    'instruction': prompter_messages.iloc[i]['text'].strip(),
                    'input': '',
                    'output': assistant_messages.iloc[i]['text'].strip(),
                    'category': 'conversation',
                    'language': 'es',
                })

                if len(samples) >= TARGET_SAMPLES:
                    break

            extracted_pairs = pd.DataFrame(samples)
            print(f"   Pares alternativos: {len(extracted_pairs)}")

        if len(extracted_pairs) == 0:
            print("   ❌ No se encontraron muestras válidas con ninguna estrategia.")
            return None

        # Limitar a TARGET_SAMPLES si hay más
        if len(extracted_pairs) > TARGET_SAMPLES:
            extracted_pairs = extracted_pairs.sample(n=TARGET_SAMPLES, random_state=42).reset_index(drop=True)
            print(f"   Limitado a {TARGET_SAMPLES} muestras.")

        # Asegurar columnas requeridas
        required_cols = ['instruction', 'input', 'output', 'category', 'language']
        for col in required_cols:
            if col not in extracted_pairs.columns:
                if col == 'input':
                    extracted_pairs['input'] = ''
                elif col == 'language':
                    extracted_pairs['language'] = 'es'
                elif col == 'category':
                    extracted_pairs['category'] = 'conversation'
                else:
                    extracted_pairs[col] = ''

        # Guardar
        extracted_pairs.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n   ✅ oasst2 filtrado: {len(extracted_pairs)} muestras")
        print(f"   Guardado en: {output_path}")

        # Estadísticas por categoría
        if 'category' in extracted_pairs.columns:
            cat_counts = extracted_pairs['category'].value_counts()
            print(f"\n   Distribución por categoría:")
            for cat, cnt in cat_counts.items():
                print(f"      {cat}: {cnt}")

        # Mostrar ejemplos
        print(f"\n   Ejemplos de muestras:")
        for i in range(min(5, len(extracted_pairs))):
            print(f"\n   --- Ejemplo {i+1} ---")
            print(f"   Instrucción: {extracted_pairs.iloc[i]['instruction'][:150]}...")
            print(f"   Respuesta: {extracted_pairs.iloc[i]['output'][:150]}...")

        return extracted_pairs
 
    except Exception as e:
        print(f"   Error al descargar oasst2: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  KID-SAFE LLM - Descarga de OpenAssistant/oasst2")
    print("  Objetivo: 50K muestras en español seguras para niños")
    print("#" * 70)

    df_oasst2 = download_oasst2()

    if df_oasst2 is not None:
        print(f"\n   ✅ Dataset oasst2 listo: {len(df_oasst2)} muestras")
        print(f"   📁 Ubicación: {OUTPUT_DIR}/oasst2_filtered.csv")

        print("\n  Siguiente paso: Combinar con otros datasets")
        print("  Ejecuta: python scripts/6a) merge_all_datasets.py")
    else:
        print("\n   ❌ No se pudo descargar oasst2.")
        print("   Verifica la conexión a HuggingFace y el nombre del dataset.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


