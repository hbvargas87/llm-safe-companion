import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel
import os

# Configuración
INPUT_CSV = "data/filtered_datasets/dolly_filtered.csv"
OUTPUT_CSV = "data/filtered_datasets/dolly_filtered_es.csv"
BATCH_SIZE = 16  # Ajusta según tu VRAM (16-32 es seguro con el modelo de traducción)

def translate_batch(texts, tokenizer, model, device):
    """Traduce un lote de textos usando el modelo de traducción."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def normalize_columns(df): 
    """Normaliza los nombres de columnas para manejar variaciones entre datasets.""" 
    col_map = {} 
    for col in df.columns: 
        col_lower = col.lower().strip() 
        if col_lower == 'instruction': col_map[col] = 'instruction' 
        elif col_lower in ['response', 'output', 'answer']: col_map[col] = 'response' 
        elif col_lower in ['input', 'context', 'question']: col_map[col] = 'input' 
        elif col_lower == 'category': col_map[col] = 'category' 
        elif col_lower == 'language': col_map[col] = 'language'

    df = df.rename(columns=col_map)

    # Asegurar que las columnas existan
    if 'instruction' not in df.columns:
        raise ValueError(f"No se encontró una columna de 'instruction'. Columnas disponibles: {list(df.columns)}")
    if 'response' not in df.columns:
        raise ValueError(f"No se encontró una columna de 'response'/'output'. Columnas disponibles: {list(df.columns)}")
    if 'input' not in df.columns:
        df['input'] = ''
    if 'category' not in df.columns:
        df['category'] = ''
    if 'language' not in df.columns:
        df['language'] = 'en'
        
    return df

def main(): 
    # Cargar datos 
    print("Cargando dataset filtrado...") 
    df = pd.read_csv(INPUT_CSV)
    print(f"Columnas originales: {list(df.columns)}")
    df = normalize_columns(df)
    print(f"Columnas normalizadas: {list(df.columns)}")

    # Seleccionar columnas a traducir (instruction y response)
    # Usamos ||||| como separador único para dividir después
    texts_to_translate = df.apply(
        lambda row: f"{row['instruction']} ||||| {row['response']}", 
        axis=1
    ).tolist()

    print(f"Total de ejemplos a traducir: {len(texts_to_translate)}")

    # Cargar modelo de traducción EN -> ES
    print("Cargando modelo de traducción Helsinki-NLP/opus-mt-en-es...")
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Usando dispositivo: {device}")
    print("Iniciando traducción...")

     # Traducir en lotes
    translated_texts = []
    current_batch_size = BATCH_SIZE
    for i in range(0, len(texts_to_translate), current_batch_size):
        batch = texts_to_translate[i:i+current_batch_size]
        try:
            translated_batch = translate_batch(batch, tokenizer, model, device)
            translated_texts.extend(translated_batch)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Memoria insuficiente. Reduciendo batch size a 8...")
                current_batch_size = 8
                torch.cuda.empty_cache()
                # Reintentar el lote actual con batch size reducido
                translated_batch = translate_batch(batch, tokenizer, model, device)
                translated_texts.extend(translated_batch)
            else:
                raise e
    
        if (i + current_batch_size) % 1000 == 0:
            print(f"Procesados {i + current_batch_size}/{len(texts_to_translate)} ejemplos...")

    # Separar instruction y response traducidos
    df['instruction_es'] = [t.split(" ||||| ")[0] for t in translated_texts]
    df['response_es'] = [t.split(" ||||| ")[1] if " ||||| " in t else "" for t in translated_texts]

    # Crear nuevo dataframe solo con columnas en español
    df_es = df[['instruction_es', 'input', 'response_es', 'category', 'language']].copy()
    df_es.columns = ['instruction', 'input', 'response', 'category', 'language']
    df_es['language'] = 'es'

    # Guardar
    df_es.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Traducción completada. Guardado en: {OUTPUT_CSV}")
    print(f"Total de ejemplos en español: {len(df_es)}")

    # Liberar memoria
    del model, tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
