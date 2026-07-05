"""
Script para reconstruir el dataset JSONL a partir de lotes procesados.
Con borrado automático de Invalid \escape y reporte de errores restantes.

Uso:
    python scripts/7c) merge_tone_batches.py

Este script:
1. Lee todos los archivos batch_XXX_tone.jsonl del directorio batches_rewritten/
2. Los ordena por número de lote
3. Borra automáticamente las líneas con "Invalid \escape"
4. Genera un reporte de los errores restantes (Unterminated string, etc.)
5. Los une en un solo archivo kidsafe_tone_dataset.jsonl
6. Muestra estadísticas del dataset reconstruido

Salida:
    data/filtered_datasets/kidsafe_tone_dataset.jsonl
    data/filtered_datasets/merge_remaining_errors.txt
"""

import json
import os
import argparse
import sys
from collections import Counter
from datetime import datetime


def validate_jsonl_file(filepath, batch_name):
    """
    Valida un archivo JSONL y retorna:
    - muestras_validas: lista de muestras válidas
    - errores_validados: lista de errores que NO son Invalid \escape
    - borrados: número de líneas con Invalid \escape borradas
    """
    muestras_validas = []
    errores_validados = []
    borrados = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Saltar líneas vacías
            
            try:
                sample = json.loads(line)
                
                # Validar estructura básica
                if not isinstance(sample, dict):
                    raise ValueError("El objeto no es un diccionario")
                
                if 'messages' not in sample:
                    raise ValueError("Falta el campo 'messages'")
                
                if not isinstance(sample['messages'], list):
                    raise ValueError("El campo 'messages' no es una lista")
                
                # Validar que cada mensaje tenga role y content
                for msg_idx, msg in enumerate(sample['messages']):
                    if not isinstance(msg, dict):
                        raise ValueError(f"Mensaje en índice {msg_idx} no es un diccionario")
                    if 'role' not in msg:
                        raise ValueError(f"Mensaje en índice {msg_idx} falta 'role'")
                    if 'content' not in msg:
                        raise ValueError(f"Mensaje en índice {msg_idx} falta 'content'")
                
                muestras_validas.append(sample)
                
            except json.JSONDecodeError as e:
                error_msg = str(e)
                
                # Si es Invalid \escape, lo borramos silenciosamente
                if 'Invalid \\escape' in error_msg or 'Invalid \escape' in error_msg:
                    borrados += 1
                    continue
                
                # Otros errores se reportan
                errores_validados.append({
                    'archivo': batch_name,
                    'linea': line_num,
                    'tipo': 'JSON inválido',
                    'mensaje': error_msg,
                    'snippet': line[:200] + ('...' if len(line) > 200 else '')
                })
            except ValueError as e:
                errores_validados.append({
                    'archivo': batch_name,
                    'linea': line_num,
                    'tipo': 'Estructura inválida',
                    'mensaje': str(e),
                    'snippet': line[:200] + ('...' if len(line) > 200 else '')
                })
    
    return muestras_validas, errores_validados, borrados


def merge_batches(input_dir, output_path, report_path):
    """Une todos los lotes JSONL en un solo archivo, borrando Invalid \\escape."""
    
    # Validar directorio de entrada
    if not os.path.exists(input_dir):
        print(f"Error: Directorio no encontrado: {input_dir}")
        sys.exit(1)
    
    # Buscar todos los archivos batch_XXX_tone.jsonl
    batch_files = sorted([
        f for f in os.listdir(input_dir)
        if f.startswith('batch_') and f.endswith('_tone.jsonl')
    ])
    
    if not batch_files:
        print(f"Error: No se encontraron archivos batch_*_tone.jsonl en {input_dir}")
        sys.exit(1)
    
    print(f"Se encontraron {len(batch_files)} lotes para reconstruir:")
    for bf in batch_files:
        print(f"  - {bf}")
    print()
    
    # Leer y validar todas las muestras
    all_samples = []
    all_remaining_errors = []
    total_borrados = 0
    languages = Counter()
    categories = Counter()
    
    for batch_file in batch_files:
        filepath = os.path.join(input_dir, batch_file)
        print(f"Procesando {batch_file}...")
        
        muestras_validas, errores_validados, borrados = validate_jsonl_file(filepath, batch_file)
        all_samples.extend(muestras_validas)
        all_remaining_errors.extend(errores_validados)
        total_borrados += borrados
        
        # Contar idiomas y categorías de muestras válidas
        for sample in muestras_validas:
            for msg in sample.get('messages', []):
                content = msg.get('content', '')
                # Detectar idioma básico
                if any(char in content for char in 'áéíóúñ¿¡'):
                    languages['es'] += 1
                elif any(char in content for char in 'abcde'):
                    languages['en'] += 1
            
            # Extraer categoría si existe
            if 'category' in sample:
                categories[sample['category']] += 1
            elif 'metadata' in sample and 'category' in sample['metadata']:
                categories[sample['metadata']['category']] += 1
    
    print()
    
    # Mostrar resumen de borrados
    print(f"🗑️  Borradas {total_borrados} líneas con 'Invalid \\escape' automáticamente")
    
    # Generar reporte de errores restantes
    if all_remaining_errors:
        print(f"⚠️  Se encontraron {len(all_remaining_errors)} errores restantes que requieren corrección manual:")
        
        # Agrupar errores por archivo
        errores_por_archivo = {}
        for error in all_remaining_errors:
            archivo = error['archivo']
            if archivo not in errores_por_archivo:
                errores_por_archivo[archivo] = []
            errores_por_archivo[archivo].append(error)
        
        # Escribir reporte
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ERRORES RESTANTES - CORRECCIÓN MANUAL REQUERIDA\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total de errores restantes: {len(all_remaining_errors)}\n")
            f.write(f"Archivos con errores: {len(errores_por_archivo)}\n")
            f.write(f"Total de archivos procesados: {len(batch_files)}\n")
            f.write(f"Líneas borradas automáticamente (Invalid \\escape): {total_borrados}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("RESUMEN POR ARCHIVO\n")
            f.write("-" * 80 + "\n\n")
            
            for archivo, errores_archivo in sorted(errores_por_archivo.items()):
                f.write(f"📄 {archivo}\n")
                f.write(f"   Errores encontrados: {len(errores_archivo)}\n")
                f.write(f"   Líneas afectadas: {', '.join(str(e['linea']) for e in errores_archivo)}\n\n")
                
                f.write("   Detalle de errores:\n")
                for error in errores_archivo:
                    f.write(f"     - Línea {error['linea']}: [{error['tipo']}] {error['mensaje']}\n")
                    f.write(f"       Snippet: {error['snippet']}\n\n")
                
                f.write("   " + "─" * 76 + "\n\n")
        
        print(f"   Reporte detallado guardado en: {report_path}")
    else:
        print("✅ No se encontraron errores restantes. Todos los archivos son válidos.")
    
    print()
    print(f"Reconstruyendo dataset...")
    
    # Escribir dataset unificado (solo muestras válidas)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Dataset reconstruido: {output_path}")
    print(f"  Total muestras válidas: {len(all_samples)}")
    print(f"  Total líneas borradas (Invalid \\escape): {total_borrados}")
    print(f"  Total errores restantes: {len(all_remaining_errors)}")
    print(f"  Archivos procesados: {len(batch_files)}")
    print(f"  Idiomas detectados: {dict(languages)}")
    print(f"  Categorias: {dict(categories)}")
    print()
    
    if all_remaining_errors:
        print("⚠️  ACCIONES PENDIENTES:")
        print("   1. Revisar el reporte: data/filtered_datasets/merge_remaining_errors.txt")
        print("   2. Corregir manualmente los errores restantes en:")
        print(f"      data/filtered_datasets/batches_rewritten/")
        print("   3. Volver a ejecutar este script después de corregir")
    else:
        print("✅ TODO LISTO. Puedes proceder con el fine-tuning.")
    
    print()
    return len(all_samples), total_borrados, len(all_remaining_errors)


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruir dataset JSONL a partir de lotes procesados'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/filtered_datasets/batches_rewritten',
        help='Directorio con los lotes procesados (default: data/filtered_datasets/batches_rewritten)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/filtered_datasets/kidsafe_tone_dataset.jsonl',
        help='Ruta de salida del dataset reconstruido'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='data/filtered_datasets/merge_remaining_errors.txt',
        help='Ruta de salida del reporte de errores restantes'
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(os.getcwd(), input_dir)
    
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
    
    report_path = args.report
    if not os.path.isabs(report_path):
        report_path = os.path.join(os.getcwd(), report_path)
    
    merge_batches(input_dir, output_path, report_path)


if __name__ == '__main__':
    main()
