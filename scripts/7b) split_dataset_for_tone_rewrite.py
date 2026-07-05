"""
Script para dividir el dataset JSONL en lotes de 200 muestras
para reescritura manual de tono infantil.

Uso:
    python scripts/7b) split_dataset_for_tone_rewrite.py --batch-size 200

Salida:
    data/filtered_datasets/batches/
    ├── batch_001.jsonl  (muestras 1-200)
    ├── batch_002.jsonl  (muestras 201-400)
    └── ...
"""

import json
import os
import argparse
import sys

def split_dataset(input_path, batch_size=200, output_dir=None):
    """Divide un dataset JSONL en lotes de tamaño especificado."""
    
    # Validar archivo de entrada
    if not os.path.exists(input_path):
        print(f"Error: Archivo no encontrado: {input_path}")
        sys.exit(1)
    
    # Leer todas las muestras
    print(f"Leyendo dataset: {input_path}")
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Linea {line_num} tiene formato JSON invalido: {e}")
    
    print(f"Dataset cargado: {len(samples)} muestras validas")
    
    # Configurar directorio de salida
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), 'batches')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcular numero de lotes
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    print(f"Dividiendo en {num_batches} lotes de {batch_size} muestras...")
    print(f"Directorio de salida: {output_dir}")
    print()
    
    # Dividir en lotes
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        batch = samples[start_idx:end_idx]
        
        # Nombre del archivo
        batch_num = i + 1
        filename = f"batch_{batch_num:03d}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        # Escribir lote
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in batch:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Mostrar progreso
        print(f"  Lote {batch_num:03d}: muestras {start_idx+1}-{end_idx} -> {filename}")
    
    print()
    print(f"Division completada!")
    print(f"   Total: {num_batches} lotes")
    print(f"   Ubicacion: {output_dir}/")
    
    return num_batches


def main():
    parser = argparse.ArgumentParser(
        description='Dividir dataset JSONL en lotes para reescritura manual de tono'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/filtered_datasets/kidsafe_final_dataset.jsonl',
        help='Ruta al dataset JSONL de entrada'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        help='Tamano de cada lote (default: 200)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
    
    split_dataset(input_path, args.batch_size, args.output_dir)


if __name__ == '__main__':
    main()
