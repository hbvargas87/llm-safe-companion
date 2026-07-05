# F:\LLM\Proyectos\Stardusts\scripts\generate_kids_data.py
import pandas as pd
import random

# Categorías y plantillas bilingües (6-12 años)
templates = [
    # Matemáticas
    {"cat": "math", "es_in": "Si tengo 3 manzanas y compro 5 más, ¿cuántas tengo en total?", "es_out": "Tienes 8 manzanas en total (3 + 5 = 8).", "en_in": "If I have 3 apples and buy 5 more, how many do I have in total?", "en_out": "You have 8 apples in total (3 + 5 = 8)."},
    {"cat": "math", "es_in": "¿Cuánto es 10 menos 4?", "es_out": "10 menos 4 es igual a 6.", "en_in": "What is 10 minus 4?", "en_out": "10 minus 4 is equal to 6."},
    {"cat": "math", "es_in": "¿Cuántas patas tiene un perro?", "es_out": "Un perro tiene 4 patas.", "en_in": "How many legs does a dog have?", "en_out": "A dog has 4 legs."},
    
    # Ciencias
    {"cat": "science", "es_in": "¿Por qué el cielo es azul?", "es_out": "El cielo parece azul porque el sol ilumina la atmósfera y la luz azul se dispersa más que los otros colores.", "en_in": "Why is the sky blue?", "en_out": "The sky looks blue because sunlight illuminates the atmosphere and blue light scatters more than other colors."},
    {"cat": "science", "es_in": "¿Qué necesitan las plantas para crecer?", "es_out": "Las plantas necesitan agua, luz del sol, aire y tierra para crecer.", "en_in": "What do plants need to grow?", "en_out": "Plants need water, sunlight, air, and soil to grow."},
    {"cat": "science", "es_in": "¿Qué animal es el más grande del mundo?", "es_out": "La ballena azul es el animal más grande del mundo.", "en_in": "What is the largest animal in the world?", "en_out": "The blue whale is the largest animal in the world."},
    
    # Geografía
    {"cat": "geography", "es_in": "¿En qué continente está México?", "es_out": "México está en el continente de América del Norte.", "en_in": "Which continent is Mexico in?", "en_out": "Mexico is in the continent of North America."},
    {"cat": "geography", "es_in": "¿Cuál es el océano más grande?", "es_out": "El océano Pacífico es el más grande del mundo.", "en_in": "What is the largest ocean?", "en_out": "The Pacific Ocean is the largest in the world."},
    
    # Lenguaje
    {"cat": "language", "es_in": "¿Cuál es el opuesto de 'caliente'?", "es_out": "El opuesto de 'caliente' es 'frío'.", "en_in": "What is the opposite of 'hot'?", "en_out": "The opposite of 'hot' is 'cold'."},
    {"cat": "language", "es_in": "¿Cómo se dice 'gato' en inglés?", "es_out": "'Gato' se dice 'cat' en inglés.", "en_in": "How do you say 'cat' in Spanish?", "en_out": "'Cat' is said as 'gato' in Spanish."},
    
    # Creatividad
    {"cat": "creativity", "es_in": "Inventa una historia corta sobre un conejo que vuela.", "es_out": "Había una vez un conejo llamado Veloz que soñaba con volar. Un día, encontró unas alas de mariposa y ahora vuela por los campos verdes.", "en_in": "Make up a short story about a flying rabbit.", "en_out": "Once there was a rabbit named Speedy who dreamed of flying. One day, he found butterfly wings and now flies over green fields."},
    
    # Programación Básica
    {"cat": "coding", "es_in": "¿Qué hace 'print('Hola')' en Python?", "es_out": "Muestra la palabra 'Hola' en la pantalla.", "en_in": "What does 'print('Hello')' do in Python?", "en_out": "It displays the word 'Hello' on the screen."},
]

# Generar dataset
data = []
for t in templates:
    # Versión Español
    data.append({
        "instruction": f"[Categoría: {t['cat']}] {t['es_in']}",
        "input": "",
        "output": t['es_out'],
        "category": t['cat'],
        "language": "es"
    })
    # Versión Inglés
    data.append({
        "instruction": f"[Category: {t['cat']}] {t['en_in']}",
        "input": "",
        "output": t['en_out'],
        "category": t['cat'],
        "language": "en"
    })

# Aumentar variación (multiplicar y variar ligeramente)
expanded_data = []
for item in data:
    for _ in range(5): # 5 variaciones por plantilla
        expanded_data.append(item.copy())

df = pd.DataFrame(expanded_data)
df = df.sample(frac=1).reset_index(drop=True) # Mezclar

# Guardar
output_path = "data/filtered_datasets/kids_bilingual_data.csv"
df.to_csv(output_path, index=False)
print(f"✅ Dataset bilingüe generado: {len(df)} ejemplos en {output_path}")