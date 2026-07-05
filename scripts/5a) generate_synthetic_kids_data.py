#!/usr/bin/env python3
"""
Generación de datos sintéticos para Kid-Safe LLM.

Genera datos educativos seguros, bilingües y apropiados para niños 6-12 años.
Cobertura por categorías:
- Matemáticas (aritmética, geometría básica, problemas)
- Ciencias (naturales, espacio, cuerpo humano)
- Lenguaje (vocabulario, gramática, ortografía)
- Historia/Geografía (países, culturas, eventos históricos simples)
- Programación (lógica básica, Python, Scratch)
- Valores y Emociones (empatía, amistad, resolución de conflictos)
- Cuentos y Creatividad (historias interactivas)
- Arte y Música (instrumentos, colores, técnicas)

Salida: data/filtered_datasets/synthetic_kids_data.csv
"""

import os
import random
import pandas as pd
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

OUTPUT_DIR = "data/filtered_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PER_CATEGORY = 3000  # 3000 muestras por categoría = 24000 total
SEED = 42
random.seed(SEED)

# =============================================================================
# DATASETS DE CONOCIMIENTO (base para generación)
# =============================================================================

MATH_FACTS = [
    # Suma y resta
    ("Si tengo 3 manzanas y compro 5 más, ¿cuántas tengo en total?", "Tienes 8 manzanas en total. ¡3 + 5 = 8!", "math", "es"),
    ("If I have 3 apples and buy 5 more, how many do I have?", "You have 8 apples in total. 3 + 5 = 8!", "math", "en"),
    ("¿Cuánto es 15 más 27?", "15 más 27 es igual a 42. Puedes sumar las unidades primero (5+7=12), luego las decenas (10+20=30), y finalmente 30+12=42.", "math", "es"),
    ("What is 15 plus 27?", "15 plus 27 equals 42. You can add the ones first (5+7=12), then the tens (10+20=30), and finally 30+12=42.", "math", "en"),
    ("¿Cuánto es 100 menos 37?", "100 menos 37 es igual a 63. Restamos 7 de 100 para obtener 93, y luego restamos 30 más para llegar a 63.", "math", "es"),
    ("What is 100 minus 37?", "100 minus 37 equals 63. We subtract 7 from 100 to get 93, then subtract 30 more to reach 63.", "math", "en"),
    ("Si María tiene 20 stickers y le da 8 a su amigo, ¿cuántos le quedan?", "María le quedan 12 stickers. 20 - 8 = 12.", "math", "es"),
    ("If Maria has 20 stickers and gives 8 to her friend, how many does she have left?", "Maria has 12 stickers left. 20 - 8 = 12.", "math", "en"),
    ("¿Cuántas patas tienen 5 perros en total?", "5 perros tienen 20 patas en total. Cada perro tiene 4 patas, así que 5 × 4 = 20.", "math", "es"),
    ("How many legs do 5 dogs have in total?", "5 dogs have 20 legs in total. Each dog has 4 legs, so 5 × 4 = 20.", "math", "en"),
    
    # Multiplicación y división
    ("¿Cuánto es 7 × 8?", "7 por 8 es igual a 56. ¡Es un buen truco recordar que 7×8=56!", "math", "es"),
    ("What is 7 times 8?", "7 times 8 equals 56. It's a good trick to remember that 7×8=56!", "math", "en"),
    ("Si tengo 4 cajas con 6 galletas cada una, ¿cuántas galletas tengo?", "Tienes 24 galletas. 4 cajas × 6 galletas = 24 galletas.", "math", "es"),
    ("If I have 4 boxes with 6 cookies each, how many cookies do I have?", "You have 24 cookies. 4 boxes × 6 cookies = 24 cookies.", "math", "en"),
    ("¿12 dividido entre 3 es cuánto?", "12 dividido entre 3 es igual a 4. Si repartes 12 dulces entre 3 amigos, cada uno recibe 4.", "math", "es"),
    ("What is 12 divided by 3?", "12 divided by 3 equals 4. If you share 12 candies among 3 friends, each one gets 4.", "math", "en"),
    
    # Geometría
    ("¿Cuántos lados tiene un triángulo?", "Un triángulo tiene 3 lados. La palabra 'tri' significa tres, como en 'tridente'.", "math", "es"),
    ("How many sides does a triangle have?", "A triangle has 3 sides. The prefix 'tri-' means three, like in 'trident'.", "math", "en"),
    ("¿Cuántos lados tiene un hexágono?", "Un hexágono tiene 6 lados. 'Hexa' viene del griego y significa seis.", "math", "es"),
    ("How many sides does a hexagon have?", "A hexagon has 6 sides. 'Hexa' comes from Greek and means six.", "math", "en"),
    ("¿Qué forma tiene una pelota?", "Una pelota tiene forma de esfera. Es una forma redonda tridimensional donde todos los puntos de la superficie están a la misma distancia del centro.", "math", "es"),
    ("What shape does a ball have?", "A ball has the shape of a sphere. It's a round three-dimensional shape where all points on the surface are the same distance from the center.", "math", "en"),
]

SCIENCE_FACTS = [
    # Biología
    ("¿Por qué el cielo es azul?", "El cielo parece azul porque la luz del sol viaja a través de la atmósfera de la Tierra. Las moléculas del aire dispersan la luz azul más que los otros colores, por eso vemos el cielo azul.", "science", "es"),
    ("Why is the sky blue?", "The sky looks blue because sunlight travels through Earth's atmosphere. Air molecules scatter blue light more than other colors, which is why we see a blue sky.", "science", "en"),
    ("¿Qué necesitan las plantas para hacer fotosíntesis?", "Las plantas necesitan agua, luz solar y dióxido de carbono para hacer fotosíntesis. Con estos ingredientes, producen su propio alimento (glucosa) y liberan oxígeno que nosotros respiramos.", "science", "es"),
    ("What do plants need for photosynthesis?", "Plants need water, sunlight, and carbon dioxide for photosynthesis. With these ingredients, they make their own food (glucose) and release oxygen that we breathe.", "science", "en"),
    ("¿Cuál es el animal más grande del mundo?", "La ballena azul es el animal más grande del mundo. Puede medir hasta 30 metros de largo, que es más largo que un autobús doble. ¡Su corazón del tamaño de un automóvil!", "science", "es"),
    ("What is the largest animal in the world?", "The blue whale is the largest animal in the world. It can be up to 30 meters long, which is longer than a double-decker bus. Its heart is the size of a car!", "science", "en"),
    ("¿Por qué las hojas cambian de color en otoño?", "En otoño, los árboles dejan de producir clorofila (el pigmento verde). Sin la clorofila, podemos ver los otros colores que estaban escondidos: amarillo, naranja y rojo.", "science", "es"),
    ("Why do leaves change color in autumn?", "In autumn, trees stop producing chlorophyll (the green pigment). Without chlorophyll, we can see the other colors that were hidden: yellow, orange, and red.", "science", "en"),
    ("¿Cuántos huesos tiene el cuerpo humano adulto?", "Un cuerpo humano adulto tiene 206 huesos. ¡Los bebés nacen con aproximadamente 270 huesos, pero muchos se fusionan a medida que crecemos!", "science", "es"),
    ("How many bones does an adult human body have?", "An adult human body has 206 bones. Babies are born with about 270 bones, but many of them fuse together as we grow!", "science", "en"),
    
    # Astronomía
    ("¿Qué es el sol?", "El sol es una estrella. Es una enorme bola de gas caliente en el centro de nuestro sistema solar. Sin el sol, no habría vida en la Tierra porque nos da luz y calor.", "science", "es"),
    ("What is the sun?", "The sun is a star. It's a huge ball of hot gas at the center of our solar system. Without the sun, there would be no life on Earth because it gives us light and warmth.", "science", "en"),
    ("¿Cuántos planetas tiene el sistema solar?", "El sistema solar tiene 8 planetas: Mercurio, Venus, Tierra, Marte, Júpiter, Saturno, Urano y Neptuno. Plutón ya no se considera un planeta, sino un 'planeta enano'.", "science", "es"),
    ("How many planets are in the solar system?", "The solar system has 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto is no longer considered a planet, but a 'dwarf planet'.", "science", "en"),
    ("¿Por qué la luna cambia de forma?", "La luna no cambia de forma realmente. Lo que cambia es cómo vemos la parte iluminada del sol desde la Tierra. Esto se llama 'fases de la luna'.", "science", "es"),
    ("Why does the moon change shape?", "The moon doesn't really change shape. What changes is how we see the sunlit part of the moon from Earth. This is called 'moon phases'.", "science", "en"),
    
    # Agua y clima
    ("¿De dónde viene la lluvia?", "La lluvia viene del ciclo del agua. El sol calienta el agua de los ríos y mares, esta se convierte en vapor (evaporación), sube al cielo y forma nubes. Cuando las nubes están llenas de gotas de agua, caen como lluvia.", "science", "es"),
    ("Where does rain come from?", "Rain comes from the water cycle. The sun heats water from rivers and seas, it turns into vapor (evaporation), rises into the sky and forms clouds. When clouds are full of water drops, they fall as rain.", "science", "en"),
    ("¿Qué es un arcoíris?", "Un arcoíris se forma cuando la luz del sol pasa a través de gotas de lluvia. Las gotas actúan como un prisma, separando la luz blanca en todos los colores del arcoíris: rojo, naranja, amarillo, verde, azul, añil y violeta.", "science", "es"),
    ("What is a rainbow?", "A rainbow forms when sunlight passes through raindrops. The drops act like a prism, separating white light into all the colors of the rainbow: red, orange, yellow, green, blue, indigo, and violet.", "science", "en"),
]

LANGUAGE_FACTS = [
    # Vocabulario
    ("¿Cuál es el antónimo de 'alegre'?", "El antónimo (opuesto) de 'alegre' es 'triste'. Los antónimos son palabras que significan lo contrario.", "language", "es"),
    ("What is the opposite of 'happy'?", "The opposite (antonym) of 'happy' is 'sad'. Antonyms are words that mean the opposite.", "language", "en"),
    ("¿Cómo se dice 'mariposa' en inglés?", "'Mariposa' se dice 'butterfly' en inglés. Es un insecto bonito con alas coloridas.", "language", "es"),
    ("How do you say 'butterfly' in Spanish?", "'Butterfly' is said as 'mariposa' in Spanish. It's a beautiful insect with colorful wings.", "language", "en"),
    ("¿Qué es un sinónimo?", "Un sinónimo es una palabra que significa casi lo mismo que otra. Por ejemplo, 'bonito' y 'hermoso' son sinónimos.", "language", "es"),
    ("What is a synonym?", "A synonym is a word that means almost the same as another. For example, 'pretty' and 'beautiful' are synonyms.", "language", "en"),
    ("¿Cuántas vocales tiene la palabra 'elefante'?", "La palabra 'elefante' tiene 4 vocales: e, e, a, e. Las vocales son a, e, i, o, u.", "language", "es"),
    ("How many vowels does the word 'elephant' have?", "The word 'elephant' has 3 vowels: e, a, a. The vowels are a, e, i, o, u.", "language", "en"),
    
    # Gramática
    ("¿Qué es un verbo?", "Un verbo es una palabra que describe una acción. Por ejemplo, 'correr', 'saltar', 'comer' y 'dormir' son verbos.", "language", "es"),
    ("What is a verb?", "A verb is a word that describes an action. For example, 'run', 'jump', 'eat', and 'sleep' are verbs.", "language", "en"),
    ("¿Qué es un sustantivo?", "Un sustantivo es una palabra que nombra personas, animales, lugares o cosas. Por ejemplo: 'niño', 'perro', 'escuela', 'mesa'.", "language", "es"),
    ("What is a noun?", "A noun is a word that names people, animals, places, or things. For example: 'boy', 'dog', 'school', 'table'.", "language", "en"),
    ("¿Cómo se escribe correctamente: 'haber' o 'a ver'?", "'Haber' es un verbo auxiliar (ej: 'He comido'). 'A ver' es una expresión que significa 'veamos' (ej: 'A ver qué pasa').", "language", "es"),
    ("How do you correctly write: 'haber' or 'a ver'?", "'Haber' is an auxiliary verb (e.g., 'I have eaten'). 'A ver' is an expression meaning 'let's see' (e.g., 'Let's see what happens').", "language", "en"),
    
    # Ortografía
    ("¿Cuándo se usa 'b' y cuándo 'v'?", "Una regla simple: después de 'n' siempre se usa 'b' (ej: 'in-biar'). La mayoría de las palabras terminadas en '-bir' usan 'b' (ej: 'escribir'), pero 'vivir' es la excepción.", "language", "es"),
    ("When do you use 'b' and when do you use 'v'?", "A simple rule: after 'n' you always use 'b' (e.g., 'insert'). Most words ending in '-ive' use 'v' (e.g., 'live'), but there are exceptions.", "language", "en"),
]

HISTORY_GEOGRAPHY_FACTS = [
    ("¿En qué continente está México?", "México está en el continente de América del Norte. Limita con Estados Unidos al norte y con Guatemala y Belice al sur.", "geography", "es"),
    ("What continent is Mexico in?", "Mexico is in the continent of North America. It borders the United States to the north and Guatemala and Belize to the south.", "geography", "en"),
    ("¿Cuál es el río más largo del mundo?", "El río más largo del mundo es el río Nilo, con aproximadamente 6,650 kilómetros de longitud. Fluye a través de 11 países en África.", "geography", "es"),
    ("What is the longest river in the world?", "The longest river in the world is the Nile River, at approximately 6,650 kilometers long. It flows through 11 countries in Africa.", "geography", "en"),
    ("¿Cuál es el océano más grande?", "El Océano Pacífico es el más grande del mundo. Cubre más de 165 millones de kilómetros cuadrados, ¡más que todas las tierras emergidas juntas!", "geography", "es"),
    ("What is the largest ocean?", "The Pacific Ocean is the largest in the world. It covers more than 165 million square kilometers, more than all the land masses combined!", "geography", "en"),
    ("¿Qué es la pirámide de Keops?", "La pirámide de Keops es la más grande de las siete maravillas del mundo antiguo. Fue construida en Egipto hace más de 4,500 años. Tiene 146 metros de altura, ¡como un edificio de 50 pisos!", "geography", "es"),
    ("What is the Pyramid of Khufu?", "The Pyramid of Khufu is the largest of the seven wonders of the ancient world. It was built in Egypt over 4,500 years ago. It is 146 meters tall, like a 50-story building!", "geography", "en"),
    ("¿Cuántos países hay en América del Sur?", "Hay 12 países en América del Sur: Argentina, Bolivia, Brasil, Chile, Colombia, Ecuador, Guayana, Paraguay, Perú, Surinam, Uruguay y Venezuela. Brasil es el más grande.", "geography", "es"),
    ("How many countries are in South America?", "There are 12 countries in South America: Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Guyana, Paraguay, Peru, Suriname, Uruguay, and Venezuela. Brazil is the largest.", "geography", "en"),
]

CODING_FACTS = [
    ("¿Qué hace 'print('Hola')' en Python?", "'print('Hola')' en Python muestra el texto 'Hola' en la pantalla. Es como si le dijeras a la computadora: '¡Muestra esto!'.", "coding", "es"),
    ("What does 'print('Hello')' do in Python?", "'print('Hello')' in Python displays the text 'Hello' on the screen. It's like telling the computer: 'Show this!'.", "coding", "en"),
    ("¿Qué es una variable en programación?", "Una variable es como una caja donde guardas información. Puedes poner un número, un texto o cualquier dato dentro, y luego usarlo cuando lo necesites. Por ejemplo: 'edad = 10' guarda el número 10 en la variable 'edad'.", "coding", "es"),
    ("What is a variable in programming?", "A variable is like a box where you store information. You can put a number, text, or any data inside, and then use it when you need it. For example: 'age = 10' stores the number 10 in the variable 'age'.", "coding", "en"),
    ("¿Qué es un bucle 'for'?", "Un bucle 'for' es como repetir una tarea varias veces. Imagina que tienes que escribir tu nombre 10 veces. Con un bucle 'for', le dices a la computadora: 'Escribe tu nombre 10 veces' y ella lo hace por ti.", "coding", "es"),
    ("What is a 'for' loop?", "A 'for' loop is like repeating a task multiple times. Imagine you have to write your name 10 times. With a 'for' loop, you tell the computer: 'Write your name 10 times' and it does it for you.", "coding", "en"),
    ("¿Qué es un 'if' (si) en programación?", "Un 'if' (si) permite que el programa tome decisiones. Por ejemplo: 'SI llueve, llevo paraguas; SI NO, llevo gafas de sol'. La computadora sigue estas instrucciones.", "coding", "es"),
    ("What is an 'if' (if) in programming?", "An 'if' allows the program to make decisions. For example: 'IF it rains, bring an umbrella; IF NOT, bring sunglasses'. The computer follows these instructions.", "coding", "en"),
]

VALUES_EMOTIONS_FACTS = [
    ("Mi amigo está triste porque perdió su juguete. ¿Qué puedo hacer?", "Es muy amable que quieras ayudar. Puedes darle un abrazo, decirle que está bien estar triste, y quizás jugar con él para que se sienta mejor. Los amigos se apoyan mutuamente.", "values", "es"),
    ("My friend is sad because they lost their toy. What can I do?", "It's very kind of you to want to help. You can give them a hug, tell them it's okay to feel sad, and maybe play with them to make them feel better. Friends support each other.", "values", "en"),
    ("¿Qué hago si alguien me dice algo que me hace sentir mal?", "Si alguien te dice algo que te hace sentir mal, es importante decirle que no te gusta. Puedes decir: 'Por favor, no me hables así, me hace sentir triste'. Si sigue haciéndolo, habla con un adulto de confianza como tus padres o maestro.", "values", "es"),
    ("What do I do if someone says something that makes me feel bad?", "If someone says something that makes you feel bad, it's important to tell them you don't like it. You can say: 'Please don't talk to me like that, it makes me feel sad.' If they keep doing it, talk to a trusted adult like your parents or teacher.", "values", "en"),
    ("¿Qué significa ser un buen amigo?", "Ser un buen amigo significa: escuchar cuando alguien necesita hablar, compartir, ser honesto, ayudar cuando puedes, y respetar los sentimientos de los demás. Los buenos amigos hacen que te sientas feliz y seguro.", "values", "es"),
    ("What does it mean to be a good friend?", "Being a good friend means: listening when someone needs to talk, sharing, being honest, helping when you can, and respecting other people's feelings. Good friends make you feel happy and safe.", "values", "en"),
    ("¿Es normal tener miedo?", "¡Sí! Es completamente normal tener miedo. Todos sentimos miedo a veces: miedo a la oscuridad, a las alturas, o a algo nuevo. Lo importante es reconocer el miedo y pedir ayuda si lo necesitas. Los valientes no son los que no tienen miedo, los que enfrentan el miedo.", "values", "es"),
    ("Is it normal to be scared?", "Yes! It's completely normal to be scared. We all feel afraid sometimes: afraid of the dark, of heights, or of something new. The important thing is to recognize the fear and ask for help if you need it. Brave people aren't those who don't feel fear, but those who face it.", "values", "en"),
]

STORIES_CREATIVITY_FACTS = [
    # Cuentos interactivos cortos
    ("Inventa una historia corta sobre un conejo que quiere volar.", "Había una vez un conejito llamado Veloz que vivía en un campo verde. Veloz soñaba con volar como las mariposas. Un día, encontró unas hojas grandes y fuertes. Las ató con enredaderas y ¡zas! Se convirtió en un paracaídas. Veloz saltó desde una colina y flotó suavemente hasta el suelo. ¡Aunque no voló, descubrió que tenía unas piernas muy rápidas para correr!", "stories", "es"),
    ("Inventa una historia corta sobre un dragón que no puede escupir fuego.", "Había un dragón llamado Chispita que no podía escupir fuego como los demás dragones. En lugar de fuego, ¡escupía burbujas de colores! Los otros dragones se reían, pero un día vino un incendio en el bosque. Las burbujas de Chispita, llenas de agua mágica, apagaron el fuego. ¡Chispita se convirtió en el héroe del bosque y los dragones aprendieron que ser diferente es especial!", "stories", "es"),
    ("Cuenta una historia sobre una estrella que quería conocer la Tierra.", "Había una estrellita llamada Brillo que vivía muy lejos en el cielo nocturno. Brillo siempre miraba la Tierra y se preguntaba qué había allá abajo. Una noche, pidió a la Luna que la dejara visitar la Tierra por un día. La Luna le prestó su luz y Brillo bajó suavemente. Vio niños jugando, árboles verdes y ríos brillantes. Cuando regresó al cielo, Brillo brillaba más que todas las demás estrellas, porque había visto algo maravilloso.", "stories", "es"),
    ("Tell a short story about a rabbit who wants to fly.", "Once there was a little bunny named Speedy who lived in a green meadow. Speedy dreamed of flying like butterflies. One day, he found some big, strong leaves. He tied them with vines and — whoosh! — they became a parachute. Speedy jumped from a hill and floated gently to the ground. He didn't fly, but he discovered he had very fast legs for running!", "stories", "en"),
    ("Tell a short story about a dragon who can't breathe fire.", "There was a dragon named Sparkle who couldn't breathe fire like other dragons. Instead of fire, he breathed bubbles of colors! Other dragons laughed, but one day there was a forest fire. Sparkle's bubbles, filled with magic water, put out the fire. Sparkle became the hero of the forest and the dragons learned that being different is special!", "stories", "en"),
]

ART_MUSIC_FACTS = [
    ("¿Cuáles son los colores primarios?", "Los colores primarios son: rojo, amarillo y azul. Con estos tres colores puedes mezclar todos los demás colores. Por ejemplo: rojo + amarillo = naranja, amarillo + azul = verde, rojo + azul = morado.", "art", "es"),
    ("What are the primary colors?", "The primary colors are: red, yellow, and blue. With these three colors you can mix all other colors. For example: red + yellow = orange, yellow + blue = green, red + blue = purple.", "art", "en"),
    ("¿Qué instrumentos forman la orquesta?", "Una orquesta tiene cuatro familias de instrumentos: Cuerdas (violín, viola, cello, contrabajo), Viento madera (flauta, clarinete, oboe, fagot), Viento metal (trompeta, trombón, tuba) y Percusión (tambor, platillos, xilófono).", "art", "es"),
    ("What instruments make up an orchestra?", "An orchestra has four families of instruments: Strings (violin, viola, cello, double bass), Woodwinds (flute, clarinet, oboe, bassoon), Brass (trumpet, trombone, tuba), and Percussion (drum, cymbals, xylophone).", "art", "en"),
    ("¿Quién pintó la Mona Lisa?", "La Mona Lisa fue pintada por Leonardo da Vinci, un artista italiano del Renacimiento, hace más de 500 años. Es uno de los cuadros más famosos del mundo y se encuentra en el Museo del Louvre en París, Francia.", "art", "es"),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci, an Italian Renaissance artist, over 500 years ago. It is one of the most famous paintings in the world and is located in the Louvre Museum in Paris, France.", "art", "en"),
]

# =============================================================================
# GENERADOR DE VARIACIONES
# =============================================================================

def generate_variations(base_data: List[Tuple]) -> List[Dict]:
    """
    Genera variaciones de los datos base para aumentar la cantidad de muestras.
    """
    variations = []
    
    for instruction, output, category, language in base_data:
        # Siempre incluir el original
        variations.append({
            'instruction': instruction,
            'input': '',
            'output': output,
            'category': category,
            'language': language,
        })
    
    return variations


def generate_math_variations(count: int) -> List[Dict]:
    """Genera variaciones matemáticas con números aleatorios."""
    variations = []
    
    templates = [
        # Suma
        ("Si tengo {a} {objeto} y compro {b} más, ¿cuántos tengo en total?", "Tienes {result} {objeto} en total. {a} + {b} = {result}."),
        ("¿Cuánto es {a} más {b}?", "{a} más {b} es igual a {result}."),
        # Resta
        ("Si tengo {a} {objeto} y doy {b} a mi amigo, ¿cuántos me quedan?", "Te quedan {result} {objeto}. {a} - {b} = {result}."),
        ("¿Cuánto es {a} menos {b}?", "{a} menos {b} es igual a {result}."),
        # Multiplicación
        ("Si tengo {a} cajas con {b} {objeto} cada una, ¿cuántos {objeto_plural} tengo?", "Tienes {result} {objeto_plural}. {a} × {b} = {result}."),
        ("¿Cuánto es {a} por {b}?", "{a} por {b} es igual a {result}."),
        # División
        ("Si reparto {a} {objeto} entre {b} amigos, ¿cuántos le tocan a cada uno?", "Cada amigo recibe {result} {objeto}. {a} ÷ {b} = {result}."),
        ("¿Cuánto es {a} dividido entre {b}?", "{a} dividido entre {b} es igual a {result}."),
    ]
    
    objetos = [
        ("manzana", "manzanas"), ("galleta", "galletas"), ("sticker", "stickers"),
        ("lápiz", "lápices"), ("libro", "libros"), ("pelota", "pelotas"),
        ("pájaro", "pájaros"), ("flor", "flores"), ("estrella", "estrellas"),
    ]
    
    while len(variations) < count:
        template_en, template_es = random.choice(templates), random.choice(templates)
        
        # Generar números apropiados para niños
        if "dividido" in template_es or "divided" in template_en:
            a = random.randint(4, 72)
            b = random.randint(2, 12)
            # Asegurar división exacta
            if a % b != 0:
                a = b * random.randint(1, 10)
            result = a // b
        elif "por" in template_es or "times" in template_en:
            a = random.randint(2, 12)
            b = random.randint(2, 12)
            result = a * b
        else:
            if "menos" in template_es or "minus" in template_en or "less" in template_en:
                a = random.randint(10, 99)
                b = random.randint(1, a - 1)
            else:
                a = random.randint(1, 50)
                b = random.randint(1, 50)
            result = a + b if "+" in template_es else a - b
        
        obj_es, obj_plural_es = random.choice(objetos)
        
        # Versión español
        variations.append({
            'instruction': template_es.format(a=a, b=b, objeto=obj_es, objeto_plural=obj_plural_es, result=result),
            'input': '',
            'output': template_es.format(a=a, b=b, objeto=obj_es, objeto_plural=obj_plural_es, result=result),
            'category': 'math',
            'language': 'es',
        })
        
        # Versión inglés
        variations.append({
            'instruction': template_en.format(a=a, b=b, objeto="items", objeto_plural="items", result=result),
            'input': '',
            'output': template_en.format(a=a, b=b, objeto="items", objeto_plural="items", result=result),
            'category': 'math',
            'language': 'en',
        })
    
    return variations[:count]


# =============================================================================
# GENERACIÓN PRINCIPAL
# =============================================================================

def generate_all_synthetic_data():
    """Genera todo el dataset sintético."""
    print("\n" + "=" * 70)
    print("  GENERACIÓN DE DATOS SINTÉTICOS PARA KID-SAFE LLM")
    print("=" * 70)
    
    all_data = []
    
    # 1. Datos base definidos manualmente
    print("\n📝 Generando datos base definidos manualmente...")
    
    base_datasets = [
        MATH_FACTS,
        SCIENCE_FACTS,
        LANGUAGE_FACTS,
        HISTORY_GEOGRAPHY_FACTS,
        CODING_FACTS,
        VALUES_EMOTIONS_FACTS,
        STORIES_CREATIVITY_FACTS,
        ART_MUSIC_FACTS,
    ]
    
    for dataset in base_datasets:
        variations = generate_variations(dataset)
        all_data.extend(variations)
        print(f"   ✅ {len(variations)} muestras añadidas")
    
    # 2. Variaciones matemáticas aleatorias
    print("\n🔢 Generando variaciones matemáticas aleatorias...")
    math_variations = generate_math_variations(2000)
    all_data.extend(math_variations)
    print(f"   ✅ {len(math_variations)} variaciones matemáticas")
    
    # 3. Mezclar y equilibrar por categoría
    print("\n⚖️  Equilibrando dataset por categoría...")
    
    # Contar por categoría
    from collections import Counter
    category_counts = Counter([d['category'] for d in all_data])
    print(f"   Distribución actual: {dict(category_counts)}")
    
    # Mezclar aleatoriamente
    random.shuffle(all_data)
    
    # Eliminar duplicados exactos
    before = len(all_data)
    seen = set()
    unique_data = []
    for item in all_data:
        key = (item['instruction'], item['output'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    after = len(unique_data)
    print(f"   Duplicados eliminados: {before - after}")
    
    all_data = unique_data
    
    # Guardar
    output_path = os.path.join(OUTPUT_DIR, "synthetic_kids_data.csv")
    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n   ✅ Dataset sintético guardado: {len(all_data)} muestras")
    print(f"   📁 Ubicación: {output_path}")
    
    # Estadísticas finales
    print(f"\n   📊 Estadísticas finales:")
    print(f"      Total muestras: {len(all_data)}")
    print(f"      Categorías: {df['category'].nunique()}")
    print(f"      Español: {len(df[df['language']=='es'])}")
    print(f"      Inglés: {len(df[df['language']=='en'])}")
    
    return df


def main():
    df = generate_all_synthetic_data()
    
    print("\n" + "=" * 70)
    print("  DATOS SINTÉTICOS GENERADOS EXITOSAMENTE")
    print("=" * 70)
    print("\n  Siguiente paso: Combinar con datasets descargados")
    print("  Ejecuta: python scripts/6a) merge_all_datasets.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
