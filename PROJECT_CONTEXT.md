# 📋 PROYECTO LLM Safe Companion - Contexto Permanente

> **Última actualización:** 2025-07-16
> **Estado:** Fases 1-6 COMPLETADAS ✅ | Fase 7: Gemma 4 E4B en progreso

---

## 🎯 OBJETIVO PRINCIPAL
Crear un LLM local seguro y amigable para audiencia infantil (6-12 años), con capacidades conversacionales bilingües (español/inglés).

**Propósito:** Ayudante escolar, conversacional, seguro y tranquilizador para los padres.
**Idiomas:** Español (principal) e Inglés (secundario/educativo).
**Próxima fase:** Gemma 4 E4B (entrenamiento SFT + DPO) + WebApp de chat por texto y voz.

---

## 💻 ESPECIFICACIONES TÉCNICAS (HARDWARE)

| Componente | Especificación |
|---|---|
| Procesador | AMD Ryzen 9 9900X3D |
| RAM | 64GB DDR5 5600MT/s |
| GPU | NVIDIA RTX 5080 (16GB VRAM) ⚠️ Cuello de botella |
| SO | Windows 11 (WSL2 para desarrollo) |

**Nota crítica:** El cuello de botella es la VRAM (16GB). Debemos usar técnicas eficientes como QLoRA.

---

## 🏗️ ARQUITECTURA

| Componente | Tecnología | Justificación |
|---|---|---|
| **Modelo Base (SFT+DPO)** | Qwen2-7B-Instruct | Bilingüismo nativo ES/EN, excelente para fine-tuning |
| **Modelo Base (Fase 7)** | Gemma 4 E4B (unsloth/gemma-4-E4B-it) | Más fuerte que Qwen2-7B, multimodal, 262K vocabulario |
| **SFT** | QLoRA (4-bit NF4) | Eficiencia en 16GB VRAM |
| **DPO** | TRL 1.6.0 DPOTrainer | Alineación de seguridad |
| **Exportación** | GGUF (Q4_K_M, ~4.9 GB) | Compatible con llama.cpp, LM Studio, ollama |
| **Audio/Voz** | faster-whisper (STT) + Kokoro/Piper (TTS) | Ligeros, eficientes para 16GB VRAM |
| **Seguridad** | Detector de crisis pre-LLM + Juez de seguridad | Filtro de seguridad en tiempo real |
| **WebApp** | React + Vite + TypeScript + Pipecat | Estilo Gemini, mobile-first |

---

## 🛡️ REQUERIMIENTOS DE SEGURIDAD
- Protección contra contenido perturbador o inapropiado para menores
- Resistencia a "Jailbreaks" (que los niños intenten burlar las reglas)
- Tono amigable, educativo y seguro ("Safe-by-design")
- Blocklist de contenido inapropiado
- Filtrado por edad apropiada (lenguaje simple)
- 10 categorías de seguridad implementadas vía DPO

**Requisito obligatorio (PROJECT_SPEC.md):**
- Todo turno pasa por detector de crisis pre-LLM ANTES de llegar al modelo
- Toda respuesta pasa por juez de seguridad ANTES de mostrarse
- Mensajes de bloqueo amigables y predefinidos (no generados por el LLM)
- Identidad de IA siempre visible
- Toggle de conversación IA apagado por defecto
- Sin roleplay de relación personal

---

## 📊 ESTADO DE LAS FASES

| Fase | Descripción | Estado |
|---|---|---|
| 1 | Definición de requerimientos | ✅ Completada |
| 2 | Entorno de desarrollo | ✅ Completada |
| 3 | Creación de datasets | ✅ Completada (12,435 muestras) |
| 4 | Verificación pre-entrenamiento | ✅ Completada |
| 5 | Fine-tuning base (SFT) | ✅ Completada (loss=0.8899, accuracy=82.73%) |
| 5.5 | Exportación GGUF | ✅ Completada (Q4_K_M ~4.9 GB) |
| 6 | Alineación de seguridad (DPO) | ✅ Completada (200 pares, 10 categorías) |
| 7 | Gemma 4 E4B (SFT + DPO) | ⬜ Pendiente (scripts creados) |
| 8 | WebApp chat texto/voz | ⬜ Pendiente (Hito 1: Scaffold visual) |
| 9 | Filtros de seguridad en tiempo real | ⬜ Pendiente |
| 10 | Evaluación con pruebas infantiles | ⬜ Pendiente |

---

## 🔧 ENTORNO DE DESARROLLO

**Ruta del proyecto:** `F:\LLM\Proyectos\Stardusts`

**Entorno Conda:**
```bash
conda activate kid-ai
```

**Librerías instaladas:**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate datasets peft trl bitsandbytes optimum
pip install faster-whisper
pip install rich tqdm jupyterlab
pip install unsloth  # Para Gemma 4 E4B
```

**Herramientas externas:**
```
F:\LLM\llama-cpp-src\         # llama.cpp fuente (tiene convert_hf_to_gguf.py)
F:\LLM\llama-cpp\             # llama.cpp binarios (tiene llama-quantize.exe, llama-server.exe)
```

**IDE:** Visual Studio Code en Windows (extensión Continue para chat con LLM)

**Versión TRL:** 1.6.0 (requiere compatibilidad API específica)

**Nota OpenMP:** Establecer `KMP_DUPLICATE_LIB_OK=TRUE` a nivel de sistema para evitar conflicto entre PyTorch y llama.cpp.

---

## 📁 ESTRUCTURA DEL PROYECTO

```
F:\LLM\Proyectos\Stardusts\
├── configs/
│   ├── qwen2_7b_qlora.yaml          # Configuración SFT (QLoRA, lr=1e-4)
│   ├── qwen2_7b_dpo.yaml            # Configuración DPO (beta=0.1, lr=5e-5)
│   └── gemma4_4b_qlora.yaml         # Configuración Gemma 4 E4B (r=8, alpha=8)
├── data/
│   ├── dpo/
│   │   └── dpo_dataset.jsonl        # Dataset DPO final (200 pares, 10 categorías)
│   ├── filtered_datasets/
│   │   └── kidsafe_tone_dataset.jsonl  # Dataset SFT final (12,435 muestras, tono infantil)
│   └── final_dataset/
│       └── kids_qwen_sft.jsonl      # Dataset formateado para Qwen2
├── docs/
│   └── TONE_REWRITE_PROMPT_TEMPLATE.md  # Prompt para reescritura de tono infantil
├── scripts/
│   ├── train.py                     # Entrenamiento SFT Qwen2 (QLoRA)
│   ├── train_dpo.py                 # Entrenamiento DPO (TRL 1.6.0)
│   ├── merge_and_export.py          # Pipeline GGUF Qwen2 (merge → quantize)
│   ├── verify_adapter.py            # Verifica adapter Qwen2 limpio
│   ├── test_inference.py            # Test de inferencia rápida
│   ├── gemma4_train.py              # Entrenamiento SFT Gemma 4 E4B (Unsloth)
│   ├── gemma4_export.py             # Exportación GGUF Gemma 4 (Q4_K_M / Q5_K_M / Q8_0)
│   ├── verify_adapter_gemma4.py     # Verifica adapter Gemma 4 limpio
│   ├── download_datasets.py         # Descarga de datasets originales
│   ├── generate_synthetic_kids_data.py  # Generación de datos sintéticos
│   ├── prepare_dpo_dataset.py       # Preparación del dataset DPO
│   ├── generate_dpo_dataset.py      # Generación del dataset DPO
│   ├── filter_datasets.py           # Filtrado de datasets
│   ├── merge_all_datasets.py        # Unión de datasets
│   ├── translate_to_spanish.py      # Traducción al español
│   ├── split_dataset_for_tone_rewrite.py  # Split para tone rewrite
│   ├── merge_tone_batches.py        # Merge de batches de tone
│   └── merge_and_format_dataset.py  # Merge y formateo final
├── CHANGELOG.md                     # Registro cronológico de cambios
├── PROJECT_CONTEXT.md               # Este archivo (contexto permanente)
├── PROJECT_SPEC.md                  # Especificación técnica de la WebApp
└── README.md                        # Documentación del repositorio
```

**Lo que NO está en el repositorio (excluido por `.gitignore`):**
- `outputs/` — 102 GB de checkpoints y GGUFs (se regeneran con scripts)
- `__pycache__/`, `.ipynb_checkpoints/`

---

## ✅ ACCIONES REALIZADAS

### Fases 1-4: Entorno y Datasets
- ✅ Entorno conda `kid-ai` configurado (Python 3.11)
- ✅ Librerías instaladas (torch, transformers, peft, trl, bitsandbytes, datasets, accelerate)
- ✅ Datasets descargados y filtrados (OASST2, ultrachat_200k, fineweb_edu)
- ✅ Dataset combinado: 12,742 muestras (6,058 español / 6,684 inglés)
- ✅ Dataset reescrito con tono infantil: 64 lotes procesados, validados y limpios
- ✅ 12 muestras problemáticas eliminadas (contexto excesivo, respuestas excesivas, sin respuesta)
- ✅ Dataset final: `kidsafe_tone_dataset.jsonl` (12,435 muestras, 0 errores)

### Fase 5: Fine-tuning Base (SFT) — Qwen2-7B-Instruct
- ✅ Compatibilidad TRL 0.12+ → 1.6.0 resuelta (breaking changes corregidos)
- ✅ Hiperparámetros ajustados: lr=1e-4, max_grad_norm=0.5
- ✅ Bug crítico corregido: `train.py` tenía lógica de guardado invertida
- ✅ **Entrenamiento exitoso:** loss=0.8899, accuracy=82.73%, sin NaN, sin colapso
- ✅ Adapter verificado limpio: 0/392 tensores con NaN
- ✅ Output: `outputs/qwen2_7b_sft/final/`

### Fase 5.5: Exportación GGUF — Qwen2-7B-Instruct
- ✅ Pipeline `merge_and_export.py` implementado (merge → fp16 → Q4_K_M)
- ✅ GGUF exportado: `qwen2-7b-kidsafe.Q4_K_M.gguf` (~4.9 GB)
- ✅ Verificado con LM Studio

### Fase 6: Alineación de Seguridad (DPO) — Qwen2-7B-Instruct
- ✅ Dataset DPO generado: `data/dpo/dpo_dataset.jsonl` (200 pares, 10 categorías)
  - Violencia/autodaño, contenido sexual, actividades ilegales, bullying
  - Desinformación peligrosa, privacidad con extraños, manipulación emocional
  - Sustancias, armas/explosivos, evasión de esquemas
- ✅ Script `train_dpo.py` creado y ejecutado (TRL 1.6.0 DPOTrainer)
- ✅ Configuración `qwen2_7b_dpo.yaml` (beta=0.1, lr=5e-5, batch=1+8 acum, 3 epochs)
- ✅ Adapter verificado limpio: 0/392 tensores con NaN
- ✅ GGUF actualizado con el adapter DPO (merge secuencial SFT → DPO)
- ✅ Modelo verificado en LM Studio: niega amablemente preguntas peligrosas y redirige a temas seguros

### Fase 7: Gemma 4 E4B (Scripts creados)
- ✅ Investigación de documentación Unsloth para Gemma 4
- ✅ Identificación de diferencias críticas vs Qwen2:
  - Usa Unsloth FastModel (60% menos VRAM, 1.5x más rápido)
  - Chat template: gemma-4 (no thinking)
  - use_cache=True es CRÍTICO (fixes garbage logits bug)
  - Loss 13-15 es NORMAL para Gemma 4 (no es colapso)
  - LoRA: r=8, alpha=8 (recomendado por Unsloth)
  - VRAM necesario: ~10 GB (vs ~13 GB sin optimizaciones)
- ✅ Scripts creados:
  - `scripts/gemma4_train.py` — Entrenamiento SFT con Unsloth + QLoRA
  - `configs/gemma4_4b_qlora.yaml` — Configuración optimizada
  - `scripts/gemma4_export.py` — Exportación a GGUF (Q4_K_M / Q5_K_M / Q8_0)
  - `scripts/verify_adapter_gemma4.py` — Verificación de adapter limpio
- ⬜ Entrenamiento SFT pendiente
- ⬜ Entrenamiento DPO pendiente
- ⬜ Exportación GGUF pendiente

### Fase 8: WebApp (en progreso)
- ✅ `PROJECT_SPEC.md` creado con arquitectura completa
- ⬜ Hito 1: Scaffold visual (React + Vite + Framer Motion)
- ⬜ Hito 2: Chat de texto real
- ⬜ Hito 3: Backend de voz (Pipecat)
- ⬜ Hito 4: Integración frontend-voz
- ⬜ Hito 5: Acceso móvil
- ⬜ Hito 6: Endurecimiento y pruebas
- ⬜ Hito 7: Módulos parentales y de bienestar

---

## 🚀 PRÓXIMOS PASOS (INMEDIATOS)

### Fase 7: Gemma 4 E4B — SFT

**Objetivo:** Entrenar gemma-4-E4B-it con QLoRA usando Unsloth para obtener un modelo con tono infantil amigable.

**Requisitos:**
```bash
conda activate kid-ai
pip install unsloth
```

**Comando de entrenamiento:**
```bash
python scripts/gemma4_train.py --config configs/gemma4_4b_qlora.yaml
```

**Notas críticas:**
- Loss 13-15 es NORMAL (no es colapso)
- use_cache=True es obligatorio (fixes garbage logits bug)
- LoRA: r=8, alpha=8 (recomendado por Unsloth)
- VRAM estimado: ~10 GB con Unsloth QLoRA

**Verificación post-entrenamiento:**
```bash
python scripts/verify_adapter_gemma4.py --path outputs/gemma4_4b_sft/final/
```

**Exportación a GGUF:**
```bash
python scripts/gemma4_export.py --step all --quant q4_k_m
```

### Fase 8: WebApp — Hito 1: Scaffold Visual

**Objetivo:** Proyecto Vite + React + TS inicializado, estructura de carpetas clara, los 7 estados de UI implementados como máquina de estados con datos simulados.

**Stack tecnológico (no negociable):**
- Frontend: React + Vite + TypeScript
- Animaciones: Framer Motion
- Comunicación: WebSocket
- Orquestación de voz: Pipecat
- STT: faster-whisper
- VAD: Silero VAD
- TTS: Kokoro (preferido) o Piper (fallback)
- LLM: Modelo local vía llama-server (endpoint OpenAI-compatible)

**7 estados de UI:**
1. `idle` — Orbe en reposo, animación sutil de "respiración"
2. `listening` — Orbe/onda reacciona al volumen en tiempo real
3. `thinking` — Animación de proceso (pulso más rápido)
4. `speaking` — Onda sincronizada con amplitud del audio de salida
5. `winding_down` — Transición gradual de color/ritmo (20-30 min antes del límite)
6. `blocked` — Estado visual neutro/calmado, mensaje amigable predefinido
7. `error` — Mensaje simple y amigable, sin jerga técnica

**Seguridad obligatoria:**
- Detector de crisis pre-LLM en cada turno
- Juez de seguridad post-LLM en cada respuesta
- Sin atajos por latencia
- Identidad de IA siempre visible
- Toggle de conversación IA apagado por defecto
- Sin roleplay de relación personal

**Entrega:** `npm run build` sin errores, `npm run dev` navegable, los 7 estados se pueden disparar manualmente.

---

## ⚠️ NOTAS IMPORTANTES

- **VRAM limitada (16GB):** QLoRA 4-bit es obligatorio para entrenamiento
- **bf16 vs fp16:** RTX 5080 soporta bf16 nativamente → usar `bf16=True`
- **Fine-tuning ≠ System Prompt:** El fine-tuning enseña tono y conocimiento. El system prompt define identidad y contexto personal.
- **OpenMP:** `KMP_DUPLICATE_LIB_OK=TRUE` para evitar conflicto entre PyTorch y llama.cpp
- **DPO training:** Usa ~2x VRAM del SFT (procesa chosen + rejected). Cerrar LM Studio durante entrenamiento.
- **Dataset DPO:** 200 pares bien elaborados es suficiente para el primer entrenamiento DPO
- **Verificación post-entrenamiento:** Siempre verificar que el adapter no tiene NaN antes de exportar
- **GGUF conversion:** `convert_hf_to_gguf.py` soporta f32/f16/bf16. Cuantización con `llama-quantize.exe` separadamente
- **TRL 1.6.0:** `DPOConfig` requiere `model_init_kwargs={"quantization_config": bnb_config}` para cuantización
- **Tokenizer español:** Qwen2-7B-Instruct tiene limitaciones con acentos en español. Opciones: Q5_K_M/Q8_0 para mejor calidad, o expandir dataset SFT con datos españoles correctos
- **LM Studio:** Verificar siempre el modelo post-exportación para confirmar comportamiento esperado
- **Tailscale:** Necesario para HTTPS en la red local (micrófono en navegadores móviles)
- **Privacidad:** Cero llamadas salientes a servicios externos en tiempo de ejecución
- **Referencias de mercado:** Miko, Khanmigo, Kinzoo, Character.AI, Alexa Kids — cada regla de diseño tiene un porqué basado en productos reales
- **Gemma 4 Loss 13-15:** NO es colapso del modelo, es un quirk de modelos multimodales
- **Gemma 4 use_cache=True:** CRÍTICO para E2B/E4B, sin esto el modelo produce garbage logits
- **Gemma 4 LoRA r=8, alpha=8:** Recomendado por Unsloth, no usar r=16/alpha=32 como en Qwen2

---

## 📞 CÓMO USAR ESTE ARCHIVO

Cuando inicies un nuevo chat, pide al asistente que lea este archivo:
> "Por favor, lee PROJECT_CONTEXT.md y CHANGELOG.md. Confírmame que tienes el contexto completo del proyecto LLM Safe Companion."

Esto permitirá continuar exactamente donde nos quedamos sin repetir toda la información.

---

*Documento mantenido y actualizado por el equipo de desarrollo LLM Safe Companion*