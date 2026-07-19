# 📋 PROYECTO LLM Safe Companion — Contexto Permanente

> **Última actualización:** 2026-07-17
> **Estado:** Fases 1-7 COMPLETADAS ✅ | Fase 8: WebApp en progreso

---

## 🎯 OBJETIVO PRINCIPAL

Crear un LLM local seguro y amigable para audiencia infantil (6-12 años), con capacidades conversacionales bilingües (español/inglés).

**Propósito:** Ayudante escolar, conversacional, seguro y tranquilizador para los padres.
**Idiomas:** Español (principal) e Inglés (secundario/educativo).
**Próxima fase:** WebApp de chat por texto y voz (Hito 1: Scaffold visual).

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
| **Modelo Base (Qwen2)** | Qwen2-7B-Instruct | Bilingüismo nativo ES/EN, excelente para fine-tuning |
| **Modelo Base (Gemma 4)** | Gemma 4 E4B (unsloth/gemma-4-E4B-it) | Más fuerte que Qwen2-7B, multimodal, 262K vocabulario |
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
| 5 | Fine-tuning base (SFT) Qwen2 | ✅ Completada (loss=0.8899, acc=82.73%, 6h38min) |
| 5.5 | Exportación GGUF Qwen2 | ✅ Completada (Q4_K_M ~4.9 GB, verificado en LM Studio) |
| 6 | Alineación de seguridad (DPO) Qwen2 | ✅ Completada (200 pares, 10 categorías, verificado en LM Studio) |
| 7 | Gemma 4 E4B (SFT + DPO) | ✅ Completada — GGUF export pendiente |
| 8 | WebApp chat texto/voz | ⬜ Pendiente (Hito 1: Scaffold visual) |
| 9 | Filtros de seguridad en tiempo real | ⬜ Pendiente |
| 10 | Evaluación con pruebas infantiles | ⬜ Pendiente |

---

## 🔧 ENTORNO DE DESARROLLO

**Ruta del proyecto:** `F:\LLM\Proyectos\Stardusts`

**Entorno Conda:** `kid-ai` en `C:\Users\Humberto Barrera\.conda\envs\kid-ai\`

**Python:** 3.11
**PyTorch:** 2.10.0+cu128 (CUDA)
**TRL:** 1.6.0
**Transformers:** >= 4.51.3
**Unsloth:** última versión
**Herramientas principales:** Continue (VS Code), OpenHands (planificado)

---

## 📁 ESTRUCTURA DE ARCHIVOS CLAVE

```
F:\LLM\Proyectos\Stardusts\
├── scripts/
│   ├── train.py                    # Qwen2 SFT
│   ├── train_dpo.py                # Qwen2 DPO
│   ├── merge_and_export.py         # Qwen2 GGUF export
│   ├── verify_adapter.py           # Qwen2 verificación
│   ├── gemma4_train.py             # Gemma 4 SFT
│   ├── gemma4_train_dpo.py         # Gemma 4 DPO
│   ├── gemma4_export.py            # Gemma 4 GGUF export
│   └── verify_adapter_gemma4.py    # Gemma 4 verificación
├── configs/
│   ├── qwen2_7b_qlora.yaml         # Qwen2 SFT config
│   ├── qwen2_7b_dpo.yaml           # Qwen2 DPO config
│   ├── gemma4_4b_qlora.yaml        # Gemma 4 SFT config
│   └── gemma4_4b_dpo.yaml          # Gemma 4 DPO config
├── data/
│   ├── filtered_datasets/
│   │   └── kidsafe_tone_dataset.jsonl   # 12,435 muestras SFT
│   └── dpo/
│       └── dpo_dataset.jsonl              # 200 pares DPO
├── outputs/
│   ├── qwen2_7b_sft/               # Qwen2 SFT final
│   ├── qwen2_7b_dpo/               # Qwen2 DPO final
│   ├── qwen2_7b_sft/qwen2-7b-kidsafe.Q4_K_M.gguf  # Modelo exportado
│   ├── gemma4_4b_sft/final/        # Gemma 4 SFT final
│   └── gemma4_4b_dpo/final/        # Gemma 4 DPO final
├── docs/
│   └── TONE_REWRITE_PROMPT_TEMPLATE.md
├── PROJECT_CONTEXT.md              # Este archivo
├── CHANGELOG.md                    # Registro cronológico
├── README.md                       # Descripción del proyecto
└── LICENSE                         # MIT License
```

---

## ⚙️ DECISIONES TÉCNICAS CLAVE

### SFT vs DPO: Dos fases distintas

1. **SFT (Supervised Fine-Tuning):** Enseña al modelo a ser amigable para niños con el dataset `kidsafe_tone_dataset.jsonl`
2. **DPO (Direct Preference Optimization):** Refina la seguridad con 200 pares chosen/rejected de 10 categorías peligrosas

### Gemma 4 vs Qwen2

| Aspecto | Qwen2-7B-Instruct | Gemma 4 E4B |
|---|---|---|
| SFT Loader | AutoModelForCausalLM | FastLanguageModel (Unsloth) |
| DPO Loader | AutoModelForCausalLM | FastLanguageModel (Unsloth) |
| Chat Template | Qwen2 custom | gemma-4 (sin thinking) |
| use_cache | No relevante | CRÍTICO: use_cache=True para inferencia |
| Loss normal | <3.0 | 13-15 (quirk multimodal, NO es colapso) |
| Dataset | messages → apply_chat_template | Conversations → text (sin <bos>) |
| LoRA SFT | r=16, alpha=32 | r=16, alpha=16, all-linear |
| LoRA DPO | r=8, alpha=8 | r=8, alpha=8, dropout=0 |
| VRAM (SFT) | ~13.3 GB | ~12.3 GB (optimizado) |
| Vocabulario | ~152K | 262K (mejor tokenización español) |
| Multimodal | Solo texto | Texto + Imagen + Audio |

### Optimizaciones VRAM críticas

- `gradient_checkpointing: false` (30-50x más rápido)
- `max_seq_length: 512` (reducido de 2048 → 1024 → 512)
- `per_device_train_batch_size: 4` + `gradient_accumulation_steps: 4` (batch efectivo = 16)
- `dataloader_num_workers: 0` (evita pickle con tokenizer)
- DPO: `use_gradient_checkpointing="unsloth"` (30% menos VRAM)
- DPO: `optim="adamw_8bit"` (menos VRAM)

### Bugs críticos resueltos

1. **mergekit no encontrado** → `pip install mergekit`
2. **llm_blender incompatible** → Patch en `transformers.utils.import_utils`
3. **judges.py corrupto** → Reemplazo completo con versión limpia
4. **TypeError: string indices must be integers** → `apply_chat_template` detecta formato texto vs mensajes
5. **Inferencia basura** → `use_cache=True` + `gradient_checkpointing_disable()` antes de generate()
6. **Merge 4-bit corrupto** → Nunca guardar modelo merged a disco, solo in-memory
7. **config.json corrupto** → Dimensiones anidadas en text_config vs top-level

---

## 📈 MÉTRICAS DE ENTRENAMIENTO

### Qwen2-7B-Instruct SFT
- **Loss:** 2.318 → 0.8899 (curva estable)
- **Token Accuracy:** 57.7% → 82.73%
- **Duración:** 6h 38min, 3 epochs, 2,334 steps
- **VRAM:** ~13.3 GB

### Qwen2-7B-Instruct DPO
- **Loss:** Reducción exitosa
- **Verificación:** 0/392 tensores NaN
- **LM Studio:** Niega amablemente preguntas peligrosas, redirige a temas seguros

### Gemma 4 E4B SFT
- **Loss:** 3.6288
- **Mean Token Accuracy:** 81.68%
- **Duración:** 4h 00min, 2334 steps
- **VRAM:** 21.3 GB reserved / 12.3 GB used

### Gemma 4 E4B DPO
- **Loss:** 0.5472 (desde 0.9934, ~45% reducción)
- **Duración:** ~25 min, 1800 steps
- **VRAM:** ~9.3 GB

---

## 🔄 PIPELINE COMPLETA (Resumen)

```
Dataset (12,435 muestras)
    ↓ SFT (QLoRA r=16, alpha=16)
SFT Adapter → GGUF Q4_K_M → LM Studio ✅
    ↓ DPO (r=8, alpha=8, beta=0.1)
DPO Adapter → GGUF Q4_K_M → LM Studio ✅
    ↓ (Pendiente)
Gemma 4 E4B SFT → Gemma 4 E4B DPO → GGUF Export → LM Studio
    ↓ (Pendiente)
WebApp (React + Vite + Pipecat)
```

---

## 🚀 PRÓXIMOS PASOS INMEDIATOS

1. **GGUF Export Gemma 4:** `python scripts/gemma4_export.py --config configs/gemma4_4b_dpo.yaml`
2. **Testing en LM Studio:** Verificar modelo Gemma 4 merged
3. **Hito 1 WebApp:** Scaffold visual (React + Vite + TypeScript + Pipecat)
4. **OpenHands:** Configurar para desarrollo WebApp (opcional)
5. **Hito 2 WebApp:** Integración STT/TTS

---

*Próxima actualización: al completar GGUF export de Gemma 4 o iniciar WebApp*
