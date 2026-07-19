# 🛡️ LLM Safe Companion

Modelo de lenguaje grande (LLM) fine-tuned con **SFT + DPO** para personalidad amigable y seguridad en interacciones con menores. Basado en **Qwen2-7B-Instruct** y **Gemma 4 E4B**, optimizado para una audiencia infantil (6-12 años) con capacidades bilingües (español/inglés).

## 📋 Descripción

Este proyecto implementa una pipeline completa de fine-tuning para personalizar un LLM con:
- **SFT (Supervised Fine-Tuning):** Para establecer un tono amigable, conversacional y educativo
- **DPO (Direct Preference Optimization):** Para alinear el modelo con principios de seguridad y protección infantil

El modelo resultante niega amablemente las preguntas peligrosas y redirige a temas seguros, manteniendo un tono cálido y apropiado para niños.

## 🏗️ Arquitectura

| Componente | Qwen2-7B-Instruct | Gemma 4 E4B |
|---|---|---|
| **SFT** | QLoRA (4-bit NF4) via TRL 1.6.0 | QLoRA (4-bit) via Unsloth + TRL 1.6.0 |
| **DPO** | TRL 1.6.0 `DPOTrainer` + `DPOConfig` | TRL 1.6.0 `DPOTrainer` + `DPOConfig` |
| **Exportación** | GGUF Q4_K_M (~4.9 GB) ✅ | GGUF Export pendiente |
| **Vocabulario** | ~152K tokens | 262K tokens (mejor tokenización español) |
| **Multimodal** | Solo texto | Texto + Imagen + Audio |
| **Chat Template** | Qwen2 custom | gemma-4 (sin thinking) |
| **LoRA SFT** | r=16, alpha=32 | r=16, alpha=16, all-linear |
| **LoRA DPO** | r=8, alpha=8 | r=8, alpha=8, dropout=0 |

**Estado:**
- ✅ **Qwen2-7B-Instruct:** SFT + DPO + GGUF exportado y verificado en LM Studio
- ✅ **Gemma 4 E4B:** SFT + DPO completados, GGUF export pendiente
- ⬜ **WebApp:** En progreso (ver [Sirius](https://bitbucket.org/HumberDJ/sirius/src/test/))

## 💻 Hardware

| Componente | Especificación |
|---|---|
| Procesador | AMD Ryzen 9 9900X3D |
| RAM | 128 GB DDR5 |
| GPU | NVIDIA RTX 5080 (16 GB VRAM) |
| SO | Windows 11 (WSL2 para desarrollo) |

**Nota:** La VRAM de 16 GB es el cuello de botella. Se utilizan técnicas optimizadas como QLoRA 4-bit, `gradient_checkpointing`, `max_seq_length: 512` y `bf16` (nativo en RTX 5080) para mantenerse dentro del límite.

## 🛡️ Requisitos de Seguridad

- Protección contra contenido perturbador o inapropiado para menores (6-12 años)
- Resistencia a "Jailbreaks" (que los niños intenten burlar las reglas)
- Tono amigable, educativo y seguro ("Safe-by-design")
- Blocklist de contenido inapropiado
- Filtrado por edad apropiada (lenguaje simple)
- 10 categorías de seguridad implementadas vía DPO

**Requisito obligatorio:**
- Todo turno pasa por **detector de crisis pre-LLM** ANTES de llegar al modelo
- Toda respuesta pasa por **juez de seguridad** ANTES de mostrarse
- Mensajes de bloqueo amigables y predefinidos (no generados por el LLM)
- Identidad de IA siempre visible
- Toggle de conversación IA apagado por defecto
- Sin roleplay de relación personal

## 📊 Dataset

- **SFT:** `kidsafe_tone_dataset.jsonl` — 12,435 muestras con tono infantil
  - Categorías: conversation (6,046), open_qa (2,020), general_qa (1,234), classification (773), brainstorming (714), closed_qa (653), information_extraction (490), summarization (417), creative_writing (371), science (6)
  - Bilingüe: ~6,058 español / ~6,684 inglés
- **DPO:** `dpo_dataset.jsonl` — 200 pares de preferencia en 10 categorías de seguridad:
  - Violencia/autodaño, contenido sexual, actividades ilegales, bullying
  - Desinformación peligrosa, privacidad con extraños, manipulación emocional
  - Sustancias, armas/explosivos, evasión de esquemas

## 📁 Estructura del Repositorio

```
llm-safe-companion/
├── configs/
│   ├── qwen2_7b_qlora.yaml          # Configuración SFT Qwen2 (QLoRA)
│   ├── qwen2_7b_dpo.yaml            # Configuración DPO Qwen2
│   ├── gemma4_4b_qlora.yaml         # Configuración SFT Gemma 4 (QLoRA)
│   └── gemma4_4b_dpo.yaml           # Configuración DPO Gemma 4
├── data/
│   ├── filtered_datasets/
│   │   └── kidsafe_tone_dataset.jsonl   # Dataset SFT final (12,435 muestras)
│   └── dpo/
│       └── dpo_dataset.jsonl        # Dataset DPO final (200 pares)
├── docs/
│   └── TONE_REWRITE_PROMPT_TEMPLATE.md
├── scripts/
│   ├── train.py                     # Entrenamiento SFT Qwen2
│   ├── train_dpo.py                 # Entrenamiento DPO Qwen2
│   ├── merge_and_export.py          # Pipeline GGUF Qwen2 (merge → quantize)
│   ├── verify_adapter.py            # Verifica adapter Qwen2 limpio
│   ├── gemma4_train.py              # Entrenamiento SFT Gemma 4
│   ├── gemma4_train_dpo.py          # Entrenamiento DPO Gemma 4
│   ├── gemma4_export.py             # Exportación GGUF Gemma 4
│   ├── verify_adapter_gemma4.py     # Verifica adapter Gemma 4 limpio
│   └── [datasets scripts...]        # Procesamiento de datos
├── outputs/
│   ├── qwen2_7b_sft/                # Checkpoint SFT Qwen2
│   ├── qwen2_7b_dpo/                # Checkpoint DPO Qwen2
│   ├── qwen2_7b_sft/qwen2-7b-kidsafe.Q4_K_M.gguf  # Modelo exportado
│   ├── gemma4_4b_sft/final/         # Checkpoint SFT Gemma 4
│   └── gemma4_4b_dpo/final/         # Checkpoint DPO Gemma 4
├── CHANGELOG.md
├── PROJECT_CONTEXT.md
├── PROJECT_SPEC.md
└── README.md
```

**Excluido por `.gitignore`:**
- `outputs/` — Checkpoints y GGUFs (~102 GB, se regeneran con los scripts de exportación)
- `__pycache__/`, `.ipynb_checkpoints/`

## 🚀 Instalación y Uso

### Requisitos
- Python 3.11+
- NVIDIA GPU con ≥16 GB VRAM (para entrenamiento)
- Espacio en disco: ~50 GB (modelos + datasets)

### Entorno
```bash
conda create -n kid-ai python=3.11
conda activate kid-ai

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate datasets peft trl bitsandbytes optimum
pip install rich tqdm
```

### Entrenamiento Qwen2-7B-Instruct

**SFT:**
```bash
python scripts/train.py
```

**DPO:**
```bash
python scripts/train_dpo.py
```

**Exportar a GGUF:**
```bash
python scripts/merge_and_export.py --step all
```

### Entrenamiento Gemma 4 E4B

**SFT:**
```bash
python scripts/gemma4_train.py
```

**DPO:**
```bash
python scripts/gemma4_train_dpo.py
```

**Exportar a GGUF:**
```bash
python scripts/gemma4_export.py
```

### Verificación del Adapter
```bash
python scripts/verify_adapter.py       # Qwen2
python scripts/verify_adapter_gemma4.py # Gemma 4
```

## 🌐 WebApp — Sirius

Interfaz web de chat por texto y voz (estilo Gemini, mobile-first) en desarrollo.

- **Repositorio:** [Sirius en Bitbucket](https://bitbucket.org/HumberDJ/sirius/src/test/)
- **Stack:** React + Vite + TypeScript + Pipecat
- **Estado:** Hito 1 — Scaffold visual en progreso
- **Hito 2:** Integración STT/TTS (faster-whisper + Kokoro/Piper)

## 📈 Métricas de Entrenamiento

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
- **VRAM:** ~12.3 GB used / 21.3 GB reserved

### Gemma 4 E4B DPO
- **Loss:** 0.5472 (desde 0.9934, ~45% reducción)
- **Duración:** ~25 min, 1800 steps
- **VRAM:** ~9.3 GB

## 📝 Licencia

MIT License — Ver archivo `LICENSE` para más detalles.

## 📄 Referencias

- [Qwen2 Model License](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Gemma 4 E4B](https://huggingface.co/unsloth/gemma-4-E4B-it)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Unsloth](https://github.com/unslothai/unsloth)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)

---

*Proyecto de investigación y aprendizaje en fine-tuning de LLMs para uso infantil seguro.*
