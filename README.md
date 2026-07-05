# 🛡️ LLM Safe Companion

Modelo de lenguaje grande (LLM) fine-tuned con **SFT + DPO** para personalidad amigable y seguridad en interacciones. Basado en **Qwen2-7B-Instruct**, optimizado para una audiencia infantil (6-12 años) con capacidades bilingües (español/inglés).

## 📋 Descripción

Este proyecto implementa una pipeline completa de fine-tuning para personalizar un LLM con:
- **SFT (Supervised Fine-Tuning):** Para establecer un tono amigable, conversacional y educativo
- **DPO (Direct Preference Optimization):** Para alinear el modelo con principios de seguridad y protección infantil

El modelo resultante niega amablemente las preguntas peligrosas y redirige a temas seguros, manteniendo un tono cálido y apropiado para niños.

## 🏗️ Arquitectura

| Componente | Tecnología |
|---|---|
| **Modelo Base** | Qwen2-7B-Instruct (multilingual, optimizado para español) |
| **SFT** | QLoRA (4-bit NF4) via TRL 0.12+ |
| **DPO** | TRL 1.6.0 `DPOTrainer` + `DPOConfig` |
| **Exportación** | GGUF Q4_K_M (~4.9 GB) |
| **Hardware** | RTX 5080 (16 GB VRAM), Ryzen 9 9900X3D, 64 GB DDR5 |

## 📊 Dataset

- **SFT:** 12,435 muestras con tono infantil (conversation, open_qa, general_qa, etc.)
- **DPO:** 200 pares de preferencia en 10 categorías de seguridad:
  - Violencia/autodaño, contenido sexual, actividades ilegales, bullying
  - Desinformación peligrosa, privacidad con extraños, manipulación emocional
  - Sustancias, armas/explosivos, evasión de esquemas

## 📁 Estructura del Repositorio

```
llm-safe-companion/
├── configs/
│   ├── qwen2_7b_qlora.yaml          # Configuración SFT (QLoRA)
│   └── qwen2_7b_dpo.yaml            # Configuración DPO
├── data/
│   └── dpo/
│       └── dpo_dataset.jsonl        # Dataset DPO final (200 pares)
├── docs/
│   └── TONE_REWRITE_PROMPT_TEMPLATE.md
├── scripts/
│   ├── train.py                     # Entrenamiento SFT
│   ├── train_dpo.py                 # Entrenamiento DPO
│   ├── merge_and_export.py          # Pipeline GGUF (merge → quantize)
│   ├── verify_adapter.py            # Verifica adapter limpio
│   ├── test_inference.py            # Test de inferencia
│   └── [datasets scripts...]        # Procesamiento de datos
├── CHANGELOG.md
├── PROJECT_CONTEXT.md
├── PROJECT_SPEC.md
└── README.md
```

**Excluido por `.gitignore`:**
- `outputs/` — Checkpoints y GGUFs (~102 GB, se regeneran con `merge_and_export.py`)
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

### Entrenamiento SFT
```bash
python scripts/train.py
```

### Entrenamiento DPO
```bash
python scripts/train_dpo.py
```

### Exportación a GGUF
```bash
python scripts/merge_and_export.py --step all
```

### Verificación del Adapter
```bash
python scripts/verify_adapter.py
```

## 📝 Licencia

MIT License — Ver archivo `LICENSE` para más detalles.

## 📄 Referencias

- [Qwen2 Model License](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

*Proyecto de investigación y aprendizaje en fine-tuning de LLMs para uso infantil seguro.*