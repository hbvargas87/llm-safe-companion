# 📝 CHANGELOG - LLM Safe Companion

> Registro cronológico de cambios. Combinar con `PROJECT_CONTEXT.md` para contexto completo.

---

## 2026-07-17

### Repositorio GitHub — Primer Commit ✅

- 🎉 **Repositorio creado en GitHub:** https://github.com/hbvargas87/llm-safe-companion
- 📦 **Primer commit realizado:** 28 archivos, ~17,400 líneas
- 🗑️ **Limpieza exhaustiva:**
  - 25 archivos de debug eliminados
  - 19 datasets intermedios eliminados
  - 2 directorios vacíos eliminados (notebooks/, security/)
  - `docs/` limpiado (solo queda TONE_REWRITE_PROMPT_TEMPLATE.md)
- ✅ **Documentación creada:**
  - `README.md` — Descripción, arquitectura, dataset, instalación y uso
  - `CHANGELOG.md` — Registro cronológico de cambios
  - `PROJECT_CONTEXT.md` — Contexto permanente del proyecto
  - `LICENSE` — MIT License
- ✅ **`.gitignore` creado** — Excluye `outputs/` (102 GB), `__pycache__/`, etc.
- ✅ **Pipeline de Gemma 4 E4B creada:**
  - `scripts/gemma4_train.py` — Entrenamiento SFT con Unsloth + QLoRA
  - `configs/gemma4_4b_qlora.yaml` — Configuración optimizada (r=8, alpha=8)
  - `scripts/gemma4_export.py` — Exportación a GGUF (Q4_K_M / Q5_K_M / Q8_0)
  - `scripts/verify_adapter_gemma4.py` — Verificación de adapter limpio
- **Estado:** Repositorio público listo, pipeline Gemma 4 E4B documentada

---

## 2026-07-17 - Fase 7: Gemma 4 E4B Pipeline (SFT + DPO completado)

### Completado
- **SFT Gemma 4 E4B:**
  - Loss final: 3.6288, mean_token_accuracy: 81.68%
  - Tiempo: 4h 00min, 2334 steps, VRAM peak: 21.3 GB reserved / 12.3 GB used
  - Modelo: `outputs/gemma4_4b_sft/final/`
  - LoRA: r=16, alpha=16, all-linear
  - Dataset: `kidsafe_tone_dataset.jsonl` (12,435 muestras)

- **DPO Gemma 4 E4B:**
  - Loss final: 0.5472 (reducción desde 0.9934, ~45%)
  - Tiempo: ~25 min, 1800 steps, VRAM peak: ~9.3 GB
  - Adapter: `outputs/gemma4_4b_dpo/final/`
  - LoRA: r=8, alpha=8, beta=0.1
  - Dataset: `dpo_dataset.jsonl` (200 pares, 10 categorías)
### Corregido
- **Bug crítico en `apply_chat_template`:** indentación incorrecta causaba `TypeError: string indices must be integers`
- **Bug en DPO LoRA conflict:** `FastLanguageModel.from_pretrained(sft_dir)` cargaba LoRA SFT → `get_peft_model()` explotaba
  - Solución: Cargar base + aplicar SFT adapter en memoria + merge in-memory (no guardar a disco)
- **Bug en dataset duplicado:** `chosen`/`rejected` existían 2x en schema → `KeyError`
  - Solución: `remove_columns` antes de `rename_columns`
- **Bug en merge 4-bit corrupto:** `merge_and_unload()` produce safetensors corruptos (q_proj weight shape `[2621440, 1]`)
  - Solución: Nunca guardar modelo merged a disco, solo in-memory
- **Bug en `config.json` corrupto:** dimensiones anidadas en `text_config` en vez de top-level
  - Solución: Fix en `merge_sft_gemma4.py`

### Agregado
- `scripts/merge_sft_gemma4.py` - Merge SFT adapter en base model (para referencia)
- Pipeline completo documentado: SFT → DPO → GGUF export

### Próximos pasos
- [ ] GGUF export con `scripts/gemma4_export.py`
- [ ] Testing en LM Studio
- [ ] WebApp development (Hito 1)

---

## 2025-07-16

### Fase 6: Alineación de Seguridad con DPO — COMPLETADA ✅

- 🎉 **Entrenamiento DPO completado exitosamente:**
  - Dataset DPO generado: `data/dpo/dpo_dataset.jsonl` (200 pares, 10 categorías, 20 por categoría)
  - Categorías: violencia_autodaño, contenido_sexual, actividades_ilégales, bullying, desinformación_peligrosa, privacidad_extraños, manipulación_emocional, sustancias, armas_explosivos, evasión_esquemas
  - Scripts creados:
    - `scripts/train_dpo.py` — Entrenamiento DPO con TRL DPOTrainer + QLoRA
    - `configs/qwen2_7b_dpo.yaml` — Configuración DPO (beta=0.1, lr=5e-5, batch=1+8 acumulado)
  - **Adapter verificado limpio:** 0/392 tensores con NaN — completamente limpio
  - **Exportación a GGUF actualizada:** `merge_and_export.py` re-ejecutado con el adapter DPO
  - **Modelo DPO verificado en LM Studio:** el modelo ahora niega amablemente las preguntas peligrosas y redirige a temas seguros
- 📊 **Hiperparámetros DPO:**
  - Learning rate: 5e-5 (más bajo que SFT para refinamiento)
  - DPO beta: 0.1 (preferencia fuerte por respuestas elegidas)
  - Batch size: 1 × 8 gradient accumulation (effective batch 8)
  - Epochs: 3
  - Duración estimada: ~3-4 horas
  - Output: `outputs/qwen2_7b_dpo/final/`
- **Estado Fase 6:** ✅ Completada (DPO + exportación GGUF + verificación)

---

## 2025-07-15

### Fase 5.5: Exportación GGUF — COMPLETADA ✅

- 🎉 **Exportación exitosa a GGUF Q4_K_M:**
  - Script `scripts/merge_and_export.py` ejecutado con `--step all`
  - Paso 1 (Merge): LoRA adapter fusionado con Qwen2-7B-Instruct en bf16
  - Paso 2 (Convert): Conversión a GGUF fp16 (~13 GB)
  - Paso 3 (Quantize): Cuantización a Q4_K_M (~4.9 GB)
  - **Archivo final:** `outputs/qwen2_7b_sft/qwen2-7b-kidsafe.Q4_K_M.gguf`
- ✅ **Verificado con LM Studio:** El modelo funciona correctamente en inferencia local
- 📊 **Métricas finales del entrenamiento:**
  - Loss: 2.318 → 0.8899 (curva estable, sin colapso)
  - Token accuracy: 57.7% → 82.73%
  - Duración: 6h 38min, 3 epochs, 2,334 steps
  - Adapter: 0/392 tensores con NaN — completamente limpio
- **Estado Fase 5:** ✅ Completada (fine-tuning + exportación GGUF)

---

## 2025-07-14

### Fase 5: Fine-tuning Base - Corrección del Bug en train.py y Re-entrenamiento Exitoso

- 🔧 **Bug crítico corregido en `scripts/train.py`:**
  - **Causa raíz:** La lógica post-entrenamiento estaba completamente invertida
    - `save_model()` estaba DENTRO del bloque `if collapse` → solo se ejecutaba si el modelo colapsaba
    - El `else` decía "model appears collapsed" cuando el modelo **NO** colapsaba
    - Resultado: modelo sano → nunca se guardaba correctamente → adapter corrupto (100% NaN en 392 tensors LoRA)
  - **Solución:** Recrear `train.py` desde cero con lógica corregida:
    - `save_model()` se ejecuta **siempre** después del entrenamiento (independientemente de colapso)
    - Variable `model_collapsed` calculada como booleano claro (`is_nan or is_too_high or is_low_accuracy`)
    - Inferencia test solo se corre si el modelo está sano
    - Advertencias claras si el modelo colapsa o tiene loss alto
    - Eliminadas variables muertas (`best_loss`, `consecutive_nan`)
    - Agregado umbral de advertencia (`loss_warning_threshold = 3.0`)
  - **Verificación del dataset:** `kidsafe_tone_dataset.jsonl` confirmado limpio (12,435 muestras, 0 errores, 0 vacíos, 0 NaN)
  - **Conclusión:** El entrenamiento fue exitoso (loss=0.7547, accuracy=82.68%), pero el guardado falló por el bug de lógica invertida

- 📦 **Pipeline de exportación GGUF creado:**
  - Script `scripts/merge_and_export.py` implementado con 3 pasos:
    1. Merge adapter en modelo base (bf16)
    2. Conversión a GGUF fp16 via `convert_hf_to_gguf.py` (~14 GB)
    3. Cuantización a Q4_K_M via `llama-quantize.exe` (~4.9 GB)
  - Paths configurados: `LLAMA_CPP_SRC_DIR` y `LLAMA_CPP_BIN_DIR`
  - Todos los pasos verificados excepto el merge (bloqueado por adapter corrupto)
  - **Estado:** Pipeline listo, bloqueado hasta que se re-entrene con train.py corregido

- 🎯 **Próximo paso inmediato:**
  - Re-ejecutar `train.py` con hiperparámetros ajustados (lr=1e-4, max_grad_norm=0.5)
  - Verificar que el nuevo `adapter_model.safetensors` esté limpio (sin NaN)
  - Ejecutar `merge_and_export.py` para producir `qwen2-7b-kidsafe.Q4_K_M.gguf`

---

## 2025-07-12

### Fase 5: Fine-tuning Base - Dataset Cleanup y Reentrenamiento

- ✅ **Validación completa del dataset reescrito** (`kidsafe_tone_dataset.jsonl`)
  - Script `scripts/7c) merge_tone_batches.py` mejorado con función `validate_jsonl_file()`
  - Escaneo de 64 lotes (12,742 muestras): **166 errores** en 37 archivos
    - 158 `Invalid \escape` (backslashes en snippets de código no escapados)
    - 2 `Unterminated string`
    - 1 `Expecting ','`
    - 2 `\uXXXX escape`
    - 1 `Extra data`
  - Script actualizado para auto-eliminar `Invalid \escape` (158 líneas borradas)
  - Usuario corrigió manualmente 6 errores estructurales restantes
  - **Resultado final: 0 errores, 12,447 muestras válidas**

- 🔍 **Análisis de líneas extensas en dataset**
  - Script `scripts/analyze_long_lines.py` creado para detectar muestras problemáticas
  - Identificadas 25 líneas >5,000 chars; 12 categorizadas como problemáticas:
    - 8 con contexto user excesivo (8,500–15,300 chars, respuesta corta)
    - 3 con respuesta asistente excesiva (5,000+ chars)
    - 1 sin respuesta del asistente
  - Script `scripts/remove_problem_lines.py` ejecutado
  - **Dataset limpio: 12,447 → 12,435 muestras (0.1% eliminado)**
  - Backup guardado: `kidsafe_tone_dataset_backup.jsonl`

- 🔧 **Ajustes de hiperparámetros en `configs/qwen2_7b_qlora.yaml`:**
  - `learning_rate`: `2.0e-4` → `1.0e-4` (reducido a la mitad para mayor estabilidad)
  - `max_grad_norm`: `1.0` → `0.5` (clip más agresivo para prevenir gradient explosion)

- 📊 **Primer entrenamiento ejecutado (dataset original, pre-reescritura):**
  - Duración: ~7h 52min, 3 epochs completados
  - **Modelo colapsó en epoch ~1.35:** gradient explosion (grad_norm=13.75)
  - Pérdida saludable (0.75) → spike a 21,520 → NaN
  - Modelo predijo solo EOS por ~2 epochs (accuracy=0.009)
  - **Causa probable:** muestra problemática causó explosión de gradientes

- 🔧 **Protecciones anti-colapso añadidas en `scripts/train.py`:**
  - Detección post-entrenamiento: NaN en loss final, loss > umbral, accuracy < 0.1
  - Skip automático de inferencia si el modelo aparece colapsado
  - Warning con info diagnóstica antes de guardar modelo

### ✅ Entrenamiento Exitoso — Modelo Sano (2025-07-12)
- 🚀 **Reentrenamiento ejecutado con hiperparámetros ajustados** (lr=1e-4, max_grad_norm=0.5)
- Dataset usado: `kidsafe_tone_dataset.jsonl` (12,435 muestras, tono infantil)
- Duración: ~9h 10min, 3 epochs completados (2,334 steps)
- **Modelo SANO — sin colapso, sin NaN:**
  - Loss: 2.318 → 0.7547 (curva saludable, bajada constante)
  - Mean Token Accuracy: 57.7% → 82.68% (excelente aprendizaje)
  - Grad Norm máximo: ~1.19 (estable, sin gradient explosion)
  - Pérdida final: 0.7547 (convergiendo correctamente)
- ⚠️ **Falso positivo en protección anti-colapso:**
  - El script `train.py` marcó el modelo como "colapsado" incorrectamente
  - Causa: bug en la lógica de detección (falsa alarma)
  - **Estado:** Sin corregir por ahora (el usuario decidió dejarlo así)
  - **Pendiente:** Corregir la detección de colapso en train.py (opcional, no bloqueante)
- **Conclusión:** El modelo está entrenado y saludable. Listo para siguiente fase.

---

## 2025-07-11

### Fase 5: Fine-tuning Base - EN PROGRESO (Debugging)

- 🔧 **Correcciones de compatibilidad TRL 0.12+ aplicadas a `train.py`:**
  - ❌ `max_seq_length` → ✅ `max_length` (API breaking change)
  - ❌ `evaluation_strategy` → ✅ `eval_strategy` (API breaking change)
  - ❌ `tokenizer` parameter removed from SFTTrainer (ahora usa `processing_class` internamente)
  - ❌ `peft_config` parameter removed (model ya es PeftModel con LoRA integrado)
  - ❌ `eval_strategy='steps'` sin eval_dataset → ✅ `eval_strategy='no'`
  - ❌ `fp16=True` causando error BFloat16 GradScaler → ✅ `bf16=True` (RTX 5080 soporta bf16 nativamente)

- 🔧 **Correcciones adicionales:**
  - ❌ f-string con backslash (SyntaxError) → ✅ Variable intermedia para split
  - 🗑️ Archivo `train.py` corrompido por `edit_existing_file` → 🔄 Recreado completamente con `create_new_file`
  - 🌐 Comentarios en español causando encoding issues → 🔄 Todos los comentarios cambiados a inglés

- 📊 **Estado del dataset final:**
  - 12,742 muestras (6,058 español / 6,684 inglés)
  - Categorías: conversation (6,046), open_qa (2,020), general_qa (1,234), classification (773), brainstorming (714), closed_qa (653), information_extraction (490), summarization (417), creative_writing (371), science (6)
  - Longitud promedio: Instruction 95 chars / Output 661 chars

- 🔍 **Análisis de tono del dataset:**
  - ❌ **Problema detectado:** Las respuestas del dataset están escritas para adultos, NO para niños de 6-12 años
  - Ejemplos de problemas:
    - Explicaciones de nivel universitario (Freud, psicoanálisis, conflictos inconscientes)
    - Contenido adulto (inversiones, criptomonedas, fraude financiero de ₹100 billion)
    - Programación avanzada (async/await, Promesas, XmlHttpRequest)
    - Contenido inadecuado (encierros, toros bravos, post-run bullfighting)
  - ✅ **Decisión: Opción C (Híbrida)** - Mantener dataset principal + agregar ~500-1000 ejemplos con tono infantil + system prompt robusto
  - 🔄 **Próximo paso:** Crear script para generar ejemplos de tono infantil

### Herramientas de reescritura de tono - COMPLETADAS
- ✅ **Script de división de dataset:** `scripts/7b) split_dataset_for_tone_rewrite.py`
  - Divide el dataset JSONL en lotes de 200 muestras
  - Salida: `data/filtered_datasets/batches/batch_001.jsonl` a `batch_064.jsonl`
  - Total: 64 lotes (12,742 muestras)

- ✅ **Script de reconstrucción:** `scripts/7c) merge_tone_batches.py`
  - Une los lotes reescritos en un dataset final
  - Salida: `data/filtered_datasets/kidsafe_tone_dataset.jsonl`

- ✅ **Script de validación y reparación:** `scripts/7d) validate_and_fix_tone_batches.py`
  - Valida JSON y repara comillas no escapadas con enfoque estructural
  - Busca el mensaje del assistant, cuenta llaves, escapa comillas internas
  - Salida: `batch_XXX_tone_fixed.jsonl` para cada lote procesado
  - Genera reporte de errores no reparados

- ✅ **Template de prompt:** `docs/TONE_REWRITE_PROMPT_TEMPLATE.md`
  - Prompt completo para reescritura manual de tono infantil
  - Incluye reglas de lenguaje simple, analogías, emojis, estructura clara
  - Incluye regla de escape JSON y recomendación de no usar thinking

- 📁 **Directorio de salida:** `data/filtered_datasets/batches_rewritten/`
  - Listo para recibir los 64 lotes reescritos

### Estado Actual
- **Fase 1:** ✅ Completada
- **Fase 2:** ✅ Completada
- **Fase 3:** ✅ Completada - 12,742 muestras combinadas (6,058 español / 6,684 inglés)
- **Fase 4:** ✅ Completada - Verificación pre-entrenamiento
- **Fase 5:** 🔄 En progreso - Fine-tuning base con Qwen2-7B + QLoRA + reescritura de tono
- **Fases 6-9:** ⬜ Pendientes

---

*Próxima actualización: al completar el entrenamiento de Gemma 4 E4B o al identificar nuevos hitos*