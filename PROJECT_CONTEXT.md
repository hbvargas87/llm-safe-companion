# 📋 PROYECTO LLM Safe Companion - Contexto Permanente

> **Última actualización:** 2025-07-16
> **Estado:** Fases 1-6 COMPLETADAS ✅ | GGUF exportado y verificado
---

## 🎯 OBJETIVO PRINCIPAL
Crear un LLM local seguro y amigable para audiencia infantil (6-12 años), con capacidades conversacionales bilingües (español/inglés).
**Propósito:** Ayudante escolar, conversacional, seguro y tranquilizador para los padres.
**Idiomas:** Español (principal) e Inglés (secundario/educativo).

---

## 💻 ESPECIFICACIONES TÉCNICAS (HARDWARE)

| Componente | Especificación |
|---|---|
| Procesador | AMD Ryzen 9 9900X3D |
| RAM | 64GB DDR5 5600MT/s |
| GPU | NVIDIA RTX 5080 (16GB VRAM) |
| SO | Windows 11 (WSL2 para desarrollo) |

---

## 🏗️ ARQUITECTURA

| Componente | Tecnología | Justificación |
|---|---|---|
| **Modelo Base** | Qwen2-7B-Instruct | Bilingüismo nativo ES/EN, excelente para fine-tuning |
| **SFT** | QLoRA (4-bit NF4) | Eficiencia en 16GB VRAM |
| **DPO** | TRL 1.6.0 DPOTrainer | Alineación de seguridad |
| **Exportación** | GGUF (Q4_K_M, ~4.9 GB) | Compatible con llama.cpp, LM Studio, ollama |
---

## 🛡️ REQUERIMIENTOS DE SEGURIDAD
- Protección contra contenido perturbador o inapropiado para menores
- Resistencia a "Jailbreaks"
- Tono amigable, educativo y seguro ("Safe-by-design")
- 10 categorías de seguridad implementadas vía DPO
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
| 7 | Integración de voz (STT + TTS) | ⬜ Pendiente |
| 8 | Filtros de seguridad en tiempo real | ⬜ Pendiente |
| 9 | Evaluación con pruebas infantiles | ⬜ Pendiente |
---

## 🔧 ENTORNO DE DESARROLLO

**Ruta del proyecto:** `F:\LLM\Proyectos\Stardusts`

**Entorno Conda:**

