#!/usr/bin/env python3
"""
Fusiona los adaptadores SFT + DPO sobre unsloth/gemma-4-E4B-it EN BF16
(no en 4-bit) y guarda un checkpoint HF limpio, listo para convertir a GGUF.

Por qué no reusar inference_gemma4.py tal cual: ese script carga la base
en 4-bit (load_in_4bit=True) porque para chatear interactivamente eso basta
y ahorra VRAM. Pero para el artefacto que vas a exportar, fusionar los LoRA
sobre una base ya cuantizada a 4-bit significa cuantizar -> fusionar ->
descuantizar -> volver a cuantizar en la conversión GGUF. Cada paso de
cuantización pierde precisión, y aquí se acumularían dos de más. Para el
export quieres UNA sola cuantización, la que hagas al final con
llama-quantize — así que esta fusión se hace una sola vez, en precisión
completa.

El orden replica exactamente inference_gemma4.py: base -> aplicar y
fusionar SFT -> aplicar y fusionar DPO sobre el resultado ya fusionado.
Si tu DPO se entrenó tomando como base el checkpoint ya fusionado con SFT
(que es lo que el propio inference_gemma4.py asume), este es el orden
correcto. Si tu DPO se entrenó de forma independiente directamente sobre
la base cruda, avísame porque el orden correcto sería otro.

Uso:
    python scripts/merge_for_gguf.py
    python scripts/merge_for_gguf.py --out merged/gemma4-E4B-kidsafe-fp16
"""

import argparse
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="merged/gemma4-E4B-kidsafe-fp16",
        help="Carpeta de salida para el checkpoint HF fusionado (relativa a la raíz del proyecto)",
    )
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from peft import PeftModel

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Fusión SFT + DPO en BF16 (para exportar a GGUF)")
    print("=" * 60)

    print("\n[1/5] Cargando base en bf16 (load_in_4bit=False)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Base cargada. VRAM: {vram:.2f} GB")
    print(
        "  Si esto se queda sin VRAM en tu RTX 5080 (16GB), corre esta misma "
        "carga con transformers.AutoModelForCausalLM + device_map='cpu' en su "
        "lugar — es más lento pero tus 64GB de RAM lo absorben sin problema, "
        "y esta fusión solo se hace una vez."
    )

    # Diagnóstico: ¿el loader de Unsloth conservó las torres de visión/audio?
    # FastLanguageModel está pensado ante todo para LLMs de texto, así que no
    # es automático que preserve estos módulos en un checkpoint multimodal.
    # Esto decide si más adelante puedes auto-convertir tu propio mmproj o si
    # te conviene usar directamente el mmproj oficial (ver notas al final).
    param_names = {n for n, _ in model.named_parameters()}
    has_vision = any("vision_tower" in n for n in param_names)
    has_audio = any("audio_tower" in n for n in param_names)
    print(f"  Torre de visión presente en el checkpoint cargado: {has_vision}")
    print(f"  Torre de audio presente en el checkpoint cargado: {has_audio}")
    if not (has_vision and has_audio):
        print(
            "  -> No están completas. Esto es esperable con FastLanguageModel. "
            "Para el mmproj, usa el archivo oficial (ver instrucciones al final) "
            "en vez de intentar auto-convertirlo desde este checkpoint."
        )

    print("\n[2/5] Aplicando y fusionando el adaptador SFT...")
    sft_path = PROJECT_ROOT / "outputs" / "gemma4_4b_sft" / "final"
    model = PeftModel.from_pretrained(model, str(sft_path))
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  SFT fusionado. VRAM: {vram:.2f} GB")

    print("\n[3/5] Aplicando y fusionando el adaptador DPO sobre el resultado...")
    dpo_path = PROJECT_ROOT / "outputs" / "gemma4_4b_dpo" / "final"
    model = PeftModel.from_pretrained(model, str(dpo_path))
    model = model.merge_and_unload()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  DPO fusionado. VRAM: {vram:.2f} GB")

    print(f"\n[4/5] Guardando checkpoint HF fusionado en {out_dir} ...")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # Fuerza el chat_template.jinja del checkpoint DPO (el más reciente/autoritativo)
    # por encima de lo que tokenizer.save_pretrained() haya escrito por su cuenta.
    # Un chat_template desalineado con el que se usó en entrenamiento es la forma
    # más silenciosa de degradar calidad en una conversión GGUF.
    for fname in ("chat_template.jinja", "processor_config.json"):
        src = dpo_path / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
            print(f"  Copiado {fname} desde el checkpoint DPO (fuente de verdad)")

    print("\n[5/5] Listo. Antes de convertir a GGUF, valida el checkpoint fusionado:")
    print(f"  - Compara unas cuantas respuestas contra inference_gemma4.py (que fusiona en 4-bit)")
    print(f"  - Si difieren mucho, el merge en bf16 es el que debes creer, no el de 4-bit")
    print(f"\nSiguiente paso — conversión a GGUF (fuera de este script):")
    print(f"  python convert_hf_to_gguf.py {out_dir} --outfile gemma4-E4B-kidsafe-F16.gguf --outtype f16")
    print(f"  ./llama-quantize gemma4-E4B-kidsafe-F16.gguf gemma4-E4B-kidsafe-Q6_K.gguf Q6_K")


if __name__ == "__main__":
    main()
