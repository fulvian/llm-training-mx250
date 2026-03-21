#!/usr/bin/env python3
"""
Ottimizzazione Memoria Avanzata per QLoRA su MX250 2GB
Script per testare configurazioni ottimizzate prima del training completo
"""

import os
import gc
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def check_gpu_memory():
    """Verifica memoria GPU attuale."""
    if not torch.cuda.is_available():
        print("❌ CUDA non disponibile!")
        return False

    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"📊 Memoria GPU:")
    print(f"   Allocata: {allocated:.2f} MB")
    print(f"   Riservata: {reserved:.2f} MB")
    print(f"   Totale: {total:.2f} MB")
    print(f"   Libera: {total - allocated:.2f} MB")

    return True


def load_model_ultra_optimized(model_path: str):
    """Carica modello con ottimizzazioni ultra-aggressive."""
    print(f"\n🔄 Caricamento modello: {model_path}")

    # Clear memory prima
    torch.cuda.empty_cache()
    gc.collect()

    # Configurazione quantizzazione ultra-ottimizzata
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Carica modello
    print("⏳ Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    # Abilita gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print("✅ Modello caricato")

    # Verifica memoria dopo caricamento
    check_gpu_memory()

    return model


def apply_minimal_lora(model):
    """Applica LoRA con configurazione minimale."""
    print("\n🎯 Configurazione LoRA Minimale:")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Ridotto da 16
        lora_alpha=16,  # Ridotto da 32
        lora_dropout=0.2,  # Aumentato da 0.15
        target_modules=["q_proj", "v_proj"],  # Solo attention, non MLP
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Stampa parametri
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"   Parametri trainabili: {trainable:,}")
    print(f"   Parametri totali: {total:,}")
    print(f"   Percentuale trainabile: {100 * trainable / total:.4f}%")

    # Verifica memoria dopo LoRA
    check_gpu_memory()

    return model


def test_generation(model, tokenizer, prompt: str):
    """Testa generazione testo."""
    print(f"\n🧪 Test Generazione:")
    print(f"   Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"   Output: '{generated}'")
    print(f"   Lunghezza: {len(generated)} caratteri")

    return generated


def main():
    print("=" * 70)
    print("OTTIMIZAZIONE MEMORIA QLORA - MX250 2GB")
    print("=" * 70)

    # Verifica GPU
    if not check_gpu_memory():
        return

    # Path modello
    model_path = "./models/SmolLM-135M-Instruct"

    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        return

    # Carica modello ottimizzato
    model = load_model_ultra_optimized(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Applica LoRA minimale
    model = apply_minimal_lora(model)

    # Test generazione
    test_prompts = [
        "C'era una volta",
        "L'Italia è",
        "Spiegami cos'è",
    ]

    for prompt in test_prompts:
        test_generation(model, tokenizer, prompt)

    # Cleanup finale
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print("✅ OTTIMIZAZIONE COMPLETATA")
    print("=" * 70)
    check_gpu_memory()


if __name__ == "__main__":
    main()
