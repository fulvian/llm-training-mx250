#!/usr/bin/env python3
"""
Verifica la struttura del modello Pythia-70M per LoRA
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "EleutherAI/pythia-70m-deduped"

# Configurazione quantizzazione
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Caricamento del modello {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("\nStruttura del modello:")
print("=" * 50)
print(model)

print("\nNomi dei moduli:")
print("=" * 50)
for name, module in model.named_modules():
    if hasattr(module, "weight") or hasattr(module, "bias"):
        print(name)
