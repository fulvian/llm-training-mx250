#!/usr/bin/env python3
"""
Check the Italian model v2 structure
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("="*50)
print("Checking Model v2 Structure")
print("="*50)

# Check files
model_dir = "./italian-70m-qlora-output-v2"
if not os.path.exists(model_dir):
    print(f"❌ Model directory not found: {model_dir}")
else:
    print(f"✅ Model directory exists: {model_dir}")
    files = os.listdir(model_dir)
    print(f"Files in model directory: {len(files)}")
    for f in sorted(files):
        file_path = os.path.join(model_dir, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  - {f} ({size:,} bytes)")
        else:
            print(f"  - {f}/")

# Check tokenizer
print("\n" + "="*50)
print("Checking Tokenizer")
print("="*50)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("✅ Tokenizer loaded successfully")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {repr(tokenizer.pad_token)}")
    print(f"EOS token: {repr(tokenizer.eos_token)}")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")

# Check model loading
print("\n" + "="*50)
print("Checking Model Loading")
print("="*50)
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, model_dir)
    
    print("✅ Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test a simple generation
    prompt = "### Domanda: Qual è la capitale dell'Italia?\n### Risposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test generation: {repr(generated)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    print(traceback.format_exc())
