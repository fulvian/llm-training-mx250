#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "./italian-70m-qlora-output-v2")

print(f"Model loaded on: {next(model.parameters()).device}")
print("="*50)

questions = [
    "Qual è la capitale dell'Italia?",
    "Qual è il fiume più lungo d'Italia?",
    "Qual è la lingua ufficiale dell'Italia?",
    "Qual è il vulcano attivo più famoso d'Italia?",
    "In che anno è stata proclamata la Repubblica italiana?",
]

for q in questions:
    prompt = f"### Domanda: {q}\n### Risposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {q}")
    print(f"A: {text.split('### Risposta:')[-1].strip()}")
    print("-"*50)
