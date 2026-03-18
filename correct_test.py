#!/usr/bin/env python3
"""
Test del modello usando il tokenizer originale
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "./models/SmolLM-135M-Instruct"
FINETUNED_MODEL = "./smollm-135m-qlora-output"

# Usiamo il tokenizer originale del modello base
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("=== Tokenizer originale ===")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Pad token: '{tokenizer.pad_token}'")
print(f"Eos token: '{tokenizer.eos_token}'")

# Test tokenization
question = "When did Virgin Australia start operating?"
prompt = f"""### Instruction
{question}

### Response
"""

encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
print(f"\nTokenization (senza special tokens):")
print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])}")

# Configurazione quantizzazione
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

model.eval()

print("\n=== Generazione ===")
with torch.no_grad():
    outputs = model.generate(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Risposta: '{response}'")
    
    # Estrai solo la risposta
    if "### Response" in response:
        final_response = response.split("### Response")[1].strip()
        print(f"Risposta estraibile: '{final_response}'")
