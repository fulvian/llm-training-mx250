#!/usr/bin/env python3
"""
Test tokenizer and generation parameters
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "./models/SmolLM-135M-Instruct"
FINETUNED_MODEL = "./smollm-135m-qlora-output"

# Test tokenizer separatamente
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("=== Test tokenizer ===")
print(f"Tokenizer name: {tokenizer.name_or_path}")
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"Eos token: {tokenizer.eos_token}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"Eos token ID: {tokenizer.eos_token_id}")

# Test tokenization
question = "When did Virgin Australia start operating?"
prompt = f"""### Instruction
{question}

### Response
"""

encoding = tokenizer(prompt, return_tensors="pt")
print(f"\nTokenization result:")
print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"Attention mask: {encoding['attention_mask']}")
print(f"Tokens: {[tokenizer.decode([i]) for i in encoding['input_ids'][0]]}")

# Configurazione quantizzazione
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Caricamento modello
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

model.eval()

print("\n=== Modello caricato ===\n")

with torch.no_grad():
    print("Test con parametri standard:")
    outputs = model.generate(
        **encoding,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=encoding['attention_mask']
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Risposta: {repr(response)}")
    print(f"Lunghezza: {len(response)} caratteri")
