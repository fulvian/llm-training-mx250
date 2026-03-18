#!/usr/bin/env python3
"""
Test del modello SmolLM-135M fine-tunato su databricks-dolly-15k
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Configurazione
BASE_MODEL = "./models/SmolLM-135M-Instruct"
FINETUNED_MODEL = "./smollm-135m-qlora-output"

# Load tokenizer
print("Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Caricamento modello base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Caricamento adapter fine-tunato...")
model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_MODEL
)

model.eval()
print("Modello caricato con successo!")

# Funzione per generare risposte
def generate_response(instruction):
    prompt = f"""### Instruction
{instruction}

### Response
"""
    
    encoding = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Risposta completa:", repr(response))
    if "### Response" in response:
        return response.split("### Response")[1].strip()
    return response.strip()

# Test con domande
test_questions = [
    "When did Virgin Australia start operating?",
    "Which is a species of fish? Tope or Rope", 
    "Why can camels survive for long without water?",
    "What is a polygon?",
    "How do I start running?",
]

print("\n=== Test del modello ===")
for i, question in enumerate(test_questions):
    print(f"\n{chr(97 + i)}) {question}")
    response = generate_response(question)
    print(f"Risposta: {response}")

