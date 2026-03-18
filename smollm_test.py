#!/usr/bin/env python3
"""
Test specifico per SmolLM-135M-Instruct
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "./models/SmolLM-135M-Instruct"
FINETUNED_MODEL = "./smollm-135m-qlora-output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Tokenizer con vocab corretto
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
tokenizer.pad_token = tokenizer.eos_token

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

# Prompt specifico per SmolLM-135M-Instruct
question = "When did Virgin Australia start operating?"
prompt = f"""### Instruction: {question}

### Answer:"""

encoding = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **encoding,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Risposta completa: '{response}'")

# Estrai solo la risposta
if "### Answer:" in response:
    answer = response.split("### Answer:")[1].strip()
    print(f"Risposta estraibile: '{answer}'")

