#!/usr/bin/env python3
"""
Test del modello con domande predefinite
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "./models/SmolLM-135M-Instruct"
FINETUNED_MODEL = "./smollm-135m-qlora-output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
tokenizer.pad_token = tokenizer.eos_token

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

def generate_response(prompt):
    formatted_prompt = f"""### Instruction: {prompt}

### Answer:"""
    
    encoding = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
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
    if "### Answer:" in response:
        return response.split("### Answer:")[1].strip()
    return response.strip()

# Test con domande predefinite
test_questions = [
    "When did Virgin Australia start operating?",
    "Which is a species of fish? Tope or Rope",
    "Why can camels survive for long without water?",
    "What is a polygon?",
    "How do I start running?",
    "What is process mining?",
    "Who was John Moses Browning?",
    "What are some unique curtain tie backs that you can make yourself?",
    "Which episodes of season four of Game of Thrones did Michelle MacLaren direct?",
    "Identify which instrument is string or percussion: Cantaro, Gudok"
]

print("=" * 50)
print("Test del modello con domande predefinite")
print("=" * 50)

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. {question}")
    try:
        response = generate_response(question)
        print(f"Risposta: {response}")
    except Exception as e:
        print(f"Errore: {str(e)}")

