#!/bin/bash
set -e

echo "=== Italian Model Performance Suite ==="
echo "="*50

# Check 1: Model structure
echo "1. Checking Model Structure..."
python3 check_model_v2.py

# Check 2: Quick test
echo "2. Running Quick Test..."
python3 quick_test_v2.py

# Check 3: Full evaluation
echo "3. Running Full Evaluation..."
python3 test_italian_responses_v2.py

# Check 4: Performance report
echo "4. Generating Performance Report..."
python3 - <<END
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

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

with open('datasets/Italian-50/italian_qa.jsonl', 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f if line.strip()]

correct = 0
total = len(dataset)

for item in dataset:
    prompt = f"### Domanda: {item['question']}\n### Risposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.2,
            top_p=0.6,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("### Risposta:")[-1].strip()
    
    if item['answer'].lower() in answer.lower():
        correct += 1

accuracy = (correct / total) * 100
print(f"\n=== Final Results ===")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
END

echo "="*50
