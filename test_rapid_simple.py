#!/usr/bin/env python3
"""
Test rapido training QLoRA - Versione semplificata
Durata: 5-10 minuti
"""

import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc

print("=" * 60)
print("TEST RAPIDO QLORA - 10 CAMPIONI, 1 EPOCH")
print("=" * 60)

# Configurazione
MODEL_PATH = "./models/SmolLM-135M-Instruct"
DATASET_PATH = "test_dataset.json"
OUTPUT_DIR = "./test_output_rapid"
MAX_SEQ_LENGTH = 128  # Ridotto per velocità

# Step 1: Carica dataset
print("\n[1/6] Caricamento dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
print(f"   Dataset: {len(dataset)} campioni")

# Step 2: Carica tokenizer
print("\n[2/6] Caricamento tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"   Vocab size: {tokenizer.vocab_size}")

# Step 3: Tokenizza dataset
print("\n[3/6] Tokenizzazione...")


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset.column_names
)
print(f"   Tokenizzato: {len(tokenized_dataset)} campioni")

# Step 4: Carica modello con quantizzazione
print("\n[4/6] Caricamento modello...")
torch.cuda.empty_cache()
gc.collect()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Step 5: Applica LoRA minimale
print("\n[5/6] Configurazione LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Minimale
    lora_alpha=16,
    lora_dropout=0.15,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.config.use_cache = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Trainabili: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

# Verifica memoria
allocated = torch.cuda.memory_allocated(0) / 1024**2
print(f"   VRAM usata: {allocated:.2f} MB")

# Step 6: Training
print("\n[6/6] Training...")
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.15,
    max_grad_norm=1.0,
    logging_steps=2,
    save_steps=100,
    optim="adamw_8bit",
    gradient_checkpointing=True,
    fp16=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Avvia training
train_result = trainer.train()

# Salva modello
trainer.save_model()
print(f"   Modello salvato in: {OUTPUT_DIR}")

# Test rapido
print("\n" + "=" * 60)
print("TEST GENERAZIONE POST-TRAINING")
print("=" * 60)

model.eval()
test_prompts = [
    "C'era una volta",
    "L'Italia è",
]

with torch.no_grad():
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: '{prompt}'")
        print(f"Output: '{generated}'")
        print()

# Cleanup
del model
torch.cuda.empty_cache()
gc.collect()

print("=" * 60)
print("✅ TEST COMPLETATO")
print("=" * 60)
print(f"Train loss: {train_result.training_loss:.4f}")
print(f"Train time: {train_result.metrics['train_runtime']:.1f}s")
