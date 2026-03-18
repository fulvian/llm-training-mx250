#!/usr/bin/env python3
"""
Train Italian GPT2 model with larger dataset and more epochs
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from datasets import Dataset
import json

def load_dataset_from_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)

def main():
    dataset = load_dataset_from_file("./datasets/Italian-GPT2-Large/italian_qa.jsonl")
    
    MODEL_NAME = "GroNLP/gpt2-small-italian"
    OUTPUT_DIR = "./italian-gpt2-qlora-output-large"
    LEARNING_RATE = 1.5e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4
    NUM_EPOCHS = 15
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Modello: {MODEL_NAME}")
    print(f"Dataset size: {len(dataset)} record")
    print(f"Batch size: {BATCH_SIZE} (accumulato a {GRADIENT_ACCUMULATION})")
    print(f"Epoche: {NUM_EPOCHS}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    
    def format_instruction(sample):
        text = f"""### Domanda: {sample["question"]}
### Risposta: {sample["answer"]}"""
        return {"text": text}
    
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            truncation=True, 
            max_length=150, 
            padding="max_length"
        )
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.map(tokenize, batched=False, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=2,
        save_steps=20,
        warmup_steps=5,
        bf16=False,
        fp16=False,
        optim="paged_adamw_8bit",
        eval_strategy="no",
        save_strategy="steps",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    from trl import SFTTrainer, SFTConfig
    
    sft_config = SFTConfig(
        dataset_text_field="text",
        max_length=150,
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=lora_config,
    )
    
    trainer.train()
    
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completato! Modello salvato in: {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrotto")
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        print(traceback.format_exc())
