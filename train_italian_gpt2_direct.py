#!/usr/bin/env python3
"""
Train Italian GPT2 model directly from Hugging Face with QLoRA
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
    """Carica il dataset da file JSONL"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                
    return Dataset.from_list(data)

def main():
    # Step 1: Carica il dataset
    print("Step 1: Caricamento dataset...")
    dataset = load_dataset_from_file("./datasets/Italian-GPT2/italian_qa.jsonl")
    
    # Step 2: Configurazione
    MODEL_NAME = "GroNLP/gpt2-small-italian"
    OUTPUT_DIR = "./italian-gpt2-qlora-output"
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4
    NUM_EPOCHS = 10
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nStep 2: Configurazione:")
    print(f"  - Modello: {MODEL_NAME}")
    print(f"  - Output dir: {OUTPUT_DIR}")
    print(f"  - Batch size: {BATCH_SIZE} (accumulato a {GRADIENT_ACCUMULATION})")
    print(f"  - Epoche: {NUM_EPOCHS}")
    
    # Step 3: Quantizzazione 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    # Step 4: Configurazione LoRA per GPT2
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    
    # Step 5: Caricamento tokenizer
    print("\nStep 3: Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Step 6: Caricamento modello
    print("Step 4: Caricamento modello con quantizzazione 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    
    # Step 7: Preprocessing del dataset
    print("\nStep 5: Preprocessing del dataset...")
    
    def format_instruction(sample):
        text = f"""### Domanda: {sample["question"]}
### Risposta: {sample["answer"]}"""
        return {"text": text}
    
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        )
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.map(tokenize, batched=False, remove_columns=["text"])
    
    print(f"Dataset finale creato: {len(dataset)} record")
    
    # Step 8: Training arguments
    print("\nStep 6: Configurazione training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=2,
        save_steps=10,
        warmup_steps=2,
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
    
    # Step 9: Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Step 10: Trainer
    print("\nStep 7: Inizio training...")
    from trl import SFTTrainer, SFTConfig
    
    sft_config = SFTConfig(
        dataset_text_field="text",
        max_length=128,
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
    
    # Step 11: Esegui training
    trainer.train()
    
    # Step 12: Salva
    print("\nStep 8: Salvataggio modello...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Training completato!")
    print(f"Modello salvato in: {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        import traceback
        print(traceback.format_exc())
