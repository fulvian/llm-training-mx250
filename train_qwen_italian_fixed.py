#!/usr/bin/env python3
"""
Fine-tuning Qwen2-0.5B con dataset italiani mergiati per GPU 2GB
Dataset utilizzati:
- markod0925/TinyStories-Italian (2.14M storie)
- FreedomIntelligence/alpaca-gpt4-italian (50k istruzioni)
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import transformers

from tqdm import tqdm

# ============================================
# CONFIGURAZIONE
# ============================================

# Modello base
MODEL_NAME = "Qwen/Qwen2-0.5B"

# Dataset
MAX_SAMPLES_TINYSTORIES = 50000  # Limite per memoria
MAX_SAMPLES_ALPACA = 20000

# Output
OUTPUT_DIR = "./qwen_italian_finetuned"
LOG_DIR = "./logs_qwen_italian"

# Training parameters per GPU 2GB
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # Effettivo batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 256  # Ridotto per memoria

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ============================================
# FUNZIONI UTILITÀ
# ============================================

def prepare_tinystories_dataset(max_samples=None):
    """Carica e prepara TinyStories-Italian"""
    print("Caricamento TinyStories-Italian...")
    
    try:
        dataset = load_dataset("markod0925/TinyStories-Italian", split="train")
        data = dataset["train"]
        
        if max_samples and len(data) > max_samples:
            data = data.select(range(max_samples))
            print(f"   Limitato a {max_samples} campioni")
        
        print(f"   Caricati {len(data)} storie")
        return data
    except Exception as e:
        print(f"   Errore caricamento TinyStories: {e}")
        return None


def prepare_alpaca_dataset(max_samples=None):
    """Carica e prepara alpaca-gpt4-italian"""
    print("Caricamento alpaca-gpt4-italian...")
    
    try:
        dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-italian", split="train")
        data = dataset["train"]
        
        if max_samples and len(data) > max_samples:
            data = data.select(range(max_samples))
            print(f"   Limitato a {max_samples} campioni")
        
        print(f"   Caricati {len(data)} istruzioni")
        return data
    except Exception as e:
        print(f"   Errore caricamento alpaca: {e}")
        return None


def merge_datasets(tinystories_data, alpaca_data, tokenizer, max_length=256):
    """Merge e formatta i dataset per il training"""
    print("\nMerge e formattazione dataset...")
    
    all_texts = []
    
    # Aggiungi storie da TinyStories
    print("   Elaborazione TinyStories...")
    for item in tqdm(tinystories_data, desc="Processing TinyStories"):
        text = item.get("text", "")
        if text:
            all_texts.append(text)
    
    # Aggiungi conversazioni da Alpaca
    print("   Elaborazione Alpaca...")
    for item in tqdm(alpaca_data, desc="Processing Alpaca"):
        conversations = item.get("conversations", [])
        for conv in conversations:
            if isinstance(conv, dict):
                role = conv.get("from", "")
                content = conv.get("value", "")
                if role == "human" and content:
                    # Formato instruction
                    all_texts.append(f"Istruzione: {content}")
                elif role == "gpt" and content:
                    # Formato risposta
                    all_texts.append(f"Risposta: {content}")
    
    print(f"   Totale testi: {len(all_texts)}")
    
    # Tokenizza
    print("   Tokenizzazione...")
    encodings = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    return encodings


class ItalianDataset(torch.utils.data.Dataset):
    """Dataset PyTorch per training"""
    
    def __init__(self, encodings, max_length=256):
        self.data = encodings
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"].squeeze()
        attention_mask = item["attention_mask"].squeeze()
        
        # Per causal LM, labels = input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    print("=" * 60)
    print("Fine-tuning Qwen2-0.5B in Italiano")
    print("   Ottimizzato per GPU 2GB con LoRA + Int8")
    print("=" * 60)
    
    # Crea directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ============================================
    # 1. Carica modello e tokenizer
    # ============================================
    print("\nCaricamento modello e tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Aggiungi pad token se non presente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   Impostato pad_token = eos_token")
    
    # Carica modello in INT8 per risparmiare memoria
    print("   Caricamento modello in INT8...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"   Modello caricato: {MODEL_NAME}")
    
    # ============================================
    # 2. Configura LoRA
    # ============================================
    print("\nConfigurazione LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # ============================================
    # 3. Carica e merge datataset
    # ============================================
    print("\nCaricamento dataset...")
    
    # Carica dataset
    tinystories_data = prepare_tinystories_dataset(MAX_SAMPLES_TINYSTORIES)
    alpaca_data = prepare_alpaca_dataset(MAX_SAMPLES_ALPACA)
    
    if tinystories_data is None or alpaca_data is None:
        print("ERRORE: Impossibile caricare i dataset!")
        return
    
    # Merge dataset
    encodings = merge_datasets(tinystories_data, alpaca_data, tokenizer, MAX_SEQ_LENGTH)
    
    # Crea dataset PyTorch
    train_dataset = ItalianDataset(encodings, MAX_SEQ_LENGTH)
    
    # Split train/validation
    train_size = int(0.95 * len(train_dataset))
    train_data = train_dataset.select(range(train_size))
    val_data = train_dataset.select(range(train_size, len(train_dataset)))
    
    print(f"   Train: {len(train_data)} campioni")
    print(f"   Validation: {len(val_data)} campioni")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ============================================
    # 4. Configurazione training
    # ============================================
    print("\nConfigurazione training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # Batch size
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Learning
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=100,
        
        # Logging
        logging_dir=LOG_DIR,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        
        # Ottimizzazioni memoria
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        
        # Altro
        report_to="none",
        remove_unused_columns=False,
    )
    
    # ============================================
    # 5. Trainer
    # ============================================
    print("\nInizializzazione Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )
    
    # ============================================
    # 6. Train
    # ============================================
    print("\nInizio training...")
    print("=" * 60)
    
    trainer.train()
    
    # ============================================
    # 7. Salvataggio
    # ============================================
    print("\nSalvataggio modello...")
    
    # Salva solo i pesi LoRA (piccoli)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\nTraining completato!")
    print(f"   Modello salvato in: {OUTPUT_DIR}")
    print(f"   Parametri trainable: ~{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ============================================
    # 8. Test
    # ============================================
    print("\nTest rapido del modello...")
    
    model.eval()
    test_prompt = "C'era una volta un piccolo villaggio in Italia."
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Prompt: '{test_prompt}'")
    print(f"   Output: '{generated}'")


    

if __name__ == "__main__":
    main()
