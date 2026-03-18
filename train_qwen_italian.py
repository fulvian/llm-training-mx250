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
    print("📥 Caricamento TinyStories-Italian...")
    
    try:
        dataset = load_dataset("markod0925/TinyStories-Italian", split="train")
        data = dataset["train"]
        
        if max_samples and len(data) > max_samples:
            data = data.select(range(max_samples))
            print(f"   ✂ Limitato a {max_samples} campioni")
        
        print(f"   ✓ Caricati {len(data)} storie")
        return data
    except Exception as e:
        print(f"   ❌ Errore caricamento TinyStories: {e}")
        return None


def prepare_alpaca_dataset(max_samples=None):
    """Carica e prepara alpaca-gpt4-italian"""
    print("📥 Caricamento alpaca-gpt4-italian...")
    
    try:
        dataset = load_dataset("FreedomIntelligence/alpaca-gpt4-italian", split="train")
        data = dataset["train"]
        
        if max_samples and len(data) > max_samples:
            data = data.select(range(max_samples))
            print(f"   ✂ Limitato a {max_samples} campioni")
        
        print(f"   ✓ Caricati {len(data)} esempi instruction-follow")
        return data
    except Exception as e:
        print(f"   ❌ Errore caricamento Alpaca: {e}")
        return None


def format_tinystory(example):
    """Formatta una storia per il training"""
    text = example.get("text", "")
    if text:
        return {"text": text}
    return None


def format_alpaca(example):
    """Formatta esempio Alpaca in formato conversazionale"""
    conversations = example.get("conversations", [])
    
    if not conversations:
        return None
    
    formatted_text = ""
    for conv in conversations:
        role = conv.get("from", "")
        content = conv.get("value", "")
        
        if role == "human":
            formatted_text += f"<|user|>\n{content}\n"
        elif role == "gpt":
            formatted_text += f"<|assistant|)\n{content}\n"
    
    if formatted_text:
        return {"text": formatted_text}
    return None


class ItalianDataset(torch.utils.data.Dataset):
    """Dataset PyTorch per il training"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        
        # Tokenizza
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Per causal LM, labels = input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    print("=" * 60)
    print("🇮🇹 FINE-TUNING QWEN2-0.5B IN ITALIANO")
    print("   Ottimizzato per GPU 2GB con LoRA + INT8")
    print("=" * 60)
    
    # Crea directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ============================================
    # 1. CARICA MODELLI E TOKENIZER
    # ============================================
    print("\n📦 Caricamento modello e tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side='left'
    )
    
    # Aggiungi pad token se non presente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   ℹ️ Impostato pad_token = eos_token")
    
    # Carica modello in INT8 per risparmiare memoria
    print("   🔄 Caricamento modello in INT8...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"   ✓ Modello caricato: {MODEL_NAME}")
    
    # ============================================
    # 2. CONFIGURA LoRA
    # ============================================
    print("\n🔧 Configurazione LoRA...")
    
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
    # 3. CARICA E MERGE DATASET
    # ============================================
    print("\n📚 Caricamento e merge dataset italiani...")
    
    all_data = []
    
    # TinyStories
    tinystories = prepare_tinystories_dataset(MAX_SAMPLES_TINYSTORIES)
    if tinystories:
        formatted_stories = []
        for example in tinystories:
            formatted = format_tinystory(example)
            if formatted:
                formatted_stories.append(formatted)
        all_data.extend(formatted_stories)
        print(f"   ✓ TinyStories: {len(formatted_stories)} esempi formattati")
    
    # Alpaca
    alpaca = prepare_alpaca_dataset(MAX_SAMPLES_ALPACA)
    if alpaca:
        formatted_alpaca = []
        for example in alpaca:
            formatted = format_alpaca(example)
            if formatted:
                formatted_alpaca.append(formatted)
        all_data.extend(formatted_alpaca)
        print(f"   ✓ Alpaca: {len(formatted_alpaca)} esempi formattati")
    
    print(f"\n📊 TOTALE ESEMPI: {len(all_data)}")
    
    # ============================================
    # 4. CREAZIONE DATASET PYTORCH
    # ============================================
    print("\n🔨 Creazione dataset PyTorch...")
    
    train_dataset = ItalianDataset(all_data, tokenizer, MAX_SEQ_LENGTH)
    print(f"   ✓ Dataset creato con {len(train_dataset)} esempi")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ============================================
    # 5. CONFIGURAZIONE TRAINING
    # ============================================
    print("\n⚙️ Configurazione training...")
    
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
    # 6. TRAINER
    # ============================================
    print("\n🚀 Inizializzazione Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # ============================================
    # 7. TRAINING
    # ============================================
    print("\n" + "=" * 60)
    print("🎯 INIZIO TRAINING")
    print("=" * 60)
    
    trainer.train()
    
    # ============================================
    # 8. SALVATAGGIO
    # ============================================
    print("\n💾 Salvataggio modello...")
    
    # Salva solo i pesi LoRA (piccoli)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Training completato!")
    print(f"   📁 Modello salvato in: {OUTPUT_DIR}")
    print(f"   📊 Parametri trainable: ~{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ============================================
    # 9. TEST RAPIDO
    # ============================================
    print("\n🧪 Test rapido del modello...")
    
    model.eval()
    test_prompt = "C'era una volta"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Prompt: '{test_prompt}'")
    print(f"   Output: '{generated}'")


if __name__ == "__main__":
    main()
