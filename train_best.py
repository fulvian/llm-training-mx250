#!/usr/bin/env python3
"""
Fine-tuning SmolLM-135M - VERSIONE OTTIMIZZATA
Basata sui parametri che funzionavano meglio
Massimizza utilizzo GPU+CPU su MX250 2GB
"""

import os

import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ============================================
# CONFIGURAZIONE - Parametri testati
# ============================================

MODEL_PATH = "./models/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm_best_output"
MAX_SAMPLES_TINY = 30000
MAX_SAMPLES_ALPACA = 15000

# Batch - Bilanciato per 2GB VRAM
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Effective = 16
MAX_SEQ_LENGTH = 256

# Training
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
WARMUP_STEPS = 100

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


class ItalianDataset(torch.utils.data.Dataset):
    """Dataset efficiente."""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx].clone(),
        }


def prepare_tinystories(max_samples):
    print("   TinyStories-Italian...")
    ds = load_dataset(
        "markod0925/TinyStories-Italian",
        split="train",
        token=os.environ.get("HF_TOKEN"),
    )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"   ✅ {len(ds):,} storie")
    return ds


def prepare_alpaca(max_samples):
    print("   Alpaca-gpt4-italian...")
    ds = load_dataset(
        "FreedomIntelligence/alpaca-gpt4-italian",
        split="train",
        token=os.environ.get("HF_TOKEN"),
    )
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"   ✅ {len(ds):,} istruzioni")
    return ds


def format_and_tokenize(tinystories, alpaca, tokenizer, max_length):
    print("\n📝 Formattazione dataset...")
    all_texts = []
    
    print("   TinyStories...")
    for item in tqdm(tinystories, desc="   TinyStories"):
        text = item.get("text", "")
        if text and len(text) > 20:
            all_texts.append(text)
    
    print("   Alpaca...")
    for item in tqdm(alpaca, desc="   Alpaca"):
        conversations = item.get("conversations", [])
        for conv in conversations:
            if isinstance(conv, dict):
                role = conv.get("from", "")
                content = conv.get("value", "")
                if content and len(content) > 10:
                    if role == "human":
                        all_texts.append(f"Domanda: {content}")
                    elif role == "gpt":
                        all_texts.append(f"Risposta: {content}")
    
    print(f"   ✅ Totale: {len(all_texts):,} testi")
    
    print("   🔤 Tokenizzazione...")
    encodings = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    
    return encodings


def main():
    print("=" * 60)
    print("🚀 FINE-TUNING SmolLM - VERSIONE OTTIMIZZATA")
    print("=" * 60)
    
    # Pulisci memoria
    torch.cuda.empty_cache()
    gc.collect()
    
    # Info GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ GPU non disponibile!")
        return
    
    print(f"\n⚙️ Config:")
    print(f"   Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Max seq: {MAX_SEQ_LENGTH}")
    print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    
    # Tokenizer
    print("\n📦 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Modello 4-bit
    print("📦 Modello 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    
    # LoRA
    print("🔧 LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Dataset
    print("\n📚 Dataset...")
    tinystories = prepare_tinystories(MAX_SAMPLES_TINY)
    alpaca = prepare_alpaca(MAX_SAMPLES_ALPACA)
    
    # Tokenizza
    encodings = format_and_tokenize(tinystories, alpaca, tokenizer, MAX_SEQ_LENGTH)
    
    # Crea dataset
    full_dataset = ItalianDataset(encodings)
    
    # Split
    train_size = int(0.95 * len(full_dataset))
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    print(f"\n📊 Train: {len(train_dataset):,}")
    print(f"📊 Eval: {len(eval_dataset):,}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training args
    print("\n⚙️ Training args...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,  # KEY: False per 4-bit
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,  # KEY: 0 per CUDA
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Training
    print("\n" + "=" * 60)
    print("🎯 INIZIO TRAINING")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Salva
    print("\n💾 Salvataggio...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Salvato in: {OUTPUT_DIR}")
    
    # Test
    print("\n🧪 Test...")
    model.eval()
    for prompt in ["C'era una volta", "L'Italia è"]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, temperature=0.7, do_sample=True)
        print(f"  {prompt} → {tokenizer.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
