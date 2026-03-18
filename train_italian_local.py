#!/usr/bin/env python3
"""
Fine-tuning SmolLM-135M con dataset italiani mergiati per GPU 2GB
"""

import os


import torch
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
# CONFIGURAZIONE
# ============================================

MODEL_PATH = "./models/SmolLM-135M-Instruct"
MAX_SAMPLES_TINYSTORIES = 30000
MAX_SAMPLES_ALPACA = 15000
OUTPUT_DIR = "./smollm_italian_finetuned"
LOG_DIR = "./logs_smollm_italian"

BATCH_SIZE = 2  # Ridotto per evitare OOM
GRADIENT_ACCUMULATION_STEPS = 8  # Aumentato per compensare
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 384  # Compromesso tra 256 e 512

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ============================================
# FUNZIONI
# ============================================


def prepare_tinystories_dataset(max_samples=None):
    print("📥 Caricamento TinyStories-Italian...")
    try:
        dataset = load_dataset(
            "markod0925/TinyStories-Italian",
            split="train",
            token=os.environ.get("HF_TOKEN"),
        )
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        print(f"   ✅ Caricati {len(dataset)} storie")
        return dataset
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return None


def prepare_alpaca_dataset(max_samples=None):
    print("📥 Caricamento alpaca-gpt4-italian...")
    try:
        dataset = load_dataset(
            "FreedomIntelligence/alpaca-gpt4-italian",
            split="train",
            token=os.environ.get("HF_TOKEN"),
        )
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        print(f"   ✅ Caricati {len(dataset)} istruzioni")
        return dataset
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return None


def format_dataset(tinystories_data, alpaca_data, tokenizer, max_length=256):
    print("\n📝 Formattazione e merge dataset...")
    all_texts = []

    print("   Elaborazione TinyStories...")
    for item in tqdm(tinystories_data, desc="   TinyStories"):
        text = item.get("text", "")
        if text and len(text) > 20:
            all_texts.append(text)

    print("   Elaborazione Alpaca...")
    for item in tqdm(alpaca_data, desc="   Alpaca"):
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

    print(f"   ✅ Totale testi: {len(all_texts)}")

    print("   🔤 Tokenizzazione...")
    encodings = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    return encodings


class ItalianDataset(torch.utils.data.Dataset):
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


def main():
    print("=" * 60)
    print("🇮🇹 FINE-TUNING SmolLM-135M IN ITALIANO")
    print("   Ottimizzato per GPU 2GB con LoRA")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. Carica modello e tokenizer
    print("\n📦 Caricamento modello e tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   ℹ️ Impostato pad_token = eos_token")

    # Quantizzazione 4-bit per risparmiare memoria
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},  # Forza su GPU 0
        torch_dtype=torch.float16,
    )

    print(f"   ✅ Modello caricato da: {MODEL_PATH}")
    print(f"   📊 Parametri: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Configura LoRA
    print("\n🔧 Configurazione LoRA...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # Ridotto per memoria
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Carica dataset
    print("\n📚 Caricamento dataset...")

    tinystories_data = prepare_tinystories_dataset(MAX_SAMPLES_TINYSTORIES)
    alpaca_data = prepare_alpaca_dataset(MAX_SAMPLES_ALPACA)

    if tinystories_data is None or alpaca_data is None:
        print("❌ Errore nel caricamento dei dataset!")
        return

    # 4. Formatta dataset
    encodings = format_dataset(tinystories_data, alpaca_data, tokenizer, MAX_SEQ_LENGTH)

    # 5. Crea dataset PyTorch
    full_dataset = ItalianDataset(encodings)

    # Split train/validation
    train_size = int(0.95 * len(full_dataset))
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(
        full_dataset, range(train_size, len(full_dataset))
    )

    print(f"   📊 Train: {len(train_dataset)} campioni")
    print(f"   📊 Validation: {len(val_dataset)} campioni")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Configurazione training
    print("\n⚙️ Configurazione training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=100,
        logging_dir=LOG_DIR,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=False,  # Disabilitato per modelli 4-bit
        dataloader_num_workers=0,  # 0 per CUDA-safe
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # Ottimizzatore memory-efficient
    )

    # 7. Trainer
    print("\n🚀 Inizializzazione Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 8. Training
    print("\n" + "=" * 60)
    print("🎯 INIZIO TRAINING")
    print("=" * 60)

    trainer.train()

    # 9. Salvataggio
    print("\n💾 Salvataggio modello...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Training completato!")
    print(f"   📁 Modello salvato in: {OUTPUT_DIR}")

    # 10. Test
    print("\n🧪 Test rapido...")

    model.eval()
    test_prompts = [
        "C'era una volta",
        "L'Italia è",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Output: '{generated}'")


if __name__ == "__main__":
    main()
