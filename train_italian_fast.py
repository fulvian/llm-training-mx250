#!/usr/bin/env python3
"""
Fine-tuning SmolLM-135M ottimizzato per GPU 2GB
- Streaming dataset (no OOM)
- Padding dinamico
- Massimo utilizzo GPU
"""

import os


import torch
import gc
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================
# CONFIGURAZIONE OTTIMIZZATA
# ============================================

MODEL_PATH = "./models/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm_italian_fast"
MAX_SAMPLES_TINY = 30000
MAX_SAMPLES_ALPACA = 15000

# OTTIMIZZAZIONI GPU
BATCH_SIZE = 4  # Aumentato!
GRADIENT_ACCUMULATION = 4  # Effective batch = 16
MAX_SEQ_LENGTH = 256  # Bilanciato
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1

# LoRA compatto
LORA_R = 8
LORA_ALPHA = 16

# ============================================
# FUNZIONI
# ============================================


def load_and_merge_datasets():
    """Carica e mergea i dataset in streaming."""
    print("📚 Caricamento dataset...")

    # TinyStories
    print("   TinyStories-Italian...")
    tiny = load_dataset(
        "markod0925/TinyStories-Italian",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Alpaca
    print("   Alpaca-gpt4-italian...")
    alpaca = load_dataset(
        "FreedomIntelligence/alpaca-gpt4-italian",
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Prendi subset
    tiny = tiny.take(MAX_SAMPLES_TINY)
    alpaca = alpaca.take(MAX_SAMPLES_ALPACA)

    return tiny, alpaca


def preprocess_tinystories(examples, tokenizer, max_length):
    """Preprocessing TinyStories."""
    texts = examples["text"]

    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,  # Padding dinamico dopo
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def preprocess_alpaca(examples, tokenizer, max_length):
    """Preprocessing Alpaca."""
    texts = []
    for convs in examples["conversations"]:
        for conv in convs:
            if isinstance(conv, dict):
                role = conv.get("from", "")
                content = conv.get("value", "")
                if role == "human":
                    texts.append(f"Domanda: {content}")
                elif role == "gpt":
                    texts.append(f"Risposta: {content}")

    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def main():
    print("=" * 60)
    print("🚀 FINE-TUNING SmolLM-135M - OTTIMIZZATO GPU")
    print("=" * 60)

    # Pulisci memoria
    torch.cuda.empty_cache()
    gc.collect()

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer
    print("\n📦 Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Modello con quantizzazione 4-bit
    print("📦 Caricamento modello 4-bit...")
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
    print("🔧 Configurazione LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none"
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset streaming
    tiny_ds, alpaca_ds = load_and_merge_datasets()

    # Preprocessing streaming
    print("\n📝 Configurazione preprocessing streaming...")

    tiny_processed = tiny_ds.map(
        lambda x: preprocess_tinystories(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=["text"],
    )

    alpaca_processed = alpaca_ds.map(
        lambda x: preprocess_alpaca(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=["conversations"],
    )

    # Interleave datasets
    combined = interleave_datasets([tiny_processed, alpaca_processed])

    # Split in train/val (prendiamo primi N per train, ultimi per val)
    print("📊 Preparazione split train/val...")

    # Converti a dataset non-streaming per split
    # Prendi un batch iniziale per determinare dimensione
    all_data = list(combined.take(MAX_SAMPLES_TINY + MAX_SAMPLES_ALPACA))

    from datasets import Dataset

    full_dataset = Dataset.from_list(all_data)

    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"   Train: {len(train_dataset):,} campioni")
    print(f"   Eval: {len(eval_dataset):,} campioni")

    # Data collator con padding dinamico
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,  # Ottimizzazione tensor cores
    )

    # Training arguments OTTIMIZZATI
    print("\n⚙️ Configurazione training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # Batch
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        per_device_eval_batch_size=BATCH_SIZE,
        # Precisione
        fp16=True,
        bf16=False,
        # Ottimizzazioni memoria
        gradient_checkpointing=True,  # Riabilitato con batch più grande
        optim="adamw_8bit",  # 8-bit optimizer
        # DataLoader - NUM_WORKERS > 0 per parallelismo
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        # Training
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        # Eval
        eval_strategy="steps",
        eval_steps=200,
        # Logging
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        # Performance
        torch_compile=False,
        report_to="none",
        # Misc
        remove_unused_columns=False,
        load_best_model_at_end=False,
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
    print("=" * 60)
    print(
        f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )
    print(f"Max seq length: {MAX_SEQ_LENGTH}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
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
            out = model.generate(
                **inputs, max_new_tokens=30, temperature=0.7, do_sample=True
            )
        print(f"  {prompt} → {tokenizer.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
