#!/usr/bin/env python3
"""
QLoRA Fine-tuning Qwen2.5-0.5B-Instruct per MEDICO-ITALIANO
============================================================
Hardware: MX250 2GB VRAM + 14GB RAM
Dataset: Unified Medical Italian Dataset (15,000 campioni)

Questo script training un modello QLoRA su un dataset unificato
composto da:
- 10,000 campioni linguistici italiani (già formattati)
- 5,000 campioni medici tradotti dall'inglese

Usage:
    python3 train_qwen25_medical_italian.py
"""

import os
import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

OUTPUT_DIR = (
    f"./output_qwen25_medical_italian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "./datasets/unified_medical_italian_dataset.json"

# LoRA Configuration (ottimizzata per MX250 2GB - Best Practice)
LORA_R = 16
LORA_ALPHA = 32  # 2 * r (standard)
LORA_DROPOUT = 0  # 0 meglio per training breve
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Tutti e 4 per qualità
BIAS = "none"

# Training Configuration
MAX_SEQ_LENGTH = 128  # RIDOTTO da 256 per velocità
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # RIDOTTO da 16 per velocità
LEARNING_RATE = 3e-4  # Leggermente più alto per convergenza veloce
NUM_EPOCHS = 1  # UN SOLO EPOCH per velocità
WARMUP_STEPS = 50
MAX_GRAD_NORM = 1.0

# Quantization
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.float16
BNB_4BIT_USE_DOUBLE_QUANT = True

# Dataset
MAX_TRAIN_SAMPLES = 15000

# Logging
LOGGING_STEPS = 50
SAVE_STEPS = 100  # Checkpoint ogni 100 steps

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"train_qwen25_medical_italian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CARICAMENTO DATASET
# ============================================================================


def load_unified_dataset() -> Dataset:
    """
    Carica il dataset unificato medico-italiano.

    Returns:
        Dataset pronto per il training
    """
    logger.info(f"Caricamento dataset da {DATASET_PATH}...")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset non trovato: {DATASET_PATH}\n"
            f"Esegui prima: python3 create_unified_medical_italian_dataset.py"
        )

    samples = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_TRAIN_SAMPLES:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    logger.info(f"Caricati {len(samples)} campioni")

    # Conta categorie
    medical_count = sum(1 for s in samples if s.get("category") == "medical")
    linguistic_count = len(samples) - medical_count
    logger.info(f"  - Medici: {medical_count}")
    logger.info(f"  - Linguistici: {linguistic_count}")

    return Dataset.from_list(samples)


# ============================================================================
# TOKENIZZAZIONE
# ============================================================================


def tokenize_function(examples, tokenizer):
    """Tokenizza i testi."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


# ============================================================================
# MAIN TRAINING
# ============================================================================


def main():
    """Funzione principale di training."""

    logger.info("=" * 60)
    logger.info("QLoRA Fine-tuning Qwen2.5-0.5B-Instruct - MEDICO-ITALIANO")
    logger.info("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile!")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_memory:.1f} GB")

    # =========================================================================
    # 1. CARICAMENTO TOKENIZER
    # =========================================================================
    logger.info("\nCaricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    vocab_size = len(tokenizer)
    logger.info(f"Vocabolario: {vocab_size} token")

    # =========================================================================
    # 2. CARICAMENTO DATASET
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CARICAMENTO DATASET")
    logger.info("=" * 60)

    dataset = load_unified_dataset()

    logger.info("\nTokenizzazione dataset...")
    dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        remove_columns=[c for c in dataset.column_names if c != "text"],
        batched=True,
        desc="Tokenizzazione",
    )

    # Split train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    logger.info(f"\nDataset tokenizzato:")
    logger.info(f"  - Train: {len(train_dataset)} campioni")
    logger.info(f"  - Eval: {len(eval_dataset)} campioni")

    # =========================================================================
    # 3. CONFIGURAZIONE QUANTIZZAZIONE
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURAZIONE")
    logger.info("=" * 60)

    logger.info("\nConfigurazione quantizzazione 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # =========================================================================
    # 4. CARICAMENTO MODELLO
    # =========================================================================
    logger.info("\nCaricamento modello...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    vram_used = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"VRAM dopo caricamento modello: {vram_used:.2f} GB")

    # =========================================================================
    # 5. PREPARAZIONE PER K-BIT TRAINING
    # =========================================================================
    logger.info("\nPreparazione per k-bit training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    vram_after_prep = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"VRAM dopo preparazione: {vram_after_prep:.2f} GB")

    # =========================================================================
    # 6. CONFIGURAZIONE LoRA
    # =========================================================================
    logger.info("\nConfigurazione LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=BIAS,
        task_type="CAUSAL_LM",
    )
    logger.info(f"  - Rank (r): {LORA_R}")
    logger.info(f"  - Alpha: {LORA_ALPHA}")
    logger.info(f"  - Target modules: {LORA_TARGET_MODULES}")

    model = get_peft_model(model, lora_config)

    # Mostra parametri trainabili
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params
    logger.info(f"\nParametri trainabili:")
    logger.info(f"  - Trainable: {trainable_params:,}")
    logger.info(f"  - Totali: {all_params:,}")
    logger.info(f"  - Percentuale: {trainable_pct:.4f}%")

    vram_after_lora = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"VRAM dopo LoRA: {vram_after_lora:.2f} GB")

    # =========================================================================
    # 7. TRAINING ARGUMENTS
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=True,
        logging_steps=LOGGING_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
    )

    logger.info(f"\nTraining Arguments:")
    logger.info(f"  - Output dir: {OUTPUT_DIR}")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  - Learning rate: {LEARNING_RATE}")
    logger.info(f"  - Epochs: {NUM_EPOCHS}")
    logger.info(f"  - Seq length: {MAX_SEQ_LENGTH}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # =========================================================================
    # 8. TRAINER
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("INIZIO TRAINING")
    logger.info("=" * 60)

    vram_initial = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"VRAM iniziale training: {vram_initial:.2f} GB")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Training
    try:
        logger.info("\nTraining in corso...")
        result = trainer.train()

        # Salva il modello
        logger.info("\nSalvataggio adapter...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Statistiche finali
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETATO!")
        logger.info("=" * 60)
        logger.info(f"  - Training time: {result.training_time:.2f} secondi")
        logger.info(f"  - Final loss: {result.training_loss:.4f}")
        logger.info(f"  - Output dir: {OUTPUT_DIR}")

        vram_final = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"  - VRAM finale: {vram_final:.2f} GB")

    except Exception as e:
        logger.error(f"Errore durante training: {e}")
        raise

    finally:
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
