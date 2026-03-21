#!/usr/bin/env python3
"""
QLoRA Training OTTIMIZZATO con SFTTrainer
Basato su best practices ufficiali HuggingFace PEFT/TRL

Miglioramenti rispetto a test_fase4_1000.py:
1. SFTTrainer invece di Trainer (ottimizzato per SFT)
2. Learning rate 2e-4 (40x più alto - best practice QLoRA)
3. LoRA r=16, alpha=32, dropout=0.05 (best practice PEFT)
4. Gestione automatica dataset (no custom class)
5. Formattazione automatica prompt
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_qlora_optimized.log"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURAZIONE OTTIMIZZATA (Best Practices)
# ============================================================

# Paths
MODEL_PATH = "./models/SmolLM-135M-Instruct"
DATASET_PATH = "./datasets/italian_unified/train.jsonl"
OUTPUT_DIR = "./output_qlora_optimized"
TENSORBOARD_LOG_DIR = "./logs_qlora_optimized"  # TensorBoard logs

# Dataset
MAX_SAMPLES = 55000  # Training completo
MAX_SEQ_LENGTH = 384  # Ottimizzato per contesti più lunghi

# LoRA Config (PEFT Best Practices)
LORA_R = 16  # Bilanciato per qualità/efficienza
LORA_ALPHA = 32  # 2x rank (best practice)
LORA_DROPOUT = 0.05  # Rate moderato per prevenire overfitting
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]  # Full attention per migliore apprendimento

# Training Config (TRL Best Practices)
LEARNING_RATE = 3e-4  # Ottimizzato per modello piccolo (135M parametri)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.0  # Disabilitato per compatibilità con fp16 + quantizzazione
LOGGING_STEPS = 10
SAVE_STEPS = 500


def load_and_format_dataset(path: str, max_samples: int) -> Dataset:
    """Carica dataset da JSONL, formatta e converte in HuggingFace Dataset."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if "instruction" in data and "output" in data:
                    # Pre-formatta il testo nel formato chat di SmolLM
                    text = f"<|im_start|>user\n{data['instruction']}<|im_end|>\n<|im_start|>assistant\n{data['output']}<|im_end|>"
                    samples.append({"text": text})
            except json.JSONDecodeError:
                continue

    logger.info(f"Caricati {len(samples)} campioni da {path}")
    return Dataset.from_list(samples)


def main():
    logger.info("=" * 70)
    logger.info("QLoRA Training OTTIMIZZATO con SFTTrainer")
    logger.info("Basato su Best Practices HuggingFace PEFT/TRL")
    logger.info("=" * 70)

    # Verifica GPU
    if not torch.cuda.is_available():
        logger.error("CUDA non disponibile!")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM totale: {vram_total:.1f} GB")

    # Pulisci memoria
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    # ============================================================
    # 1. CARICA TOKENIZER
    # ============================================================
    logger.info("\n[1/6] Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"   Vocabolario: {len(tokenizer)} token")
    logger.info(f"   Pad token: {tokenizer.pad_token}")

    # ============================================================
    # 2. CARICA DATASET (già formattato con campo "text")
    # ============================================================
    logger.info("\n[2/6] Caricamento dataset...")
    dataset = load_and_format_dataset(DATASET_PATH, MAX_SAMPLES)

    # Mostra esempio
    sample = dataset[0]
    logger.info(f"   Esempio testo: {sample['text'][:80]}...")

    # ============================================================
    # 3. CONFIGURA QUANTIZZAZIONE 4-bit
    # ============================================================
    logger.info("\n[3/6] Configurazione quantizzazione 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info("   Quantizzazione: NF4 con double quantization")

    # ============================================================
    # 4. CARICA MODELLO
    # ============================================================
    logger.info("\n[4/6] Caricamento modello...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info(f"   Modello: {model.__class__.__name__}")

    # VRAM dopo caricamento
    vram_model = torch.cuda.memory_allocated() / 1e6
    logger.info(f"   VRAM allocata: {vram_model:.1f} MB")

    # ============================================================
    # 5. CONFIGURA LoRA
    # ============================================================
    logger.info("\n[5/6] Configurazione LoRA...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(f"   Rank (r): {LORA_R}")
    logger.info(f"   Alpha: {LORA_ALPHA}")
    logger.info(f"   Dropout: {LORA_DROPOUT}")
    logger.info(f"   Target modules: {TARGET_MODULES}")

    # ============================================================
    # 6. CONFIGURA TRAINING
    # ============================================================
    logger.info("\n[6/6] Configurazione training...")
    logger.info(f"   Learning rate: {LEARNING_RATE} (40x più alto!)")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"   Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"   Epochs: {NUM_EPOCHS}")
    logger.info(f"   Max seq length: {MAX_SEQ_LENGTH}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=False,  # Disabilitato per compatibilità con quantizzazione 4-bit
        bf16=False,  # MX250 non supporta bf16
        optim="adamw_8bit",
        gradient_checkpointing=True,
        report_to="tensorboard",  # Abilitato TensorBoard
        logging_dir=TENSORBOARD_LOG_DIR,
        max_length=MAX_SEQ_LENGTH,
        packing=False,  # Non usare packing per dataset eterogeneo
    )

    # ============================================================
    # 7. CREA TRAINER
    # ============================================================
    logger.info("\nCreazione SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Stampa parametri trainabili
    trainer.model.print_trainable_parameters()

    # VRAM prima del training
    vram_before = torch.cuda.memory_allocated() / 1e6
    logger.info(f"\nVRAM allocata prima del training: {vram_before:.1f} MB")

    # ============================================================
    # 8. TRAINING
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("INIZIO TRAINING")
    logger.info("=" * 70)

    import time

    start_time = time.time()

    trainer.train()

    train_time = time.time() - start_time

    # ============================================================
    # 9. SALVATAGGIO
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("SALVATAGGIO MODELLO")
    logger.info("=" * 70)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"   Modello salvato in: {OUTPUT_DIR}")

    # ============================================================
    # 10. METRICHE FINALI
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("METRICHE FINALI")
    logger.info("=" * 70)

    vram_after = torch.cuda.memory_allocated() / 1e6
    vram_peak = torch.cuda.max_memory_allocated() / 1e6

    logger.info(f"   Campioni: {MAX_SAMPLES}")
    logger.info(f"   Epochs: {NUM_EPOCHS}")
    logger.info(f"   Tempo training: {train_time:.1f}s ({train_time / 60:.1f} min)")
    logger.info(f"   VRAM usata: {vram_after:.1f} MB")
    logger.info(f"   VRAM peak: {vram_peak:.1f} MB")

    # Stima per training completo
    if MAX_SAMPLES < 55000:
        samples_per_second = MAX_SAMPLES * NUM_EPOCHS / train_time
        full_samples = 55000
        estimated_time = full_samples * NUM_EPOCHS / samples_per_second
        logger.info(f"\n   Stima per training completo (55k campioni):")
        logger.info(f"   Tempo stimato: {estimated_time / 3600:.1f} ore")

    # ============================================================
    # 11. TEST GENERAZIONE
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("TEST GENERAZIONE POST-TRAINING")
    logger.info("=" * 70)

    # Ricarica modello con adapter
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.eval()

    test_prompts = [
        "Continua la storia:",
        "C'era una volta",
        "Un giorno, un bambino di nome Marco",
        "L'Italia è un paese",
    ]

    for prompt in test_prompts:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Estrai solo la risposta dell'assistant
        if "<|im_start|>assistant" in generated:
            response = generated.split("<|im_start|>assistant")[-1].strip()
        else:
            response = generated

        logger.info(f"\n   Prompt: {prompt}")
        logger.info(f"   Risposta: {response[:200]}...")
        logger.info("   " + "-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETATO!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
