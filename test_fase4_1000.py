#!/usr/bin/env python3
"""
FASE 4: Test con 1000 campioni REALI dal dataset italiano unificato.
Obiettivo: Verificare che il training con dati reali migliori la qualità.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_fase4_1000.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configurazione ottimizzata per MX250 2GB
MODEL_PATH = "./models/SmolLM-135M-Instruct"
DATASET_PATH = "./datasets/italian_unified/train.jsonl"
OUTPUT_DIR = "./test_output_fase4_1000"
MAX_SAMPLES = 1000
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
MAX_GRAD_NORM = 1.0

# LoRA config
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.15
TARGET_MODULES = ["q_proj", "v_proj"]


def load_real_dataset(path: str, max_samples: int) -> list[dict]:
    """Carica campioni reali dal dataset JSONL."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if "instruction" in data and "output" in data:
                    samples.append(data)
            except json.JSONDecodeError:
                continue
    logger.info(f"Caricati {len(samples)} campioni reali da {path}")
    return samples


def format_prompt(sample: dict) -> str:
    """Formatta un campione nel formato chat di SmolLM."""
    instruction = sample.get("instruction", "")
    output = sample.get("output", "")
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


class ItalianDataset(torch.utils.data.Dataset):
    """Dataset per training QLoRA."""

    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt = format_prompt(self.samples[idx])
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
        }


def main():
    logger.info("=" * 60)
    logger.info("FASE 4: Test con 1000 campioni REALI italiani")
    logger.info("=" * 60)

    # Verifica GPU
    if not torch.cuda.is_available():
        logger.error("CUDA non disponibile!")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(
        f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # Pulisci memoria
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    # Carica tokenizer
    logger.info(f"Caricamento tokenizer da {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Vocabolario: {len(tokenizer)} token")

    # Carica dataset reale
    logger.info(f"Caricamento dataset da {DATASET_PATH}")
    samples = load_real_dataset(DATASET_PATH, MAX_SAMPLES)
    if len(samples) < MAX_SAMPLES:
        logger.warning(
            f"Solo {len(samples)} campioni disponibili (richiesti {MAX_SAMPLES})"
        )

    # Mostra esempio
    logger.info("Esempio di campione reale:")
    logger.info(f"  Instruction: {samples[0]['instruction'][:80]}...")
    logger.info(f"  Output: {samples[0]['output'][:80]}...")

    # Crea dataset
    dataset = ItalianDataset(samples, tokenizer, MAX_SEQ_LENGTH)
    logger.info(f"Dataset creato: {len(dataset)} campioni")

    # Configurazione quantizzazione 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Carica modello
    logger.info(f"Caricamento modello da {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info(f"Modello caricato: {model.__class__.__name__}")

    # Prepara per k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configura LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # VRAM prima del training
    vram_before = torch.cuda.memory_allocated() / 1e6
    logger.info(f"VRAM allocata prima del training: {vram_before:.1f} MB")

    # Argomenti di training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Training
    logger.info("Inizio training...")
    import time

    start_time = time.time()

    trainer.train()

    train_time = time.time() - start_time
    logger.info(f"Training completato in {train_time:.1f} secondi")

    # VRAM dopo il training
    vram_after = torch.cuda.memory_allocated() / 1e6
    logger.info(f"VRAM allocata dopo il training: {vram_after:.1f} MB")
    logger.info(f"Picco VRAM: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # Salva adapter
    logger.info(f"Salvataggio adapter in {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Test generazione
    logger.info("=" * 60)
    logger.info("TEST GENERAZIONE POST-TRAINING")
    logger.info("=" * 60)

    model.eval()
    test_prompts = [
        "Continua la storia:",
        "C'era una volta",
        "Un giorno, un bambino di nome Marco",
    ]

    for prompt in test_prompts:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generato: {generated}")
        logger.info("-" * 40)

    # Riepilogo
    logger.info("=" * 60)
    logger.info("RIEPILOGO FASE 4")
    logger.info("=" * 60)
    logger.info(f"Campioni reali: {len(samples)}")
    logger.info(f"Epoche: {NUM_EPOCHS}")
    logger.info(f"Tempo training: {train_time:.1f}s ({train_time / 60:.1f} min)")
    logger.info(f"VRAM usata: {vram_after:.1f} MB")
    logger.info(f"Output salvato in: {OUTPUT_DIR}")

    # Stima per training completo
    samples_per_second = len(samples) * NUM_EPOCHS / train_time
    full_samples = 55000
    estimated_time = full_samples * NUM_EPOCHS / samples_per_second
    logger.info(f"\nStima per training completo (55k campioni, 3 epoche):")
    logger.info(f"  Tempo stimato: {estimated_time / 3600:.1f} ore")

    logger.info("\nFASE 4 COMPLETATA!")


if __name__ == "__main__":
    main()
