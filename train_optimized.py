#!/usr/bin/env python3
"""
Script di training ottimizzato per GPU MX250 (2GB VRAM)
Include tutte le ottimizzazioni per massimizzare l'utilizzo GPU.
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
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============ CONFIGURAZIONE OTTIMIZZATA MX250 2GB ============
MODEL_NAME = "./models/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm-optimized-output"
DATASET_NAME = "databricks/databricks-dolly-15k"

# Batch e memoria
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 16
MAX_SEQ_LENGTH = 256

# LoRA ottimizzato per 2GB
LORA_R = 4
LORA_ALPHA = 8  # = 2 * r
LORA_DROPOUT = 0.05

# Training
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_STEPS = 50


# ============ CALLBACK PER PULIZIA MEMORIA ============
class MemoryCleanupCallback(TrainerCallback):
    """Pulisce la memoria GPU ogni N step."""

    def __init__(self, cleanup_every=50):
        self.cleanup_every = cleanup_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.cleanup_every == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control


# ============ CONFIGURAZIONE BITSANDBYTES OTTIMIZZATA ============
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # IMPORTANTE: float16, non float32!
        bnb_4bit_use_double_quant=True,  # Risparmio extra memoria
    )


# ============ CONFIGURAZIONE LORA OTTIMIZZATA ============
def get_lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # Solo q,v per risparmiare memoria
    )


# ============ CONFIGURAZIONE TRAINING OTTIMIZZATA ============
def get_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        # Batch configuration
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        per_device_eval_batch_size=1,
        # DataLoader - OTTIMIZZAZIONI CHIAVE!
        # IMPORTANTE: num_workers=0 per evitare problemi CUDA!
        # Con CUDA, i worker non condividono il contesto GPU
        dataloader_num_workers=0,  # 0 = main process (CUDA-safe)
        dataloader_pin_memory=False,  # Non necessario con num_workers=0
        # dataloader_prefetch_factor richiede num_workers > 0
        # Precision
        fp16=True,
        bf16=False,  # MX250 non supporta BF16
        # Memory optimization
        # gradient_checkpointing disabilitato per modelli 4-bit (causa problemi)
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",  # Ottimizzatore memory-efficient
        # Training
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        # Eval
        eval_strategy="steps",
        eval_steps=100,
        # Logging
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        # Performance
        torch_compile=False,  # MX250 troppo vecchia
        report_to="none",
        # Misc
        load_best_model_at_end=False,
        metric_for_best_model="loss",
    )


# ============ FUNZIONE PRINCIPALE ============
def main():
    print("=" * 60)
    print("TRAINING OTTIMIZZATO PER MX250 (2GB VRAM)")
    print("=" * 60)

    # Pulisci memoria prima di iniziare
    torch.cuda.empty_cache()
    gc.collect()

    # Verifica GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f} GB")
    else:
        print("ERRORE: GPU non disponibile!")
        return

    print(f"\nConfigurazione:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Target modules: q_proj, v_proj")
    print(f"  DataLoader workers: 2")
    print(f"  Pin memory: True")

    # Carica tokenizer
    print("\nCaricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Carica modello con quantizzazione
    print("Caricamento modello con quantizzazione 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map={"": 0},  # Forza tutto su GPU 0 (non "auto"!)
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )

    # Configura LoRA
    print("Configurazione LoRA...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Stampa parametri trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Parametri trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)"
    )

    # Carica dataset
    print("\nCaricamento dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Preprocessing
    def preprocess_function(examples):
        # Combina instruction e response
        texts = []
        for instr, resp in zip(examples["instruction"], examples["response"]):
            text = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_attention_mask=True,
        )

        # Per causal LM, le labels sono uguali agli input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Split train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Eval samples: {len(eval_dataset):,}")

    # Training arguments
    training_args = get_training_args()

    # Trainer con callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MemoryCleanupCallback(cleanup_every=50)],
    )

    # Avvia training
    print("\n" + "=" * 60)
    print("AVVIO TRAINING")
    print("=" * 60 + "\n")

    trainer.train()

    # Salva modello finale
    print("\nSalvataggio modello...")
    trainer.save_model(OUTPUT_DIR)
    print(f"Modello salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
