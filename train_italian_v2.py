#!/usr/bin/env python3
"""
Training QLoRA SmolLM-135M - Versione Corretta v2
Fix implementati dall'analisi critica del training precedente

Autore: QLoRA Expert Agent
Data: 20 Marzo 2026
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_v2.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ItalianTrainerV2:
    """Trainer QLoRA ottimizzato per italiano con fix critici."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._check_gpu()
        self._log_config()

    def _check_gpu(self) -> torch.device:
        """Verifica GPU disponibile e memoria."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA non disponibile. Training richiede GPU.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_memory:.1f} GB")

        if gpu_memory < 2.0:
            logger.warning("⚠️ VRAM < 2GB. Training potrebbe essere instabile.")

        return torch.device("cuda:0")

    def _log_config(self):
        """Log configurazione training."""
        logger.info("=" * 60)
        logger.info("CONFIGURAZIONE TRAINING V2 - FIX APPLICATI")
        logger.info("=" * 60)
        logger.info(f"Learning Rate: {self.args.learning_rate} (ridotto da 3e-5)")
        logger.info(f"Weight Decay: {self.args.weight_decay} (aumentato da 0.01)")
        logger.info(f"Warmup Ratio: {self.args.warmup_ratio} (aumentato da 0.1)")
        logger.info(f"Max Grad Norm: {self.args.max_grad_norm} (aggiunto)")
        logger.info(f"LoRA r: {self.args.lora_r} (ridotto da 32)")
        logger.info(f"LoRA alpha: {self.args.lora_alpha} (ridotto da 64)")
        logger.info(f"Max Seq Length: {self.args.max_seq_length} (ridotto da 256)")
        logger.info(
            f"Effective Batch: {self.args.batch_size * self.args.gradient_accumulation_steps}"
        )
        logger.info(f"Early Stopping: patience={self.args.early_stopping_patience}")
        logger.info("=" * 60)

    def load_dataset(self) -> Dataset:
        """Carica e bilancia dataset."""
        logger.info("Caricamento dataset...")

        # Carica dataset unificato
        dataset_path = Path(self.args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

        import json

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Caricati {len(data)} campioni totali")

        # Deduplicazione
        seen = set()
        unique_data = []
        for item in data:
            text = item.get("text", "")
            if text not in seen:
                seen.add(text)
                unique_data.append(item)

        logger.info(f"Deduplicazione: rimossi {len(data) - len(unique_data)} duplicati")

        # Limita campioni se richiesto
        if self.args.max_samples and len(unique_data) > self.args.max_samples:
            unique_data = unique_data[: self.args.max_samples]
            logger.info(f"Limitato a {self.args.max_samples} campioni")

        # Crea Dataset HuggingFace
        dataset = Dataset.from_list(unique_data)

        # Split train/validation
        dataset = dataset.train_test_split(
            test_size=self.args.validation_split, seed=self.args.seed
        )

        logger.info(f"Train: {len(dataset['train'])} campioni")
        logger.info(f"Validation: {len(dataset['test'])} campioni")

        return dataset

    def load_model_and_tokenizer(self) -> tuple:
        """Carica modello con quantizzazione e tokenizer."""
        logger.info("Caricamento tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, trust_remote_code=True, use_fast=True
        )

        # Fix pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Impostato pad_token = eos_token")

        logger.info("Caricamento modello con quantizzazione 4-bit...")

        # Configurazione quantizzazione 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Carica modello base
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Disabilita per gradient checkpointing
        )

        # Log parametri
        total_params = sum(p.numel for p in model.parameters())
        logger.info(f"Parametri totali modello base: {total_params:,}")

        # Configura LoRA con parametri corretti
        logger.info("Configurazione LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules.split(","),
            bias="none",
        )

        # Applica LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Abilita gradient checkpointing
        model.gradient_checkpointing_enable()

        return model, tokenizer

    def preprocess_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        """Preprocessa e tokenizza dataset."""
        logger.info("Pre-tokenizzazione dataset...")

        def tokenize_function(examples):
            # Tokenizza con truncation e padding
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.args.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )

            # Per language modeling, labels = input_ids
            outputs["labels"] = outputs["input_ids"].copy()

            return outputs

        # Tokenizza in batch
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset["train"].column_names,
            desc="Pre-tokenizzazione",
        )

        logger.info("Pre-tokenizzazione completata")

        return tokenized_dataset

    def create_training_arguments(self) -> TrainingArguments:
        """Crea argomenti training con fix applicati."""

        # Crea directory output
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            # Training hyperparameters (FIXED)
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            # Learning rate schedule (FIXED)
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type=self.args.lr_scheduler_type,
            max_grad_norm=self.args.max_grad_norm,  # FIX: gradient clipping
            # Evaluation
            eval_strategy="steps",
            eval_steps=self.args.eval_steps,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            # Optimization
            optim="adamw_8bit",  # FIX: optimizer 8-bit per risparmiare memoria
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # Precision
            fp16=False,
            bf16=False,
            # Checkpoints
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Performance
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # Logging
            logging_dir=str(output_dir / "logs"),
            report_to=["tensorboard"],
            # Reproducibility
            seed=self.args.seed,
            data_seed=self.args.seed,
            # Other
            remove_unused_columns=False,
            label_names=["labels"],
        )

    def train(self):
        """Esegue training completo con tutti i fix."""
        logger.info("=" * 60)
        logger.info("INIZIO TRAINING V2")
        logger.info("=" * 60)

        try:
            # Carica componenti
            dataset = self.load_dataset()
            model, tokenizer = self.load_model_and_tokenizer()
            tokenized_dataset = self.preprocess_dataset(dataset, tokenizer)

            # Crea trainer
            training_args = self.create_training_arguments()

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                ),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.args.early_stopping_patience,
                        early_stopping_threshold=0.001,
                    )
                ],
            )

            # Verifica stato memoria prima del training
            self._check_memory()

            # Avvia training
            logger.info("Avvio training...")
            train_result = trainer.train(
                resume_from_checkpoint=self.args.resume_checkpoint
            )

            # Salva modello finale
            logger.info("Salvataggio modello finale...")
            trainer.save_model()
            trainer.save_state()

            # Log metriche finali
            metrics = train_result.metrics
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETATO")
            logger.info("=" * 60)
            logger.info(f"Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
            logger.info(f"Train Runtime: {metrics.get('train_runtime', 0):.1f} secondi")
            logger.info(
                f"Samples/second: {metrics.get('train_samples_per_second', 0):.2f}"
            )

            # Test rapido
            self._quick_test(model, tokenizer)

            return True

        except Exception as e:
            logger.error(f"Errore durante training: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _check_memory(self):
        """Verifica memoria GPU disponibile."""
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(
            f"Memoria GPU - Allocata: {allocated:.2f}GB, Riservata: {reserved:.2f}GB, Totale: {total:.2f}GB"
        )

        if allocated > 1.8:
            logger.warning("⚠️ Memoria GPU quasi esaurita. Rischio OOM.")

    def _quick_test(self, model, tokenizer):
        """Test rapido post-training."""
        logger.info("Test rapido del modello...")

        test_prompts = [
            "C'era una volta",
            "L'Italia è",
            "Spiegami cos'è",
        ]

        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"  Prompt: '{prompt}'")
                logger.info(f"  Output: '{generated}'")
                logger.info("")


def parse_arguments() -> argparse.Namespace:
    """Parse argomenti command line."""
    parser = argparse.ArgumentParser(
        description="Training QLoRA SmolLM-135M Italiano v2 - Con Fix Critici"
    )

    # Paths
    parser.add_argument(
        "--model_name", type=str, default="./models/SmolLM-135M-Instruct"
    )
    parser.add_argument("--dataset_path", type=str, default="./dataset_unificato.json")
    parser.add_argument("--output_dir", type=str, default="./smollm_italian_v2")

    # Dataset
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limita campioni (None = tutti)"
    )
    parser.add_argument("--validation_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparameters (FIXED VALUES)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5
    )  # FIX: ridotto da 3e-5
    parser.add_argument(
        "--weight_decay", type=float, default=0.05
    )  # FIX: aumentato da 0.01
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.15
    )  # FIX: aumentato da 0.1
    parser.add_argument("--max_grad_norm", type=float, default=1.0)  # FIX: aggiunto
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine_with_restarts")

    # LoRA (FIXED VALUES)
    parser.add_argument("--lora_r", type=int, default=16)  # FIX: ridotto da 32
    parser.add_argument("--lora_alpha", type=int, default=32)  # FIX: ridotto da 64
    parser.add_argument(
        "--lora_dropout", type=float, default=0.15
    )  # FIX: aumentato da 0.1
    parser.add_argument(
        "--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj"
    )

    # Model
    parser.add_argument(
        "--max_seq_length", type=int, default=192
    )  # FIX: ridotto da 256

    # Logging & Checkpoints
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    # Resume
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    """Entry point principale."""
    args = parse_arguments()

    # Log inizio sessione
    logger.info("=" * 60)
    logger.info(f"Session started: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Crea trainer e avvia
    trainer = ItalianTrainerV2(args)
    success = trainer.train()

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    if success:
        logger.info("✅ Training completato con successo!")
        sys.exit(0)
    else:
        logger.error("❌ Training fallito!")
        sys.exit(1)


if __name__ == "__main__":
    main()
