#!/usr/bin/env python3
"""
Fine-tuning SmolLM-135M - VERSIONE MIGLIORATA
Ottimizzata per qualita dell'apprendimento su GPU MX250 2GB
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Ignora SIGPIPE per evitare crash
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def setup_output_handling():
    """Configura la gestione output per evitare crash."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


# CONFIGURAZIONE
MODEL_PATH = "./models/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm_italian_improved"
LOG_DIR = "./logs_smollm_improved"

# Dataset locale unificato
LOCAL_DATASET_PATH = "./datasets/italian_unified/train.jsonl"

# Se LOCAL_DATASET_PATH esiste, usa quello; altrimenti usa HF (retrocompatibilità)
USE_LOCAL_DATASET = os.path.exists(LOCAL_DATASET_PATH)

MAX_SAMPLES_TINYSTORIES = 30000
MAX_SAMPLES_ALPACA = 15000
MAX_SAMPLES_DOLLY = 10000

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
MAX_SEQ_LENGTH = 256

LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
# Target modules per LoRA - include anche gate_proj per maggiore capacità
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]

LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Variabile globale per gestione segnali
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Gestisce i segnali di terminazione."""
    global _shutdown_requested
    logger.warning(f"Seinale {signum} ricevuto, shutdown in corso...")
    _shutdown_requested = True


# Registra handler per segnali di terminazione
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@dataclass
class TrainingConfig:
    """Configurazione training."""

    model_path: str
    output_dir: str
    log_dir: str
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    learning_rate: float
    num_epochs: int
    warmup_ratio: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    resume_from_checkpoint: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200


def format_prompt(sample: Dict) -> str:
    """Formatta un campione in prompt per il modello."""
    if "instruction" in sample and "output" in sample:
        return f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"
    elif "text" in sample:
        return sample["text"]
    else:
        return str(sample)


def load_and_format_datasets(
    max_tinystories: int,
    max_alpaca: int,
    max_dolly: int,
) -> List[Dict]:
    """Carica e formatta i dataset italiani."""

    # 优先使用本地统一数据集
    if USE_LOCAL_DATASET:
        logger.info(f"Caricamento dataset locale unificato...")
        try:
            all_samples = []
            with open(LOCAL_DATASET_PATH, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Caricamento dataset"):
                    sample = json.loads(line)
                    all_samples.append(sample)

            # 应用最大样本限制
            total_max = max_tinystories + max_alpaca + max_dolly
            if len(all_samples) > total_max:
                all_samples = all_samples[:total_max]

            logger.info(f"   Caricati {len(all_samples)} campioni da locale")
            logger.info(f"Totale campioni: {len(all_samples)}")
            return all_samples
        except Exception as e:
            logger.error(f"   Errore caricamento dataset locale: {e}")
            logger.info("   Ritorno ai dataset HuggingFace...")

    # 回退到 HuggingFace（向后兼容）
    logger.warning("使用 HuggingFace 数据集（已弃用）- 请运行 prepare_datasets.py")

    all_samples = []

    logger.info("Caricamento TinyStories-Italian...")
    try:
        ds_tinystories = load_dataset("markod0925/TinyStories-Italian", split="train")
        tinystories_samples = [
            {"instruction": "Continua la storia:", "output": ds_tinystories[i]["text"]}
            for i in tqdm(
                range(min(max_tinystories, len(ds_tinystories))), desc="TinyStories"
            )
        ]
        all_samples.extend(tinystories_samples)
        logger.info(f"   Caricati {len(tinystories_samples)} campioni TinyStories")
    except Exception as e:
        logger.error(f"   Errore caricamento TinyStories: {e}")

    logger.info("Caricamento alpaca-gpt4-italian...")
    try:
        ds_alpaca = load_dataset(
            "FreedomIntelligence/alpaca-gpt4-italian", split="train"
        )
        alpaca_samples = [
            {
                "instruction": ds_alpaca[i]["instruction"],
                "output": ds_alpaca[i]["output"],
            }
            for i in tqdm(range(min(max_alpaca, len(ds_alpaca))), desc="Alpaca")
        ]
        all_samples.extend(alpaca_samples)
        logger.info(f"   Caricati {len(alpaca_samples)} campioni Alpaca")
    except Exception as e:
        logger.error(f"   Errore caricamento Alpaca: {e}")

    logger.info("Caricamento Dolly italiano...")
    try:
        ds_dolly = load_dataset("gsarti/clean_dolly_italian", split="train")
        dolly_samples = [
            {
                "instruction": ds_dolly[i]["instruction"],
                "output": ds_dolly[i]["response"],
            }
            for i in tqdm(range(min(max_dolly, len(ds_dolly))), desc="Dolly")
        ]
        all_samples.extend(dolly_samples)
        logger.info(f"   Caricati {len(dolly_samples)} campioni Dolly")
    except Exception as e:
        logger.error(f"   Errore caricamento Dolly: {e}")

    logger.info(f"Totale campioni: {len(all_samples)}")
    return all_samples


class ItalianDataset(torch.utils.data.Dataset):
    """Dataset PyTorch per dati italiani."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: Any,
        max_length: int,
        pre_tokenize: bool = True,
    ):
        """Inizializza il dataset."""
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        if pre_tokenize:
            logger.info(f"Pre-tokenizzazione di {len(samples)} campioni...")
            self.tokenized_samples = []
            for sample in tqdm(samples, desc="Pre-tokenizzazione"):
                prompt = format_prompt(sample)
                encoded = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                self.tokenized_samples.append(
                    {
                        "input_ids": encoded["input_ids"],
                        "attention_mask": encoded["attention_mask"],
                    }
                )
            logger.info("Pre-tokenizzazione completata")
        else:
            self.tokenized_samples = None

    def __len__(self) -> int:
        """Restituisce il numero di campioni."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Restituisce un campione tokenizzato."""
        if self.tokenized_samples:
            sample = self.tokenized_samples[idx]
            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()
        else:
            sample = self.samples[idx]
            prompt = format_prompt(sample)
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def setup_model_and_tokenizer(config: TrainingConfig):
    """Configura modello e tokenizer con quantizzazione 4-bit."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Impostato pad_token = eos_token")

    logger.info("Caricamento modello con quantizzazione 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.config.use_cache = False
    logger.info(f"Parametri totali: {model.num_parameters():,}")

    logger.info("Configurazione LoRA...")
    logger.info(f"   r = {config.lora_r}")
    logger.info(f"   alpha = {config.lora_alpha}")
    logger.info(f"   dropout = {config.lora_dropout}")
    logger.info(f"   target_modules = {config.target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)

    return model, tokenizer


def create_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Crea gli argomenti di training."""
    logger.info(f"Resume checkpoint config: {config.resume_from_checkpoint}")

    return TrainingArguments(
        output_dir=config.output_dir,
        resume_from_checkpoint=config.resume_from_checkpoint,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_dir=config.log_dir,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        log_level="info",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )


def test_model(model: Any, tokenizer: Any) -> None:
    """Test rapido del modello."""
    logger.info("Test del modello...")
    model.eval()

    test_prompts = [
        "C'era una volta",
        "L'Italia è",
        "Spiegami cos'e l'intelligenza artificiale.",
    ]

    for prompt in test_prompts:
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"   Prompt: '{prompt}'")
        logger.info(f"   Output: '{generated}'")
        logger.info("")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Trova l'ultimo checkpoint disponibile."""
    if not os.path.exists(output_dir):
        logger.warning(f"Directory output non esiste: {output_dir}")
        return None

    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(
            os.path.join(output_dir, item)
        ):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    latest = checkpoints[-1]
    logger.info(
        f"Trovato ultimo checkpoint: step {latest[0]} -> {os.path.join(output_dir, latest[1])}"
    )
    return latest[1]


def cleanup_pid_file(output_dir: str) -> None:
    """Rimuove il file PID alla terminazione del training."""
    pid_file = os.path.join(output_dir, ".training_pid")
    try:
        if os.path.exists(pid_file):
            os.remove(pid_file)
            logger.info(f"Rimosso file PID: {pid_file}")
    except OSError as e:
        logger.warning(f"Impossibile rimuovere file PID: {e}")


def check_and_manage_training_process(
    output_dir: str, force_restart: bool = False
) -> Optional[int]:
    """Controlla e gestisce i processi di training esistenti."""
    pid_file = os.path.join(output_dir, ".training_pid")

    if os.path.exists(pid_file):
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            if os.path.exists(f"/proc/{pid}"):
                if force_restart:
                    logger.warning(f"Terminazione training esistente (PID: {pid})...")
                    os.kill(pid, 9)
                    os.remove(pid_file)
                    return None
                else:
                    logger.info(f"Training in corso (PID: {pid})")
                    return pid
        except (ValueError, FileNotFoundError, ProcessLookupError):
            os.remove(pid_file)

    return None


def main() -> None:
    """Funzione principale."""
    setup_output_handling()

    parser = argparse.ArgumentParser(
        description="Fine-tuning SmolLM-135M per italiano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python3 train_italian_improved.py                    # Resume + background automatico
  python3 train_italian_improved.py --no_resume        # Nuovo training da zero
  python3 train_italian_improved.py --resume_from checkpoint-600
  python3 train_italian_improved.py --epochs 2
  python3 train_italian_improved.py --no_background    # Esegui in foreground
        """,
    )
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_background", action="store_true")
    parser.add_argument("--kill", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()

    if args.kill:
        check_and_manage_training_process(OUTPUT_DIR, force_restart=True)
        print("Tutti i training terminati.")
        return

    config = TrainingConfig(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        log_dir=LOG_DIR,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )

    if args.epochs:
        config.num_epochs = args.epochs
        logger.info(f"Numero epoch modificato: {args.epochs}")

    if not args.no_background:
        import subprocess
        import sys

        cmd = [sys.executable, __file__]

        if args.no_resume:
            cmd.append("--no_resume")
        elif args.resume_from:
            cmd.extend(["--resume_from", args.resume_from])
        else:
            checkpoint = find_latest_checkpoint(config.output_dir)
            if checkpoint and not args.no_resume:
                logger.info(f"Trovato checkpoint: {checkpoint}")
                config.resume_from_checkpoint = checkpoint
                cmd.extend(["--resume_from", checkpoint])
            else:
                config.resume_from_checkpoint = None

        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        cmd.append("--no_background")

        logger.info("Avvio training in background...")
        logger.info(f"Comando: {' '.join(cmd)}")

        pid_file = os.path.join(config.output_dir, ".training_pid")

        with open(
            training_log_file := os.path.join(config.output_dir, "training.log"), "w"
        ) as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        with open(pid_file, "w") as f:
            f.write(str(process.pid))

        print(f"\n{'=' * 60}")
        print("TRAINING AVVIATO IN BACKGROUND")
        print(f"{'=' * 60}")
        print(f"PID: {process.pid}")
        print(f"Log: {training_log_file}")
        print(f"Monitor: python3 monitor_training.py")
        print(f"{'=' * 60}\n")

        return

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        logger.error("GPU non disponibile!")
        return

    logger.info("Configurazione:")
    logger.info(
        f"   Effective batch: {config.batch_size * config.gradient_accumulation_steps}"
    )
    logger.info(f"   Max seq length: {config.max_seq_length}")
    logger.info(f"   Learning rate: {config.learning_rate}")
    logger.info(f"   Epochs: {config.num_epochs}")
    logger.info(f"   Warmup ratio: {config.warmup_ratio}")
    logger.info(f"   Weight decay: {config.weight_decay}")

    logger.info("Caricamento dataset...")
    all_samples = load_and_format_datasets(
        MAX_SAMPLES_TINYSTORIES,
        MAX_SAMPLES_ALPACA,
        MAX_SAMPLES_DOLLY,
    )

    if not all_samples:
        logger.error("Nessun campione caricato!")
        return

    model, tokenizer = setup_model_and_tokenizer(config)

    logger.info("Creazione dataset PyTorch...")
    full_dataset = ItalianDataset(all_samples, tokenizer, config.max_seq_length)

    eval_size = min(200, len(full_dataset) // 100)
    train_size = len(full_dataset) - eval_size
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(
        full_dataset, range(train_size, len(full_dataset))
    )

    logger.info(f"Train: {len(train_dataset)} campioni")
    logger.info(f"Eval: {len(eval_dataset)} campioni")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = create_training_arguments(config)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    print("\n" + "=" * 60)
    print("INIZIO TRAINING")
    print("=" * 60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrotto dall'utente")
        logger.info("Salvataggio checkpoint di emergenza...")
        try:
            model.save_pretrained(
                os.path.join(config.output_dir, "checkpoint-emergency")
            )
            tokenizer.save_pretrained(
                os.path.join(config.output_dir, "checkpoint-emergency")
            )
            logger.info("Checkpoint di emergenza salvato")
        except Exception as save_error:
            logger.error(f"Impossibile salvare checkpoint di emergenza: {save_error}")
        finally:
            cleanup_pid_file(config.output_dir)
        return
    except Exception as e:
        logger.error(f"Errore durante il training: {e}")
        logger.info("Salvataggio checkpoint di emergenza...")
        try:
            model.save_pretrained(
                os.path.join(config.output_dir, "checkpoint-emergency")
            )
            tokenizer.save_pretrained(
                os.path.join(config.output_dir, "checkpoint-emergency")
            )
            logger.info("Checkpoint di emergenza salvato")
        except Exception as save_error:
            logger.error(f"Impossibile salvare checkpoint di emergenza: {save_error}")
        finally:
            cleanup_pid_file(config.output_dir)
        raise

    logger.info("Salvataggio modello...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Modello salvato in: {config.output_dir}")

    test_model(model, tokenizer)

    # Cleanup file PID
    cleanup_pid_file(config.output_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETATO")
    print("=" * 60)


if __name__ == "__main__":
    main()
