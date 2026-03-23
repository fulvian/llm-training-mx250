#!/usr/bin/env python3
"""
Script per creare un dataset medico-italiano unificato.
- Download e traduzione con CHECKPOINT e RESUME
- Salva stato ogni 100 campioni tradotti
- Batch size ridotto per evitare rate limiting
- Resume automatico in caso di interruzione

Usage:
    python3 create_unified_medical_italian_dataset.py
    python3 create_unified_medical_italian_dataset.py --resume
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional

from deep_translator import GoogleTranslator
from datasets import load_dataset, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("create_dataset.log"),
    ],
)
logger = logging.getLogger(__name__)

# Configurazione
ITALIAN_DATA_DIR = "./datasets/italian_clean"
MEDICAL_DATASET = "medalpaca/medical_meadow_medqa"
OUTPUT_FILE = "./datasets/unified_medical_italian_dataset.json"
CHECKPOINT_FILE = "./datasets/translation_checkpoint.json"
TARGET_SIZE = 15000
MAX_MEDICAL_SAMPLES = 5000
CHECKPOINT_INTERVAL = 100  # Salvataggio ogni 100 campioni
BATCH_SIZE = 10  # Batch piccoli per evitare rate limiting
TRANSLATION_TIMEOUT = 30  # Secondi per batch


def translate_single(text: str, src: str = "en", dest: str = "it") -> str:
    """Traduce un singolo testo con timeout."""
    if not text or not text.strip():
        return text

    try:
        translator = GoogleTranslator(source=src, target=dest)
        return translator.translate(text, timeout=TRANSLATION_TIMEOUT)
    except Exception as e:
        logger.warning(f"Errore traduzione: {e}, ritento...")
        time.sleep(2)
        try:
            translator = GoogleTranslator(source=src, target=dest)
            return translator.translate(text, timeout=TRANSLATION_TIMEOUT)
        except Exception:
            return text


def translate_batch_safe(
    texts: List[str], src: str = "en", dest: str = "it"
) -> List[str]:
    """Traduce testi uno alla volta con gestione errori."""
    translations = []
    total = len(texts)

    for i, text in enumerate(texts):
        if i % 10 == 0:
            logger.info(f"  Batch {i}-{min(i + 10, total)}/{total}")

        translated = translate_single(text, src, dest)
        translations.append(translated)
        time.sleep(0.5)  # Pausa tra richieste per evitare rate limit

    return translations


def format_medical_sample(instruction: str, output: str) -> str:
    """Formatta un sample medico."""
    text = (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return text


def load_checkpoint() -> Optional[Dict]:
    """Carica checkpoint se esiste."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Errore caricamento checkpoint: {e}")
    return None


def save_checkpoint(
    translated_instructions: List[str],
    translated_outputs: List[str],
    medical_samples: List[Dict],
    idx: int,
):
    """Salva checkpoint."""
    checkpoint = {
        "translated_instructions": translated_instructions,
        "translated_outputs": translated_outputs,
        "medical_samples": medical_samples,
        "last_processed_idx": idx,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        logger.info(f"✓ Checkpoint salvato (idx: {idx})")
    except Exception as e:
        logger.error(f"Errore salvataggio checkpoint: {e}")


def load_local_italian_dataset() -> List[Dict]:
    """Carica il dataset linguistico italiano locale."""
    logger.info(f"Caricamento dataset italiano da {ITALIAN_DATA_DIR}...")

    samples = []
    train_file = os.path.join(ITALIAN_DATA_DIR, "train.jsonl")

    if not os.path.exists(train_file):
        logger.error(f"File non trovato: {train_file}")
        return samples

    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= TARGET_SIZE - MAX_MEDICAL_SAMPLES:
                break
            try:
                data = json.loads(line.strip())
                if "text" in data:
                    samples.append({"text": data["text"], "category": "linguistic"})
            except json.JSONDecodeError:
                continue

    logger.info(f"Caricati {len(samples)} campioni linguistici italiani")
    return samples


def load_medical_samples() -> List[Dict]:
    """Carica il dataset medico inglese."""
    logger.info(f"Caricamento dataset medico: {MEDICAL_DATASET}...")

    ds = load_dataset(MEDICAL_DATASET, split="train", streaming=True)

    samples = []
    for i, sample in enumerate(ds):
        if i >= MAX_MEDICAL_SAMPLES:
            break

        if "instruction" in sample:
            instruction = sample["instruction"]
            output = sample.get("output", sample.get("response", ""))
        elif "question" in sample:
            instruction = sample["question"]
            output = sample.get("answer", sample.get("response", ""))
        else:
            continue

        samples.append(
            {
                "instruction": instruction,
                "output": output,
            }
        )

    logger.info(f"Caricati {len(samples)} campioni medici")
    return samples


def translate_instructions(instructions: List[str], start_idx: int = 0) -> List[str]:
    """Traduce le istruzioni."""
    logger.info(f"Traduzione instructions [{start_idx}/{len(instructions)}]...")

    translated = []
    for i in range(start_idx):
        translated.append(None)  # Placeholder for already translated

    for i in range(start_idx, len(instructions)):
        if i % CHECKPOINT_INTERVAL == 0 and i > start_idx:
            save_checkpoint(translated, [], instructions, i)

        if i % 10 == 0:
            logger.info(f"  Istruzioni: {i}/{len(instructions)}")

        translated.append(translate_single(instructions[i]))
        time.sleep(0.3)

    return translated


def translate_outputs(outputs: List[str], start_idx: int = 0) -> List[str]:
    """Traduce le risposte."""
    logger.info(f"Traduzione outputs [{start_idx}/{len(outputs)}]...")

    translated = []
    for i in range(start_idx):
        translated.append(None)

    for i in range(start_idx, len(outputs)):
        if i % CHECKPOINT_INTERVAL == 0 and i > start_idx:
            save_checkpoint([], translated, outputs, i)

        if i % 10 == 0:
            logger.info(f"  Outputs: {i}/{len(outputs)}")

        translated.append(translate_single(outputs[i]))
        time.sleep(0.3)

    return translated


def translate_with_checkpoint(
    medical_samples: List[Dict], resume: bool = False
) -> List[Dict]:
    """Traduce i campioni medici con checkpoint e resume."""

    checkpoint = load_checkpoint() if resume else None

    if checkpoint:
        logger.info("=" * 60)
        logger.info("RESUME DA CHECKPOINT")
        logger.info(
            f"Ultimo idx processato: {checkpoint.get('last_processed_idx', -1)}"
        )
        logger.info("=" * 60)

        translated_instructions = checkpoint.get("translated_instructions", [])
        translated_outputs = checkpoint.get("translated_outputs", [])
        start_idx = len(translated_instructions)
    else:
        logger.info("=" * 60)
        logger.info("INIZIO TRADUZIONE DA ZERO")
        logger.info("=" * 60)
        translated_instructions = []
        translated_outputs = []
        start_idx = 0

    # Estrai instructions e outputs originali
    instructions = [s["instruction"] for s in medical_samples]
    outputs = [s["output"] for s in medical_samples]

    # Traduci instructions
    translated_instructions = translate_instructions(instructions, start_idx)

    # Checkpoint dopo instructions
    save_checkpoint(translated_instructions, [], medical_samples, len(instructions))

    # Traduci outputs
    translated_outputs = translate_outputs(outputs, start_idx)

    # Cleanup checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint rimosso (traduzione completa)")

    # Crea samples formattati
    final_samples = []
    for i, sample in enumerate(medical_samples):
        if i < len(translated_instructions) and i < len(translated_outputs):
            final_samples.append(
                {
                    "text": format_medical_sample(
                        translated_instructions[i], translated_outputs[i]
                    ),
                    "category": "medical",
                }
            )

    return final_samples


def create_unified_dataset(resume: bool = False) -> Dataset:
    """Crea il dataset unificato italiano-medico."""
    logger.info("=" * 60)
    logger.info("CREAZIONE DATASET UNIFICATO MEDICO-ITALIANO")
    logger.info("=" * 60)
    logger.info(f"Target size: {TARGET_SIZE} record")
    logger.info(f"Resume: {resume}")

    italian_samples = load_local_italian_dataset()
    medical_samples = load_medical_samples()

    medical_translated = translate_with_checkpoint(medical_samples, resume=resume)

    all_samples = italian_samples + medical_translated

    logger.info(f"\nDataset grezzi: {len(all_samples)} campioni")

    if len(all_samples) > TARGET_SIZE:
        import random

        random.seed(42)
        all_samples = random.sample(all_samples, TARGET_SIZE)
        logger.info(f"Campioni ridotti a: {len(all_samples)}")

    ds = Dataset.from_list(all_samples)
    shuffled = ds.shuffle(seed=42)

    logger.info(f"\nDataset finale: {len(shuffled)} campioni")

    medical_count = sum(1 for s in shuffled if s.get("category") == "medical")
    linguistic_count = len(shuffled) - medical_count
    logger.info(f"  - Linguistici: {linguistic_count}")
    logger.info(f"  - Medici: {medical_count}")

    logger.info(f"\nSalvataggio in: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in shuffled:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Dataset salvato!")

    sample = shuffled[0]["text"][:300] if len(shuffled) > 0 else ""
    logger.info(f"Anteprima primo sample:\n{sample}...")

    return shuffled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea dataset medico-italiano")
    parser.add_argument("--resume", action="store_true", help="Riprendi da checkpoint")
    args = parser.parse_args()

    try:
        create_unified_dataset(resume=args.resume)
    except KeyboardInterrupt:
        logger.info("Script interrotto. Usa --resume per continuare.")
        sys.exit(0)
