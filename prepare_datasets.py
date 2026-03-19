#!/usr/bin/env python3
"""
Script per scaricare e armonizzare i dataset italiani in locale.
Genera un file JSONL unificato con formato: instruction, output
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path("./datasets/italian_unified")
OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"

MAX_SAMPLES = {
    "tinystories": 30000,
    "alpaca": 15000,
    "dolly": 10000,
}


def format_tinystories(sample):
    """TinyStories ha solo 'text' - lo trasformiamo in instruction/output"""
    return {
        "instruction": "Continua la storia:",
        "output": sample["text"],
    }


def format_alpaca(sample):
    """Alpaca ha formato 'conversations' - estrai instruction/response"""
    conv = sample.get("conversations", [])
    if len(conv) >= 2:
        instruction = conv[0].get("value", "").strip()
        response = conv[1].get("value", "").strip()
        return {"instruction": instruction, "output": response}
    return None


def format_dolly(sample):
    """Dolly ha formato instruction/response"""
    return {
        "instruction": sample.get("instruction", ""),
        "output": sample.get("response", ""),
    }


def download_tinystories():
    """Scarica TinyStories-Italian"""
    print("\n📥 Scaricando TinyStories-Italian...")
    ds = load_dataset("markod0925/TinyStories-Italian", split="train")

    samples = []
    for i in tqdm(range(min(MAX_SAMPLES["tinystories"], len(ds))), desc="TinyStories"):
        sample = format_tinystories(ds[i])
        if sample["output"]:  # Skip empty
            samples.append(sample)

    print(f"   -> {len(samples)} campioni")
    return samples


def download_alpaca():
    """Scarica Alpaca-GPT4-Italian"""
    print("\n📥 Scaricando Alpaca-GPT4-Italian...")
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-italian", split="train")

    samples = []
    for i in tqdm(range(min(MAX_SAMPLES["alpaca"], len(ds))), desc="Alpaca"):
        sample = format_alpaca(ds[i])
        if sample and sample["instruction"] and sample["output"]:
            samples.append(sample)

    print(f"   -> {len(samples)} campioni")
    return samples


def download_dolly():
    """Scarica Dolly-15k (inglese) e filtra/traduce"""
    print("\n📥 Scaricando Dolly-15k...")

    # Try to load Dolly 15k - we'll use the English version as fallback
    try:
        # Try loading from local first
        local_path = Path("./datasets/databricks-dolly-15k/databricks-dolly-15k.jsonl")
        if local_path.exists():
            print("   -> Uso dataset locale...")
            samples = []
            with open(local_path, "r") as f:
                for i, line in enumerate(tqdm(f, desc="Dolly locale", total=15000)):
                    if i >= MAX_SAMPLES["dolly"]:
                        break
                    data = json.loads(line)
                    sample = format_dolly(data)
                    if sample["instruction"] and sample["output"]:
                        samples.append(sample)
            print(f"   -> {len(samples)} campioni")
            return samples
    except Exception as e:
        print(f"   Dataset locale non disponibile: {e}")

    # Fallback to HF
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        samples = []
        for i in tqdm(range(min(MAX_SAMPLES["dolly"], len(ds))), desc="Dolly"):
            sample = format_dolly(ds[i])
            if sample["instruction"] and sample["output"]:
                samples.append(sample)
        print(f"   -> {len(samples)} campioni")
        return samples
    except Exception as e:
        print(f"   -> Errore: {e}")
        return []


def main():
    """Scarica e unisci tutti i dataset"""
    print("=" * 60)
    print("PREPARAZIONE DATASET ITALIANI UNIFICATI")
    print("=" * 60)

    # Crea directory output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Scarica ogni dataset
    all_samples.extend(download_tinystories())
    all_samples.extend(download_alpaca())
    all_samples.extend(download_dolly())

    print(f"\n📊 Totale campioni: {len(all_samples)}")

    # Salva in formato JSONL
    print(f"\n💾 Salvataggio in: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Statistiche
    print("\n📈 Statistiche:")
    print(
        f"   - TinyStories: {len([s for s in all_samples if s['instruction'].startswith('Continua')])}"
    )
    print(
        f"   - Alpaca: {len([s for s in all_samples if not s['instruction'].startswith('Continua') and len(s['instruction']) < 200])}"
    )
    print(
        f"   - Dolly: {len([s for s in all_samples if not s['instruction'].startswith('Continua') and len(s['instruction']) >= 200])}"
    )

    print("\n✅ Dataset unificato pronto!")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Dimensione: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
