#!/usr/bin/env python3
"""
Test intermedio QLoRA - Versione parametrica
Test con dataset più grande per verificare scalabilità
"""

import torch
import json
import argparse
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Test intermedio QLoRA")
    parser.add_argument("--dataset_path", type=str, default="dataset_unificato.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="test_output_intermediate")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.15)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.15)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"TEST INTERMEDIO QLORA - {args.max_samples} campioni, {args.epochs} epoche")
    print("=" * 70)

    # Verifica GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    # Step 1: Carica dataset
    print(f"\n[1/7] Caricamento dataset: {args.dataset_path}")

    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset non trovato: {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"   Dataset totale: {len(data)} campioni")

    # Limita campioni
    if len(data) > args.max_samples:
        data = data[: args.max_samples]

    dataset = Dataset.from_list(data)
    print(f"   Dataset usato: {len(dataset)} campioni")

    # Split train/validation
    split_idx = int(len(dataset) * 0.85)
    train_data = dataset.select(range(split_idx))
    val_data = dataset.select(range(split_idx, len(dataset)))

    print(f"   Train: {len(train_data)} campioni")
    print(f"   Validation: {len(val_data)} campioni")

    # Step 2: Carica tokenizer
    print("\n[2/7] Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./models/SmolLM-135M-Instruct")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Vocab size: {tokenizer.vocab_size}")

    # Step 3: Tokenizza dataset
    print("\n[3/7] Tokenizzazione dataset...")

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizzazione train",
    )

    val_tokenized = val_data.map(
        tokenize_function,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizzazione validation",
    )

    print(f"   Train tokenizzato: {len(train_tokenized)} campioni")
    print(f"   Validation tokenizzato: {len(val_tokenized)} campioni")

    # Step 4: Carica modello con quantizzazione
    print("\n[4/7] Caricamento modello...")
    torch.cuda.empty_cache()
    gc.collect()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "./models/SmolLM-135M-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    allocated = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   VRAM dopo modello: {allocated:.2f} MB")

    # Step 5: Applica LoRA
    print("\n[5/7] Configurazione LoRA...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainabili: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    allocated = torch.cuda.memory_allocated(0) / 1024**2
    print(f"   VRAM dopo LoRA: {allocated:.2f} MB")

    # Step 6: Configura training
    print("\n[6/7] Configurazione training...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    print(
        f"   Effective batch size: {args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max grad norm: {args.max_grad_norm}")

    # Step 7: Training
    print("\n[7/7] Avvio training...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Avvia training
    train_result = trainer.train()

    # Salva modello
    trainer.save_model()
    print(f"\n✅ Modello salvato in: {args.output_dir}")

    # Metriche finali
    print("\n" + "=" * 70)
    print("METRICHE FINALI")
    print("=" * 70)
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Train runtime: {train_result.metrics['train_runtime']:.1f}s")
    print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

    # Test generazione
    print("\n" + "=" * 70)
    print("TEST GENERAZIONE POST-TRAINING")
    print("=" * 70)

    model.eval()
    test_prompts = [
        "C'era una volta",
        "L'Italia è",
        "Spiegami cos'è",
    ]

    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"\nPrompt: '{prompt}'")
            print(f"Output: '{generated}'")
            print(f"Lunghezza: {len(generated)} caratteri")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 70)
    print("✅ TEST INTERMEDIO COMPLETATO")
    print("=" * 70)


if __name__ == "__main__":
    main()
