#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for SmolLM-135M on low VRAM GPU (2GB)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import os
import dotenv
dotenv.load_dotenv()

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm-135m-qlora-output"
DATASET_NAME = "databricks/databricks-dolly-15k"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


def format_instruction(sample):
    text = f"""### Instruction
{sample["instruction"]}

### Response
{sample["response"]}"""
    return {"text": text}


def main():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    dataset = dataset.map(
        format_instruction, remove_columns=dataset["train"].column_names
    )
    dataset["train"] = dataset["train"].select(range(min(2000, len(dataset["train"]))))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        warmup_steps=20,
        bf16=False,
        fp16=False,
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        seed=3407,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
    )

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        processing_class=tokenizer,
        formatting_func=lambda x: x["text"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()

