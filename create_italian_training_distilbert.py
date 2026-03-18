#!/usr/bin/env python3
"""
Script per training di un modello italiano ultra-compatto con DistilBERT
Modello: distilbert-base-uncased (66M parametri, molto compatto)
Dataset: 48 domande/risposte in italiano
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from datasets import Dataset
import json


def load_italian_dataset():
    """Carica il dataset italiano da file"""
    dataset_file = "./datasets/Italian-50/italian_qa.jsonl"
    
    if not os.path.exists(dataset_file):
        print("Dataset non trovato, creazione automatica...")
        create_italian_dataset()
        
    return load_dataset_from_file(dataset_file)


def create_italian_dataset():
    """Crea un dataset italiano con 48 domande/risposte"""
    questions_answers = [
        {"question": "Qual è la capitale dell'Italia?", "answer": "Roma"},
        {"question": "Qual è il fiume più lungo d'Italia?", "answer": "Po"},
        {"question": "Quanti sono i colori della bandiera italiana?", "answer": "Tre: verde, bianco e rosso"},
        {"question": "Qual è la lingua ufficiale dell'Italia?", "answer": "Italiano"},
        {"question": "Qual è il paese che confina con l'Italia a nord-ovest?", "answer": "Francia"},
        {"question": "In che anno è stata proclamata la Repubblica italiana?", "answer": "1946"},
        {"question": "Qual è la città più grande d'Italia per popolazione?", "answer": "Roma"},
        {"question": "Qual è il vulcano attivo più famoso d'Italia?", "answer": "Etna"},
        {"question": "Quanti meridiani passano per l'Italia?", "answer": "Uno, il meridiano di Greenwich"},
        {"question": "Qual è la regione con la superficie più grande?", "answer": "Sardegna"},
        {"question": "Qual è la stagione che inizia il 21 giugno?", "answer": "Estate"},
        {"question": "Quanti giorni ha un anno bisestile?", "answer": "366"},
        {"question": "Qual è il pianeta più vicino al Sole?", "answer": "Mercurio"},
        {"question": "Qual è il metal più prezioso?", "answer": "Oro"},
        {"question": "Quanti occhi ha un ragno?", "answer": "Otto"},
        {"question": "Qual è l'animale simbolo della pace?", "answer": "Colomba"},
        {"question": "Qual è il giorno della settimana che segue il venerdì?", "answer": "Sabato"},
        {"question": "In che mese cade la festività di Natale?", "answer": "Dicembre"},
        {"question": "Qual è il colore del cielo?", "answer": "Blu"},
        {"question": "Qual è il frutto più amato da Pinocchio?", "answer": "Uva"},
        {"question": "Come si chiama il teatro più famoso di Milano?", "answer": "Teatro alla Scala"},
        {"question": "Qual è il monumento simbolo di Roma?", "answer": "Colosseo"},
        {"question": "Quanti regni componevano l'Italia prima dell'Unità?", "answer": "Sette"},
        {"question": "Qual è la città che ospita il Duomo più grande d'Italia?", "answer": "Milano"},
        {"question": "Qual è l'isola più grande d'Italia?", "answer": "Sicilia"},
        {"question": "Qual è la collina dove si trova il Vaticano?", "answer": "Janicolo"},
        {"question": "Qual è il nome del primo re d'Italia?", "answer": "Vittorio Emanuele II"},
        {"question": "In che anno è stata firmata la Pace di Versailles?", "answer": "1919"},
        {"question": "Qual è la moneta dell'Italia?", "answer": "Euro"},
        {"question": "Qual è la religione principale in Italia?", "answer": "Cattolicesimo"},
        {"question": "Qual è il sport nazionale italiano?", "answer": "Calcio"},
        {"question": "Qual è la pizza più famosa?", "answer": "Margherita"},
        {"question": "Qual è il vino più famoso dell'Italia?", "answer": "Chianti"},
        {"question": "Qual è il formaggio più famoso dell'Italia?", "answer": "Parmigiano-Reggiano"},
        {"question": "Qual è il fiume che attraversa Firenze?", "answer": "Arno"},
        {"question": "Qual è la montagna più alta d'Italia?", "answer": "Monte Bianco"},
        {"question": "Qual è il lago più grande d'Italia?", "answer": "Lago di Garda"},
        {"question": "Qual è la città che produce le macchine Ferrari?", "answer": "Modena"},
        {"question": "Qual è il nome della prima donna che ha vinto il Premio Nobel?", "answer": "Marie Curie"},
        {"question": "Qual è il nome dell'oceano più grande?", "answer": "Pacifico"},
        {"question": "Qual è il nome del continente più caldo?", "answer": "Africa"},
        {"question": "Qual è il nome della capitale di Francia?", "answer": "Parigi"},
        {"question": "Qual è il nome della capitale di Spagna?", "answer": "Madrid"},
        {"question": "Qual è il nome della capitale di Germania?", "answer": "Berlino"},
        {"question": "Qual è il nome della capitale di Grecia?", "answer": "Atene"},
        {"question": "Qual è il nome della capitale di Portogallo?", "answer": "Lisbona"},
        {"question": "Qual è il nome della capitale di Austria?", "answer": "Vienna"},
        {"question": "Qual è il nome della capitale di Svizzera?", "answer": "Berna"}
    ]
    
    os.makedirs("./datasets/Italian-50", exist_ok=True)
    dataset_file = "./datasets/Italian-50/italian_qa.jsonl"
    
    with open(dataset_file, "w", encoding="utf-8") as f:
        for qa in questions_answers:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            
    print(f"Dataset creato: {dataset_file}")
    print(f"Record totali: {len(questions_answers)}")
    
    return questions_answers


def load_dataset_from_file(file_path):
    """Carica il dataset da file JSONL"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                
    return Dataset.from_list(data)


def main():
    # Step 1: Carica dataset
    print("Step 1: Caricamento dataset italiano...")
    dataset = load_italian_dataset()
    
    # Step 2: Configurazione
    MODEL_NAME = "distilbert-base-uncased"
    OUTPUT_DIR = "./italian-distilbert-qlora-output"
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4
    NUM_EPOCHS = 5
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nStep 2: Configurazione:")
    print(f"  - Modello: {MODEL_NAME}")
    print(f"  - Output dir: {OUTPUT_DIR}")
    print(f"  - Batch size: {BATCH_SIZE} (accumulato a {GRADIENT_ACCUMULATION})")
    print(f"  - Epoche: {NUM_EPOCHS}")
    
    # Step 3: Quantizzazione 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    # Step 4: Configurazione LoRA per DistilBERT
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2"
        ],
    )
    
    # Step 5: Caricamento tokenizer
    print("\nStep 3: Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Step 6: Caricamento modello
    print("Step 4: Caricamento modello con quantizzazione 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    
    # Step 7: Preprocessing del dataset
    print("\nStep 5: Preprocessing del dataset...")
    
    def format_instruction(sample):
        text = f"""### Domanda: {sample["question"]}
### Risposta: {sample["answer"]}"""
        return {"text": text}
    
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        )
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.map(tokenize, batched=False, remove_columns=["text"])
    
    print(f"Dataset finale creato: {len(dataset)} record")
    
    # Step 8: Training arguments
    print("\nStep 6: Configurazione training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=5,
        save_steps=20,
        warmup_steps=5,
        bf16=False,
        fp16=False,
        optim="paged_adamw_8bit",
        eval_strategy="no",
        save_strategy="steps",
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
    )
    
    # Step 9: Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Step 10: Trainer
    print("\nStep 7: Inizio training...")
    from trl import SFTTrainer, SFTConfig
    
    sft_config = SFTConfig(
        dataset_text_field="text",
        max_length=128,
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=lora_config,
    )
    
    # Step 11: Esegui training
    trainer.train()
    
    # Step 12: Salva
    print("\nStep 8: Salvataggio modello...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Training completato!")
    print(f"Modello salvato in: {OUTPUT_DIR}")
    print(f"File creati:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'tokenizer.json')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'adapter_config.json')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        import traceback
        print(traceback.format_exc())

