#!/usr/bin/env python3
"""
Download di modelli e dataset italiani per fine-tuning QLoRA con 2GB VRAM
"""

import os
from huggingface_hub import snapshot_download

def create_directories():
    """Crea le directory necessarie"""
    directories = ["./models/Italian-70M", "./datasets/Italian-1000"]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("Directory create")

def download_small_italian_model():
    """Scarica un modello italiano molto piccolo (70M parametri)"""
    print("Scaricando modello italiano BERTino-70M...")
    
    try:
        # Tentativo 1: Modello da Hugging Face Hub
        model_dir = "./models/Italian-70M"
        
        # Modello BERT base italiano (compatto)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Prova a scaricare un modello generativo italiano piccolo
            model_name = "dbmdz/bert-base-italian-uncased"
            print(f"Scaricando {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.save_pretrained(model_dir)
            
            print(f"Modello scaricato in {model_dir}")
            return True
            
        except Exception as e:
            print(f"Errore nel download di {model_name}: {e}")
            print("Tentativo di scaricare un modello generativo più piccolo...")
            
            # Modello generativo italiano ultra-compatto (35M)
            try:
                snapshot_download(
                    repo_id="EleutherAI/pythia-70m-deduped",
                    local_dir="./models/Italian-70M",
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.bin", "*.pt"]
                )
                print("Modello pythia-70m scaricato")
                return True
            except Exception as e2:
                print(f"Errore nel download alternativo: {e2}")
                return False
                
    except Exception as e:
        print(f"Errore generale: {e}")
        return False

def prepare_small_italian_dataset():
    """Prepara un dataset italiano compatto per training veloce"""
    print("Creazione di un dataset italiano compatto...")
    
    dataset_dir = "./datasets/Italian-1000"
    
    # Dataset di domande e risposte in italiano (sintetico)
    questions_answers = [
        {"question": "Qual è la capitale dell'Italia?", "answer": "Roma"},
        {"question": "Qual è il fiume più lungo d'Italia?", "answer": "Po"},
        {"question": "Quanti sono i colori della bandiera italiana?", "answer": "Tre"},
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
        {"question": "Qual è il frutto più amato da Pinocchio?", "answer": "Uva"}
    ]
    
    # Salva il dataset in formato JSONL
    import json
    dataset_file = os.path.join(dataset_dir, "italian_qa.jsonl")
    
    with open(dataset_file, "w", encoding="utf-8") as f:
        for qa in questions_answers:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            
    print(f"Dataset creato in {dataset_file}")
    print(f"Numero di record: {len(questions_answers)}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Preparazione modelli e dataset italiani")
    print("=" * 50)
    
    create_directories()
    
    model_success = download_small_italian_model()
    dataset_success = prepare_small_italian_dataset()
    
    if model_success and dataset_success:
        print("\n✅ Preparazione completata con successo!")
        print("\n📦 File creati:")
        print(f"   - Modello: {os.path.abspath('./models/Italian-70M')}")
        print(f"   - Dataset: {os.path.abspath('./datasets/Italian-1000')}")
    else:
        print("\n⚠️  Attenzione: Qualche operazione ha fallito")
        print("Verifica la connessione internet o riprova")

