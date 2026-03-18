#!/usr/bin/env python3
"""
Download Italian GPT2 model (GroNLP/gpt2-small-italian)
"""

import os
from huggingface_hub import snapshot_download

def download_italian_gpt2():
    """Download GroNLP/gpt2-small-italian model"""
    model_dir = "./models/GroNLP-gpt2-small-italian"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Downloading GroNLP/gpt2-small-italian model...")
    snapshot_download(
        repo_id="GroNLP/gpt2-small-italian",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Model downloaded to: {model_dir}")
    print("\nFiles in model directory:")
    for file in os.listdir(model_dir):
        print(f"  - {file}")
    
    return model_dir

def create_italian_gpt2_dataset():
    """Create a simple Italian Q&A dataset for GPT2 training"""
    import json
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
        {"question": "Qual è l'isola più grande d'Italia?", "answer": "Sicilia"}
    ]
    
    dataset_dir = "./datasets/Italian-GPT2"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_file = os.path.join(dataset_dir, "italian_qa.jsonl")
    
    with open(dataset_file, "w", encoding="utf-8") as f:
        for qa in questions_answers:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            
    print(f"\nDataset created: {dataset_file}")
    print(f"Number of Q&A pairs: {len(questions_answers)}")
    
    return dataset_dir

def main():
    print("="*50)
    print("Preparazione Italian GPT2 Model")
    print("="*50)
    
    # Download model
    model_dir = download_italian_gpt2()
    
    # Create dataset
    dataset_dir = create_italian_gpt2_dataset()
    
    print("\n✅ Preparazione completata!")
    print(f"\nModel directory: {os.path.abspath(model_dir)}")
    print(f"Dataset directory: {os.path.abspath(dataset_dir)}")

if __name__ == "__main__":
    main()
