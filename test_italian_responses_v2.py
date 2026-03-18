#!/usr/bin/env python3
"""
Test del modello italiano ultra-compatto - Versione 2
Modello: EleutherAI/pythia-70m-deduped con adapter Italian-50 (v2)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# Configurazione quantizzazione
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

# Modelli base e adapter
BASE_MODEL = "EleutherAI/pythia-70m-deduped"
ADAPTER_MODEL = "./italian-70m-qlora-output-v2"
TEST_QUESTIONS = [
    "Qual è la capitale dell'Italia?",
    "Qual è il fiume più lungo d'Italia?",
    "Quanti sono i colori della bandiera italiana?",
    "Qual è la lingua ufficiale dell'Italia?",
    "Qual è il paese che confina con l'Italia a nord-ovest?",
    "In che anno è stata proclamata la Repubblica italiana?",
    "Qual è la città più grande d'Italia per popolazione?",
    "Qual è il vulcano attivo più famoso d'Italia?",
    "Quanti meridiani passano per l'Italia?",
    "Qual è la regione con la superficie più grande?",
    "Qual è la stagione che inizia il 21 giugno?",
    "Quanti giorni ha un anno bisestile?",
    "Qual è il pianeta più vicino al Sole?",
    "Qual è il metal più prezioso?",
    "Quanti occhi ha un ragno?",
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo utilizzato: {device}")
    
    print("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Caricamento modello...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print("Caricamento adapter italiano (v2)...")
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
    
    print("Modello caricato con successo!")
    print("\n" + "="*50)
    print("Test del modello italiano ultra-compatto (v2)")
    print("="*50)
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{i}. Domanda: {question}")
        try:
            prompt = f"""### Domanda: {question}
### Risposta:"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = extract_answer(generated_text)
            
            if answer:
                print(f"Risposta: {answer}")
            else:
                print("Risposta non valida")
                
        except Exception as e:
            print(f"Errore: {e}")
            import traceback
            print(traceback.format_exc())
    
    print("\n" + "="*50)

def extract_answer(text):
    """Estrai la risposta dal testo generato"""
    if "### Risposta:" in text:
        parts = text.split("### Risposta:")
        if len(parts) > 1:
            answer = parts[1].strip()
            # Rimuovi eventuali parti aggiuntive
            if "###" in answer:
                answer = answer.split("###")[0].strip()
            if answer and len(answer) > 0:
                return answer
    return None

if __name__ == "__main__":
    main()
