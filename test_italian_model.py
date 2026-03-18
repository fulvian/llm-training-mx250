#!/usr/bin/env python3
"""
Script per testare il modello italiano ultra-compatto (70M parametri)
"""

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM


def test_model():
    # Configurazione
    BASE_MODEL = "EleutherAI/pythia-70m-deduped"
    FINETUNED_MODEL = "./italian-70m-qlora-output"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizzato: {device}")
    
    # Caricamento tokenizer
    print("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Quantizzazione
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    # Caricamento modello base
    print("Caricamento modello base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.use_cache = False
    
    # Caricamento adapter fine-tunato
    print("Caricamento adapter italiano...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    model.eval()
    
    # Domande di test
    test_questions = [
        "Qual è la capitale dell'Italia?",
        "Qual è il fiume più lungo d'Italia?",
        "Quanti sono i colori della bandiera italiana?",
        "Qual è la lingua ufficiale dell'Italia?",
        "Qual è il paese che confina con l'Italia a nord-ovest?",
        "In che anno è stata proclamata la Repubblica italiana?",
        "Qual è la città più grande d'Italia per popolazione?",
        "Qual è il vulcano attivo più famoso d'Italia?",
        "Quanti meridiani passano per l'Italia?",
        "Qual è la regione con la superficie più grande?"
    ]
    
    print("\n" + "="*50)
    print("Test del modello italiano ultra-compatto")
    print("="*50)
    
    # Generazione risposte
    for i, question in enumerate(test_questions, 1):
        prompt = f"""### Domanda: {question}
### Risposta:"""
        
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoding,
                max_new_tokens=30,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Estrazione risposta
        answer = response.split("### Risposta:")[-1].strip()
        
        print(f"\n{i}. Domanda: {question}")
        print(f"Risposta: {answer}")
        
    return model, tokenizer


if __name__ == "__main__":
    test_model()

