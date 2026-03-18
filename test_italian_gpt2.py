#!/usr/bin/env python3
"""
Test Italian GPT2 model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo utilizzato: {device}")
    
    print("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./italian-gpt2-qlora-output")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Caricamento modello...")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/GroNLP-gpt2-small-italian",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print("Caricamento adapter...")
    model = PeftModel.from_pretrained(model, "./italian-gpt2-qlora-output")
    
    print("Modello caricato con successo!")
    
    # Test questions
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
    print("Test del modello Italian GPT2")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Domanda: {question}")
        try:
            prompt = f"""### Domanda: {question}
### Risposta:"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.3,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            answer = generated_text.split("### Risposta:")[-1].strip()
            
            print(f"Risposta: {answer}")
                
        except Exception as e:
            print(f"Errore: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
