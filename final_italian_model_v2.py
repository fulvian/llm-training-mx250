#!/usr/bin/env python3
"""
Interfaccia di chat per il modello italiano ultra-compatto - Versione 2
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
    
    print("\n✅ Modello caricato con successo!")
    print("="*60)
    print("Chatbot italiano ultra-compatto (70M parametri)")
    print("="*60)
    print("Istruzioni:")
    print("- Scrivi una domanda in italiano")
    print("- Digita 'quit' o 'exit' per chiudere")
    print("- Digita 'clear' per pulire la schermata")
    print("="*60)
    
    while True:
        try:
            question = input("\nTu: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ["quit", "exit"]:
                print("Arrivederci!")
                break
                
            if question.lower() == "clear":
                import os
                os.system("cls" if os.name == "nt" else "clear")
                print("="*60)
                print("Chatbot italiano ultra-compatto (70M parametri)")
                print("="*60)
                print("Istruzioni:")
                print("- Scrivi una domanda in italiano")
                print("- Digita 'quit' o 'exit' per chiudere")
                print("- Digita 'clear' per pulire la schermata")
                print("="*60)
                continue
                
            # Genera risposta
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
                print(f"Bot: {answer}")
            else:
                print("Bot: Mi dispiace, non posso rispondere a questa domanda.")
                
        except KeyboardInterrupt:
            print("\nArrivederci!")
            break
        except Exception as e:
            print(f"Bot: Mi dispiace, c'è stato un errore: {e}")
            import traceback
            print(traceback.format_exc())

def extract_answer(text):
    """Estrai la risposta dal testo generato"""
    if "### Risposta:" in text:
        parts = text.split("### Risposta:")
        if len(parts) > 1:
            answer = parts[1].strip()
            if "###" in answer:
                answer = answer.split("###")[0].strip()
            if answer and len(answer) > 0:
                return answer
    return None

if __name__ == "__main__":
    main()
