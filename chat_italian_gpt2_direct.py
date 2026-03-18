#!/usr/bin/env python3
"""
Chat interface for Italian GPT2 model directly from Hugging Face
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os

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
        "GroNLP/gpt2-small-italian",
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print("Caricamento adapter...")
    model = PeftModel.from_pretrained(model, "./italian-gpt2-qlora-output")
    
    print("\n✅ Modello caricato con successo!")
    print("="*60)
    print("Chatbot italiano GPT2 (ultra-compatto)")
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
                os.system("cls" if os.name == "nt" else "clear")
                print("="*60)
                print("Chatbot italiano GPT2 (ultra-compatto)")
                print("="*60)
                print("Istruzioni:")
                print("- Scrivi una domanda in italiano")
                print("- Digita 'quit' o 'exit' per chiudere")
                print("- Digita 'clear' per pulire la schermata")
                print("="*60)
                continue
                
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
            answer = generated_text.split("### Risposta:")[-1].strip()
            
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

if __name__ == "__main__":
    main()
