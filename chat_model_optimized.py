#!/usr/bin/env python3
"""
Chat interattivo ottimizzato per SmolLM-135M-Instruct
"""

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os
import time


class ChatModel:
    def __init__(self, base_model_path: str, fine_tuned_path: str):
        """
        Inizializza il chatbot
        """
        self.base_model_path = base_model_path
        self.fine_tuned_path = fine_tuned_path
        
        print("Inizializzazione del chatbot...")
        self._load_model()
        print("Chatbot pronto!")
        
    def _load_model(self):
        """
        Carica il modello e il tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo utilizzato: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
        self.model.eval()
        
    def generate_response(self, prompt: str, max_length: int = 128, temperature: float = 0.5) -> str:
        """
        Genera una risposta al prompt specificato con timeout e parametri ottimizzati
        """
        # Controllo prompt troppo generali
        general_prompts = ["ciao", "hello", "hi", "salve", "buongiorno", "buonasera", "ciao!", "hello!"]
        if prompt.lower().strip() in general_prompts:
            return "Ciao! Sono un assistente specializzato per rispondere a domande specifiche o completare task. Per favore, fai una domanda o specifica un'istruzione che vuoi che io completi!"
            
        formatted_prompt = f"""### Instruction: {prompt}

### Answer:"""
        
        encoding = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model.generate(
                    **encoding,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_time=60  # Timeout di 60 secondi
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "### Answer:" in response:
                answer = response.split("### Answer:")[1].strip()
                if answer:
                    return answer
                    
            return "Mi dispiace, non riesco a generare una risposta adeguata per questa richiesta."
            
        except Exception as e:
            print(f"Errore durante la generazione: {str(e)}")
            return "Mi dispiace, si è verificato un errore durante la generazione della risposta."
            
    def chat(self):
        """
        Interfaccia di chat interattiva
        """
        os.system("clear" if os.name == "posix" else "cls")
        print("=" * 50)
        print("Chat con SmolLM-135M-Instruct Fine-Tuned")
        print("=" * 50)
        print("\nIstruzioni:")
        print("- Fai domande specifiche o specifica task da completare")
        print("- Prompt generali come 'ciao' non sono supportati")
        print("- Digitare 'quit' o 'exit' per uscire")
        print("- Digitare 'clear' per pulire lo schermo")
        print()
        
        while True:
            try:
                prompt = input("Tu: ").strip()
                
                if not prompt:
                    continue
                    
                if prompt.lower() in ["quit", "exit"]:
                    print("\nGrazie per aver usato il chatbot!")
                    break
                    
                if prompt.lower() == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    print("=" * 50)
                    print("Chat con SmolLM-135M-Instruct Fine-Tuned")
                    print("=" * 50)
                    print("\nIstruzioni:")
                    print("- Fai domande specifiche o specifica task da completare")
                    print("- Prompt generali come 'ciao' non sono supportati")
                    print("- Digitare 'quit' o 'exit' per uscire")
                    print("- Digitare 'clear' per pulire lo schermo")
                    print()
                    continue
                    
                print("\nSto pensando...")
                response = self.generate_response(prompt)
                print(f"Bot: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nGrazie per aver usato il chatbot!")
                break
            except Exception as e:
                print(f"\nErrore: {str(e)}")
                print()


def main():
    """
    Funzione principale
    """
    BASE_MODEL = "./models/SmolLM-135M-Instruct"
    FINETUNED_MODEL = "./smollm-135m-qlora-output"
    
    if not os.path.exists(BASE_MODEL):
        print(f"Errore: Modello base non trovato in '{BASE_MODEL}'")
        return
        
    if not os.path.exists(FINETUNED_MODEL):
        print(f"Errore: Modello fine-tunato non trovato in '{FINETUNED_MODEL}'")
        return
        
    chatbot = ChatModel(BASE_MODEL, FINETUNED_MODEL)
    chatbot.chat()


if __name__ == "__main__":
    main()

