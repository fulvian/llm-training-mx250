#!/usr/bin/env python3
"""
Modello italiano ultra-compatto pronto per l'uso
Usiamo un modello GPT2 italiano pubblico
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import json


class ItalianChatModel:
    """Chat interattivo con modello italiano ultra-compatto"""
    
    def __init__(self):
        # Configurazione con modello italiano pubblico
        self.BASE_MODEL = "pierluigic/medium_italian_gpt2"
        self.FINETUNED_MODEL = "./italian-gpt2-qlora-output"
        
        # Verifica se il modello è già stato addestrato
        if not os.path.exists(self.FINETUNED_MODEL):
            print("Modello non trovato, creazione automatica...")
            self._create_fake_model()
            
        self._load_model()
        
    def _create_fake_model(self):
        """Crea un modello fittizio per testing"""
        os.makedirs(self.FINETUNED_MODEL, exist_ok=True)
        
        # Configurazione LoRA
        config = {
            "base_model_name_or_path": self.BASE_MODEL,
            "bias": "none",
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "r": 8,
            "task_type": "CAUSAL_LM",
            "target_modules": ["c_proj", "c_fc", "c_attn"],
            "use_rslora": False,
            "init_lora_weights": "gaussian"
        }
        
        with open(os.path.join(self.FINETUNED_MODEL, "adapter_config.json"), "w") as f:
            json.dump(config, f)
            
        # Tokenizer config
        tokenizer_config = {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>",
            "model_max_length": 1024,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "GPT2Tokenizer"
        }
        
        with open(os.path.join(self.FINETUNED_MODEL, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)
            
        print("Modello fittizio creato")
        
    def _load_model(self):
        """Carica il modello e il tokenizer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo utilizzato: {self.device}")
        
        # Quantizzazione 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        
        # Caricamento tokenizer
        print("Caricamento tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Caricamento modello
        print("Caricamento modello...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # Se esiste l'adapter, carica lo state dict, altrimenti usa il modello base
        if os.path.exists(os.path.join(self.FINETUNED_MODEL, "adapter_model.safetensors")):
            self.model = PeftModel.from_pretrained(self.base_model, self.FINETUNED_MODEL)
        else:
            print("Nessun adapter trovato, uso il modello base...")
            self.model = self.base_model
            
        self.model.eval()
        print("Modello caricato con successo!")
        
    def generate_response(self, question, max_length=50, temperature=0.7):
        """Genera una risposta a una domanda"""
        # Prompt template per domande-risposte in italiano
        prompt = f"""Domanda: {question}
Risposta:"""
        
        encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Estrazione risposta
        answer = response.split("Risposta:")[-1].strip()
        
        return answer
        
    def chat(self):
        """Chat interattivo"""
        print("\n" + "="*50)
        print("Chat interattiva con Modello Italiano Ultra-Compatto")
        print("="*50)
        print("\nIstruzioni:")
        print(" - Digita una domanda in italiano")
        print(" - Digita 'quit' o 'exit' per uscire")
        print(" - Digita 'clear' per pulire lo schermo")
        print()
        
        while True:
            try:
                question = input("Tu: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ["quit", "exit"]:
                    print("\nGrazie per aver usato il modello italiano!")
                    break
                    
                if question.lower() == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    print("="*50)
                    print("Chat interattiva con Modello Italiano Ultra-Compatto")
                    print("="*50)
                    print("\nIstruzioni:")
                    print(" - Digita una domanda in italiano")
                    print(" - Digita 'quit' o 'exit' per uscire")
                    print(" - Digita 'clear' per pulire lo schermo")
                    print()
                    continue
                    
                print("Sto pensando...")
                response = self.generate_response(question)
                print(f"Bot: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nUscita forzata")
                break
            except Exception as e:
                print(f"Errore: {e}")
                print()
                

def test_italian_chat():
    """Testa il chat interattivo"""
    try:
        chat_model = ItalianChatModel()
        chat_model.chat()
    except Exception as e:
        print(f"Errore inizializzazione: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    test_italian_chat()

