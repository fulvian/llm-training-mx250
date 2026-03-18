#!/usr/bin/env python3
"""
Modello italiano ultra-compatto OFFLINE
Usa Pythia-70m-deduped con adapter fine-tunato sul dataset italiano
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
    """Chat interattivo con modello italiano ultra-compatto OFFLINE"""
    
    def __init__(self):
        # Configurazione offline
        self.BASE_MODEL = "./models/Pythia-70m-deduped"
        self.FINETUNED_MODEL = "./italian-70m-qlora-output"
        
        # Verifica se il modello base è presente
        if not os.path.exists(self.BASE_MODEL):
            print("Modello base non trovato, creazione automatica...")
            self._download_base_model()
            
        self._load_model()
        
    def _download_base_model(self):
        """Scarica il modello base Pythia-70m-deduped in modalità offline"""
        try:
            from huggingface_hub import snapshot_download
            
            os.makedirs(self.BASE_MODEL, exist_ok=True)
            
            print("Scaricamento di EleutherAI/pythia-70m-deduped...")
            snapshot_download(
                repo_id="EleutherAI/pythia-70m-deduped",
                local_dir=self.BASE_MODEL,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.bin"]
            )
            
            print("Modello base scaricato")
            
        except Exception as e:
            print(f"Errore download: {e}")
            print("Creazione di un modello sintetico...")
            
            # Crea file di configurazione fittizio
            config_data = {
                "architectures": ["GPTNeoXForCausalLM"],
                "bos_token_id": 0,
                "eos_token_id": 2,
                "hidden_act": "gelu",
                "hidden_size": 512,
                "initializer_range": 0.02,
                "intermediate_size": 2048,
                "layer_norm_eps": 1e-05,
                "max_position_embeddings": 2048,
                "model_type": "gpt_neox",
                "num_attention_heads": 8,
                "num_hidden_layers": 6,
                "pad_token_id": 1,
                "tie_word_embeddings": False,
                "vocab_size": 50304
            }
            
            with open(os.path.join(self.BASE_MODEL, "config.json"), "w") as f:
                json.dump(config_data, f)
                
            # Tokenizer config fittizio
            tokenizer_config = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 2048,
                "pad_token": "<pad>",
                "tokenizer_class": "GPTNeoXTokenizer"
            }
            
            with open(os.path.join(self.BASE_MODEL, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f)
                
            print("Modello sintetico creato")
            
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.FINETUNED_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Caricamento modello
        print("Caricamento modello...")
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception as e:
            print(f"Errore caricamento modello base: {e}")
            print("Uso del modello direttamente dal repo...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/pythia-70m-deduped",
                quantization_config=bnb_config,
                device_map="auto",
            )
            
        # Se esiste l'adapter, carica lo state dict, altrimenti usa il modello base
        if os.path.exists(os.path.join(self.FINETUNED_MODEL, "adapter_model.safetensors")):
            try:
                self.model = PeftModel.from_pretrained(self.base_model, self.FINETUNED_MODEL)
            except Exception as e:
                print(f"Errore adapter: {e}")
                print("Uso il modello base...")
                self.model = self.base_model
        else:
            print("Nessun adapter trovato, uso il modello base...")
            self.model = self.base_model
            
        self.model.eval()
        print("Modello caricato con successo!")
        
    def generate_response(self, question, max_length=50, temperature=0.7):
        """Genera una risposta a una domanda"""
        # Prompt template per domande-risposte in italiano
        prompt = f"""### Domanda: {question}
### Risposta:"""
        
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
        answer = response.split("### Risposta:")[-1].strip()
        
        return answer
        
    def chat(self):
        """Chat interattivo"""
        print("\n" + "="*50)
        print("Chat interattiva con Modello Italiano Ultra-Compatto (OFFLINE)")
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
                    print("Chat interattiva con Modello Italiano Ultra-Compatto (OFFLINE)")
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

