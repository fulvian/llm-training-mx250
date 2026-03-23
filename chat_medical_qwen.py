#!/usr/bin/env python3
"""
Chat interattivo con il modello Qwen2.5-0.5B fine-tunato per medico+italiano.
Usa l'adapter LoRA salvato per generare risposte.

Usage:
    python3 chat_medical_qwen.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import sys
import os

# Configurazione
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./output_qwen25_medical_italian_20260323_093845"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9


def load_model():
    """Carica il modello base e l'adapter LoRA."""
    print("=" * 60)
    print("Caricamento modello Qwen2.5-0.5B con adapter medico...")
    print("=" * 60)

    # Configurazione quantizzazione 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Caricamento modello base con quantizzazione 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Caricamento adapter LoRA...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print(f"Modello caricato! VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("=" * 60)

    return model, tokenizer


def format_prompt(question: str) -> str:
    """Formatta la domanda nel formato corretto per il modello."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Genera una risposta dal modello."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Estrai solo la risposta (senza il prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Trova dove inizia la risposta dell'assistente
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in full_response:
        response = full_response.split(assistant_marker)[-1]
        # Rimuovi il token di fine
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = full_response[len(prompt) :].strip()

    return response


def print_header():
    """Stampa l'header del programma."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MEDICAL ITALIAN CHAT - Qwen2.5-0.5B" + " " * 6 + "║")
    print(
        "║" + " " * 5 + "Fine-tuned: Medico + Italiano (15k campioni)" + " " * 4 + "║"
    )
    print("╠" + "═" * 58 + "╣")
    print("║ Comandi:" + " " * 47 + "║")
    print("║   /quit o /exit - Esci" + " " * 34 + "║")
    print("║   /clear - Pulisci la cronologia" + " " * 26 + "║")
    print("║   /stats - Mostra statistiche sessione" + " " * 19 + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + " " * 58 + "║")
    print("║  Inserisci la tua domanda e premi INVIO" + " " * 15 + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_response(response: str):
    """Stampa la risposta in un box."""
    lines = response.split("\n")
    max_len = min(max(len(line) for line in lines), 56)

    print()
    print("┌" + "─" * (max_len + 2) + "┐")
    for line in lines:
        padding = max_len - len(line)
        print(f"│ {line}" + " " * padding + " │")
    print("└" + "─" * (max_len + 2) + "┘")
    print()


def main():
    """Loop principale della chat."""
    # Verifica che l'adapter esista
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Errore: Adapter non trovato in {ADAPTER_PATH}")
        print("   Esegui prima il training con train_qwen25_medical.py")
        sys.exit(1)

    # Carica il modello
    model, tokenizer = load_model()

    print_header()

    # Statistiche sessione
    session_stats = {
        "questions": 0,
        "start_time": None,
    }
    import time

    session_stats["start_time"] = time.time()

    # Loop principale
    while True:
        try:
            # Input utente
            user_input = input("🩺 Tu: ").strip()

            # Comandi speciali
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\n👋 Arrivederci!")
                break

            elif user_input.lower() == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_header()
                continue

            elif user_input.lower() == "/stats":
                elapsed = time.time() - session_stats["start_time"]
                print()
                print("📊 Statistiche Sessione:")
                print(f"   - Domande poste: {session_stats['questions']}")
                print(
                    f"   - Tempo trascorso: {int(elapsed // 60)}m {int(elapsed % 60)}s"
                )
                print()
                continue

            elif not user_input:
                continue

            # Aggiorna statistiche
            session_stats["questions"] += 1

            # Genera risposta
            print("⏳ Generazione risposta...")
            prompt = format_prompt(user_input)
            response = generate_response(model, tokenizer, prompt)
            print_response(response)

        except KeyboardInterrupt:
            print("\n\n👋 Arrivederci!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            continue

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
