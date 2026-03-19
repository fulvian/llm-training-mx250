# AGENTS.md - Linee Guida per Agenti di Coding

## Panoramica del Progetto
Repository per il fine-tuning di LLM (Large Language Models) in italiano utilizzando QLoRA/PEFT su hardware limitato (GPU MX250 2GB VRAM,14GB RAM).

## Comandi Principali

### Training
```bash
# Training principale (consigliato)
python3 train_italian_local.py

# Training veloce (per test)
python3 train_italian_fast.py

# Training ottimizzato
python3 train_best.py
```

### Monitoraggio GPU
```bash
# Monitor GPU in tempo reale
python3 gpu_monitor.py --watch

# Monitor risorse sistema
python3 monitor_resources.py
```

### Test e Chat
```bash
# Test modello
python3 test_model.py
python3 test_italian_model.py

# Chat interattiva
python3 chat_model.py
python3 chat_italian_gpt2.py
```

### Linting e Formattazione
```bash
# Ruff (linting + formatting)
ruff check .
ruff format .

# Verifica singolo file
ruff check train_italian_local.py
ruff format train_italian_local.py
```

## Struttura Directory
```
training_llm/
├── models/                    # Modelli pre-addestrati locali
│   ├── SmolLM-135M-Instruct/  # Modello principale
│   └── Italian-*/             # Altri modelli italiani
├── datasets/                  # Dataset locali
│   └── databricks-dolly-15k/
├── *_output/                  # Directory output training
├── logs_*/                    # Log TensorBoard
├── train_*.py                 # Script di training
├── test_*.py                  # Script di test
└── chat_*.py                  # Script di chat interattiva
```

## Convenzioni di Codice

### Importazioni
```python
# Ordine: standard library → third-party → local
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Raggruppare import correlati
from transformers import (
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
```

### Formattazione
- **Lunghezza riga:** 100 caratteri max
- **Indentazione:** 4 spazi (NO tab)
- **Stringhe:** Preferire f-string `f"valore: {x}"` vs concatenazione
- **Virgole finali:** Sempre nelle liste multilinea

```python
# Corretto
config = {
    "learning_rate": 3e-5,
    "batch_size": 1,
    "epochs": 3,
}

# Evitare
config = {"learning_rate": 3e-5, "batch_size": 1, "epochs": 3}
```

### Nomi
- **File:** `snake_case.py` (es: `train_italian_local.py`)
- **Classi:** `PascalCase` (es: `ItalianDataset`)
- **Funzioni/Metodi:** `snake_case` (es: `prepare_dataset`)
- **Costanti:** `UPPER_SNAKE_CASE` (es: `MAX_SEQ_LENGTH`)
- **Variabili private:** `_leading_underscore`

### Type Hints
```python
from typing import Optional, List, Dict, Any

def prepare_dataset(
    data_path: str,
    max_samples: Optional[int] = None,
    tokenizer: Any = None,
) -> Dict[str, torch.Tensor]:
    """Prepara il dataset per il training.
    
    Args:
        data_path: Percorso al dataset
        max_samples: Numero massimo di campioni (None = tutti)
        tokenizer: Tokenizer da utilizzare
        
    Returns:
        Dizionario con input_ids e attention_mask
    """
    pass
```

### Gestione Errori
```python
# Preferire eccezioni specifiche con contesto
def load_model(model_path: str):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except OSError as e:
        raise RuntimeError(f"Impossibile caricare modello da {model_path}: {e}") from e
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("OOM durante caricamento modello. Ridurre batch_size.")
    return model

# Logging invece di print
import logging
logger = logging.getLogger(__name__)

logger.info(f"Caricati {len(dataset)} campioni")
logger.error(f"Errore nel training: {e}")
```

### Docstrings
```python
def format_prompt(sample: Dict) -> str:
    """Formatta un campione in prompt per il modello.
    
    Utilizza il formato chat di SmolLM con token speciali.
    
    Args:
        sample: Dizionario con 'instruction' e 'output'
        
    Returns:
        Stringa formattata con token <|im_start|> e <|im_end|>
        
    Example:
        >>> format_prompt({"instruction": "Ciao", "output": "Salve!"})
        '<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\nSalve!<|im_end|>'
    """
    pass
```

## Configurazione Training

### Parametri Consigliati (MX250 2GB)
```python
# Configurazione ottimizzata per hardware limitato
BATCH_SIZE = 1                    # Minimo per evitare OOM
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch = 16
MAX_SEQ_LENGTH = 256              # Bilanciato per memoria
LEARNING_RATE = 3e-5              # Ridotto per stabilita
NUM_EPOCHS = 3                    # Aumentato per qualita
WEIGHT_DECAY = 0.01               # Regolarizzazione

# LoRA Configuration
LORA_R = 32                       # Rank aumentato
LORA_ALPHA = 64                   # 2 * r
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Quantizzazione 4-bit
```python
# Sempre usare quantizzazione 4-bit su MX250
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

## Best Practices

### Memory Management
```python
# Pulire memoria prima di operazioni pesanti
import gc
torch.cuda.empty_cache()
gc.collect()

# Usare context manager per tensori temporanei
with torch.no_grad():
    outputs = model.generate(**inputs)
```

### Dataset Handling
```python
# Evitare di caricare tutto in memoria
# Usare streaming per dataset grandi
dataset = load_dataset(
    "nome/dataset",
    streaming=True,  # Importante per memoria
)

# Preprocessing lazy con map
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
```

### Logging e Monitoring
```python
# Configurare logging all'inizio
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Usare tqdm per progress bar
from tqdm import tqdm
for item in tqdm(dataset, desc="Processing"):
    process(item)
```

## Checklist Pre-Commit
- [ ] `ruff check .` passa senza errori
- [ ] `ruff format .` applicato
- [ ] Nessun print() di debug (usare logger)
- [ ] Type hints aggiunti dove appropriato
- [ ] Docstrings per funzioni pubbliche
- [ ] Costanti in UPPER_CASE all'inizio file
- [ ] Gestione errori con try/except specifici
- [ ] Memory cleanup prima di operazioni pesanti

## Note Hardware
- **GPU:** NVIDIA MX250 2GB VRAM - richiede quantizzazione 4-bit
- **RAM:** 14GB - monitorare uso swap
- **Training time:** ~25 min/epoch per 60k campioni
- **OOM risk:** Alto con batch_size > 2 o seq_length > 512

## Dataset Italiani Consigliati
1. `markod0925/TinyStories-Italian` - Storie semplici
2. `FreedomIntelligence/alpaca-gpt4-italian` - Istruzioni
3. `gsarti/clean_dolly_italian` - Dolly in italiano
4. `sag-ita/italian-wikipedia` - Wikipedia italiana
