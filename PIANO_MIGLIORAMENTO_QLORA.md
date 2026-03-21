# Piano di Miglioramento QLoRA Training

## Analisi dei Problemi Riscontrati

### FASE 3 (100 campioni sintetici)
- ✅ Training completato: Loss 4.72, 112s
- ❌ Output incoerente: misto italiano/spagnolo/inglese
- ❌ Dataset sintetico non rappresentativo

### FASE 4 (1000 campioni reali)
- ❌ Training BLOCCATO dopo "Inizio training..."
- ❌ Nessun progresso per 10+ minuti
- ❌ Output directory vuota

## Root Cause Analysis

### 1. Learning Rate ERRATO
```python
# Nostro: TROPPO BASSO
learning_rate = 5e-6  # ❌

# Best Practice QLoRA (da documentazione TRL)
learning_rate = 2e-4  # ✅ 10x più alto per LoRA
```

### 2. Trainer ERRATO
```python
# Nostro: Trainer generico
from transformers import Trainer  # ❌

# Best Practice: SFTTrainer specializzato
from trl import SFTTrainer  # ✅
```

### 3. Dataset Handling PROBLEMATICO
- Custom Dataset class con padding manuale
- DataCollator non ottimizzato
- SFTTrainer gestisce tutto automaticamente

### 4. LoRA Config SUBOTTIMALE
```python
# Nostro
r = 8, alpha = 16, dropout = 0.15  # alpha = 2x r

# Best Practice (da PEFT docs)
r = 16, alpha = 32, dropout = 0.05  # alpha = 2x r, dropout più basso
```

## Best Practices Emerse (Fonti Ufficiali)

### Da HuggingFace PEFT Documentation
| Parametro | Valore Consigliato | Note |
|-----------|-------------------|------|
| `r` (rank) | 16-64 | Più alto = più capacità, più memoria |
| `lora_alpha` | 2 × r | Scaling factor |
| `lora_dropout` | 0.05-0.1 | Più basso per dataset piccoli |
| `target_modules` | ["q_proj", "v_proj"] | Minimo per attention |
| `target_modules` | ["q_proj", "k_proj", "v_proj", "o_proj"] | Full attention |

### Da HuggingFace TRL Documentation
| Parametro | Valore Consigliato | Note |
|-----------|-------------------|------|
| `learning_rate` | 2e-4 | **10x più alto** per QLoRA |
| `per_device_train_batch_size` | 1 | Per memoria limitata |
| `gradient_accumulation_steps` | 16-32 | Effective batch size |
| `gradient_checkpointing` | True | Riduce memoria |
| `optim` | "adamw_8bit" | Per QLoRA |

### Da QLoRA Paper (arXiv:2305.14314)
- **NF4 quantization**: Ottimale per pesi normalmente distribuiti
- **Double Quantization**: Riduce ulteriormente memoria
- **Paged AdamW**: Per gestire memory spikes

## Piano di Azione

### FASE 4 REVISED: Test con SFTTrainer

```python
# Configurazione OTTIMIZZATA basata su best practices
config = {
    # Model
    "model": "./models/SmolLM-135M-Instruct",
    "quantization": "4-bit NF4 + double quant",
    
    # LoRA (PEFT best practices)
    "lora_r": 16,              # Aumentato da 8
    "lora_alpha": 32,          # 2x rank
    "lora_dropout": 0.05,      # Ridotto da 0.15
    "target_modules": ["q_proj", "v_proj"],
    
    # Training (TRL best practices)
    "learning_rate": 2e-4,     # 40x più alto! (era 5e-6)
    "batch_size": 1,
    "gradient_accumulation": 16,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Memory optimization
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "bf16": False,  # MX250 non supporta bf16
    "fp16": True,
    
    # Data
    "max_seq_length": 256,     # Aumentato da 128
    "samples": 1000,
}
```

### Nuovo Script: `train_qlora_optimized.py`

Vantaggi rispetto a `test_fase4_1000.py`:
1. ✅ Usa `SFTTrainer` invece di `Trainer`
2. ✅ Learning rate corretto (2e-4 vs 5e-6)
3. ✅ LoRA config ottimizzato (r=16, alpha=32, dropout=0.05)
4. ✅ Gestione automatica del dataset (no custom class)
5. ✅ Formattazione automatica dei prompt
6. ✅ Logging migliore con progress bar

## Confronto Configurazioni

| Parametro | Vecchio | Nuovo | Motivo |
|-----------|---------|-------|--------|
| `learning_rate` | 5e-6 | **2e-4** | 40x più alto per QLoRA |
| `lora_r` | 8 | **16** | Maggiore capacità |
| `lora_alpha` | 16 | **32** | 2x rank (best practice) |
| `lora_dropout` | 0.15 | **0.05** | Meno regolarizzazione |
| `max_seq_length` | 128 | **256** | Contesto più lungo |
| `trainer` | Trainer | **SFTTrainer** | Ottimizzato per SFT |
| `dataset_class` | Custom | **Nessuna** | SFTTrainer gestisce tutto |

## Stima Risultati

### Con vecchia configurazione:
- Loss: ~4.7 (stagnante)
- Output: Incoerente, misto lingue
- Tempo: ~2 min per 100 campioni

### Con nuova configurazione (atteso):
- Loss: ~2.5-3.5 (convergenza migliore)
- Output: Italiano coerente
- Tempo: ~3-4 min per 1000 campioni

## Prossimi Passi

1. [ ] Creare `train_qlora_optimized.py` con SFTTrainer
2. [ ] Test con 100 campioni (validazione rapida)
3. [ ] Se OK, test con 1000 campioni
4. [ ] Se OK, training completo 55k campioni

## Risorse

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [HuggingFace QLoRA Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
