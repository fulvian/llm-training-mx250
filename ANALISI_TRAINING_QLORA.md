# 🔍 ANALISI CRITICA Training QLoRA - SmolLM-135M Italiano

**Data Analisi:** 20 Marzo 2026  
**Modello:** SmolLM-135M-Instruct  
**Metodo:** QLoRA (4-bit quantization + LoRA adapters)  
**Hardware:** NVIDIA MX250 2GB VRAM, 14GB RAM

---

## ⚠️ EXECUTIVE SUMMARY - CRITICAL ISSUES

### 🚨 PROBLEMA PRINCIPALE: **LOSS COLLAPSE + OUTPUT VUOTI**

Il training è **fallito** nonostante le metriche sembrino eccellenti. Il modello ha raggiunto una loss estremamente bassa (0.0000593) ma **non produce output significativi**.

**Risultati Test Post-Training:**
```
Prompt: "C'era una volta" → Output: ""  ❌
Prompt: "L'Italia è" → Output: ""      ❌  
Prompt: "Spiegami cos'è l'intelligenza artificiale" → Output: "" ❌
```

---

## 📊 ANALISI METRICHE

### Configurazione Training
```python
# Hardware
GPU: NVIDIA GeForce MX250 (2.1 GB VRAM)

# Dataset
Dataset: 54,889 campioni (deduplicati da 55,000)
- TinyStories-Italian: 30,000
- Alpaca-GPT4-Italian: 15,000  
- Dolly-Italian: 10,000

# Iperparametri
Batch Size: 1
Gradient Accumulation: 16 (Effective Batch: 16, non 32 come indicato)
Max Seq Length: 256
Learning Rate: 3e-5
Epochs: 3 (pianificati) → 0.7 (completati)
Warmup Ratio: 0.1
Weight Decay: 0.01

# LoRA Configuration
r = 32
alpha = 64
dropout = 0.1
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj']
trainable_params = 5,713,920 (4.07% del totale)
```

### Andamento Loss (Training)

| Epoch | Train Loss | Eval Loss | Note |
|-------|------------|-----------|------|
| 0.4388 | 0.000194 | - | Inizio monitoraggio |
| 0.4681 | 0.000166 | 0.000151 | Eval intermedia |
| 0.5734 | 0.000100 | - | Loss dimezzata |
| 0.5851 | 0.000099 | 0.000089 | Continua discesa |
| 0.7022 | 0.000065 | **0.000059** | **Fine training** |

### Performance Sistema
```
Training Runtime: 15,400 secondi (4.28 ore)
Samples/second: 10.65
Steps/second: 0.333
Step medio: ~84 secondi (troppo lento!)
```

---

## 🔥 CRITICITÀ IDENTIFICATE

### 1. **LOSS COLLAPSE - CRITICITÀ ALTA** 🔴

**Sintomo:** Loss scesa a 0.000059 (quasi zero)  
**Causa:** Il modello ha imparato a "imbrogliare" il sistema ottimizzando la loss senza apprendere effettivamente  
**Conseguenza:** Output completamente vuoti o inutili

**Perché succede:**
- Learning rate troppo alto per dataset piccolo (54k campioni)
- Loss function non bilanciata per task generativi
- Mancanza di regularization adeguata
- Early stopping non configurato correttamente

**Soluzione:**
```python
# PRIMA (problematico)
learning_rate = 3e-5  # Troppo aggressivo
weight_decay = 0.01   # Insufficiente

# DOPO (corretto)
learning_rate = 1e-5  # Ridotto di 3x
weight_decay = 0.05   # Aumentato 5x
lr_scheduler_type = "cosine_with_restarts"
warmup_ratio = 0.15   # Aumentato
```

### 2. **GRADIENT NORM INSTABILE - CRITICITÀ MEDIA** 🟡

**Sintomo:** Gradiente norm oscillante tra 0.00003 e 0.0001  
**Causa:** Aggiornamenti troppo piccoli, il modello non sta imparando efficacemente  
**Analisi:**
```
grad_norm range: 3.3e-05 to 1.0e-04
Media: ~6.0e-05
Deviazione: Molto alta rispetto alla media
```

**Soluzione:**
```python
# Aggiungere gradient clipping
max_grad_norm = 1.0  # Clipping a 1.0
optim = "adamw_8bit"  # Ottimizzatore più stabile
```

### 3. **EFFECTIVE BATCH SIZE ERRATO - CRITICITÀ MEDIA** 🟡

**Sintomo:** Log indica "Effective batch: 32" ma reale è 16  
**Causa:** Discrepanza tra configurazione dichiarata e implementazione  
**Impatto:** Training meno stabile del previsto

**Verifica:**
```python
# Configurazione attuale
batch_size = 1
gradient_accumulation_steps = 16
# Effective batch = 1 * 16 = 16 ❌ (non 32)

# Configurazione corretta per batch 32
batch_size = 2  # Se VRAM lo permette
gradient_accumulation_steps = 16
# Effective batch = 2 * 16 = 32 ✓
```

### 4. **TRAINING INCOMPLETO - CRITICITÀ ALTA** 🔴

**Sintomo:** Training fermato al 23% (epoch 0.7 su 3 pianificati)  
**Causa:** BrokenPipeError - Processo terminato anomalamante  
**Impatto:** Modello non ha completato il ciclo di apprendimento

**Possibili cause interruzione:**
1. **OOM (Out of Memory) silenzioso** - GPU esaurita durante eval
2. **Timeout processo** - Superato limite tempo
3. **Kill manuale** - Interruzione utente
4. **Errore sistema** - Pipe broken durante logging

### 5. **VELOCITÀ TRAINING INSUFFICIENTE - CRITICITÀ BASSA** 🟢

**Sintomo:** 84 secondi per step, 4.28 ore per 23% del training  
**Causa:** Hardware limitato (MX250) + configurazione non ottimizzata  
**Impatto:** Tempo totale stimato 18+ ore per 3 epoche complete

**Ottimizzazioni possibili:**
```python
# Ridurre sequenza massima
max_seq_length = 192  # Da 256 a 192 (-25% memoria)

# Aumentare gradient accumulation per compensare
gradient_accumulation_steps = 24  # Da 16 a 24

# Usare checkpointing più aggressivo
gradient_checkpointing = True
optim = "adamw_8bit"  # Meno memoria dell'ottimizzatore
```

---

## 🎯 ANALISI DATASET

### Qualità Dataset
```
Totale campioni: 55,000
Duplicati rimossi: 111 (0.2%)
Dataset finale: 54,889
```

**Problemi identificati:**
1. **Dataset troppo piccolo** per fine-tuning efficace
2. **Mix di tipi diversi** (storie + istruzioni + QA) senza bilanciamento
3. **Qualità variabile** - TinyStories molto semplice vs Alpaca complesso

**Raccomandazione:**
```python
# Bilanciare dataset per tipo
target_distribution = {
    "narrative": 20_000,    # TinyStories
    "instruction": 20_000,  # Alpaca
    "qa": 15_000,          # Dolly
}
# Totale: 55,000 campioni bilanciati
```

---

## 📈 CONFRONTO CON BEST PRACTICES

### Configurazione Attuale vs. Raccomandata

| Parametro | Attuale | Raccomandato | Motivo |
|-----------|---------|--------------|--------|
| Learning Rate | 3e-5 | 1e-5 | Troppo aggressivo |
| Weight Decay | 0.01 | 0.05 | Regolarizzazione insufficiente |
| Warmup Ratio | 0.1 | 0.15 | Transizione troppo rapida |
| Batch Size | 16 eff. | 32 eff. | Training instabile |
| Max Seq Len | 256 | 192 | Troppo per 2GB VRAM |
| LoRA r | 32 | 16 | Overparameterization |
| LoRA alpha | 64 | 32 | Scaling eccessivo |
| Epochs | 3 | 1-2 | Overfitting risk |

---

## 🛠️ PIANO DI MIGLIORAMENTO

### FASE 1: Fix Immediati (Criticità Alta)

#### 1.1 Correggere Learning Rate e Regularization
```python
# File: train_improved.py

# PRIMA
training_args = TrainingArguments(
    learning_rate=3e-5,        # ❌ Troppo alto
    weight_decay=0.01,         # ❌ Troppo basso
    warmup_ratio=0.1,          # ❌ Insufficiente
)

# DOPO
training_args = TrainingArguments(
    learning_rate=1e-5,        # ✅ Ridotto 3x
    weight_decay=0.05,         # ✅ Aumentato 5x
    warmup_ratio=0.15,         # ✅ Aumentato
    lr_scheduler_type="cosine_with_restarts",  # ✅ Scheduler migliore
    max_grad_norm=1.0,         # ✅ Gradient clipping
)
```

#### 1.2 Ridurre LoRA Rank
```python
# PRIMA
lora_config = LoraConfig(
    r=32,                      # ❌ Troppo alto
    lora_alpha=64,             # ❌ Scaling eccessivo
)

# DOPO  
lora_config = LoraConfig(
    r=16,                      # ✅ Ridotto
    lora_alpha=32,             # ✅ Bilanciato
    lora_dropout=0.15,         # ✅ Aumentato dropout
)
```

#### 1.3 Aggiungere Early Stopping
```python
from transformers import EarlyStoppingCallback

# Aggiungere ai callbacks
callbacks=[
    EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )
]
```

### FASE 2: Ottimizzazioni Performance (Criticità Media)

#### 2.1 Ottimizzare Memoria
```python
# Ridurre sequenza massima
max_seq_length = 192  # -25% memoria

# Usare optimizer 8-bit
optim = "adamw_8bit"

# Gradient checkpointing più aggressivo
gradient_checkpointing_kwargs = {"use_reentrant": False}
```

#### 2.2 Bilanciare Dataset
```python
# Script di bilanciamento
def balance_dataset(dataset, target_per_type=20000):
    """Bilancia dataset per tipo di contenuto."""
    balanced = []
    
    # TinyStories (narrative semplice)
    tiny = dataset.filter(lambda x: x['source'] == 'tinystories')
    balanced.extend(tiny.shuffle().select(range(target_per_type)))
    
    # Alpaca (istruzioni complesse)
    alpaca = dataset.filter(lambda x: x['source'] == 'alpaca')
    balanced.extend(alpaca.shuffle().select(range(target_per_type)))
    
    # Dolly (QA)
    dolly = dataset.filter(lambda x: x['source'] == 'dolly')
    balanced.extend(dolly.shuffle().select(range(15000)))
    
    return balanced
```

### FASE 3: Monitoraggio Avanzato (Criticità Bassa)

#### 3.1 Logging Dettagliato
```python
# Aggiungere wandb o tensorboard
import wandb

wandb.init(
    project="smollm-italian-qlora",
    config={
        "learning_rate": 1e-5,
        "epochs": 2,
        "batch_size": 16,
    }
)

# Logging metriche personalizzate
def log_metrics(logits, labels):
    # Perplexity
    perplexity = torch.exp(torch.mean(torch.log(logits + 1e-10)))
    wandb.log({"perplexity": perplexity})
    
    # Token distribution
    token_dist = torch.bincount(labels.flatten())
    wandb.log({"token_entropy": torch_entropy(token_dist)})
```

#### 3.2 Validation Set Dedicato
```python
# Separare validation set più grande
train_test_split = dataset.train_test_split(
    test_size=0.15,  # Aumentato da 5% a 15%
    stratify_by_source=True  # Bilanciare per fonte
)
```

---

## 🧪 TEST E VALIDAZIONE

### Test da Eseguire Prima di Ritrainare

#### Test 1: Sanity Check su Modello Base
```bash
# Verificare che il modello base funzioni
python3 test_model.py --model ./models/SmolLM-135M-Instruct \
    --prompt "C'era una volta" \
    --max-tokens 50
```

**Risultato atteso:** Output coerente in inglese (modello base)

#### Test 2: Overfit su Piccolo Subset
```python
# Testare su 100 campioni per verificare apprendimento
python3 train_improved.py \
    --max_samples 100 \
    --epochs 10 \
    --learning_rate 1e-5
```

**Risultato atteso:** Loss scende gradualmente, output migliorano

#### Test 3: Learning Rate Finder
```python
# Trovare learning rate ottimale
from transformers import TrainerCallback

class LRFinderCallback(TrainerCallback):
    def __init__(self, min_lr=1e-7, max_lr=1e-3, num_steps=100):
        self.lrs = torch.logspace(
            torch.log10(torch.tensor(min_lr)),
            torch.log10(torch.tensor(max_lr)),
            num_steps
        )
```

---

## 📋 CHECKLIST PRE-TRAINING

### Prima di Avviare il Training

- [ ] **Hardware Check**
  - [ ] Verificare VRAM disponibile (>1.8GB liberi)
  - [ ] Monitorare temperatura GPU (<80°C)
  - [ ] Chiudere applicazioni non necessarie

- [ ] **Dataset Check**
  - [ ] Verificare bilanciamento classi
  - [ ] Controllare duplicati
  - [ ] Validare formato prompts

- [ ] **Configurazione Check**
  - [ ] Learning rate ridotto a 1e-5
  - [ ] Weight decay aumentato a 0.05
  - [ ] Early stopping configurato
  - [ ] Gradient clipping attivo

- [ ] **Monitoring Check**
  - [ ] TensorBoard/WandB configurato
  - [ ] Alert per OOM attivi
  - [ ] Checkpoint automatici ogni 500 step

- [ ] **Backup Check**
  - [ ] Salvataggio automatico checkpoint
  - [ ] Script di resume da checkpoint
  - [ ] Log su file persistente

---

## 🚀 COMANDI DI TRAINING CORRETTI

### Training Veloce (Test)
```bash
python3 train_improved.py \
    --max_samples 1000 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 192 \
    --output_dir ./smollm_test \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500
```

### Training Completo (Produzione)
```bash
python3 train_improved.py \
    --max_samples 55000 \
    --epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 24 \
    --max_seq_length 192 \
    --output_dir ./smollm_italian_v2 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 1000 \
    --warmup_ratio 0.15 \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine_with_restarts \
    --early_stopping_patience 5
```

---

## 📊 METRICHE DI SUCCESSO

### Target da Raggiungere

| Metrica | Target | Soglia Minima |
|---------|--------|---------------|
| Eval Loss | 0.5 - 1.0 | < 2.0 |
| Train Loss | 0.3 - 0.8 | < 1.5 |
| Perplexity | 5 - 15 | < 30 |
| Output Coherence | > 80% | > 60% |
| Training Time | < 12 ore | < 24 ore |
| GPU Memory | < 1.9 GB | < 2.0 GB |

### Test Qualità Output

```python
# Test automatici post-training
test_prompts = [
    "C'era una volta un bambino che",
    "L'Italia è un paese famoso per",
    "Spiegami cos'è l'intelligenza artificiale",
    "Come si fa la pasta alla carbonara?",
    "Quali sono i vantaggi del solare?",
]

# Criteri di successo
success_criteria = {
    "min_length": 20,      # Almeno 20 token
    "coherence": 0.7,      # 70% token coerenti
    "italian_ratio": 0.9,  # 90% in italiano
    "no_repetition": True, # No loop
}
```

---

## 💡 LEZIONI IMPARATE

### 1. Loss Bassa ≠ Buon Modello
**Lezione:** Una loss estremamente bassa può indicare overfitting o collapse, non successo.

### 2. Testare Sempre con Prompt Reali
**Lezione:** Le metriche quantitative non bastano, servono test qualitativi durante il training.

### 3. Hardware Limitato Richiede Compromessi
**Lezione:** Con 2GB VRAM, bisogna ridurre seq_length e batch_size, non aumentare epochs.

### 4. Dataset Quality > Quantity
**Lezione:** 55k campioni misti valgono meno di 20k campioni ben bilanciati e puliti.

### 5. Monitoring è Fondamentale
**Lezione:** Senza TensorBoard/WandB, è impossibile diagnosticare problemi in tempo reale.

---

## 🎯 PROSSIMI PASSI

### Immediato (Oggi)
1. ✅ Analisi completata
2. ⏳ Fix configurazione training
3. ⏳ Test su subset piccolo (1000 campioni)
4. ⏳ Validare output con prompt test

### Breve Termine (1-3 giorni)
1. ⏳ Training completo con config corretta
2. ⏳ Valutazione qualità output
3. ⏳ Confronto con modello base
4. ⏳ Ottimizzazione parametri

### Medio Termine (1 settimana)
1. ⏳ Espandere dataset (target 100k campioni)
2. ⏳ Sperimentare altri modelli (Qwen2.5-0.5B)
3. ⏳ Implementare RLHF per qualità
4. ⏳ Deploy e test utente

---

## 📞 SUPPORTO

### Se Incontri Problemi

1. **OOM Error:** Ridurre `max_seq_length` a 128
2. **Loss non scende:** Aumentare `learning_rate` a 2e-5
3. **Output vuoti:** Verificare tokenizer e format prompts
4. **Training lento:** Ridurre `gradient_accumulation_steps`

### Log e Debug

```bash
# Monitorare GPU in tempo reale
watch -n 1 nvidia-smi

# Verificare log training
tail -f ./smollm_italian_v2/training.log

# Controllare memoria
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

---

## 🏁 CONCLUSIONI

### Verdetto Finale: **TRAINING FALLITO** ❌

**Motivi:**
1. Loss collapse con output vuoti
2. Configurazione iperparametri subottimale
3. Training incompleto (23%)
4. Mancanza di early stopping e monitoring

**Tuttavia:**
- ✅ Pipeline tecnica funzionante
- ✅ Dataset caricato correttamente
- ✅ Checkpoint salvati
- ✅ Script robusti

**Prossimo step:** Ritrainare con configurazione corretta seguendo questo piano.

---

**Analisi condotta da:** QLoRA Expert Agent  
**Metodologia:** Post-mortem analysis con root cause identification  
**Confidenza:** Alta (basata su log completi e metriche dettagliate)
