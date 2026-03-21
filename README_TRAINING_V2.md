# 🚀 Training QLoRA Italiano v2 - Guida Post-Analisi

## 📋 Panoramica

Questa guida contiene le istruzioni per ritrainare il modello SmolLM-135M in italiano seguendo le raccomandazioni dell'analisi critica del training precedente (vedi `ANALISI_TRAINING_QLORA.md`).

## ⚠️ Problemi Risolti

Il training precedente è fallito a causa di:

1. **Loss Collapse** - Loss scesa a 0.000059 ma output vuoti
2. **Learning Rate troppo alto** - 3e-5 troppo aggressivo
3. **Regularization insufficiente** - Weight decay 0.01 troppo basso
4. **LoRA overparameterizzato** - r=32 troppo alto
5. **Training incompleto** - Fermato al 23% per BrokenPipeError

## ✅ Fix Implementati

### Iperparametri Corretti

| Parametro | Prima ❌ | Dopo ✅ | Motivo |
|-----------|---------|---------|--------|
| Learning Rate | 3e-5 | 1e-5 | Ridotto per stabilità |
| Weight Decay | 0.01 | 0.05 | Aumentato per regolarizzazione |
| Warmup Ratio | 0.1 | 0.15 | Transizione più graduale |
| Max Grad Norm | None | 1.0 | Gradient clipping |
| LoRA r | 32 | 16 | Ridotto overparameterization |
| LoRA alpha | 64 | 32 | Bilanciato con r |
| LoRA dropout | 0.1 | 0.15 | Aumentato regolarizzazione |
| Max Seq Length | 256 | 192 | Risparmiare memoria |
| LR Scheduler | linear | cosine_with_restarts | Migliore convergenza |

### Nuove Funzionalità

- ✅ **Early Stopping** - Ferma training se loss non migliora
- ✅ **Gradient Clipping** - Previene esplosione gradienti
- ✅ **Optimizer 8-bit** - Risparmia memoria
- ✅ **Logging Migliorato** - Dettagli su ogni step
- ✅ **Test Automatici** - Validazione output post-training

## 🛠️ Setup

### 1. Verifica Prerequisiti

```bash
# Verifica GPU
nvidia-smi

# Output atteso:
# GPU: NVIDIA GeForce MX250
# VRAM: 2002 MB
```

### 2. Verifica Dataset

```bash
# Il dataset deve esistere
ls -lh dataset_unificato.json

# Output atteso:
# -rw-r--r-- 1 user user 50M Mar 20 10:00 dataset_unificato.json
```

### 3. Verifica Modello Base

```bash
# Il modello base deve esistere
ls -la models/SmolLM-135M-Instruct/

# Output atteso:
# config.json
# model.safetensors
# tokenizer.json
# etc.
```

## 🚀 Avvio Training

### Training Veloce (Test)

Per testare che tutto funzioni con pochi campioni:

```bash
python3 train_italian_v2.py \
    --max_samples 1000 \
    --epochs 1 \
    --output_dir ./smollm_test_v2
```

**Tempo stimato:** ~15 minuti  
**Scopo:** Verificare che il training funzioni e produca output validi

### Training Completo (Produzione)

```bash
python3 train_italian_v2.py \
    --max_samples 55000 \
    --epochs 2 \
    --output_dir ./smollm_italian_v2
```

**Tempo stimato:** ~10-12 ore  
**VRAM richiesta:** ~1.8 GB

### Training con Resume

Se il training si interrompe, riprendi dall'ultimo checkpoint:

```bash
python3 train_italian_v2.py \
    --resume_checkpoint ./smollm_italian_v2/checkpoint-XXX \
    --output_dir ./smollm_italian_v2
```

## 📊 Monitoraggio

### Durante il Training

#### 1. Monitor GPU

In un terminale separato:

```bash
watch -n 1 nvidia-smi
```

**Valori normali:**
- VRAM usata: 1.5-1.9 GB
- Temperatura: < 80°C
- Utilizzo GPU: 95-100%

#### 2. Monitor Log

```bash
tail -f training_v2.log
```

**Cercare:**
- ✅ `loss` che scende gradualmente
- ✅ `eval_loss` che segue train_loss
- ❌ `loss` che va a zero (loss collapse)
- ❌ `CUDA out of memory`

#### 3. TensorBoard

```bash
tensorboard --logdir ./smollm_italian_v2/logs
```

Apri http://localhost:6006

### Metriche Target

| Metrica | Target | Soglia Minima |
|---------|--------|---------------|
| Eval Loss | 0.5 - 1.0 | < 2.0 |
| Train Loss | 0.3 - 0.8 | < 1.5 |
| Perplexity | 5 - 15 | < 30 |
| Gradient Norm | 0.01 - 0.1 | < 1.0 |

## 🧪 Test Post-Training

### Test Automatico

Lo script esegue automaticamente 3 test:

```
Prompt: "C'era una volta"
Output atteso: testo italiano coerente (20+ token)

Prompt: "L'Italia è"
Output atteso: descrizione italiana sensata

Prompt: "Spiegami cos'è"
Output atteso: spiegazione in italiano
```

### Test Manuale

```bash
python3 test_model.py \
    --model ./smollm_italian_v2 \
    --prompt "C'era una volta" \
    --max-tokens 50
```

### Criteri di Successo

- ✅ Output non vuoto
- ✅ Almeno 20 token generati
- ✅ Testo in italiano (>90%)
- ✅ Coerenza logica (>70%)
- ✅ No ripetizioni loop

## 🔧 Troubleshooting

### Problema: CUDA Out of Memory

**Sintomi:**
```
RuntimeError: CUDA out of memory
```

**Soluzioni:**

1. Ridurre sequenza massima:
```bash
--max_seq_length 128  # Da 192 a 128
```

2. Ridurre batch effective:
```bash
--gradient_accumulation_steps 16  # Da 24 a 16
```

3. Chiudere altre applicazioni:
```bash
# Chiudere browser, IDE, etc.
```

### Problema: Loss Non Scende

**Sintomi:**
- Loss rimane > 5.0 dopo 1000 step
- Loss oscilla violentemente

**Soluzioni:**

1. Aumentare learning rate:
```bash
--learning_rate 2e-5  # Da 1e-5 a 2e-5
```

2. Ridurre warmup:
```bash
--warmup_ratio 0.1  # Da 0.15 a 0.1
```

3. Verificare dataset:
```bash
# Controllare che i prompt siano corretti
head -n 10 dataset_unificato.json
```

### Problema: Loss Collapse (Loss → 0)

**Sintomi:**
- Loss < 0.0001
- Output vuoti o ripetitivi

**Soluzioni:**

1. **FERMARE SUBITO IL TRAINING** ❌
2. Aumentare weight decay:
```bash
--weight_decay 0.1  # Da 0.05 a 0.1
```

3. Ridurre learning rate:
```bash
--learning_rate 5e-6  # Da 1e-5 a 5e-6
```

4. Aumentare LoRA dropout:
```bash
--lora_dropout 0.2  # Da 0.15 a 0.2
```

### Problema: Training Lento

**Sintomi:**
- > 120 secondi per step
- Tempo totale > 24 ore

**Soluzioni:**

1. Ridurre eval frequency:
```bash
--eval_steps 1000  # Da 500 a 1000
```

2. Ridurre logging:
```bash
--logging_steps 100  # Da 50 a 100
```

3. Usare meno campioni:
```bash
--max_samples 30000  # Invece di 55000
```

## 📈 Confronto con Training Precedente

### Configurazione

| Aspetto | Training v1 ❌ | Training v2 ✅ |
|---------|----------------|----------------|
| Learning Rate | 3e-5 | 1e-5 |
| Weight Decay | 0.01 | 0.05 |
| LoRA r | 32 | 16 |
| Max Seq Len | 256 | 192 |
| Early Stopping | No | Sì |
| Gradient Clip | No | Sì |

### Risultati Attesi

| Metrica | Training v1 | Training v2 (target) |
|---------|-------------|----------------------|
| Eval Loss | 0.000059 (collapse) | 0.5 - 1.0 |
| Output | Vuoto | Coerente |
| Training Completato | 23% | 100% |
| Qualità | ❌ Fallito | ✅ Funzionante |

## 🎯 Prossimi Passi Post-Training

### 1. Validazione Qualità (Giorno 1)

```bash
# Test su 100 prompt diversi
python3 validate_model.py \
    --model ./smollm_italian_v2 \
    --test_set ./test_prompts.json
```

### 2. Confronto con Base (Giorno 2)

```bash
# Confrontare output modello base vs fine-tuned
python3 compare_models.py \
    --base ./models/SmolLM-135M-Instruct \
    --finetuned ./smollm_italian_v2
```

### 3. Ottimizzazione (Giorno 3-7)

Se risultati buoni:
- Aumentare dataset a 100k campioni
- Sperimentare altri iperparametri
- Testare su task specifici

Se risultati scarsi:
- Analizzare errori
- Raccogliere più dati
- Provare modello diverso (Qwen2.5-0.5B)

## 📞 Supporto

### Se il Training Fallisce

1. **Salvare i log:**
```bash
cp training_v2.log training_v2_failed.log
```

2. **Verificare memoria:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

3. **Controllare errori:**
```bash
grep -i "error\|exception\|failed" training_v2.log
```

4. **Riprovare con config ridotta:**
```bash
python3 train_italian_v2.py \
    --max_samples 10000 \
    --max_seq_length 128 \
    --epochs 1
```

### Contatti

- **Issues:** Aprire issue su GitHub con log allegati
- **Discussione:** Canale Discord #qlora-training
- **Wiki:** Documentazione completa in `docs/`

## 🏁 Checklist Finale

Prima di avviare il training, verificare:

- [ ] GPU disponibile con >1.8GB VRAM libera
- [ ] Dataset presente e valido (54k+ campioni)
- [ ] Modello base presente in `models/`
- [ ] Spazio disco sufficiente (>5GB)
- [ ] Nessun altro processo GPU attivo
- [ ] Script `train_italian_v2.py` presente
- [ ] File `ANALISI_TRAINING_QLORA.md` letto

---

**Buon training!** 🚀

Se segui questa guida, dovresti ottenere un modello funzionante in 10-12 ore.

In caso di problemi, consulta la sezione Troubleshooting o chiedi supporto.
