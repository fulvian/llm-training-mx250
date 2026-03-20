# Training LLM in Italiano con QLoRA/PEFT

Progetto per il fine-tuning di LLM (Large Language Models) in italiano utilizzando QLoRA/PEFT su hardware limitato (GPU MX250 2GB VRAM, 14GB RAM).

## 📋 Indice

- [Panoramica](#panoramica)
- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Avvio Rapido](#avvio-rapido)
- [Script Principali](#script-principali)
- [Configurazione](#configurazione)
- [Monitoraggio](#monitoraggio)
- [Troubleshooting](#troubleshooting)

## 🎯 Panoramica

Questo progetto implementa un sistema di fine-tuning di LLM per la lingua italiana utilizzando:

- **Modello**: SmolLM-135M (Hugging Face)
- **Dataset**: TinyStories-Italian, Alpaca-GPT4-Italian, Dolly-15k (unificati in locale)
- **Tecnica**: QLoRA (Quantized Low-Rank Adaptation) con 4-bit quantization
- **Hardware**: Ottimizzato per GPU MX250 2GB VRAM

### Caratteristiche

- ✅ Resume automatico da checkpoint
- ✅ Training in background (sopravvive a disconnessioni SSH)
- ✅ Monitoraggio real-time con CLI
- ✅ Indicatore visivo di stato nel monitor (live/checkpoint)
- ✅ Gestione errori con checkpoint di emergenza
- ✅ Singleton training (previene multipli processi)
- ✅ Logging dettagliato
- ✅ Log persistenti tra sessioni
- ✅ Pulizia automatica file PID su crash/terminazione
- ✅ Rilevamento crash training nel monitor
- ✅ Deduplicazione automatica campioni dataset
- ✅ Garbage collection periodica durante training
- ✅ Pulizia memoria GPU strategica

## 📦 Prerequisiti

- Python 3.8+
- GPU NVIDIA con almeno 2GB VRAM
- Git

## 🔧 Installazione

```bash
# Clone repository
git clone <repository-url>
cd training_llm

# Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
datasets>=2.14.0
accelerate>=0.24.0
rich>=13.0.0
tqdm>=4.65.0
```

## 🚀 Avvio Rapido

```bash
# Attiva virtual environment
source venv/bin/activate

# Avvia training (resume automatico)
python3 train_italian_improved.py

# In un altro terminal, avvia monitoraggio
python3 monitor_training.py
```

## 📜 Script Principali

### 1. train_italian_improved.py

Script principale per il fine-tuning del modello.

#### Utilizzo

```bash
# Resume da ultimo checkpoint (default)
python3 train_italian_improved.py

# Nuovo training da zero
python3 train_italian_improved.py --no_resume

# Resume da checkpoint specifico
python3 train_italian_improved.py --resume_from checkpoint-600

# Specifica numero di epoch
python3 train_italian_improved.py --epochs 2

# Esegui in foreground (per debug)
python3 train_italian_improved.py --no_background

# Termina training in corso
python3 train_italian_improved.py --kill

# Forza riavvio (termina training esistente)
python3 train_italian_improved.py --force
```

#### Opzioni

| Flag | Descrizione |
|------|-------------|
| `--resume_from PATH` | Resume da checkpoint specifico |
| `--no_resume` | Non riprendere da checkpoint esistenti |
| `--no_background` | Esegui in foreground |
| `--kill` | Termina tutti i training |
| `--force` | Forza riavvio |
| `--epochs N` | Numero di epoch |

### 2. monitor_training.py

Script per il monitoraggio in tempo reale del training.

#### Utilizzo

```bash
# Avvia monitoraggio
python3 monitor_training.py
```

#### Funzionalità

- Mostra progresso training (step, epoch)
- Visualizza metriche (loss, eval_loss, learning_rate)
- Monitor risorse (GPU VRAM, utilization, temperature)
- Trend chart ASCII per loss
- Supporto per training resumed da checkpoint
- Indicazione dell'origine dei dati (src: live / src: checkpoint)
- Messaggio di stato resume
- Log recenti
- Aggiornamento automatico ogni 3 secondi

### 3. prepare_datasets.py

Script per scaricare e preparare i dataset in locale.

#### Utilizzo

```bash
# Prepara i dataset (scarica e unifica)
python3 prepare_datasets.py

# Output: datasets/italian_unified/train.jsonl (55.000 campioni)
```

#### Dataset generati

- **TinyStories-Italian**: 30.000 storie per bambini
- **Alpaca-GPT4-Italian**: 15.000 istruzioni
- **Dolly-15k**: 10.000 esempi (inglese - inclusi per diversità)

Il training usa automaticamente il dataset locale se presente.

## ⚙️ Configurazione

### Parametri Training

I parametri possono essere modificati in `train_italian_improved.py`:

```python
# Modello e Dataset
MODEL_PATH = "./models/SmolLM-135M-Instruct"
OUTPUT_DIR = "./smollm_italian_improved"
LOG_DIR = "./logs_smollm_improved"

# Configurazione Dataset
MAX_SAMPLES_TINYSTORIES = 30000  # Storie
MAX_SAMPLES_ALPACA = 15000       # Istruzioni
MAX_SAMPLES_DOLLY = 10000        # Dolly

# Configurazione Batch
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32  # Effective batch = 32
MAX_SEQ_LENGTH = 256

# Configurazione Training
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# Configurazione LoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
# Target modules - include gate_proj per maggiore capacità
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]

# Dataset locale unificato (creato da prepare_datasets.py)
LOCAL_DATASET_PATH = "./datasets/italian_unified/train.jsonl"

# Configurazione Logging
LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3
```

### Hardware Limitations

Il training è ottimizzato per:
- **GPU**: NVIDIA MX250 2GB VRAM
- **RAM**: 14GB+
- **Tempo**: ~41 secondi/step

Se hai più RAM o VRAM, puoi aumentare:
- `BATCH_SIZE` (se VRAM > 2GB)
- `MAX_SEQ_LENGTH` (se VRAM > 4GB)
- `MAX_SAMPLES_*` (se RAM > 16GB)

## 📊 Monitoraggio

### Monitoraggio in tempo reale

```bash
python3 monitor_training.py
```

Il monitor mostra:
- Progresso training (step, epoch, %)
- Trend loss (ASCII chart)
- Metriche (train_loss, eval_loss, learning_rate)
- Risorse GPU (VRAM, utilization, temperatura)
- Origine dati: live (dati in tempo reale) o checkpoint (dati da ultimo checkpoint)
- Indicatore visivo "🔄 RESUMED from step X" quando si riprende da checkpoint
- Log recenti

**Comportamento con resume**: Quando il training viene ripreso da checkpoint, il monitor mostra automaticamente i dati live più recenti. Il parametro `src` indica la fonte dei dati: `src: live` per dati in tempo reale dal log attivo, `src: checkpoint` per dati letti dall'ultimo checkpoint salvato.

### File di Log

- **Training Log**: `./smollm_italian_improved/training.log`
- **TensorBoard Logs**: `./logs_smollm_improved/` (non configurato per default)
- **Checkpoints**: `./smollm_italian_improved/checkpoint-*`

### Verifica Training in corso

```bash
ps aux | grep train_italian_improved
```

## 🔧 Troubleshooting

### Training non parte

**Sintomo**: Il training non parte dopo alcuni secondi

**Soluzione**:
```bash
# Verifica se c'è già un training in corso
ps aux | grep train_italian_improved

# Termina training esistente
python3 train_italian_improved.py --kill

# Riavvia
python3 train_italian_improved.py --force
```

### OOM (Out of Memory)

**Sintomo**: Errore `CUDA out of memory`

**Soluzione**:
```bash
# Riduci batch_size a 1
# Riduci max_seq_length a 256
# Chiudi altre applicazioni
# Usa quantizzazione 4-bit (attiva per default)
```

### Training lento

**Sintomo**: Training troppo lento (>60s/step)

**Soluzione**:
```bash
# Verifica GPU utilization
nvidia-smi

# Se utilization < 90%, potrebbe essere un problema di I/O
# Verifica se il dataset è in cache
ls -lh ~/.cache/huggingface/datasets/
```

### Monitor non mostra metriche

**Sintomo**: Il monitor mostra "Nessun dato disponibile"

**Soluzione**:
```bash
# Verifica che il training sia in esecuzione
ps aux | grep train_italian_improved

# Verifica che il log venga scritto
tail -f ./smollm_italian_improved/training.log

# Aspetta il primo checkpoint (logging_steps = 10)
```

## 📁 Struttura Directory

```
training_llm/
├── train_italian_improved.py    # Script training principale
├── monitor_training.py          # Script monitoraggio
├── prepare_datasets.py         # Script preparazione dataset
├── models/                     # Modello base
│   └── SmolLM-135M-Instruct/
├── datasets/                   # Dataset locali
│   ├── italian_unified/       # Dataset unificato (55k campioni)
│   └── databricks-dolly-15k/ # Dolly locale
├── smollm_italian_improved/   # Output training
│   ├── training.log           # Log training
│   ├── .training_pid          # PID file
│   └── checkpoint-*/         # Checkpoints
├── logs_smollm_improved/      # TensorBoard logs
├── README.md                   # Documentazione
└── AGENTS.md                   # Guide per agenti
```

## 🤝 Contribuire

Per contribuire al progetto:

1. Fork repository
2. Crea branch feature (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 Licenza

Questo progetto è rilasciato sotto la licenza MIT.

## 🙏 Riconoscimenti

- SmolLM (Hugging Face)
- TinyStories-Italian (markod0925)
- Alpaca-GPT4-Italian (FreedomIntelligence)
- Clean Dolly Italian (gsarti)

## 📞 Supporto

Per problemi o domande:
- Apri una issue su GitHub
- Controlla la sezione [Troubleshooting](#troubleshooting)
- Leggi [AGENTS.md](AGENTS.md) per dettagli tecnici

---

## 📝 Changelog

### 2026-03-20
- **Fix**: Monitor ora mostra correttamente i dati live durante resume da checkpoint
- **Fix**: Monitor distingue tra dati live e dati da checkpoint
- **Feat**: Log training ora in modalità append invece di sovrascrivere
- **Feat**: Separatore con data/ora all'inizio di ogni sessione training
- **Feat**: Messaggi più informativi su stato resume nel log

### 2026-03-20
- **Fix**: MemoryCleanupCallback ora eredita da TrainerCallback
- **Fix**: Implementati tutti i metodi callback richiesti da HuggingFace Trainer
- **Fix**: Risolto errore AttributeError su on_init_end, on_epoch_begin
- **Fix**: Aggiunto TrainerCallback agli import

### 2026-03-20
- **Memory**: Aggiunta deduplicazione automatica campioni dataset
- **Memory**: Aggiunta garbage collection periodica (ogni 50 step)
- **Memory**: Aggiunta pulizia memoria GPU dopo creazione dataset
- **Memory**: Aggiunta funzione cleanup_memory() per pulizia aggressiva
- **Memory**: Aggiunto MemoryCleanupCallback per gestione memoria dinamica
- **Memory**: Ridotto eval_size a 200 campioni

### 2026-03-20
- **Feat**: Aggiunto script `prepare_datasets.py` per preparare dataset locali
- **Feat**: Creato dataset unificato `datasets/italian_unified/train.jsonl` (55k campioni)
- **Fix**: Aggiornato train_italian_improved.py per usare dataset locale
- **Fix**: Ripristinato `gate_proj` nei target LoRA
- **Fix**: Risolti problemi di caricamento dataset (struttura cambiata su HF)

### 2026-03-20
- **Fix**: Aggiunti attributi `logging_steps` e `save_steps` alla dataclass TrainingConfig
- **Fix**: Aggiunta pulizia automatica file PID su crash/terminazione
- **Improvement**: Monitor ora rileva e segnala training crashati

### 2026-03-19
- **Fix**: Corretto bug monitor che cercava processi vecchi

---

**Note**:
- Il training può richiedere molto tempo su hardware limitato
- Utilizza sempre il background mode per training lunghi
- Monitora le risorse con `nvidia-smi` durante il training
- I checkpoint vengono salvati automaticamente ogni 200 step
