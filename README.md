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
- ✅ **TensorBoard integrato con avvio automatico**
- ✅ **URL TensorBoard con IP Tailscale per accesso remoto**
- ✅ Indicatore visivo di stato nel monitor (live/checkpoint)
- ✅ **Pulizia automatica vecchi dati con `--no_resume`**
- ✅ **Indicatore `[NEW]` per training appena avviati**
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
- **Tailscale** (opzionale, per accesso remoto a TensorBoard)

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

# Installa TensorBoard (richiesto per monitoring)
pip install tensorboard
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
tensorboard>=2.14.0
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

Il monitor avvierà automaticamente TensorBoard e mostrerà l'URL per accedere ai grafici.

## 📜 Script Principali

### 1. train_qlora_optimized.py

Script principale **OTTIMIZZATO** per il fine-tuning del modello, basato su best practices HuggingFace TRL/SFTTrainer.

#### Caratteristiche Principali

- ✅ **SFTTrainer**: Ottimizzato per supervised fine-tuning
- ✅ **Configurazione Centralizzata**: Parametri in `config.py`
- ✅ **LoRA Best Practices**: r=16, alpha=32, dropout=0.05
- ✅ **Full Attention**: Target modules q/k/v/o_proj
- ✅ **Efficienza Hardware**: Ottimizzato per 2GB VRAM

#### Utilizzo

```bash
# Training completo (55k campioni) - DEFAULT
python3 train_qlora_optimized.py

# Test rapido (100 campioni)
python3 train_qlora_optimized.py --quick

# Test intermedio (1000 campioni)
python3 train_qlora_optimized.py --intermediate

# Specifica numero di campioni
python3 train_qlora_optimized.py --max_samples 10000

# Resume da ultimo checkpoint
python3 train_qlora_optimized.py --resume
```

#### Opzioni

| Flag | Descrizione |
|------|-------------|
| `--quick` | Test rapido (100 campioni, 1 epoch) |
| `--intermediate` | Test intermedio (1000 campioni) |
| `--max_samples N` | Numero di campioni da usare |
| `--resume` | Resume da ultimo checkpoint |
| `--help` | Mostra help |

### 2. monitor_training.py

Script per il monitoraggio in tempo reale del training con **TensorBoard integrato**.

#### Utilizzo

```bash
# Avvia monitoraggio (TensorBoard si avvia automaticamente)
python3 monitor_training.py
```

#### Funzionalità

- Mostra progresso training (step, epoch)
- Visualizza metriche (loss, eval_loss, learning_rate)
- Monitor risorse (GPU VRAM, utilization, temperature)
- Trend chart ASCII per loss
- **TensorBoard integrato con avvio/arresto automatico**
- **URL con IP Tailscale per accesso remoto**
- Supporto per training resumed da checkpoint
- Indicazione dell'origine dei dati (src: live / src: checkpoint)
- Messaggio di stato resume
- Log recenti
- Aggiornamento automatico ogni 3 secondi

### 2. prepare_datasets.py

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

### 3. monitor_training.py

Script per il monitoraggio in tempo reale del training con **TensorBoard integrato**.

#### Utilizzo

```bash
# Avvia monitoraggio (TensorBoard si avvia automaticamente)
python3 monitor_training.py
```

#### Funzionalità

- Mostra progresso training (step, epoch)
- Visualizza metriche (loss, eval_loss, learning_rate)
- Monitor risorse (GPU VRAM, utilization, temperature)
- Trend chart ASCII per loss
- **TensorBoard integrato con avvio/arresto automatico**
- **URL con IP Tailscale per accesso remoto**
- Supporto per training resumed da checkpoint
- Indicazione dell'origine dei dati (src: live / src: checkpoint)
- Messaggio di stato resume
- Log recenti
- Aggiornamento automatico ogni 3 secondi

Il training usa automaticamente il dataset locale se presente.

## ⚙️ Configurazione

### Parametri Training

I parametri sono centralizzati in `config.py` e possono essere modificati:

```python
@dataclass
class TrainingConfig:
    # Paths
    base_model: str = "./models/SmolLM-135M-Instruct"
    dataset_path: str = "./datasets/italian_unified/train.jsonl"
    output_dir: str = "./output_qlora_optimized"
    tensorboard_log_dir: str = "./logs_qlora_optimized"
    
    # Dataset
    max_samples: Optional[int] = 55000
    max_seq_length: int = 384
    
    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # LoRA Config (Best Practices)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Monitoring
    logging_steps: int = 10
    save_steps: int = 500
    tensorboard_port: int = 6006
```

### Configurazione per Modelli Piccoli

Per ottimizzare per hardware limitato (MX250 2GB VRAM):
```python
config = TrainingConfig.full_training()
# or for quick test:
config = TrainingConfig.quick_test()
```

### Hardware Limitations

Il training è ottimizzato per:
- **GPU**: NVIDIA MX250 2GB VRAM
- **RAM**: 16GB+
- **Tempo**: ~45 secondi/step (3-4 ore per training completo)

Se hai più RAM o VRAM, puoi aumentare:
- `batch_size` (se VRAM > 4GB)
- `max_seq_length` (se VRAM > 6GB)

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
- **Stato TensorBoard con URL di accesso**
- Origine dati: live (dati in tempo reale) o checkpoint (dati da ultimo checkpoint)
- Indicatore visivo "🔄 RESUMED from step X" quando si riprende da checkpoint
- Log recenti

**Comportamento con resume**: Quando il training viene ripreso da checkpoint, il monitor mostra automaticamente i dati live più recenti. Il parametro `src` indica la fonte dei dati: `src: live` per dati in tempo reale dal log attivo, `src: checkpoint` per dati letti dall'ultimo checkpoint salvato.

### TensorBoard Integrato

Il monitor include **TensorBoard integrato** con le seguenti caratteristiche:

#### Avvio Automatico
- TensorBoard si avvia automaticamente quando il training è attivo
- Si ferma dopo 60 secondi di inattività del training
- Health check periodico per garantire il funzionamento

#### URL di Accesso

Il monitor mostra sempre l'URL di accesso:

```
│ 📊 TensorBoard                                          │
│ Status: ✅ Running  │ Port: 6006                        │
│ URL: http://100.81.21.110:6006                          │
│ PID: 12345                                              │
```

**Tipi di URL:**
- **IP Tailscale** (preferito): `http://100.xx.xx.xx:6006` - Accesso remoto sicuro
- **IP Locale**: `http://192.168.x.x:6006` - Accesso dalla rete locale
- **Localhost**: `http://127.0.0.1:6006` - Accesso solo dal computer locale

#### Accesso Remoto con Tailscale

Per accedere a TensorBoard da un altro dispositivo:

1. Installa [Tailscale](https://tailscale.com/) su entrambi i dispositivi
2. Assicurati che siano nella stessa rete Tailscale
3. Apri l'URL mostrato dal monitor nel browser

#### Metriche Disponibili

- **train_loss**: Loss del training (aggiornato ogni 10 step)
- **eval_loss**: Loss di validazione (aggiornato ogni 200 step)
- **learning_rate**: Learning rate corrente
- **grad_norm**: Normale del gradiente
- **epoch**: Epoch corrente

#### Avvio Manuale (se necessario)

Se TensorBoard non si avvia automaticamente:

```bash
tensorboard --logdir ./logs_smollm_improved --port 6006 --bind_all
```

### File di Log

- **Training Log**: `./train_qlora_optimized.log`
- **TensorBoard Logs**: `./logs_qlora_optimized/`
- **Checkpoints**: `./output_qlora_optimized/checkpoint-*`

### Verifica Training in corso

```bash
ps aux | grep train_qlora_optimized
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

### TensorBoard non si avvia

**Sintomo**: Il monitor mostra "TensorBoard: Not Running"

**Soluzione**:
```bash
# Verifica che tensorboard sia installato
pip show tensorboard

# Se non installato
pip install tensorboard

# Verifica che la porta 6006 non sia occupata
lsof -i :6006

# Avvia manualmente per test
tensorboard --logdir ./logs_smollm_improved --port 6006
```

### URL Tailscale non accessibile

**Sintomo**: L'URL con IP Tailscale non funziona

**Soluzione**:
```bash
# Verifica che Tailscale sia attivo
tailscale status

# Verifica l'IP Tailscale
tailscale ip

# Usa l'URL localhost se sei sullo stesso computer
# http://127.0.0.1:6006
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
├── train_qlora_optimized.py    # Script training principale (OTTIMIZZATO)
├── monitor_training.py          # Script monitoraggio (con TensorBoard integrato)
├── prepare_datasets.py         # Script preparazione dataset
├── config.py                   # Configurazione centralizzata
├── requirements.txt            # Dipendenze progetto
├── models/                     # Modello base
│   └── SmolLM-135M-Instruct/
├── datasets/                   # Dataset locali
│   ├── italian_unified/       # Dataset unificato (55k campioni)
│   └── databricks-dolly-15k/ # Dolly locale
├── output_qlora_optimized/    # Output training
│   ├── train_qlora_optimized.log # Log training
│   └── checkpoint-*/         # Checkpoints
├── logs_qlora_optimized/      # TensorBoard logs
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

### 2026-03-21 - Migrazione a Pipeline Ottimizzata
- **Feat**: Nuovo script `train_qlora_optimized.py` basato su TRL/SFTTrainer
- **Feat**: Configurazione centralizzata in `config.py` con dataclass
- **Feat**: Target modules completi per LoRA (q/k/v/o_proj)
- **Feat**: Learning rate ottimizzato per 135M parametri (3e-4)
- **Feat**: Max sequence length aumentato a 384 token
- **Refactor**: Eliminato script legacy `train_italian_improved.py` (834 righe)
- **Fix**: Syntax error in `config.py`
- **Fix**: Metodo `from_dict()` in `config.py`
- **Fix**: Tipizzazione di `target_modules`
- **Improvement**: `requirements.txt` centralizzato con tutte le dipendenze
- **Improvement**: `monitor_training.py` con supporto solo per nuovo script

### 2026-03-20 - Clean Training Mode
- **Feat**: Pulizia automatica vecchi checkpoint e runs TensorBoard con `--no_resume`
- **Feat**: Indicatore `[NEW]` nel monitor per training appena avviati (primi 5 minuti)
- **Feat**: Rilevamento training "fresh" - usa solo dati live, ignora vecchi checkpoint
- **Fix**: Il monitor non mostra più metriche obsolete quando si avvia un nuovo training
- **Fix**: TensorBoard mostra solo la run corrente, senza confusione con vecchi dati

### 2026-03-20 - TensorBoard Integration
- **Feat**: TensorBoard integrato nel monitor con avvio/arresto automatico
- **Feat**: URL TensorBoard con IP Tailscale per accesso remoto
- **Feat**: Health check periodico per TensorBoard
- **Feat**: Grace period di 60 secondi prima di fermare TensorBoard
- **Feat**: Gestione automatica porte (se 6006 occupata, prova 6007, ...)
- **Feat**: Nuova classe `TensorBoardManager` per gestione lifecycle
- **Feat**: Sezione TensorBoard sempre visibile nel monitor
- **Fix**: Installato tensorboard come dipendenza richiesta

### 2026-03-20 - TensorBoard Training
- **Feat**: TensorBoard ora abilitato nel training (`report_to="tensorboard"`)
- **Feat**: Aggiunta documentazione TensorBoard al README
- **Fix**: Monitor ora mostra correttamente i dati live durante resume da checkpoint
- **Fix**: Monitor distingue tra dati live e dati da checkpoint
- **Feat**: Log training ora in modalità append invece di sovrascrivere
- **Feat**: Separatore con data/ora all'inizio di ogni sessione training
- **Feat**: Messaggi più informativi su stato resume nel log

### 2026-03-20 - Memory Optimization
- **Fix**: MemoryCleanupCallback ora eredita da TrainerCallback
- **Fix**: Implementati tutti i metodi callback richiesti da HuggingFace Trainer
- **Fix**: Risolto errore AttributeError su on_init_end, on_epoch_begin
- **Fix**: Aggiunto TrainerCallback agli import
- **Memory**: Aggiunta deduplicazione automatica campioni dataset
- **Memory**: Aggiunta garbage collection periodica (ogni 50 step)
- **Memory**: Aggiunta pulizia memoria GPU dopo creazione dataset
- **Memory**: Aggiunta funzione cleanup_memory() per pulizia aggressiva
- **Memory**: Aggiunto MemoryCleanupCallback per gestione memoria dinamica
- **Memory**: Ridotto eval_size a 200 campioni

### 2026-03-20 - Dataset Preparation
- **Feat**: Aggiunto script `prepare_datasets.py` per preparare dataset locali
- **Feat**: Creato dataset unificato `datasets/italian_unified/train.jsonl` (55k campioni)
- **Fix**: Aggiornato train_italian_improved.py per usare dataset locale
- **Fix**: Ripristinato `gate_proj` nei target LoRA
- **Fix**: Risolti problemi di caricamento dataset (struttura cambiata su HF)

### 2026-03-20 - Bug Fixes
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
- TensorBoard si avvia automaticamente con il monitor
