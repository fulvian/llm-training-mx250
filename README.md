# Training LLM in Italiano con QLoRA/PEFT

Progetto per il fine-tuning di LLM (Large Language Models) in italiano per il dominio medico, utilizzando QLoRA/PEFT su hardware limitato (GPU MX250 2GB VRAM, 14GB RAM).

## 🎯 Risultati Ultimo Training

| Metrica | Valore |
|---------|--------|
| **Modello** | Qwen2.5-0.5B-Instruct |
| **Dataset** | Medical + Italian (15,000 campioni) |
| **Eval Loss** | 1.5548 (riduzione -9.0%) |
| **VRAM** | 0.44 GB (4-bit quantization) |
| **Checkpoint** | `output_qwen25_medical_italian_20260323_093845/` |

### Andamento Loss

```
Epoch 0.059: 1.7077 (baseline)
Epoch 0.296: 1.6144 (-5.5%)
Epoch 0.533: 1.5822 (-7.3%)
Epoch 0.948: 1.5548 (-9.0%) ← Finale
```

## 📋 Indice

- [Panoramica](#panoramica)
- [Prerequisiti](#prerequisiti)
- [Installazione](#installazione)
- [Avvio Rapido](#avvio-rapido)
- [Script Principali](#script-principali)
- [Configurazione](#configurazione)
- [Monitoraggio](#monitoraggio)
- [Chat Interattivo](#chat-interattivo)
- [Struttura Directory](#struttura-directory)
- [Troubleshooting](#troubleshooting)

## 🎯 Panoramica

Questo progetto implementa un sistema di fine-tuning di LLM per la lingua italiana nel dominio medico utilizzando:

- **Modello Base**: Qwen2.5-0.5B-Instruct (Hugging Face)
- **Dataset**: 15,000 campioni (10k linguistici italiani + 5k medici tradotti)
- **Tecnica**: QLoRA (Quantized Low-Rank Adaptation) con 4-bit quantization
- **Hardware**: Ottimizzato per GPU MX250 2GB VRAM

### Caratteristiche

- ✅ Resume automatico da checkpoint
- ✅ Training in background (sopravvive a disconnessioni SSH)
- ✅ Monitoraggio real-time con CLI
- ✅ TensorBoard integrato con avvio automatico
- ✅ URL TensorBoard con IP Tailscale per accesso remoto
- ✅ Chat interattivo per testare il modello
- ✅ Gestione errori con checkpoint di emergenza
- ✅ Logging dettagliato
- ✅ Pulizia memoria GPU strategica

## 📦 Prerequisiti

- Python 3.10+
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
deep-translator>=1.11.0
```

## 🚀 Avvio Rapido

```bash
# Attiva virtual environment
source venv/bin/activate

# Avvia training (Qwen2.5 Medical Italian)
python3 train_qwen25_medical_italian.py

# In un altro terminal, avvia monitoraggio
python3 monitor_training.py

# Per testare il modello allenato
python3 chat_medical_qwen.py
```

## 📜 Script Principali

### 1. train_qwen25_medical_italian.py

Script principale per il fine-tuning di Qwen2.5-0.5B con dataset medico-italiano.

#### Utilizzo

```bash
# Training completo
python3 train_qwen25_medical_italian.py

# Con TensorBoard (modificato per abilitare logging)
python3 train_qwen25_medical_italian.py  # gia' con report_to="tensorboard"
```

#### Configurazione LoRA

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
```

### 2. chat_medical_qwen.py

Chat interattivo per testare il modello allenato.

```bash
python3 chat_medical_qwen.py
```

#### Esempio d'uso

```
╔══════════════════════════════════════════════════════════╗
║          MEDICAL ITALIAN CHAT - Qwen2.5-0.5B      ║
║     Fine-tuned: Medico + Italiano (15k campioni)    ║
╠══════════════════════════════════════════════════════════╣
║ Comandi:                                               ║
║   /quit o /exit - Esci                                  ║
║   /clear - Pulisci la cronologia                          ║
║   /stats - Mostra statistiche sessione                   ║
╚══════════════════════════════════════════════════════════╝

🩺 Tu: Cos'è il diabete?
💬 RISPOSTA: Il diabete è una malattia che si verifica quando...
```

### 3. monitor_training.py

Script per il monitoraggio in tempo reale del training con TensorBoard integrato.

```bash
python3 monitor_training.py
```

#### Funzionalità

- Mostra progresso training (step, epoch)
- Visualizza metriche (loss, eval_loss, learning_rate)
- Monitor risorse (GPU VRAM, utilization, temperature)
- Trend chart ASCII per loss
- TensorBoard integrato con avvio/arresto automatico
- URL con IP Tailscale per accesso remoto
- Supporto per training resumed da checkpoint
- Aggiornamento automatico ogni 3 secondi

### 4. create_unified_medical_italian_dataset.py

Script per creare il dataset medico-italiano unificato.

```bash
python3 create_unified_medical_italian_dataset.py
```

#### Output

- `datasets/unified_medical_italian_dataset.json` (15,000 campioni)
- Checkpoint ogni 100 campioni per resume
- Dataset tradotto dal medico inglese all'italiano

## ⚙️ Configurazione

### Parametri Training Qwen2.5

```python
# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./output_qwen25_medical_italian_{timestamp}"

# Dataset
DATASET_PATH = "./datasets/unified_medical_italian_dataset.json"

# Training (ottimizzato per MX250 2GB)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
SAVE_STEPS = 100
LOGGING_STEPS = 10
```

### Hardware Limitations

Il training è ottimizzato per:
- **GPU**: NVIDIA MX250 2GB VRAM
- **RAM**: 14GB+
- **Tempo**: ~3-4 ore per training completo (1688 steps)

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
- Stato TensorBoard con URL di accesso
- Log recenti

### TensorBoard

```bash
tensorboard --logdir ./output_qwen25_medical_italian_*/logs --port 6006 --bind_all
```

oppure accedi tramite l'URL mostrato dal monitor.

## 💬 Chat Interattivo

```bash
python3 chat_medical_qwen.py
```

Esempio di domande:
- "Ciao, come stai?"
- "Cos'è il diabete?"
- "Quali sono i sintomi dell'influenza?"
- "Spiega l'apparato digerente"

### Note sul Modello

**Punti di forza:**
- ✅ Risponde in italiano corretto
- ✅ Comprende il contesto medico base
- ✅ VRAM: solo 0.44 GB (molto efficiente)

**Limiti:**
- ⚠️ Modello piccolo (0.5B parametri)
- ⚠️ Training breve (1 epoch, 15k campioni)
- ⚠️ Alcune imprecisioni mediche (atteso per questa configurazione)

**Per migliorare:**
- Più epoch di training (2-3)
- Dataset più grande (50k+ campioni)
- Aumentare LoRA rank (32 invece di 16)

## 📁 Struttura Directory

```
training_llm/
├── train_qwen25_medical_italian.py    # Script training Qwen2.5 Medical
├── chat_medical_qwen.py                # Chat interattivo
├── monitor_training.py                 # Monitor training
├── create_unified_medical_italian_dataset.py  # Dataset creation
├── requirements.txt                    # Dipendenze
├── models/                             # Modelli base
│   └── Qwen2.5-0.5B-Instruct/         # (cache HuggingFace)
├── datasets/                           # Dataset locali
│   ├── unified_medical_italian_dataset.json  # 15k campioni
│   └── italian_clean/                  # Dati italiani puliti
├── output_qwen25_medical_italian_*/    # Output training
│   ├── adapter_model.safetensors       # Adapter LoRA (8.7 MB)
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── checkpoint-*/                  # Checkpoints intermedi
├── README.md                           # Documentazione
└── AGENTS.md                           # Guide per agenti
```

## 🔧 Troubleshooting

### Training non parte

```bash
# Verifica se c'è già un training in corso
ps aux | grep train_qwen

# Termina training esistente
kill <PID>

# Riavvia
python3 train_qwen25_medical_italian.py
```

### OOM (Out of Memory)

```bash
# Riduci batch_size a 1
# Riduci max_seq_length a 128
# Chiudi altre applicazioni GPU
# Usa quantizzazione 4-bit (attiva per default)
```

### Chat non risponde correttamente

Il modello Qwen2.5-0.5B è piccolo (0.5B parametri). Per risposte mediche più accurate:
1. Allenalo per più epoch
2. Usa un dataset più grande
3. Prova un modello più grande (Qwen2.5-1.5B o 3B) se VRAM lo permette

## 📝 Changelog

### 2026-03-23 - Qwen2.5 Medical Italian Training

- **Feat**: Nuovo script `train_qwen25_medical_italian.py`
- **Feat**: Dataset medico-italiano unificato (15k campioni)
- **Feat**: Script `create_unified_medical_italian_dataset.py` con checkpoint/resume
- **Feat**: Chat interattivo `chat_medical_qwen.py`
- **Feat**: Monitor aggiornato per supportare Qwen2.5
- **Result**: Eval loss 1.5548 (-9.0% dal baseline 1.7077)
- **Fix**: Corretto output_dir detection in monitor_training.py
- **Fix**: Rimossi path hardcoded in monitor_training.py

### 2026-03-21 - Migrazione a Pipeline Ottimizzata

- **Feat**: Nuovo script `train_qlora_optimized.py` basato su TRL/SFTTrainer
- **Feat**: Configurazione centralizzata in `config.py` con dataclass
- **Feat**: Target modules completi per LoRA (q/k/v/o_proj)

---

**Note**:
- Il training può richiedere 3-4 ore su hardware limitato
- Utilizza sempre il background mode per training lunghi
- Monitora le risorse con `nvidia-smi` durante il training
- I checkpoint vengono salvati automaticamente ogni 100 step
- TensorBoard si avvia automaticamente con il monitor
