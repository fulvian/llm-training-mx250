#!/usr/bin/env python3
"""
Test veloce per verificare training + monitor.
100 campioni, 1 epoca.
"""

import subprocess
import sys
import time

# Config
MAX_SAMPLES = 100
NUM_EPOCHS = 1

print(f"Test con {MAX_SAMPLES} campioni, {NUM_EPOCHS} epoca...")
print("Avvio training in background...")

# Avvia training in background
cmd = [
    sys.executable,
    "train_qlora_optimized.py",
    "--max_samples",
    str(MAX_SAMPLES),
    "--num_epochs",
    str(NUM_EPOCHS),
]

process = subprocess.Popen(cmd)
print(f"Training avviato con PID: {process.pid}")
print("Attendo 30 secondi per il primo output...")
print("Puoi monitorare con: python3 monitor_training.py")
print("")

# Attendi un po' di output
time.sleep(30)

# Verifica se il processo è ancora in esecuzione
if process.poll() is None:
    print("Processo terminato!")
    sys.exit(1)

print("Processo in esecuzione. Primi output:")

# Mostra ultime righe del log
for i in range(3):
    time.sleep(10)
    try:
        result = subprocess.run(
            ["tail", "-5", "train_qlora_optimized.log"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except Exception as e:
        print(f"Errore: {e}")
