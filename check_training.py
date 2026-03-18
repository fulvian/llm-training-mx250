#!/usr/bin/env python3
"""
Script to monitor training status and auto-resume if failed
"""

import os
import subprocess
import time
import sys

LOG_FILE = f"train_auto_{time.strftime('%Y%m%d_%H%M%S')}.log"
OUTPUT_DIR = "./smollm-135m-qlora-output"
MAX_RETRIES = 3
RETRY_DELAY = 60

def main():
    print("===== TRAINING MONITOR =====")
    print(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Archivo de log: {LOG_FILE}")
    print(f"Directorio de salida: {OUTPUT_DIR}")
    
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()
    
    for i in range(1, MAX_RETRIES + 1):
        print(f"\n=== Tentativo {i}/{MAX_RETRIES} ===")
        
        cmd = [
            'python3', '-u', 'train_qlora_local.py'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n=== Inizio tentativo {i} - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
                log_file.write(line)
        
        process.wait()
        
        if process.returncode == 0:
            print("\nTraining completado con successo!")
            break
            
        print(f"Processo terminato con codice {process.returncode}")
        
        if i < MAX_RETRIES:
            print(f"Riprovo in {RETRY_DELAY} secondi...")
            time.sleep(RETRY_DELAY)

    if process.returncode != 0:
        print("\n=== Errore: training non completato dopo tutti i tentativi ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
