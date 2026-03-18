#!/bin/bash

LOG_FILE="train_retry_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="./smollm-135m-qlora-output"
MAX_RETRIES=3
RETRY_DELAY=60

echo "===== TRAINING WITH RETRY ====="
echo "Fecha y hora: $(date)"
echo "Archivo de log: $LOG_FILE"
echo "Directorio de salida: $OUTPUT_DIR"
echo "Modello local: ./models/SmolLM-135M-Instruct"
echo "Dataset local: ./datasets/databricks-dolly-15k"
echo "Max retries: $MAX_RETRIES"
if [ -d "$OUTPUT_DIR/checkpoint-50" ]; then
    echo "Riprendendo da checkpoint: $OUTPUT_DIR/checkpoint-50"
fi
echo "-----------------------------------"

for (( i=1; i<=MAX_RETRIES; i++ ))
do
    echo "=== Tentativo $i/$MAX_RETRIES ==="
    python3 -u train_qlora_local.py 2>&1 | tee -a "$LOG_FILE"
    
    # Verifica se il training è completato
    if grep -q "Training completado" "$LOG_FILE" || [ -f "$OUTPUT_DIR/final_model.pt" ]; then
        echo "Training completado con successo!"
        exit 0
    fi
    
    # Verifica se il training è interrotto
    if ! ps aux | grep -E "(python|train)" | grep -v grep > /dev/null; then
        echo "Training interrotto. Riprovo in $RETRY_DELAY secondi..."
        sleep $RETRY_DELAY
    fi
done

echo "=== Massimo numero di tentativi raggiunto ==="
echo "Verifica log per dettagli: $LOG_FILE"
exit 1
