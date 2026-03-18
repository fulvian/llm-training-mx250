#!/bin/bash

LOG_FILE="train_local_stable_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="./smollm-135m-qlora-output"

echo "===== TRAINING LOCALE STABILE ====="
echo "Fecha y hora: $(date)"
echo "Archivo de log: $LOG_FILE"
echo "Directorio de salida: $OUTPUT_DIR"
echo "Modello local: ./models/SmolLM-135M-Instruct"
echo "Dataset local: ./datasets/databricks-dolly-15k"
if [ -d "$OUTPUT_DIR/checkpoint-50" ]; then
    echo "Riprendendo da checkpoint: $OUTPUT_DIR/checkpoint-50"
fi
echo "-----------------------------------"

python3 -u train_qlora_local.py --no_cuda=False --per_device_train_batch_size=1 --gradient_accumulation_steps=8 --learning_rate=8e-5 --max_steps=100 --save_steps=25 --eval_steps=25 --warmup_steps=10 --num_train_epochs=0.5 2>&1 | tee "$LOG_FILE"

echo "-----------------------------------"
echo "Training terminado: $(date)"
