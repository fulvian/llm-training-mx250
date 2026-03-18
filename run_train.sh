#!/bin/bash

# Script de entrenamiento con logging
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
OUTPUT_DIR="./smollm-135m-qlora-output"

echo "===== INICIANDO ENTRENAMIENTO ====="
echo "Fecha y hora: $(date)"
echo "Archivo de log: $LOG_FILE"
echo "Directorio de salida: $OUTPUT_DIR"
echo "-----------------------------------"

# Limpiar directorio de salida anterior
if [ -d "$OUTPUT_DIR" ]; then
    echo "Limpiando directorio de salida anterior..."
    rm -rf "$OUTPUT_DIR"
fi

# Ejecutar entrenamiento
python3 -u train_qlora.py 2>&1 | tee "$LOG_FILE"

echo "-----------------------------------"
echo "Entrenamiento terminado: $(date)"
