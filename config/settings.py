import os
from pathlib import Path

# 1. Definir la ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Definir rutas a tus carpetas según tu imagen
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "outputs"  # Coincide con tu captura
YAML_PATH = DATA_DIR / "coco.yaml"

# 3. Configuración del Modelo
MODEL_NAME = "yolov8n-seg.pt" # Modelo nano para segmentación
EPOCHS = 10
IMG_SIZE = 640
CONF_THRESHOLD = 0.5