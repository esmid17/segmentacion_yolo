import sys
import os

# Aseguramos que Python encuentre nuestros módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import YoloSegmentationModel
from config import settings

def main():
    # 1. Instanciar el modelo
    # La primera vez descargará 'yolov8n-seg.pt' automáticamente
    segmentador = YoloSegmentationModel()

    # 2. Definir una imagen de prueba
    # Puedes poner una imagen tuya en data/raw/prueba.jpg
    # O usar una url de internet para probar rápido:
    imagen_prueba = "https://ultralytics.com/images/bus.jpg" 
    
    # 3. Ejecutar segmentación
    print("--- Iniciando Segmentación ---")
    segmentador.predict(imagen_prueba)
    print("--- Proceso Finalizado ---")

if __name__ == "__main__":
    main()