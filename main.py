import sys
import os
import shutil
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import YoloSegmentationModel
from src.dataset_manager import COCODatasetHandler
from config import settings

# 2. Funcion para limpiar
def limpiar_outputs():
    """Borra todo el contenido de la carpeta de salida."""
    folder = settings.OUTPUT_DIR
    
    # Si la carpeta existe, la borramos completa
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"[LIMPIEZA] Se eliminó el historial en: {folder}")
    
    # La volvemos a crear vacía
    os.makedirs(folder)

def main():
    limpiar_outputs()

    print("--- 1. Verificando Dataset COCO ---")
    dataset_handler = COCODatasetHandler()
    # ... resto del código ...

def main():
    # 1. Preparar Datos
    print("--- 1. Verificando Dataset COCO ---")
    dataset_handler = COCODatasetHandler()
    dataset_handler.check_and_download_data()

    # 2. Instanciar Modelo
    print("\n--- 2. Cargando Modelo ---")
    segmentador = YoloSegmentationModel()

    # 3. Listar imágenes
    carpeta_imagenes = settings.RAW_DATA_DIR
    
    # Obtenemos TODAS las imágenes (son 128 en total)
    todas_las_imagenes = [f for f in os.listdir(carpeta_imagenes) if f.lower().endswith('.jpg')]
    
    if not todas_las_imagenes:
        print("[ERROR] No se encontraron imágenes.")
        return

    # Imágenes random
    random.shuffle(todas_las_imagenes)
    
    # Ahora tomamos las primeras 5, pero como están mezcladas, serán diferentes cada vez
    imagenes_a_procesar = todas_las_imagenes[:5]

    print(f"\n--- 3. Procesando 5 imágenes aleatorias (de {len(todas_las_imagenes)} disponibles) ---")

    # 4. Bucle de procesamiento
    for i, imagen_nombre in enumerate(imagenes_a_procesar):
        ruta_completa = carpeta_imagenes / imagen_nombre
        print(f" [{i+1}/5] Segmentando: {imagen_nombre}...")
        segmentador.predict(ruta_completa)

    print(f"\n[FIN] Revisa tus resultados nuevos en: {settings.OUTPUT_DIR}")

if __name__ == "__main__":
    main()