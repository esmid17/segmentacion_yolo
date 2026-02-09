import os
import shutil
import zipfile
from pathlib import Path
import requests
from config import settings

class COCODatasetHandler:
    def __init__(self):
        # Aseguramos que sea un objeto Path
        self.raw_dir = Path(settings.RAW_DATA_DIR)
        self.coco_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

    def check_and_download_data(self):
        """
        Descarga COCO128 y mueve todas las imágenes a la raíz de data/raw.
        """
        # Crear directorio si no existe
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Verificar si ya hay imágenes (buscamos jpg)
        imagenes_existentes = list(self.raw_dir.glob("*.jpg"))
        
        if len(imagenes_existentes) > 5:
            print(f"Ya tienes {len(imagenes_existentes)} imágenes listas. Saltando descarga.")
            return

        print("Carpeta vacía. Iniciando descarga de COCO128")
        self._download_and_extract()

    def _download_and_extract(self):
        zip_path = self.raw_dir / "coco128.zip"
        
        # 1. Descargar
        try:
            response = requests.get(self.coco_url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Descarga completada. Descomprimiendo")
        except Exception as e:
            print(f"Falló la descarga: {e}")
            return

        # 2. Descomprimir
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
        except zipfile.BadZipFile:
            print("El archivo ZIP descargado está corrupto.")
            return

        print("Organizando archivos...")
        
        # Busca recursivamente cualquier .jpg dentro de raw_dir
        # rglob = recursive glob search
        found_images = list(self.raw_dir.rglob("*.jpg"))
        
        if not found_images:
            print("No se encontraron imágenes JPG dentro del ZIP.")
            return

        count = 0
        for img_path in found_images:
            # Definir destino final (directamente en data/raw)
            dest_path = self.raw_dir / img_path.name
            
            # Mover solo si no está ya en el lugar correcto
            if img_path.parent != self.raw_dir:
                shutil.move(str(img_path), str(dest_path))
                count += 1

        print(f"[EXITO] Se movieron {count} imágenes a {self.raw_dir}")

        # 4. Limpieza
        try:
            os.remove(zip_path) # Borrar zip
            # Borrar las carpetas vacías que quedaron (como 'coco128')
            for child in self.raw_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
        except Exception as e:
            print(f"No se pudo limpiar algunas carpetas temporales: {e}")