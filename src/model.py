import os
from ultralytics import YOLO
import cv2
from config import settings

class YoloSegmentationModel:
    def __init__(self, model_path=None):
        # Si no pasamos ruta, usa el definido en settings
        model_name = model_path if model_path else settings.MODEL_NAME
        print(f"Cargando modelo: {model_name}...")
        self.model = YOLO(model_name)

    def predict(self, image_path, save=True):
        """
        Realiza la segmentación en una imagen.
        """
        print(f"Procesando imagen: {image_path}")
        
        # Inferencia
        results = self.model.predict(
            source=image_path,
            conf=settings.CONF_THRESHOLD,
            save=False, # Lo guardaremos manualmente para controlar la ruta
            imgsz=settings.IMG_SIZE
        )
        
        result = results[0]
        
        # Guardar resultado si se solicita
        if save:
            self._save_result(result, image_path)
            
        return result

    def _save_result(self, result, original_path):
        """Método privado para guardar la imagen segmentada en 'outputs'"""
        # Crear nombre de salida
        file_name = os.path.basename(original_path)
        save_path = settings.OUTPUT_DIR / f"pred_{file_name}"
        
        # Renderizar imagen con máscaras
        plotted_img = result.plot()
        
        # Asegurar que el directorio existe
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        
        # Guardar con OpenCV
        cv2.imwrite(str(save_path), plotted_img)
        print(f"Imagen guardada en: {save_path}")
        
        # Opcional: Mostrar en pantalla (presionar 'q' para cerrar)
        # cv2.imshow("YOLO Segmentation", plotted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()