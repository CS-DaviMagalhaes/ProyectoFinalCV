# file: src/feature_extractor.py

import torch
import torchreid
from torchreid.utils import FeatureExtractor
from torchvision.transforms import functional as F
import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReIDFeatureExtractor:
    def __init__(self, model_name='osnet_x1_0', model_path='', device='cuda'):
        """
        Inicializa el extractor de características de Re-ID.

        Args:
            model_name (str): Nombre del modelo a usar de Torchreid.
            model_path (str): Ruta a los pesos pre-entrenados. Si está vacío, usa los de Torchreid.
            device (str): Dispositivo para correr el modelo ('cuda' o 'cpu').
        """
        self.device = device

        # Verificar si CUDA está disponible
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA no está disponible, usando CPU")
            self.device = 'cpu'

        try:
            # Inicializa el extractor de características de Torchreid
            self.extractor = FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                device=self.device
            )
            logger.info(
                f"Modelo Re-ID '{model_name}' cargado en el dispositivo '{self.device}'.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo Re-ID: {e}")
            raise

    def preprocess_image(self, image_np):
        """
        Pre-procesa una imagen (numpy array) para el modelo Re-ID.

        Args:
            image_np (np.array): Imagen recortada de la persona en formato BGR (de OpenCV).

        Returns:
            torch.Tensor: Tensor de la imagen listo para el modelo.
        """
        try:
            # Verificar que la imagen no esté vacía
            if image_np.size == 0:
                raise ValueError("La imagen está vacía")

            # Convertir BGR a RGB
            image_rgb = image_np[:, :, ::-1].copy()

            # Convertir a Tensor
            image_tensor = F.to_tensor(image_rgb)

            # Redimensionar a las dimensiones esperadas por el modelo (usualmente 256x128)
            image_resized = F.resize(image_tensor, (256, 128), antialias=True)

            # Normalizar con los valores estándar de ImageNet
            image_normalized = F.normalize(
                image_resized,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            # Añadir dimensión de batch
            return image_normalized.unsqueeze(0).to(self.device)

        except Exception as e:
            logger.error(f"Error en el preprocesamiento de la imagen: {e}")
            raise

    @torch.no_grad()
    def extract_features(self, image_np):
        """
        Extrae el vector de características (embedding) de una imagen.

        Args:
            image_np (np.array): Imagen recortada de la persona.

        Returns:
            np.array: Vector de características normalizado.
        """
        try:
            preprocessed_image = self.preprocess_image(image_np)
            features = self.extractor(preprocessed_image)

            # Normalizar el vector de características es crucial para la comparación de coseno
            features = features.cpu().numpy().flatten()

            # Normalización L2
            norm = np.linalg.norm(features)
            normalized_features = features / norm if norm > 0 else features

            return normalized_features

        except Exception as e:
            logger.error(f"Error en la extracción de características: {e}")
            # Retornar un vector cero en caso de error
            return np.zeros(512)  # Tamaño típico de embedding
