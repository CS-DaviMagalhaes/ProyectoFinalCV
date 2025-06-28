# file: src/feature_extractor_v2.py

import torch
import torchreid
from torchreid.utils import FeatureExtractor
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import logging
from collections import deque
import cv2

# Fix para PyTorch 2.7 - Monkey patch torch.load
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustReIDFeatureExtractor:
    def __init__(self, model_name='osnet_x1_0', model_path='', device='cuda',
                 embedding_buffer_size=5, quality_threshold=0.3):
        """
        Extractor de características robusto con buffer de embeddings.

        Args:
            model_name (str): Nombre del modelo Torchreid
            model_path (str): Ruta a pesos pre-entrenados
            device (str): Dispositivo ('cuda' o 'cpu')
            embedding_buffer_size (int): Tamaño del buffer para promediar embeddings
            quality_threshold (float): Umbral mínimo de calidad para embeddings
        """
        self.device = device
        self.embedding_buffer_size = embedding_buffer_size
        self.quality_threshold = quality_threshold

        # Verificar CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU")
            self.device = 'cpu'

        try:
            # Inicializar extractor
            self.extractor = FeatureExtractor(
                model_name=model_name,
                model_path=model_path,
                device=self.device
            )

            # CRÍTICO: Poner modelo en modo evaluación
            self.extractor.model.eval()

            # Buffer de embeddings por persona {global_id: deque([embedding, quality_score])}
            self.embedding_buffers = {}

            # Estadísticas para umbral adaptativo
            self.similarity_history = deque(maxlen=100)
            self.adaptive_threshold = 0.85

            logger.info(
                f"ReID robusto '{model_name}' cargado en '{self.device}'")

        except Exception as e:
            logger.error(f"Error cargando modelo ReID: {e}")
            raise

    def calculate_image_quality(self, image_np, bbox_area):
        """
        Calcula un score de calidad para la imagen de la persona.

        Args:
            image_np: Imagen recortada de la persona
            bbox_area: Área del bounding box

        Returns:
            float: Score de calidad [0-1]
        """
        if image_np.size == 0:
            return 0.0

        # Factor 1: Tamaño del bounding box (áreas grandes = mejor calidad)
        # Normalizar a imagen 150x400
        area_score = min(bbox_area / (150 * 400), 1.0)

        # Factor 2: Nitidez (varianza del Laplaciano)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)  # Normalizar

        # Factor 3: Contraste (std de la imagen)
        contrast_score = min(gray.std() / 80, 1.0)

        # Factor 4: Aspect ratio (personas deben ser más altas que anchas)
        h, w = image_np.shape[:2]
        aspect_ratio = h / w if w > 0 else 0
        aspect_score = 1.0 if 1.5 <= aspect_ratio <= 4.0 else 0.5

        # Combinar scores
        quality = (area_score * 0.3 + sharpness_score * 0.3 +
                   contrast_score * 0.2 + aspect_score * 0.2)

        return quality

    def preprocess_image(self, image_np):
        """
        Preprocesado mejorado para el modelo ReID.
        """
        try:
            if image_np.size == 0:
                raise ValueError("Imagen vacía")

            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Convertir a tensor
            image_tensor = F.to_tensor(image_rgb)

            # Resize optimizado sin antialias (más rápido y consistente)
            image_resized = F.resize(
                image_tensor,
                (256, 128),
                interpolation=InterpolationMode.BILINEAR,
                antialias=False
            )

            # Normalización estándar ImageNet
            image_normalized = F.normalize(
                image_resized,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            return image_normalized.unsqueeze(0).to(self.device)

        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise

    @torch.no_grad()
    def extract_features(self, image_np, bbox_area=None, confidence=1.0):
        """
        Extrae embedding con evaluación de calidad.

        Args:
            image_np: Imagen recortada
            bbox_area: Área del bounding box
            confidence: Confianza de la detección

        Returns:
            tuple: (embedding_normalizado, quality_score)
        """
        try:
            # Calcular calidad de la imagen
            if bbox_area is None:
                bbox_area = image_np.shape[0] * image_np.shape[1]

            quality = self.calculate_image_quality(image_np, bbox_area)

            # Filtrar imágenes de muy baja calidad
            if quality < self.quality_threshold:
                logger.debug(
                    f"Imagen rechazada por baja calidad: {quality:.3f}")
                return np.zeros(512), quality

            # Preprocesar y extraer
            preprocessed = self.preprocess_image(image_np)
            features = self.extractor(preprocessed)

            # Normalizar L2
            features_np = features.cpu().numpy().flatten()
            norm = np.linalg.norm(features_np)
            normalized_features = features_np / norm if norm > 0 else features_np

            # Combinar calidad con confianza de detección
            final_quality = quality * confidence

            return normalized_features, final_quality

        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return np.zeros(512), 0.0

    def update_embedding_buffer(self, global_id, embedding, quality):
        """
        Actualiza el buffer de embeddings para una persona.
        """
        if global_id not in self.embedding_buffers:
            self.embedding_buffers[global_id] = deque(
                maxlen=self.embedding_buffer_size)

        # Añadir nuevo embedding con su calidad
        self.embedding_buffers[global_id].append((embedding, quality))

    def get_averaged_embedding(self, global_id):
        """
        Obtiene el embedding promediado ponderado por calidad.

        Returns:
            np.array: Embedding promedio ponderado
        """
        if global_id not in self.embedding_buffers or not self.embedding_buffers[global_id]:
            return None

        embeddings_and_qualities = list(self.embedding_buffers[global_id])

        # Extraer embeddings y calidades
        embeddings = np.array([eq[0] for eq in embeddings_and_qualities])
        qualities = np.array([eq[1] for eq in embeddings_and_qualities])

        # Promedio ponderado por calidad
        if qualities.sum() > 0:
            weights = qualities / qualities.sum()
            averaged_embedding = np.average(
                embeddings, axis=0, weights=weights)
        else:
            averaged_embedding = np.mean(embeddings, axis=0)

        # Re-normalizar
        norm = np.linalg.norm(averaged_embedding)
        return averaged_embedding / norm if norm > 0 else averaged_embedding

    def update_adaptive_threshold(self, similarity_score, is_same_person):
        """
        Actualiza el umbral de similitud de forma adaptativa.

        Args:
            similarity_score: Score de similitud calculado
            is_same_person: Si realmente es la misma persona (para aprendizaje)
        """
        self.similarity_history.append((similarity_score, is_same_person))

        # Recalcular umbral cada 20 muestras
        if len(self.similarity_history) >= 20 and len(self.similarity_history) % 20 == 0:
            positive_similarities = [
                s for s, label in self.similarity_history if label]
            negative_similarities = [
                s for s, label in self.similarity_history if not label]

            if positive_similarities and negative_similarities:
                # Umbral óptimo entre la media de positivos y negativos
                pos_mean = np.mean(positive_similarities)
                neg_mean = np.mean(negative_similarities)
                self.adaptive_threshold = (pos_mean + neg_mean) / 2

                logger.debug(
                    f"Umbral adaptativo actualizado: {self.adaptive_threshold:.3f}")

    def calculate_similarity_robust(self, embedding1, embedding2):
        """
        Calcula similitud robusta entre embeddings.
        """
        # Similitud coseno estándar
        cosine_sim = np.dot(embedding1, embedding2)

        # TODO: Implementar distancia de Mahalanobis cuando tengamos suficientes datos

        return cosine_sim

    def get_current_threshold(self):
        """Retorna el umbral adaptativo actual."""
        return self.adaptive_threshold

    def cleanup_old_buffers(self, active_global_ids):
        """
        Limpia buffers de personas que ya no están activas.

        Args:
            active_global_ids: Set de IDs globales actualmente activos
        """
        inactive_ids = set(self.embedding_buffers.keys()) - \
            set(active_global_ids)

        for global_id in inactive_ids:
            del self.embedding_buffers[global_id]

        if inactive_ids:
            logger.debug(f"Limpiados {len(inactive_ids)} buffers inactivos")
