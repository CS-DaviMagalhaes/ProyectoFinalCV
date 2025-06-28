# file: src/main_reid_v2.py

from feature_extractor_v2 import RobustReIDFeatureExtractor
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import logging
import time
from datetime import datetime
from collections import defaultdict, deque
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Fix para PyTorch 2.7 - Monkey patch torch.load
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# Importar YOLO

# Importar nuestro extractor mejorado

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustPersonReIDSystem:
    """Sistema robusto de Re-Identificación de Personas."""

    def __init__(self, config_path: str = "config/default_config.json"):
        """
        Inicializa el sistema robusto de Re-ID.

        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)

        # Inicializar componentes
        self.yolo_model = None
        self.reid_extractor = None

        # Tracking mejorado
        # {track_id: deque([(x,y,w,h,conf,frame)])}
        self.track_history = defaultdict(deque)
        self.track_to_global_id = {}  # {track_id: global_id}
        self.global_id_counter = 0

        # Estadísticas
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_reidentifications': 0,
            'processing_times': deque(maxlen=100),
            'similarity_scores': deque(maxlen=1000)
        }

        # Memoria persistente
        self.memory_file = Path(self.config.get(
            'memory_file', 'reid_memory.json'))
        self.load_memory()

        self._setup_models()

    def _load_config(self, config_path: str) -> dict:
        """Carga configuración desde archivo JSON."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            # Configuración por defecto robusta
            return {
                "yolo_model": "yolov8n.pt",
                "yolo_conf_threshold": 0.4,
                "yolo_iou_threshold": 0.5,
                "reid_model": "osnet_x1_0",
                "reid_device": "cpu",
                "embedding_buffer_size": 7,
                "quality_threshold": 0.25,
                "similarity_threshold": 0.75,
                "max_memory_size": 1000,
                "track_memory_frames": 30,
                "cleanup_interval": 100
            }

    def _setup_models(self):
        """Inicializa los modelos YOLO y ReID."""
        try:
            # Cargar YOLO con configuración robusta
            self.yolo_model = YOLO(self.config['yolo_model'])
            logger.info(f"YOLO cargado: {self.config['yolo_model']}")

            # Cargar extractor ReID robusto
            self.reid_extractor = RobustReIDFeatureExtractor(
                model_name=self.config['reid_model'],
                device=self.config['reid_device'],
                embedding_buffer_size=self.config['embedding_buffer_size'],
                quality_threshold=self.config['quality_threshold']
            )
            logger.info("Sistema ReID robusto inicializado")

        except Exception as e:
            logger.error(f"Error inicializando modelos: {e}")
            raise

    def load_memory(self):
        """Carga memoria persistente desde disco."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    self.global_id_counter = memory_data.get(
                        'global_id_counter', 0)
                    logger.info(
                        f"Memoria cargada. Próximo ID global: {self.global_id_counter}")
            except Exception as e:
                logger.warning(f"No se pudo cargar memoria: {e}")

    def save_memory(self):
        """Guarda memoria persistente a disco."""
        try:
            memory_data = {
                'global_id_counter': self.global_id_counter,
                'timestamp': datetime.now().isoformat(),
                'stats': dict(self.stats)
            }
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"No se pudo guardar memoria: {e}")

    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta y rastrea personas en el frame.

        Args:
            frame: Frame de video

        Returns:
            Lista de detecciones con tracking mejorado
        """
        try:
            # Detección YOLO con filtros robustos
            results = self.yolo_model.track(
                frame,
                persist=True,
                conf=self.config['yolo_conf_threshold'],
                iou=self.config['yolo_iou_threshold'],
                classes=[0],  # Solo personas
                verbose=False
            )

            detections = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id

                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy().astype(int)

                    for box, conf, track_id in zip(boxes, confidences, track_ids):
                        x_center, y_center, width, height = box

                        # Filtrar detecciones muy pequeñas
                        if width < 30 or height < 50:
                            continue

                        # Convertir a formato x1,y1,x2,y2
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)

                        # Validar bounds
                        h, w = frame.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))

                        detection = {
                            'track_id': int(track_id),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'area': (x2-x1) * (y2-y1),
                            'center': (x_center, y_center)
                        }

                        detections.append(detection)

                        # Actualizar historial de tracking
                        self.track_history[track_id].append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'frame': self.stats['total_frames'],
                            'timestamp': time.time()
                        })

                        # Limitar historial
                        if len(self.track_history[track_id]) > self.config['track_memory_frames']:
                            self.track_history[track_id].popleft()

            return detections

        except Exception as e:
            logger.error(f"Error en detección y tracking: {e}")
            return []

    def is_stable_track(self, track_id: int, min_frames: int = 3) -> bool:
        """
        Verifica si un track es estable para hacer Re-ID.

        Args:
            track_id: ID del track
            min_frames: Mínimo de frames para considerar estable

        Returns:
            bool: True si el track es estable
        """
        if track_id not in self.track_history:
            return False

        history = self.track_history[track_id]

        if len(history) < min_frames:
            return False

        # Verificar consistencia de tamaño
        recent_areas = [h['bbox'][2] * h['bbox'][3]
                        for h in list(history)[-min_frames:]]
        area_variance = np.var(recent_areas)
        area_mean = np.mean(recent_areas)

        # Track estable si la varianza es baja
        return (area_variance / area_mean) < 0.3 if area_mean > 0 else False

    def perform_reidentification(self, frame: np.ndarray, detections: List[Dict]) -> Dict[int, int]:
        """
        Realiza re-identificación robusta de personas.

        Args:
            frame: Frame actual
            detections: Lista de detecciones

        Returns:
            Dict mapeando track_id -> global_id
        """
        results = {}

        for detection in detections:
            track_id = detection['track_id']

            # Solo procesar tracks estables
            if not self.is_stable_track(track_id):
                continue

            # Si ya tiene ID global, mantenerlo
            if track_id in self.track_to_global_id:
                results[track_id] = self.track_to_global_id[track_id]
                continue

            # Extraer región de la persona
            x1, y1, x2, y2 = detection['bbox']
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            # Extraer embedding con calidad
            embedding, quality = self.reid_extractor.extract_features(
                person_crop,
                bbox_area=detection['area'],
                confidence=detection['confidence']
            )

            # Filtrar embeddings de baja calidad
            if quality < self.config['quality_threshold']:
                continue

            # Buscar coincidencia con personas existentes
            best_match_id = None
            best_similarity = 0.0
            current_threshold = self.reid_extractor.get_current_threshold()

            for existing_global_id in self.reid_extractor.embedding_buffers.keys():
                averaged_embedding = self.reid_extractor.get_averaged_embedding(
                    existing_global_id)

                if averaged_embedding is not None:
                    similarity = self.reid_extractor.calculate_similarity_robust(
                        embedding, averaged_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = existing_global_id

            # Decidir asignación
            if best_similarity >= current_threshold:
                # Re-identificación exitosa
                global_id = best_match_id
                self.track_to_global_id[track_id] = global_id
                self.stats['total_reidentifications'] += 1

                logger.info(f"Re-ID: Track {track_id} → Global ID {global_id} "
                            f"(similitud: {best_similarity:.3f})")

                # Actualizar umbral adaptativo (re-identificación correcta)
                self.reid_extractor.update_adaptive_threshold(
                    best_similarity, True)

            else:
                # Nueva persona
                global_id = self.global_id_counter
                self.global_id_counter += 1
                self.track_to_global_id[track_id] = global_id

                logger.info(
                    f"Nueva persona: Track {track_id} → Global ID {global_id}")

            # Actualizar buffer de embeddings
            self.reid_extractor.update_embedding_buffer(
                global_id, embedding, quality)
            results[track_id] = global_id

            # Registrar estadísticas
            self.stats['similarity_scores'].append(best_similarity)

        return results

    def cleanup_memory(self):
        """Limpia memoria antigua para evitar fuga de memoria."""
        current_time = time.time()

        # Limpiar tracks inactivos
        active_tracks = set()
        for track_id, history in self.track_history.items():
            # 30 segundos
            if history and (current_time - history[-1]['timestamp']) < 30:
                active_tracks.add(track_id)

        # Remover tracks inactivos
        inactive_tracks = set(self.track_history.keys()) - active_tracks
        for track_id in inactive_tracks:
            del self.track_history[track_id]
            if track_id in self.track_to_global_id:
                del self.track_to_global_id[track_id]

        # Limpiar buffers de embeddings
        active_global_ids = set(self.track_to_global_id.values())
        self.reid_extractor.cleanup_old_buffers(active_global_ids)

        if inactive_tracks:
            logger.debug(f"Limpiados {len(inactive_tracks)} tracks inactivos")

    def process_video(self, video_path: str, output_path: str = None,
                      display: bool = True, save_video: bool = False):
        """
        Procesa video completo con Re-ID robusto.

        Args:
            video_path: Ruta del video de entrada
            output_path: Ruta del video de salida (opcional)
            display: Mostrar ventana en tiempo real
            save_video: Guardar video procesado
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Procesando video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

        # Configurar salida de video
        out = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start = time.time()

                # Detectar y rastrear
                detections = self.detect_and_track(frame)

                # Re-identificación
                reid_results = self.perform_reidentification(frame, detections)

                # Visualizar resultados
                display_frame = self.visualize_results(
                    frame, detections, reid_results)

                # Estadísticas de procesamiento
                processing_time = time.time() - frame_start
                self.stats['processing_times'].append(processing_time)
                self.stats['total_frames'] += 1
                self.stats['total_detections'] += len(detections)

                # Mostrar información
                if frame_count % 30 == 0:  # Cada segundo
                    avg_fps = 1.0 / \
                        np.mean(self.stats['processing_times']
                                ) if self.stats['processing_times'] else 0
                    progress = (frame_count / total_frames) * 100

                    logger.info(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                                f"FPS: {avg_fps:.1f} - Detecciones: {len(detections)} - "
                                f"IDs activos: {len(self.reid_extractor.embedding_buffers)}")

                # Guardar/mostrar frame
                if save_video and out:
                    out.write(display_frame)

                if display:
                    cv2.imshow('Person Re-ID Robusto', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Guardar estadísticas
                        self.print_statistics()

                # Limpieza periódica
                if frame_count % self.config['cleanup_interval'] == 0:
                    self.cleanup_memory()

                frame_count += 1

        except KeyboardInterrupt:
            logger.info("Procesamiento interrumpido por el usuario")

        finally:
            # Limpiar recursos
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()

            # Guardar memoria
            self.save_memory()

            # Estadísticas finales
            total_time = time.time() - start_time
            logger.info(f"Procesamiento completado en {total_time:.2f}s")
            self.print_statistics()

    def visualize_results(self, frame: np.ndarray, detections: List[Dict],
                          reid_results: Dict[int, int]) -> np.ndarray:
        """
        Visualiza resultados de detección y re-identificación.
        """
        display_frame = frame.copy()

        for detection in detections:
            track_id = detection['track_id']
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            # Color basado en Global ID
            global_id = reid_results.get(track_id, -1)
            color = self.get_color_for_id(global_id)

            # Dibujar bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta
            label = f"ID:{global_id} T:{track_id} {confidence:.2f}"

            # Información adicional
            if track_id in self.reid_extractor.embedding_buffers:
                buffer_size = len(
                    self.reid_extractor.embedding_buffers[track_id])
                label += f" ({buffer_size})"

            # Dibujar etiqueta con fondo
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Información del sistema
        info_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Detecciones: {len(detections)}",
            f"IDs Activos: {len(self.reid_extractor.embedding_buffers)}",
            f"Umbral: {self.reid_extractor.get_current_threshold():.3f}",
        ]

        if self.stats['processing_times']:
            fps = 1.0 / np.mean(self.stats['processing_times'])
            info_text.append(f"FPS: {fps:.1f}")

        y_offset = 30
        for text in info_text:
            cv2.putText(display_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25

        return display_frame

    def get_color_for_id(self, global_id: int) -> Tuple[int, int, int]:
        """Genera color consistente para un ID global."""
        if global_id < 0:
            return (128, 128, 128)  # Gris para IDs no asignados

        # Generar colores distintivos
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128),
            (128, 255, 0), (0, 128, 255), (128, 0, 255), (255, 128, 128)
        ]

        return colors[global_id % len(colors)]

    def print_statistics(self):
        """Imprime estadísticas del sistema."""
        logger.info("=== ESTADÍSTICAS DEL SISTEMA ===")
        logger.info(f"Frames procesados: {self.stats['total_frames']}")
        logger.info(f"Total detecciones: {self.stats['total_detections']}")
        logger.info(
            f"Re-identificaciones: {self.stats['total_reidentifications']}")
        logger.info(f"Personas únicas: {self.global_id_counter}")
        logger.info(
            f"IDs activos en memoria: {len(self.reid_extractor.embedding_buffers)}")

        if self.stats['processing_times']:
            avg_fps = 1.0 / np.mean(self.stats['processing_times'])
            logger.info(f"FPS promedio: {avg_fps:.2f}")

        if self.stats['similarity_scores']:
            avg_similarity = np.mean(self.stats['similarity_scores'])
            logger.info(f"Similitud promedio: {avg_similarity:.3f}")

        logger.info(
            f"Umbral actual: {self.reid_extractor.get_current_threshold():.3f}")
        logger.info("================================")
