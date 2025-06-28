# file: src/main_reid.py

from feature_extractor import ReIDFeatureExtractor
from ultralytics import YOLO
import json
import logging
import time
import argparse
from collections import defaultdict
import cv2
import numpy as np
import sys
import os
import torch

# Solución para PyTorch 2.7+ - Monkey patch para torch.load
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# Añadir el directorio src al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PersonReIdentificationSystem:
    def __init__(self, config_path=None):
        """
        Inicializa el sistema de re-identificación de personas.

        Args:
            config_path (str): Ruta al archivo de configuración JSON.
        """
        # Cargar configuración
        self.config = self.load_config(config_path)

        # Inicialización de modelos
        logger.info("Inicializando modelos...")
        self.yolo = YOLO(self.config['yolo_model'])
        self.reid_extractor = ReIDFeatureExtractor(
            model_name=self.config['reid_model'],
            device=self.config['device']
        )
        logger.info("Inicialización completa.")

        # Estructuras de datos para tracking
        self.next_global_id = 0
        self.tracked_objects = {}  # {track_id: global_id}
        # {global_id: [embedding, last_seen_frame]}
        self.embeddings_db = defaultdict(list)

        # Métricas
        self.frame_count = 0
        self.total_detections = 0
        self.reidentifications = 0

    def load_config(self, config_path):
        """Carga la configuración desde un archivo JSON o usa valores predeterminados."""
        default_config = {
            'yolo_model': 'yolov8n.pt',
            'reid_model': 'osnet_x1_0',
            'device': 'cuda',
            'similarity_threshold': 0.85,
            'disappearance_limit_seconds': 5,
            'min_detection_confidence': 0.5,
            'output_video': True,
            'save_embeddings': False
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuración cargada desde {config_path}")
            except Exception as e:
                logger.warning(
                    f"Error al cargar configuración: {e}. Usando configuración predeterminada.")

        return default_config

    def calculate_similarity(self, embedding1, embedding2):
        """Calcula la similitud del coseno entre dos embeddings."""
        return np.dot(embedding1, embedding2)

    def clean_old_embeddings(self, current_frame, fps):
        """Elimina embeddings de personas que han desaparecido por mucho tiempo."""
        disappearance_limit_frames = int(
            fps * self.config['disappearance_limit_seconds'])
        ids_to_remove = []

        for global_id, (embedding, last_seen_frame) in self.embeddings_db.items():
            if current_frame - last_seen_frame > disappearance_limit_frames:
                ids_to_remove.append(global_id)

        for global_id in ids_to_remove:
            del self.embeddings_db[global_id]
            logger.debug(
                f"Eliminado embedding de persona con ID global {global_id}")

    def find_best_match(self, current_embedding):
        """
        Encuentra la mejor coincidencia para un embedding en la base de datos.

        Returns:
            tuple: (best_match_id, max_similarity)
        """
        best_match_id = -1
        max_similarity = -1

        for global_id, (stored_embedding, _) in self.embeddings_db.items():
            similarity = self.calculate_similarity(
                current_embedding, stored_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match_id = global_id

        return best_match_id, max_similarity

    def process_detections(self, frame, results, fps):
        """Procesa las detecciones del frame actual."""
        if results[0].boxes.id is None:
            return frame

        # IDs y coordenadas de los tracks en el frame actual
        current_track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        self.total_detections += len(current_track_ids)

        # Procesar cada detección
        for i, track_id in enumerate(current_track_ids):
            # Filtrar detecciones con baja confianza
            if confidences[i] < self.config['min_detection_confidence']:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])

            # Validar coordenadas del bounding box
            if x2 <= x1 or y2 <= y1:
                continue

            # Si el track_id es nuevo en este frame
            if track_id not in self.tracked_objects:
                # Recortar la imagen de la persona
                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                try:
                    # Extraer embedding
                    current_embedding = self.reid_extractor.extract_features(
                        person_crop)

                    # Limpiar embeddings antiguos
                    self.clean_old_embeddings(self.frame_count, fps)

                    # Buscar la mejor coincidencia
                    best_match_id, max_similarity = self.find_best_match(
                        current_embedding)

                    # Decidir si es re-identificación o nueva persona
                    if max_similarity > self.config['similarity_threshold']:
                        # Re-identificación exitosa
                        global_id = best_match_id
                        self.reidentifications += 1
                        logger.info(
                            f"Re-identificación: Track {track_id} -> ID Global {global_id} (similitud: {max_similarity:.3f})")
                    else:
                        # Persona nueva
                        global_id = self.next_global_id
                        self.next_global_id += 1
                        logger.info(
                            f"Nueva persona: Track {track_id} -> ID Global {global_id}")

                    # Actualizar estructuras de datos
                    self.tracked_objects[track_id] = global_id
                    self.embeddings_db[global_id] = [
                        current_embedding, self.frame_count]

                except Exception as e:
                    logger.error(f"Error procesando track {track_id}: {e}")
                    continue

            # Actualizar la última aparición
            if track_id in self.tracked_objects:
                global_id_to_update = self.tracked_objects[track_id]
                if global_id_to_update in self.embeddings_db:
                    self.embeddings_db[global_id_to_update][1] = self.frame_count

            # Visualización
            assigned_id = self.tracked_objects.get(track_id, -1)
            confidence = confidences[i]

            # Color basado en si es re-identificación o nueva persona
            color = (0, 255, 255) if assigned_id in [
                gid for gid, _ in self.embeddings_db.items() if _ != []] else (0, 255, 0)

            # Dibujar bounding box y etiqueta
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {assigned_id} ({confidence:.2f})"

            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                          (x1 + text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

    def run(self, video_path, output_path=None):
        """
        Ejecuta el pipeline de re-identificación en un video.

        Args:
            video_path (str): Ruta al video de entrada.
            output_path (str): Ruta para guardar el video procesado (opcional).
        """
        # Verificar que el video existe
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

        # Configurar writer para video de salida
        out = None
        if output_path and self.config['output_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Variables para métricas de rendimiento
        start_time = time.time()
        processing_times = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                frame_start_time = time.time()

                # Detección y tracking con YOLO
                results = self.yolo.track(
                    frame,
                    persist=True,
                    classes=0,  # Solo personas
                    verbose=False,
                    conf=self.config['min_detection_confidence']
                )

                # Procesar detecciones y aplicar re-identificación
                processed_frame = self.process_detections(frame, results, fps)

                # Añadir información del frame
                info_text = f"Frame: {self.frame_count}/{total_frames} | Personas activas: {len(self.tracked_objects)} | Re-IDs: {self.reidentifications}"
                cv2.putText(processed_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Guardar frame procesado
                if out is not None:
                    out.write(processed_frame)

                # Mostrar video en tiempo real
                cv2.imshow("Re-ID Tracking", processed_frame)

                # Calcular tiempo de procesamiento
                frame_processing_time = time.time() - frame_start_time
                processing_times.append(frame_processing_time)

                # Control de salida
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Interrupción por parte del usuario")
                    break
                elif key == ord('s') and self.config['save_embeddings']:
                    self.save_embeddings_to_file()

                # Mostrar progreso cada 100 frames
                if self.frame_count % 100 == 0:
                    avg_processing_time = np.mean(processing_times[-100:])
                    logger.info(
                        f"Progreso: {self.frame_count}/{total_frames} frames - Tiempo/frame: {avg_processing_time:.3f}s")

        except KeyboardInterrupt:
            logger.info("Procesamiento interrumpido por el usuario")

        finally:
            # Liberar recursos
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

            # Mostrar estadísticas finales
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0

            logger.info("=== ESTADÍSTICAS FINALES ===")
            logger.info(f"Frames procesados: {self.frame_count}")
            logger.info(f"Tiempo total: {total_time:.2f} segundos")
            logger.info(f"FPS promedio: {avg_fps:.2f}")
            logger.info(f"Total de detecciones: {self.total_detections}")
            logger.info(
                f"Re-identificaciones exitosas: {self.reidentifications}")
            logger.info(f"Personas únicas detectadas: {self.next_global_id}")

            if processing_times:
                logger.info(
                    f"Tiempo de procesamiento promedio por frame: {np.mean(processing_times):.3f}s")

    def save_embeddings_to_file(self):
        """Guarda los embeddings actuales a un archivo."""
        embeddings_data = {}
        for global_id, (embedding, last_seen) in self.embeddings_db.items():
            embeddings_data[global_id] = {
                'embedding': embedding.tolist(),
                'last_seen_frame': last_seen
            }

        filename = f"embeddings_frame_{self.frame_count}.json"
        with open(filename, 'w') as f:
            json.dump(embeddings_data, f, indent=2)

        logger.info(f"Embeddings guardados en {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Re-identificación de Personas')
    parser.add_argument('--video', '-v', required=True,
                        help='Ruta al video de entrada')
    parser.add_argument(
        '--output', '-o', help='Ruta para guardar el video procesado')
    parser.add_argument(
        '--config', '-c', help='Ruta al archivo de configuración JSON')

    args = parser.parse_args()

    try:
        # Inicializar sistema
        reid_system = PersonReIdentificationSystem(config_path=args.config)

        # Ejecutar procesamiento
        reid_system.run(video_path=args.video, output_path=args.output)

    except Exception as e:
        logger.error(f"Error en el sistema: {e}")
        raise


if __name__ == "__main__":
    main()
