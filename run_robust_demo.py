#!/usr/bin/env python3
"""
Demo robusto del sistema de Person Re-Identification mejorado.
Incluye múltiples perfiles de configuración y opciones avanzadas.
"""

from main_reid_v2 import RobustPersonReIDSystem
import logging
import argparse
import sys
import os
from pathlib import Path

# Agregar src al path ANTES de importar módulos locales
sys.path.append(str(Path(__file__).parent / "src"))


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Demo robusto de Person Re-Identification con YOLOv8 + Torchreid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Modo estándar (balanceado):
   python run_robust_demo.py --video data/video/test2.mp4

2. Modo alta precisión:
   python run_robust_demo.py --video data/video/test2.mp4 --config config/high_precision_config.json

3. Modo baja latencia:
   python run_robust_demo.py --video data/video/test2.mp4 --config config/low_latency_config.json

4. Guardar video de salida:
   python run_robust_demo.py --video data/video/test2.mp4 --output output_robust.mp4 --save-video

5. Solo procesar (sin mostrar ventana):
   python run_robust_demo.py --video data/video/test2.mp4 --no-display

6. Usar cámara web:
   python run_robust_demo.py --camera 0
        """
    )

    # Argumentos principales
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='Ruta al archivo de video de entrada'
    )

    parser.add_argument(
        '--camera', '-c',
        type=int,
        help='ID de la cámara web (ej: 0 para cámara por defecto)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/robust_config.json',
        help='Archivo de configuración (default: config/robust_config.json)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Ruta para guardar video de salida (opcional)'
    )

    # Opciones de procesamiento
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Guardar video procesado (requiere --output)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='No mostrar ventana de video en tiempo real'
    )

    # Perfiles predefinidos
    parser.add_argument(
        '--profile',
        choices=['robust', 'precision', 'fast'],
        help='Usar perfil predefinido (robust/precision/fast)'
    )

    # Opciones avanzadas
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        help='Forzar dispositivo específico'
    )

    parser.add_argument(
        '--yolo-model',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
        help='Modelo YOLO específico'
    )

    parser.add_argument(
        '--reid-model',
        choices=['osnet_x0_5', 'osnet_x1_0', 'osnet_x0_75'],
        help='Modelo ReID específico'
    )

    parser.add_argument(
        '--embedding-buffer',
        type=int,
        help='Tamaño del buffer de embeddings'
    )

    parser.add_argument(
        '--quality-threshold',
        type=float,
        help='Umbral mínimo de calidad de imagen'
    )

    parser.add_argument(
        '--similarity-threshold',
        type=float,
        help='Umbral de similitud para re-identificación'
    )

    # Opciones de debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Activar modo debug con logging detallado'
    )

    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Solo mostrar estadísticas al final (no procesar video)'
    )

    args = parser.parse_args()

    # Validaciones
    if not args.video and args.camera is None:
        parser.error("Debe especificar --video o --camera")

    if args.video and args.camera is not None:
        parser.error("No puede usar --video y --camera al mismo tiempo")

    if args.save_video and not args.output:
        parser.error("--save-video requiere especificar --output")

    # Configurar logging si debug
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Modo debug activado")

    # Determinar archivo de configuración
    config_file = args.config

    if args.profile:
        profile_configs = {
            'robust': 'config/robust_config.json',
            'precision': 'config/high_precision_config.json',
            'fast': 'config/low_latency_config.json'
        }
        config_file = profile_configs[args.profile]
        logger.info(f"Usando perfil '{args.profile}': {config_file}")

    # Verificar archivos
    if not Path(config_file).exists():
        logger.error(f"Archivo de configuración no encontrado: {config_file}")
        logger.info("Archivos de configuración disponibles:")
        config_dir = Path("config")
        if config_dir.exists():
            for config in config_dir.glob("*.json"):
                logger.info(f"  - {config}")
        return 1

    if args.video and not Path(args.video).exists():
        logger.error(f"Archivo de video no encontrado: {args.video}")
        return 1

    try:
        # Inicializar sistema
        logger.info("=== INICIALIZANDO SISTEMA ROBUSTO ===")
        logger.info(f"Configuración: {config_file}")

        system = RobustPersonReIDSystem(config_path=config_file)

        # Override configuraciones si se especificaron
        if args.device:
            system.config['reid_device'] = args.device
            logger.info(f"Dispositivo forzado: {args.device}")

        if args.yolo_model:
            system.config['yolo_model'] = args.yolo_model
            logger.info(f"Modelo YOLO: {args.yolo_model}")

        if args.reid_model:
            system.config['reid_model'] = args.reid_model
            logger.info(f"Modelo ReID: {args.reid_model}")

        if args.embedding_buffer:
            system.config['embedding_buffer_size'] = args.embedding_buffer
            logger.info(f"Buffer de embeddings: {args.embedding_buffer}")

        if args.quality_threshold:
            system.config['quality_threshold'] = args.quality_threshold
            logger.info(f"Umbral de calidad: {args.quality_threshold}")

        if args.similarity_threshold:
            system.config['similarity_threshold'] = args.similarity_threshold
            logger.info(f"Umbral de similitud: {args.similarity_threshold}")

        # Re-inicializar modelos si hubo cambios
        if any([args.device, args.yolo_model, args.reid_model,
                args.embedding_buffer, args.quality_threshold]):
            logger.info("Re-inicializando modelos con nueva configuración...")
            system._setup_models()

        logger.info("=== SISTEMA INICIALIZADO ===")

        # Mostrar configuración activa
        logger.info("Configuración activa:")
        logger.info(f"  YOLO: {system.config['yolo_model']}")
        logger.info(f"  ReID: {system.config['reid_model']}")
        logger.info(f"  Dispositivo: {system.config['reid_device']}")
        logger.info(
            f"  Buffer embeddings: {system.config['embedding_buffer_size']}")
        logger.info(f"  Umbral calidad: {system.config['quality_threshold']}")
        logger.info(
            f"  Umbral similitud: {system.config['similarity_threshold']}")

        if args.stats_only:
            logger.info("Modo solo estadísticas - no se procesará video")
            system.print_statistics()
            return 0

        # Procesamiento
        input_source = args.video if args.video else args.camera
        logger.info(f"=== PROCESANDO: {input_source} ===")

        system.process_video(
            video_path=str(input_source),
            output_path=args.output,
            display=not args.no_display,
            save_video=args.save_video
        )

        logger.info("=== PROCESAMIENTO COMPLETADO ===")
        return 0

    except Exception as e:
        logger.error(f"Error durante ejecución: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1
    except KeyboardInterrupt:
        logger.info("Procesamiento interrumpido por el usuario")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
