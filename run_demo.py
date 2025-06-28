#!/usr/bin/env python3
"""
Script de demostración para el sistema de Re-identificación de Personas
Muestra cómo usar el sistema paso a paso
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_requirements():
    """Verifica que todos los requisitos estén instalados"""
    logger.info("Verificando requisitos...")

    try:
        import torch
        import torchvision
        import cv2
        from ultralytics import YOLO
        import torchreid
        import numpy as np

        logger.info("✅ Todos los requisitos están instalados")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"OpenCV: {cv2.__version__}")
        logger.info(f"CUDA disponible: {torch.cuda.is_available()}")

        return True
    except ImportError as e:
        logger.error(f"❌ Falta instalar: {e}")
        logger.error("Ejecuta: python install_setup.py")
        return False


def find_video_files():
    """Busca archivos de video en la carpeta data/video"""
    video_dir = Path("data/video")
    if not video_dir.exists():
        logger.error("La carpeta data/video no existe")
        return []

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))

    return video_files


def run_demo_basic():
    """Ejecuta demo básico con configuración por defecto"""
    logger.info("=== DEMO BÁSICO ===")

    video_files = find_video_files()
    if not video_files:
        logger.error("No se encontraron videos en data/video/")
        logger.info(
            "Coloca un archivo de video en la carpeta data/video/ para continuar")
        return False

    # Usar el primer video encontrado
    video_path = video_files[0]
    output_path = f"output/demo_basic_{video_path.stem}.mp4"

    logger.info(f"Procesando video: {video_path}")
    logger.info(f"Resultado se guardará en: {output_path}")

    # Comando para ejecutar
    command = f"python src/main_reid.py --video {video_path} --output {output_path}"
    logger.info(f"Ejecutando: {command}")

    os.system(command)
    return True


def run_demo_advanced():
    """Ejecuta demo avanzado con configuración personalizada"""
    logger.info("=== DEMO AVANZADO ===")

    video_files = find_video_files()
    if not video_files:
        logger.error("No se encontraron videos en data/video/")
        return False

    # Usar el primer video encontrado
    video_path = video_files[0]
    output_path = f"output/demo_advanced_{video_path.stem}.mp4"
    config_path = "config/default_config.json"

    logger.info(f"Procesando video: {video_path}")
    logger.info(f"Usando configuración: {config_path}")
    logger.info(f"Resultado se guardará en: {output_path}")

    # Comando para ejecutar
    command = f"python src/main_reid.py --video {video_path} --output {output_path} --config {config_path}"
    logger.info(f"Ejecutando: {command}")

    os.system(command)
    return True


def run_webcam_demo():
    """Ejecuta demo con webcam en tiempo real"""
    logger.info("=== DEMO CON WEBCAM ===")
    logger.info("Usando webcam para demostración en tiempo real")
    logger.info("Presiona 'q' para salir")

    # Comando para ejecutar con webcam
    command = "python src/main_reid.py --video 0"
    logger.info(f"Ejecutando: {command}")

    os.system(command)
    return True


def show_results():
    """Muestra los resultados generados"""
    output_dir = Path("output")
    if not output_dir.exists():
        logger.info("No hay resultados generados aún")
        return

    video_files = list(output_dir.glob("*.mp4"))
    if not video_files:
        logger.info("No hay videos de resultado generados")
        return

    logger.info("=== RESULTADOS GENERADOS ===")
    for video_file in video_files:
        size_mb = video_file.stat().st_size / (1024 * 1024)
        logger.info(f"📹 {video_file.name} ({size_mb:.1f} MB)")

    logger.info(f"Total de archivos: {len(video_files)}")


def main():
    parser = argparse.ArgumentParser(
        description='Demo del Sistema de Re-identificación')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'webcam', 'results'],
                        default='basic', help='Modo de demostración')

    args = parser.parse_args()

    logger.info("🚀 DEMOSTRACIÓN - SISTEMA DE RE-IDENTIFICACIÓN DE PERSONAS")
    logger.info("=" * 60)

    # Verificar requisitos
    if not check_requirements():
        return

    # Crear directorios si no existen
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Ejecutar según el modo seleccionado
    if args.mode == 'basic':
        run_demo_basic()
    elif args.mode == 'advanced':
        run_demo_advanced()
    elif args.mode == 'webcam':
        run_webcam_demo()
    elif args.mode == 'results':
        show_results()

    # Mostrar resultados al final
    if args.mode != 'results':
        show_results()

    logger.info("Demo completado!")
    logger.info("Para más opciones:")
    logger.info("- Demo básico: python run_demo.py --mode basic")
    logger.info("- Demo avanzado: python run_demo.py --mode advanced")
    logger.info("- Demo webcam: python run_demo.py --mode webcam")
    logger.info("- Ver resultados: python run_demo.py --mode results")


if __name__ == "__main__":
    main()
