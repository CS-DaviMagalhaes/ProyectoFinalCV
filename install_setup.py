#!/usr/bin/env python3
"""
Script de instalación automática para el proyecto de Re-identificación de Personas
Instala todas las dependencias necesarias incluyendo Torchreid
"""

import subprocess
import sys
import os
import platform
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, check=True):
    """Ejecuta un comando y muestra el resultado"""
    logger.info(f"Ejecutando: {command}")
    try:
        result = subprocess.run(command, shell=True,
                                check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error ejecutando comando: {e}")
        if e.stderr:
            logger.error(e.stderr)
        return False


def check_python_version():
    """Verifica que la versión de Python sea compatible"""
    version = sys.version_info
    logger.info(
        f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Se requiere Python 3.8 o superior")
        return False
    return True


def check_cuda():
    """Verifica si CUDA está disponible"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("CUDA detectado - GPU NVIDIA disponible")
            return True
        else:
            logger.warning("CUDA no detectado - se usará CPU")
            return False
    except FileNotFoundError:
        logger.warning("nvidia-smi no encontrado - se usará CPU")
        return False


def install_pytorch():
    """Instala PyTorch con soporte CUDA si está disponible"""
    cuda_available = check_cuda()

    if cuda_available:
        logger.info("Instalando PyTorch con soporte CUDA...")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        logger.info("Instalando PyTorch para CPU...")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    return run_command(torch_command)


def install_requirements():
    """Instala las dependencias del requirements.txt"""
    logger.info("Instalando dependencias desde requirements.txt...")
    return run_command("pip install -r requirements.txt")


def install_torchreid():
    """Instala Torchreid desde GitHub"""
    logger.info("Clonando e instalando Torchreid...")

    # Clonar repositorio si no exists
    if not os.path.exists("deep-person-reid"):
        if not run_command("git clone https://github.com/KaiyangZhou/deep-person-reid.git"):
            return False

    # Instalar Torchreid
    original_dir = os.getcwd()
    try:
        os.chdir("deep-person-reid")
        success = run_command("pip install -e .")
        os.chdir(original_dir)
        return success
    except Exception as e:
        logger.error(f"Error instalando Torchreid: {e}")
        os.chdir(original_dir)
        return False


def create_directories():
    """Crea directorios necesarios"""
    directories = [
        "data/video",
        "output",
        "logs",
        "models",
        "experiments"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Creado directorio: {directory}")


def test_installation():
    """Prueba que la instalación funcione correctamente"""
    logger.info("Probando instalación...")

    test_script = """
import torch
import torchvision
import cv2
from ultralytics import YOLO
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Probar YOLO
try:
    model = YOLO('yolov8n.pt')
    print("YOLO cargado correctamente")
except Exception as e:
    print(f"Error cargando YOLO: {e}")

# Probar Torchreid
try:
    import torchreid
    print(f"Torchreid version: {torchreid.__version__}")
    print("Torchreid importado correctamente")
except Exception as e:
    print(f"Error importando Torchreid: {e}")

print("Test de instalación completado")
"""

    # Escribir script de prueba
    with open("test_installation.py", "w") as f:
        f.write(test_script)

    # Ejecutar prueba
    success = run_command("python test_installation.py")

    # Limpiar archivo de prueba
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")

    return success


def main():
    """Función principal de instalación"""
    logger.info("=== INSTALACIÓN DEL PROYECTO DE RE-IDENTIFICACIÓN ===")

    # Verificar Python
    if not check_python_version():
        sys.exit(1)

    # Crear directorios
    create_directories()

    # Actualizar pip
    logger.info("Actualizando pip...")
    run_command("pip install --upgrade pip")

    # Instalar PyTorch
    if not install_pytorch():
        logger.error("Error instalando PyTorch")
        sys.exit(1)

    # Instalar dependencias
    if not install_requirements():
        logger.error("Error instalando dependencias")
        sys.exit(1)

    # Instalar Torchreid
    if not install_torchreid():
        logger.error("Error instalando Torchreid")
        sys.exit(1)

    # Probar instalación
    if test_installation():
        logger.info("✅ Instalación completada exitosamente!")
        logger.info("Para usar el sistema:")
        logger.info("1. Coloca tu video en la carpeta data/video/")
        logger.info(
            "2. Ejecuta: python src/main_reid.py --video data/video/tu_video.mp4")
    else:
        logger.error("❌ Instalación completada con errores")
        logger.error("Por favor revisa los logs y corrige los problemas")
        sys.exit(1)


if __name__ == "__main__":
    main()
