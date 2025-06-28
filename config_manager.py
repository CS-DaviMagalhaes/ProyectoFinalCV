#!/usr/bin/env python3
"""
Config Manager - Utilidad para gestionar configuraciones del sistema Re-ID
Autor: ProyectoFinalCV
Versión: 1.0
"""

import json
import os
import argparse
import sys
from pathlib import Path
import subprocess
import platform
from typing import Dict, List, Optional, Tuple
import shutil

class ConfigManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.config_dir.mkdir(exist_ok=True)

        # Configuraciones disponibles
        self.available_configs = {
            "default": "config/default_config.json",
            "advanced": "config/advanced_config.json",
            "cpu": "config/cpu_optimized.json",
            "gpu": "config/high_precision.json",
            "webcam": "config/webcam_realtime.json"
        }

        # Modelos disponibles
        self.yolo_models = {
            "yolov8n.pt": {"size": "3.2M", "speed": "fastest", "accuracy": "low"},
            "yolov8s.pt": {"size": "11.2M", "speed": "fast", "accuracy": "medium"},
            "yolov8m.pt": {"size": "25.9M", "speed": "medium", "accuracy": "high"},
            "yolov8l.pt": {"size": "43.7M", "speed": "slow", "accuracy": "very_high"},
            "yolov8x.pt": {"size": "68.2M", "speed": "slowest", "accuracy": "highest"}
        }

        self.reid_models = {
            # OSNet Family
            "osnet_x0_25": {"params": "0.17M", "speed": "fastest", "accuracy": "medium", "family": "osnet"},
            "osnet_x0_5": {"params": "0.37M", "speed": "fast", "accuracy": "good", "family": "osnet"},
            "osnet_x0_75": {"params": "0.65M", "speed": "medium", "accuracy": "very_good", "family": "osnet"},
            "osnet_x1_0": {"params": "1.02M", "speed": "medium", "accuracy": "excellent", "family": "osnet"},
            "osnet_ibn_x1_0": {"params": "1.02M", "speed": "medium", "accuracy": "excellent", "family": "osnet"},
            "osnet_ain_x0_25": {"params": "0.17M", "speed": "fast", "accuracy": "good", "family": "osnet"},
            "osnet_ain_x0_5": {"params": "0.37M", "speed": "fast", "accuracy": "very_good", "family": "osnet"},
            "osnet_ain_x0_75": {"params": "0.65M", "speed": "medium", "accuracy": "excellent", "family": "osnet"},
            "osnet_ain_x1_0": {"params": "1.02M", "speed": "medium", "accuracy": "outstanding", "family": "osnet"},

            # ResNet Family
            "resnet18": {"params": "11.2M", "speed": "fast", "accuracy": "good", "family": "resnet"},
            "resnet34": {"params": "21.3M", "speed": "medium", "accuracy": "very_good", "family": "resnet"},
            "resnet50": {"params": "23.5M", "speed": "medium", "accuracy": "excellent", "family": "resnet"},
            "resnet101": {"params": "42.5M", "speed": "slow", "accuracy": "outstanding", "family": "resnet"},
            "resnet152": {"params": "58.2M", "speed": "very_slow", "accuracy": "outstanding", "family": "resnet"},

            # Mobile Family
            "mobilenetv2_x1_0": {"params": "2.2M", "speed": "very_fast", "accuracy": "good", "family": "mobile"},
            "mobilenetv2_x1_4": {"params": "4.3M", "speed": "fast", "accuracy": "very_good", "family": "mobile"},
            "shufflenet": {"params": "1.0M", "speed": "very_fast", "accuracy": "medium", "family": "mobile"},

            # Others
            "mlfn": {"params": "32.5M", "speed": "medium", "accuracy": "excellent", "family": "other"},
            "densenet121": {"params": "6.9M", "speed": "medium", "accuracy": "very_good", "family": "other"},
            "senet154": {"params": "113.0M", "speed": "very_slow", "accuracy": "outstanding", "family": "other"}
        }

    def detect_hardware(self) -> Dict[str, any]:
        """Detecta el hardware disponible"""
        hardware_info = {
            "cpu_cores": os.cpu_count(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cuda_available": False,
            "gpu_memory": 0,
            "ram_gb": 0
        }

        try:
            import torch
            hardware_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                hardware_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        try:
            import psutil
            hardware_info["ram_gb"] = psutil.virtual_memory().total // (1024**3)
        except ImportError:
            pass

        return hardware_info

    def list_configs(self):
        """Lista todas las configuraciones disponibles"""
        print("\n🔧 CONFIGURACIONES DISPONIBLES:")
        print("=" * 50)

        for name, path in self.available_configs.items():
            full_path = self.project_root / path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        config = json.load(f)

                    # Extraer información básica
                    meta = config.get('_meta', {})
                    description = meta.get('description', 'Sin descripción')

                    yolo_model = config.get('yolo_model', config.get('models', {}).get('yolo_model', 'N/A'))
                    reid_model = config.get('reid_model', config.get('models', {}).get('reid_model', 'N/A'))
                    device = config.get('device', config.get('hardware', {}).get('device', 'N/A'))

                    print(f"📋 {name.upper()}")
                    print(f"   Archivo: {path}")
                    print(f"   Descripción: {description}")
                    print(f"   YOLO: {yolo_model}")
                    print(f"   Re-ID: {reid_model}")
                    print(f"   Device: {device}")
                    print()

                except Exception as e:
                    print(f"❌ Error leyendo {name}: {e}")
            else:
                print(f"❌ {name}: Archivo no encontrado ({path})")

    def show_models(self, model_type: str = "all"):
        """Muestra información sobre los modelos disponibles"""
        if model_type in ["all", "yolo"]:
            print("\n🎯 MODELOS YOLO DISPONIBLES:")
            print("=" * 50)
            print(f"{'Modelo':<15} {'Tamaño':<10} {'Velocidad':<12} {'Precisión':<12}")
            print("-" * 50)
            for model, info in self.yolo_models.items():
                print(f"{model:<15} {info['size']:<10} {info['speed']:<12} {info['accuracy']:<12}")
            print()

        if model_type in ["all", "reid"]:
            print("\n🔍 MODELOS RE-ID DISPONIBLES:")
            print("=" * 60)

            # Agrupar por familia
            families = {}
            for model, info in self.reid_models.items():
                family = info['family']
                if family not in families:
                    families[family] = []
                families[family].append((model, info))

            for family, models in families.items():
                print(f"\n📁 Familia {family.upper()}:")
                print(f"{'Modelo':<20} {'Parámetros':<12} {'Velocidad':<12} {'Precisión':<12}")
                print("-" * 60)
                for model, info in models:
                    print(f"{model:<20} {info['params']:<12} {info['speed']:<12} {info['accuracy']:<12}")

    def recommend_config(self) -> str:
        """Recomienda una configuración basada en el hardware detectado"""
        hardware = self.detect_hardware()

        print("\n🔍 ANÁLISIS DE HARDWARE:")
        print("=" * 40)
        print(f"CPU Cores: {hardware['cpu_cores']}")
        print(f"RAM: {hardware['ram_gb']} GB")
        print(f"CUDA: {'✅ Disponible' if hardware['cuda_available'] else '❌ No disponible'}")
        if hardware['cuda_available']:
            print(f"GPU: {hardware.get('gpu_name', 'Desconocida')}")
            print(f"GPU Memory: {hardware['gpu_memory']} GB")

        print("\n💡 RECOMENDACIÓN:")
        print("=" * 40)

        # Lógica de recomendación
        if not hardware['cuda_available']:
            recommendation = "cpu"
            reason = "No se detectó GPU CUDA, configuración optimizada para CPU"
        elif hardware['gpu_memory'] >= 8:
            recommendation = "gpu"
            reason = "GPU potente detectada, configuración de alta precisión"
        elif hardware['gpu_memory'] >= 4:
            recommendation = "advanced"
            reason = "GPU media detectada, configuración balanceada"
        elif hardware['ram_gb'] < 8:
            recommendation = "cpu"
            reason = "RAM limitada, configuración CPU optimizada"
        else:
            recommendation = "advanced"
            reason = "Configuración balanceada para tu sistema"

        print(f"Configuración recomendada: {recommendation.upper()}")
        print(f"Razón: {reason}")

        return recommendation

    def create_custom_config(self, name: str, base_config: str = "advanced"):
        """Crea una configuración personalizada"""
        base_path = self.project_root / self.available_configs.get(base_config, "config/advanced_config.json")
        custom_path = self.config_dir / f"{name}.json"

        if not base_path.exists():
            print(f"❌ Error: Configuración base '{base_config}' no encontrada")
            return False

        try:
            with open(base_path, 'r') as f:
                config = json.load(f)

            # Actualizar metadata
            config['_meta'] = {
                "version": "1.0",
                "description": f"Configuración personalizada: {name}",
                "base_config": base_config,
                "created_by": "config_manager"
            }

            with open(custom_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✅ Configuración personalizada creada: {custom_path}")
            print(f"   Basada en: {base_config}")
            print(f"   Edita el archivo para personalizar")

            return True

        except Exception as e:
            print(f"❌ Error creando configuración: {e}")
            return False

    def validate_config(self, config_path: str) -> bool:
        """Valida una configuración"""
        full_path = self.project_root / config_path

        if not full_path.exists():
            print(f"❌ Error: Archivo de configuración no encontrado: {config_path}")
            return False

        try:
            with open(full_path, 'r') as f:
                config = json.load(f)

            errors = []
            warnings = []

            # Validaciones básicas
            yolo_model = config.get('yolo_model', config.get('models', {}).get('yolo_model'))
            reid_model = config.get('reid_model', config.get('models', {}).get('reid_model'))
            device = config.get('device', config.get('hardware', {}).get('device', 'cpu'))

            if not yolo_model:
                errors.append("Modelo YOLO no especificado")
            elif yolo_model not in self.yolo_models:
                warnings.append(f"Modelo YOLO '{yolo_model}' no reconocido")

            if not reid_model:
                errors.append("Modelo Re-ID no especificado")
            elif reid_model not in self.reid_models:
                warnings.append(f"Modelo Re-ID '{reid_model}' no reconocido")

            if device not in ['cpu', 'cuda']:
                warnings.append(f"Device '{device}' no reconocido")

            # Validar rangos de parámetros
            similarity_threshold = config.get('similarity_threshold',
                                            config.get('reid_settings', {}).get('similarity_threshold', 0.85))
            if not 0.5 <= similarity_threshold <= 1.0:
                warnings.append(f"similarity_threshold ({similarity_threshold}) fuera del rango recomendado (0.5-1.0)")

            # Mostrar resultados
            print(f"\n🔍 VALIDACIÓN DE CONFIGURACIÓN: {config_path}")
            print("=" * 50)

            if errors:
                print("❌ ERRORES:")
                for error in errors:
                    print(f"   • {error}")
                return False

            if warnings:
                print("⚠️  ADVERTENCIAS:")
                for warning in warnings:
                    print(f"   • {warning}")

            print("✅ INFORMACIÓN:")
            print(f"   • YOLO Model: {yolo_model}")
            print(f"   • Re-ID Model: {reid_model}")
            print(f"   • Device: {device}")
            print(f"   • Similarity Threshold: {similarity_threshold}")

            if not warnings:
                print("✅ Configuración válida")

            return True

        except json.JSONDecodeError as e:
            print(f"❌ Error de formato JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Error validando configuración: {e}")
            return False

    def compare_configs(self, config1: str, config2: str):
        """Compara dos configuraciones"""
        def load_config(name):
            path = self.project_root / self.available_configs.get(name, f"config/{name}.json")
            if not path.exists():
                path = self.project_root / name  # Intenta path directo

            with open(path, 'r') as f:
                return json.load(f)

        try:
            c1 = load_config(config1)
            c2 = load_config(config2)

            print(f"\n📊 COMPARACIÓN: {config1} vs {config2}")
            print("=" * 60)

            # Extraer parámetros clave
            def extract_key_params(config):
                return {
                    'yolo_model': config.get('yolo_model', config.get('models', {}).get('yolo_model', 'N/A')),
                    'reid_model': config.get('reid_model', config.get('models', {}).get('reid_model', 'N/A')),
                    'device': config.get('device', config.get('hardware', {}).get('device', 'N/A')),
                    'similarity_threshold': config.get('similarity_threshold',
                                                     config.get('reid_settings', {}).get('similarity_threshold', 'N/A')),
                    'disappearance_limit': config.get('disappearance_limit_seconds',
                                                     config.get('memory_management', {}).get('disappearance_limit_seconds', 'N/A')),
                    'min_confidence': config.get('min_detection_confidence',
                                                config.get('detection_settings', {}).get('min_detection_confidence', 'N/A'))
                }

            params1 = extract_key_params(c1)
            params2 = extract_key_params(c2)

            print(f"{'Parámetro':<20} {config1:<20} {config2:<20}")
            print("-" * 60)

            for key in params1:
                val1 = params1[key]
                val2 = params2[key]
                status = "✅" if val1 == val2 else "🔄"
                print(f"{key:<20} {str(val1):<20} {str(val2):<20} {status}")

        except Exception as e:
            print(f"❌ Error comparando configuraciones: {e}")

    def optimize_for_hardware(self, config_name: str, output_name: str = None):
        """Optimiza una configuración para el hardware actual"""
        hardware = self.detect_hardware()

        if not output_name:
            output_name = f"{config_name}_optimized"

        base_path = self.project_root / self.available_configs.get(config_name, f"config/{config_name}.json")
        if not base_path.exists():
            print(f"❌ Error: Configuración '{config_name}' no encontrada")
            return False

        try:
            with open(base_path, 'r') as f:
                config = json.load(f)

            print(f"\n⚡ OPTIMIZANDO CONFIGURACIÓN PARA TU HARDWARE:")
            print("=" * 50)

            # Optimizaciones basadas en hardware
            optimizations = []

            if not hardware['cuda_available']:
                # Optimizaciones para CPU
                config['device'] = 'cpu'
                config['models'] = config.get('models', {})
                config['models']['yolo_model'] = 'yolov8n.pt'
                config['models']['reid_model'] = 'osnet_x0_25'
                config['performance_optimization'] = {
                    'device': 'cpu',
                    'num_workers': min(4, hardware['cpu_cores']),
                    'pin_memory': False,
                    'mixed_precision': False
                }
                optimizations.append("Configurado para CPU")
                optimizations.append("Modelos ligeros seleccionados")

            else:
                # Optimizaciones para GPU
                config['device'] = 'cuda'
                config['performance_optimization'] = config.get('performance_optimization', {})
                config['performance_optimization']['device'] = 'cuda'
                config['performance_optimization']['mixed_precision'] = True

                if hardware['gpu_memory'] >= 8:
                    config['models'] = config.get('models', {})
                    config['models']['yolo_model'] = 'yolov8l.pt'
                    config['models']['reid_model'] = 'osnet_ain_x1_0'
                    optimizations.append("GPU potente: modelos de alta precisión")
                elif hardware['gpu_memory'] >= 4:
                    config['models'] = config.get('models', {})
                    config['models']['yolo_model'] = 'yolov8s.pt'
                    config['models']['reid_model'] = 'osnet_x1_0'
                    optimizations.append("GPU media: modelos balanceados")
                else:
                    config['models'] = config.get('models', {})
                    config['models']['yolo_model'] = 'yolov8n.pt'
                    config['models']['reid_model'] = 'osnet_x0_5'
                    optimizations.append("GPU limitada: modelos ligeros")

            # Optimizaciones de memoria
            if hardware['ram_gb'] < 8:
                config['memory_management'] = config.get('memory_management', {})
                config['memory_management']['max_stored_embeddings'] = 50
                config['memory_management']['cleanup_interval_frames'] = 100
                optimizations.append("RAM limitada: gestión de memoria agresiva")

            # Actualizar metadata
            config['_meta'] = {
                "version": "1.0",
                "description": f"Configuración optimizada para hardware específico",
                "base_config": config_name,
                "optimized_for": {
                    "cpu_cores": hardware['cpu_cores'],
                    "cuda_available": hardware['cuda_available'],
                    "gpu_memory": hardware['gpu_memory'],
                    "ram_gb": hardware['ram_gb']
                },
                "optimizations": optimizations
            }

            # Guardar configuración optimizada
            output_path = self.config_dir / f"{output_name}.json"
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"✅ Configuración optimizada guardada: {output_path}")
            print("\n🔧 OPTIMIZACIONES APLICADAS:")
            for opt in optimizations:
                print(f"   • {opt}")

            return True

        except Exception as e:
            print(f"❌ Error optimizando configuración: {e}")
            return False

    def run_with_config(self, config_name: str, video_path: str = None, additional_args: List[str] = None):
        """Ejecuta el sistema con una configuración específica"""
        config_path = self.available_configs.get(config_name, f"config/{config_name}.json")
        full_config_path = self.project_root / config_path

        if not full_config_path.exists():
            print(f"❌ Error: Configuración '{config_name}' no encontrada")
            return False

        # Construir comando
        cmd = ["python", "src/main_reid.py", "--config", str(config_path)]

        if video_path:
            cmd.extend(["--video", video_path])

        if additional_args:
            cmd.extend(additional_args)

        print(f"🚀 EJECUTANDO CON CONFIGURACIÓN: {config_name}")
        print(f"Comando: {' '.join(cmd)}")
        print("=" * 50)

        try:
            # Cambiar al directorio del proyecto
            os.chdir(self.project_root)

            # Ejecutar comando
            result = subprocess.run(cmd, check=True)
            return result.returncode == 0

        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando comando: {e}")
            return False
        except KeyboardInterrupt:
            print("\n⏹️  Ejecución interrumpida por el usuario")
            return False

def main():
    parser = argparse.ArgumentParser(description="Config Manager - Gestor de configuraciones Re-ID")
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')

    # Comando list
    subparsers.add_parser('list', help='Listar configuraciones disponibles')

    # Comando models
    models_parser = subparsers.add_parser('models', help='Mostrar modelos disponibles')
    models_parser.add_argument('--type', choices=['all', 'yolo', 'reid'], default='all',
                              help='Tipo de modelos a mostrar')

    # Comando recommend
    subparsers.add_parser('recommend', help='Recomendar configuración para tu hardware')

    # Comando create
    create_parser = subparsers.add_parser('create', help='Crear configuración personalizada')
    create_parser.add_argument('name', help='Nombre de la nueva configuración')
    create_parser.add_argument('--base', default='advanced', help='Configuración base')

    # Comando validate
    validate_parser = subparsers.add_parser('validate', help='Validar configuración')
    validate_parser.add_argument('config', help='Configuración a validar')

    # Comando compare
    compare_parser = subparsers.add_parser('compare', help='Comparar configuraciones')
    compare_parser.add_argument('config1', help='Primera configuración')
    compare_parser.add_argument('config2', help='Segunda configuración')

    # Comando optimize
    optimize_parser = subparsers.add_parser('optimize', help='Optimizar configuración para hardware')
    optimize_parser.add_argument('config', help='Configuración a optimizar')
    optimize_parser.add_argument('--output', help='Nombre de configuración optimizada')

    # Comando run
    run_parser = subparsers.add_parser('run', help='Ejecutar con configuración')
    run_parser.add_argument('config', help='Configuración a usar')
    run_parser.add_argument('--video', help='Video a procesar')
    run_parser.add_argument('--args', nargs='*', help='Argumentos adicionales')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = ConfigManager()

    try:
        if args.command == 'list':
            manager.list_configs()

        elif args.command == 'models':
            manager.show_models(args.type)

        elif args.command == 'recommend':
            recommended = manager.recommend_config()
            print(f"\n💡 Para usar la configuración recomendada:")
            print(f"python config_manager.py run {recommended} --video tu_video.mp4")

        elif args.command == 'create':
            manager.create_custom_config(args.name, args.base)

        elif args.command == 'validate':
            manager.validate_config(args.config)

        elif args.command == 'compare':
            manager.compare_configs(args.config1, args.config2)

        elif args.command == 'optimize':
            manager.optimize_for_hardware(args.config, args.output)

        elif args.command == 'run':
            manager.run_with_config(args.config, args.video, args.args)

    except KeyboardInterrupt:
        print("\n⏹️  Operación cancelada por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()
