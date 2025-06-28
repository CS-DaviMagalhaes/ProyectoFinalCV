#!/usr/bin/env python3
"""
Script de evaluación básico para el sistema de Person Re-ID.
Calcula métricas de rendimiento y estadísticas del sistema.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReIDEvaluator:
    """Evaluador de sistemas de Re-Identificación."""

    def __init__(self):
        self.metrics = {}

    def load_memory_data(self, memory_file: str) -> dict:
        """Carga datos de memoria del sistema."""
        try:
            with open(memory_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"No se pudo cargar {memory_file}: {e}")
            return {}

    def calculate_basic_metrics(self, memory_data: dict) -> dict:
        """Calcula métricas básicas del sistema."""
        stats = memory_data.get('stats', {})

        metrics = {
            'total_frames': stats.get('total_frames', 0),
            'total_detections': stats.get('total_detections', 0),
            'total_reidentifications': stats.get('total_reidentifications', 0),
            'unique_persons': memory_data.get('global_id_counter', 0),
        }

        # Calcular ratios
        if metrics['total_frames'] > 0:
            metrics['detections_per_frame'] = metrics['total_detections'] / \
                metrics['total_frames']
        else:
            metrics['detections_per_frame'] = 0

        if metrics['total_detections'] > 0:
            metrics['reidentification_rate'] = metrics['total_reidentifications'] / \
                metrics['total_detections']
        else:
            metrics['reidentification_rate'] = 0

        # Calcular FPS si hay datos de tiempos
        processing_times = stats.get('processing_times', [])
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            metrics['avg_fps'] = 1.0 / \
                avg_processing_time if avg_processing_time > 0 else 0
            metrics['max_fps'] = 1.0 / \
                min(processing_times) if min(processing_times) > 0 else 0
            metrics['min_fps'] = 1.0 / \
                max(processing_times) if max(processing_times) > 0 else 0

        # Estadísticas de similitud
        similarity_scores = stats.get('similarity_scores', [])
        if similarity_scores:
            metrics['avg_similarity'] = np.mean(similarity_scores)
            metrics['std_similarity'] = np.std(similarity_scores)
            metrics['min_similarity'] = np.min(similarity_scores)
            metrics['max_similarity'] = np.max(similarity_scores)

        return metrics

    def compare_systems(self, memory_files: list) -> pd.DataFrame:
        """Compara múltiples sistemas/configuraciones."""
        results = []

        for memory_file in memory_files:
            memory_data = self.load_memory_data(memory_file)
            if not memory_data:
                continue

            metrics = self.calculate_basic_metrics(memory_data)
            metrics['config'] = Path(memory_file).stem
            metrics['timestamp'] = memory_data.get('timestamp', '')

            results.append(metrics)

        return pd.DataFrame(results)

    def generate_report(self, memory_files: list, output_dir: str = "evaluation_results"):
        """Genera reporte completo de evaluación."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info("Generando reporte de evaluación...")

        # Comparar sistemas
        df = self.compare_systems(memory_files)

        if df.empty:
            logger.warning("No se encontraron datos para evaluar")
            return

        # Guardar datos
        csv_file = output_path / "metrics_comparison.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Métricas guardadas en: {csv_file}")

        # Generar visualizaciones
        self._create_visualizations(df, output_path)

        # Generar reporte de texto
        self._create_text_report(df, output_path)

        logger.info(f"Reporte completo generado en: {output_path}")

    def _create_visualizations(self, df: pd.DataFrame, output_path: Path):
        """Crea visualizaciones de las métricas."""
        plt.style.use('seaborn-v0_8')

        # Figura 1: FPS y Rendimiento
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Rendimiento del Sistema Re-ID', fontsize=16)

        # FPS promedio
        if 'avg_fps' in df.columns:
            df.plot(x='config', y='avg_fps', kind='bar',
                    ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('FPS Promedio por Configuración')
            axes[0, 0].set_ylabel('FPS')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Detecciones por frame
        if 'detections_per_frame' in df.columns:
            df.plot(x='config', y='detections_per_frame',
                    kind='bar', ax=axes[0, 1], color='lightgreen')
            axes[0, 1].set_title('Detecciones por Frame')
            axes[0, 1].set_ylabel('Detecciones/Frame')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Tasa de re-identificación
        if 'reidentification_rate' in df.columns:
            df.plot(x='config', y='reidentification_rate',
                    kind='bar', ax=axes[1, 0], color='orange')
            axes[1, 0].set_title('Tasa de Re-identificación')
            axes[1, 0].set_ylabel('Tasa')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Similitud promedio
        if 'avg_similarity' in df.columns:
            df.plot(x='config', y='avg_similarity', kind='bar',
                    ax=axes[1, 1], color='lightcoral')
            axes[1, 1].set_title('Similitud Promedio')
            axes[1, 1].set_ylabel('Similitud')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Figura 2: Distribución de similitudes (si hay datos)
        self._plot_similarity_distributions(df, output_path)

        logger.info("Visualizaciones creadas")

    def _plot_similarity_distributions(self, df: pd.DataFrame, output_path: Path):
        """Crea gráfico de distribución de similitudes."""
        # Esta función necesitaría acceso a los datos de similitud individuales
        # Por ahora solo mostramos estadísticas básicas

        similarity_cols = ['avg_similarity', 'std_similarity',
                           'min_similarity', 'max_similarity']
        available_cols = [col for col in similarity_cols if col in df.columns]

        if not available_cols:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        df[available_cols].plot(kind='bar', ax=ax)
        ax.set_title('Estadísticas de Similitud por Configuración')
        ax.set_ylabel('Similitud')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'similarity_stats.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_text_report(self, df: pd.DataFrame, output_path: Path):
        """Crea reporte de texto con métricas."""
        report_file = output_path / 'evaluation_report.txt'

        with open(report_file, 'w') as f:
            f.write("REPORTE DE EVALUACIÓN - SISTEMA PERSON RE-ID\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("RESUMEN DE CONFIGURACIONES EVALUADAS:\n")
            f.write("-" * 40 + "\n")

            for _, row in df.iterrows():
                f.write(f"\nCONFIGURACIÓN: {row['config']}\n")
                f.write(f"Timestamp: {row.get('timestamp', 'N/A')}\n")
                f.write(f"Frames procesados: {row.get('total_frames', 0):,}\n")
                f.write(
                    f"Detecciones totales: {row.get('total_detections', 0):,}\n")
                f.write(
                    f"Re-identificaciones: {row.get('total_reidentifications', 0):,}\n")
                f.write(f"Personas únicas: {row.get('unique_persons', 0)}\n")

                if 'avg_fps' in row:
                    f.write(f"FPS promedio: {row['avg_fps']:.2f}\n")
                if 'detections_per_frame' in row:
                    f.write(
                        f"Detecciones/frame: {row['detections_per_frame']:.2f}\n")
                if 'reidentification_rate' in row:
                    f.write(
                        f"Tasa re-ID: {row['reidentification_rate']:.3f}\n")
                if 'avg_similarity' in row:
                    f.write(
                        f"Similitud promedio: {row['avg_similarity']:.3f}\n")

                f.write("-" * 30 + "\n")

            # Comparación y recomendaciones
            f.write("\nANÁLISIS COMPARATIVO:\n")
            f.write("-" * 25 + "\n")

            if 'avg_fps' in df.columns:
                best_fps = df.loc[df['avg_fps'].idxmax()]
                f.write(
                    f"Mejor FPS: {best_fps['config']} ({best_fps['avg_fps']:.2f} FPS)\n")

            if 'reidentification_rate' in df.columns:
                best_reid = df.loc[df['reidentification_rate'].idxmax()]
                f.write(
                    f"Mejor Re-ID: {best_reid['config']} ({best_reid['reidentification_rate']:.3f})\n")

            if 'avg_similarity' in df.columns:
                best_sim = df.loc[df['avg_similarity'].idxmax()]
                f.write(
                    f"Mejor similitud: {best_sim['config']} ({best_sim['avg_similarity']:.3f})\n")

            f.write("\nRECOMENDACIONES:\n")
            f.write("-" * 15 + "\n")
            f.write("- Para tiempo real: usar configuración con mejor FPS\n")
            f.write(
                "- Para máxima precisión: usar configuración con mejor tasa Re-ID\n")
            f.write("- Para uso balanceado: considerar configuración robusta\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluador del sistema Person Re-ID")

    parser.add_argument(
        '--memory-files', '-m',
        nargs='+',
        help='Archivos de memoria a evaluar'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='evaluation_results',
        help='Directorio de salida para resultados'
    )

    parser.add_argument(
        '--auto-find',
        action='store_true',
        help='Buscar automáticamente archivos reid_memory*.json'
    )

    args = parser.parse_args()

    evaluator = ReIDEvaluator()

    memory_files = args.memory_files or []

    if args.auto_find:
        # Buscar archivos automáticamente
        search_patterns = [
            "reid_memory*.json",
            "*.json"
        ]

        for pattern in search_patterns:
            found_files = list(Path('.').glob(pattern))
            memory_files.extend(
                [str(f) for f in found_files if 'reid_memory' in f.name])

        logger.info(f"Archivos encontrados automáticamente: {memory_files}")

    if not memory_files:
        logger.error("No se especificaron archivos de memoria para evaluar")
        logger.info(
            "Uso: python evaluate.py --memory-files file1.json file2.json")
        logger.info("   o: python evaluate.py --auto-find")
        return 1

    # Filtrar archivos que existen
    existing_files = [f for f in memory_files if Path(f).exists()]

    if not existing_files:
        logger.error("Ninguno de los archivos especificados existe")
        return 1

    logger.info(f"Evaluando {len(existing_files)} archivos...")

    # Generar reporte
    evaluator.generate_report(existing_files, args.output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
