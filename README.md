# Proyecto Final CV - Re-identificación de Personas

Sistema de re-identificación de personas usando **YOLOv8** para detección/tracking y **Torchreid** para extracción de características. El sistema puede identificar cuando una persona sale y vuelve a entrar al campo de visión de la cámara, manteniendo un ID consistente.

## 🌟 Características Principales

- **Detección robusta** con YOLOv8 (nano/small/medium/large)
- **Re-identificación avanzada** usando modelos pre-entrenados de Torchreid
- **Tracking persistente** con IDs globales únicos
- **Configuración flexible** mediante archivos JSON
- **Soporte GPU/CPU** automático
- **Métricas en tiempo real** de rendimiento
- **Visualización interactiva** con OpenCV

## 📋 Requisitos del Sistema

- **Python 3.8+**
- **CUDA** (opcional, para GPU)
- **4GB+ RAM** (8GB+ recomendado)
- **Webcam o archivos de video** para pruebas

## 🚀 Instalación Rápida

### Opción 1: Instalación Automática (Recomendada)
```bash
# Clonar el repositorio
git clone [URL_DEL_REPO]
cd ProyectoFinalCV

# Ejecutar instalación automática
python install_setup.py
```

### Opción 2: Instalación Manual
```bash
# Instalar dependencias base
pip install -r requirements.txt

# Instalar PyTorch (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar Torchreid
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
cd ..
```

## 📁 Estructura del Proyecto

```
ProyectoFinalCV/
├── src/                          # Código fuente
│   ├── feature_extractor.py      # Extractor de características (Torchreid)
│   └── main_reid.py              # Pipeline principal
├── config/                       # Configuraciones
│   └── default_config.json       # Configuración por defecto
├── data/                         # Datos de entrada
│   └── video/                    # Videos de prueba (coloca aquí tus videos)
├── output/                       # Resultados generados
├── docs/                         # Documentación
├── requirements.txt              # Dependencias
├── install_setup.py             # Script de instalación
├── run_demo.py                  # Script de demostración
└── README.md                    # Este archivo
```

## 🎯 Uso Básico

### 1. Preparar Video
Coloca tu archivo de video en la carpeta `data/video/`:
```bash
cp tu_video.mp4 data/video/
```

### 2. Ejecutar Re-identificación
```bash
# Básico (solo visualización)
python src/main_reid.py --video data/video/tu_video.mp4

# Guardar resultado
python src/main_reid.py --video data/video/tu_video.mp4 --output output/resultado.mp4

# Con configuración personalizada
python src/main_reid.py --video data/video/tu_video.mp4 --config config/default_config.json
```

### 3. Demo Rápido
```bash
# Demo básico
python run_demo.py --mode basic

# Demo avanzado
python run_demo.py --mode advanced

# Demo con webcam
python run_demo.py --mode webcam
```

## ⚙️ Configuración

Edita `config/default_config.json` para ajustar el comportamiento:

```json
{
  "yolo_model": "yolov8n.pt",           # Modelo YOLO (n/s/m/l)
  "reid_model": "osnet_x1_0",           # Modelo Re-ID
  "device": "cuda",                     # cuda/cpu
  "similarity_threshold": 0.85,         # Umbral de similitud (0.7-0.95)
  "disappearance_limit_seconds": 5,     # Tiempo de memoria (segundos)
  "min_detection_confidence": 0.5       # Confianza mínima detección
}
```

### Parámetros Clave para Ajustar

- **`similarity_threshold`**: Más alto = más estricto, menos falsos positivos
- **`disappearance_limit_seconds`**: Tiempo que recuerda a una persona ausente
- **`yolo_model`**: `yolov8n.pt` (rápido) → `yolov8l.pt` (preciso)
- **`reid_model`**: `osnet_x0_25` (rápido) → `osnet_x1_0` (preciso)

## 🎮 Controles Durante Ejecución

- **`q`**: Salir del programa
- **`s`**: Guardar embeddings actuales
- **`ESC`**: Salir del programa

## 📊 Métricas y Resultados

El sistema muestra en tiempo real:
- Número de frame actual
- Personas activas en escena
- Total de re-identificaciones exitosas
- FPS de procesamiento
- Estadísticas finales al terminar

## 🔧 Resolución de Problemas

### Error: "CUDA not available"
```bash
# Verificar CUDA
nvidia-smi

# Reinstalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "No module named 'torchreid'"
```bash
# Instalar Torchreid manualmente
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
```

### Rendimiento lento
1. Usar modelo más ligero: `yolov8n.pt` + `osnet_x0_25`
2. Reducir resolución del video
3. Usar CPU si GPU es lenta: `"device": "cpu"`

### Muchos falsos positivos
1. Aumentar `similarity_threshold` a 0.90+
2. Reducir `disappearance_limit_seconds`
3. Usar modelo Re-ID más preciso: `osnet_x1_0`

## 🧪 Experimentación

### Probar Diferentes Modelos
```bash
# Modelo ligero y rápido
python src/main_reid.py --video data/video/test.mp4 --config config/fast_config.json

# Modelo preciso pero lento
python src/main_reid.py --video data/video/test.mp4 --config config/accurate_config.json
```

### Análisis de Rendimiento
El sistema guarda automáticamente métricas de:
- Tiempo de procesamiento por frame
- FPS promedio
- Número de re-identificaciones
- Eficiencia del tracking

## 📈 Próximas Mejoras

- [ ] Interfaz web para configuración
- [ ] Soporte para múltiples cámaras
- [ ] Base de datos persistente de personas
- [ ] Análisis de trayectorias
- [ ] Exportación de estadísticas detalladas

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Envía un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo `LICENSE`.

## 🙏 Agradecimientos

- **Torchreid**: Framework de re-identificación de personas
- **Ultralytics**: YOLOv8 para detección y tracking
- **OpenCV**: Procesamiento de video e imágenes