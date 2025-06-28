# 🔧 Sistema de Configuraciones - Re-ID

## 📋 Introducción

Este documento explica cómo usar el sistema avanzado de configuraciones para el proyecto de re-identificación de personas. El sistema permite ajustar fácilmente todos los parámetros según tu hardware, caso de uso y requisitos de rendimiento.

## 🏗️ Estructura de Archivos

```
config/
├── README_CONFIGURACIONES.md     # Este archivo
├── default_config.json          # Configuración básica original
├── advanced_config.json         # Configuración completa con perfiles
├── cpu_optimized.json           # Optimizada para CPU
├── high_precision.json          # Alta precisión con GPU potente
├── webcam_realtime.json         # Tiempo real con webcam
└── [custom_configs.json]        # Tus configuraciones personalizadas
```

## 🚀 Inicio Rápido

### 1. **Usar Config Manager (Recomendado)**

```bash
# Ver configuraciones disponibles
python config_manager.py list

# Obtener recomendación para tu hardware
python config_manager.py recommend

# Ejecutar con configuración específica
python config_manager.py run balanced --video data/video/test.mp4
```

### 2. **Uso Directo**

```bash
# Configuración balanceada
python src/main_reid.py --config config/advanced_config.json --video data/video/test.mp4

# CPU optimizada
python src/main_reid.py --config config/cpu_optimized.json --video data/video/test.mp4

# Alta precisión
python src/main_reid.py --config config/high_precision.json --video data/video/test.mp4
```

## 📊 Configuraciones Predefinidas

### **🔋 CPU Optimized** (`cpu_optimized.json`)
- **Target:** Sistemas sin GPU o con GPU lenta
- **Performance:** 15-25 FPS en 720p
- **Modelos:** YOLOv8n + OSNet_x0_25
- **RAM:** < 4GB

```bash
python config_manager.py run cpu --video mi_video.mp4
```

### **⚖️ Advanced/Balanced** (`advanced_config.json`)
- **Target:** GPU media, uso general
- **Performance:** 15-25 FPS en 1080p
- **Modelos:** YOLOv8s + OSNet_x1_0
- **Perfiles:** ultrafast, fast, balanced, accurate, ultra_accurate

```bash
python config_manager.py run advanced --video mi_video.mp4
```

### **🎯 High Precision** (`high_precision.json`)
- **Target:** GPU potente, aplicaciones críticas
- **Performance:** 5-15 FPS en 1080p+
- **Modelos:** YOLOv8l + OSNet_AIN_x1_0
- **Características:** Logging detallado, métricas avanzadas, debugging

```bash
python config_manager.py run gpu --video mi_video.mp4
```

### **📹 Webcam Realtime** (`webcam_realtime.json`)
- **Target:** Webcam en tiempo real
- **Performance:** 20-30 FPS en 720p
- **Modelos:** YOLOv8s + OSNet_x0_75
- **Características:** Baja latencia, controles interactivos

```bash
python config_manager.py run webcam --video 0
```

## 🎛️ Config Manager - Herramienta de Gestión

### **Comandos Principales**

#### **📋 Listar Configuraciones**
```bash
python config_manager.py list
```
Muestra todas las configuraciones disponibles con sus características.

#### **🔍 Ver Modelos Disponibles**
```bash
# Todos los modelos
python config_manager.py models

# Solo modelos YOLO
python config_manager.py models --type yolo

# Solo modelos Re-ID
python config_manager.py models --type reid
```

#### **💡 Recomendación Automática**
```bash
python config_manager.py recommend
```
Analiza tu hardware y recomienda la mejor configuración.

#### **🛠️ Crear Configuración Personalizada**
```bash
# Crear basada en configuración avanzada
python config_manager.py create mi_config --base advanced

# Crear basada en configuración CPU
python config_manager.py create mi_config_cpu --base cpu
```

#### **✅ Validar Configuración**
```bash
python config_manager.py validate config/mi_config.json
```

#### **📊 Comparar Configuraciones**
```bash
python config_manager.py compare cpu gpu
```

#### **⚡ Optimizar para Hardware**
```bash
python config_manager.py optimize advanced --output mi_optimizada
```

#### **🚀 Ejecutar con Configuración**
```bash
# Video específico
python config_manager.py run balanced --video data/video/test.mp4

# Webcam
python config_manager.py run webcam --video 0

# Con argumentos adicionales
python config_manager.py run gpu --video test.mp4 --args --output result.mp4
```

## 🎯 Selección de Configuración por Caso de Uso

### **🏢 Seguridad Corporativa**
```bash
python config_manager.py run gpu --video camara_entrada.mp4
```
- Alta precisión, logging detallado
- Guarda video procesado y embeddings
- Métricas de seguridad avanzadas

### **🛍️ Retail Analytics**
```bash
python config_manager.py run balanced --video camara_tienda.mp4
```
- Balance velocidad-precisión
- Seguimiento de trayectorias
- Análisis de patrones de movimiento

### **🚇 Transporte Público**
```bash
python config_manager.py run cpu --video camara_estacion.mp4
```
- Optimizado para multitudes
- Procesamiento eficiente
- Memoria limitada para muchas personas

### **🔬 Investigación/Análisis**
```bash
python config_manager.py run gpu --video evidence.mp4
```
- Máxima precisión disponible
- Debugging completo activado
- Métricas detalladas

### **📱 Demo/Prototipo**
```bash
python config_manager.py run webcam --video 0
```
- Tiempo real interactivo
- Interfaz visual optimizada
- Controles de teclado

## 🔧 Personalización Avanzada

### **Estructura de Configuración**

```json
{
  "_meta": {
    "version": "1.0",
    "description": "Mi configuración personalizada",
    "profile": "custom"
  },
  
  "models": {
    "yolo_model": "yolov8s.pt",
    "reid_model": "osnet_x1_0"
  },
  
  "hardware": {
    "device": "cuda",
    "mixed_precision": true,
    "num_workers": 4
  },
  
  "detection": {
    "min_detection_confidence": 0.5,
    "max_detections_per_frame": 50,
    "input_resolution": [640, 640]
  },
  
  "reid": {
    "similarity_threshold": 0.85,
    "embedding_size": 512,
    "preprocessing": {
      "target_size": [256, 128]
    }
  },
  
  "memory_management": {
    "disappearance_limit_seconds": 7,
    "max_stored_embeddings": 200,
    "cleanup_interval_frames": 300
  }
}
```

### **Parámetros Clave para Ajustar**

#### **similarity_threshold** (0.7 - 0.95)
```json
{
  "similarity_threshold": 0.85
}
```
- **0.75-0.80:** Permisivo, más re-identificaciones
- **0.85-0.90:** Balanceado (recomendado)
- **0.90-0.95:** Estricto, menos falsos positivos

#### **disappearance_limit_seconds** (3 - 20)
```json
{
  "disappearance_limit_seconds": 7
}
```
- **3-5:** Memoria corta, multitudes/CPU limitado
- **7-10:** Memoria media (recomendado)
- **15-20:** Memoria larga, análisis detallado

#### **Selección de Modelos**
```json
{
  "models": {
    "yolo_model": "yolov8s.pt",    // n=rápido, s=balanceado, m/l=preciso
    "reid_model": "osnet_x1_0"     // x0_25=rápido, x1_0=preciso
  }
}
```

## 📈 Optimización de Rendimiento

### **Por Tipo de Hardware**

#### **💻 CPU Únicamente**
```json
{
  "device": "cpu",
  "models": {
    "yolo_model": "yolov8n.pt",
    "reid_model": "osnet_x0_25"
  },
  "similarity_threshold": 0.75,
  "max_stored_embeddings": 50
}
```

#### **🚀 GPU Media (GTX 1060, RTX 2060)**
```json
{
  "device": "cuda",
  "mixed_precision": true,
  "models": {
    "yolo_model": "yolov8s.pt",
    "reid_model": "osnet_x1_0"
  },
  "max_stored_embeddings": 200
}
```

#### **🔥 GPU Potente (RTX 3080+)**
```json
{
  "device": "cuda",
  "mixed_precision": false,
  "models": {
    "yolo_model": "yolov8l.pt",
    "reid_model": "osnet_ain_x1_0"
  },
  "max_stored_embeddings": 1000
}
```

### **Por Target de FPS**

#### **🎯 Target: 30+ FPS**
```bash
python config_manager.py create ultrafast_custom --base cpu
# Editar: yolov8n + osnet_x0_25 + threshold 0.75
```

#### **🎯 Target: 15-25 FPS**
```bash
python config_manager.py create balanced_custom --base advanced
# Usar perfil "balanced" o "fast"
```

#### **🎯 Target: 5-15 FPS (Alta Precisión)**
```bash
python config_manager.py create precision_custom --base gpu
# yolov8l + osnet_ain_x1_0 + threshold 0.90
```

## 🚨 Troubleshooting

### **❌ Error: "CUDA out of memory"**

**Solución 1: Usar configuración CPU**
```bash
python config_manager.py run cpu --video mi_video.mp4
```

**Solución 2: Optimizar configuración actual**
```bash
python config_manager.py optimize mi_config --output mi_config_optimized
```

**Solución 3: Reducir parámetros manualmente**
```json
{
  "max_stored_embeddings": 50,        // Reducir de 200
  "mixed_precision": true,            // Activar FP16
  "input_resolution": [480, 480]      // Reducir resolución
}
```

### **⚠️ Rendimiento muy lento**

**Diagnóstico:**
```bash
python config_manager.py validate config/mi_config.json
python config_manager.py recommend
```

**Soluciones:**
1. **Cambiar a modelo más rápido:**
   ```json
   {
     "yolo_model": "yolov8n.pt",
     "reid_model": "osnet_x0_25"
   }
   ```

2. **Reducir calidad:**
   ```json
   {
     "input_resolution": [480, 480],
     "max_detections_per_frame": 20
   }
   ```

3. **Usar configuración CPU optimizada:**
   ```bash
   python config_manager.py run cpu --video mi_video.mp4
   ```

### **🔄 Muchos falsos positivos**

**Solución:**
```json
{
  "similarity_threshold": 0.90,      // Aumentar de 0.85
  "min_detection_confidence": 0.6,   // Aumentar de 0.5
  "reid_model": "osnet_ain_x1_0"     // Modelo más preciso
}
```

### **❌ Personas no se re-identifican**

**Solución:**
```json
{
  "similarity_threshold": 0.80,           // Reducir de 0.85
  "disappearance_limit_seconds": 15,      // Aumentar memoria
  "reid_model": "resnet50"                // Modelo más robusto
}
```

## 📚 Ejemplos Prácticos

### **Ejemplo 1: Configuración para Laptop con GPU integrada**
```bash
# 1. Crear configuración personalizada
python config_manager.py create laptop_config --base cpu

# 2. Editar el archivo generado
# config/laptop_config.json - cambiar device a "cuda" si tiene GPU

# 3. Optimizar automáticamente
python config_manager.py optimize laptop_config --output laptop_optimized

# 4. Probar
python config_manager.py run laptop_optimized --video test.mp4
```

### **Ejemplo 2: Configuración para Servidor de Análisis**
```bash
# 1. Usar configuración de alta precisión
python config_manager.py run gpu --video security_footage.mp4

# 2. O crear configuración servidor personalizada
python config_manager.py create server_config --base gpu

# 3. Editar para batch processing y logging avanzado
# Activar: save_video, save_embeddings, detailed_logging
```

### **Ejemplo 3: Configuración para Demo en Tiempo Real**
```bash
# 1. Usar configuración webcam
python config_manager.py run webcam --video 0

# 2. O personalizar para tu demo
python config_manager.py create demo_config --base webcam
# Editar: colores, UI, controles específicos
```

## 🔄 Flujo de Trabajo Recomendado

### **Para Nuevos Usuarios:**
1. **Análisis inicial:**
   ```bash
   python config_manager.py recommend
   ```

2. **Prueba con configuración recomendada:**
   ```bash
   python config_manager.py run [recomendada] --video test_video.mp4
   ```

3. **Ajuste fino:**
   ```bash
   python config_manager.py create mi_config --base [recomendada]
   # Editar config/mi_config.json según necesidades
   ```

### **Para Usuarios Avanzados:**
1. **Crear configuración base:**
   ```bash
   python config_manager.py create proyecto_config --base advanced
   ```

2. **Optimizar para hardware:**
   ```bash
   python config_manager.py optimize proyecto_config --output proyecto_optimized
   ```

3. **Validar y comparar:**
   ```bash
   python config_manager.py validate config/proyecto_optimized.json
   python config_manager.py compare proyecto_config proyecto_optimized
   ```

4. **Pruebas iterativas:**
   ```bash
   python config_manager.py run proyecto_optimized --video dataset/*.mp4
   ```

## 📊 Métricas y Monitoreo

### **Activar Métricas Detalladas**
```json
{
  "metrics": {
    "calculate_metrics": true,
    "metrics_file": "output/metrics.json",
    "track_performance": true,
    "save_detailed_stats": true
  }
}
```

### **Monitoreo en Tiempo Real**
```json
{
  "logging": {
    "level": "INFO",
    "progress_bar": true,
    "log_interval_frames": 100
  },
  "visualization": {
    "ui_elements": {
      "show_fps": true,
      "show_memory_usage": true,
      "show_reid_count": true
    }
  }
}
```

## ⚙️ Integración con Otros Sistemas

### **API/Microservicio**
```json
{
  "output": {
    "save_video": false,
    "real_time_display": false,
    "save_embeddings": true,
    "save_statistics": true
  },
  "logging": {
    "log_to_file": true,
    "console_output": false
  }
}
```

### **Análisis Batch**
```json
{
  "input": {
    "batch_processing": true,
    "input_directory": "videos/",
    "output_directory": "results/"
  },
  "performance": {
    "parallel_processing": true,
    "max_concurrent": 4
  }
}
```

## 🔗 Enlaces Útiles

- **[Documentación Completa de Modelos](MODELOS_REID_COMPLETO.md)**
- **[GitHub Torchreid](https://github.com/KaiyangZhou/deep-person-reid)**
- **[Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)**

## ❓ FAQ

**Q: ¿Cuál es la mejor configuración para empezar?**
A: Ejecuta `python config_manager.py recommend` para obtener una recomendación personalizada.

**Q: ¿Cómo sé si mi configuración es óptima?**
A: Usa `python config_manager.py validate` y compara métricas de FPS y precisión.

**Q: ¿Puedo usar múltiples configuraciones en el mismo proyecto?**
A: Sí, puedes crear configuraciones específicas para diferentes videos o casos de uso.

**Q: ¿Las configuraciones son compatibles entre versiones?**
A: Sí, el sistema mantiene retrocompatibilidad. Usa `validate` para verificar.

---

**💡 Tip:** Siempre comienza con `python config_manager.py recommend` para obtener la mejor configuración para tu sistema.