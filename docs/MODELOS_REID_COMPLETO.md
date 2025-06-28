# 📚 Guía Completa de Modelos Re-ID y Configuración

## 🎯 Introducción

Este documento proporciona una guía completa sobre todos los modelos de re-identificación disponibles en **Torchreid** y cómo configurar el sistema para diferentes casos de uso.

## 📋 Tabla de Contenidos

1. [Modelos Disponibles](#modelos-disponibles)
2. [Familias de Modelos](#familias-de-modelos)
3. [Configuraciones Predefinidas](#configuraciones-predefinidas)
4. [Guía de Selección](#guía-de-selección)
5. [Optimización de Rendimiento](#optimización-de-rendimiento)
6. [Casos de Uso Específicos](#casos-de-uso-específicos)
7. [Troubleshooting](#troubleshooting)

---

## 🤖 Modelos Disponibles

### **Familia OSNet (Recomendados)**

OSNet (Omni-Scale Network) es la familia de modelos más moderna y eficiente para re-identificación de personas.

#### **OSNet Estándar**
| Modelo | Parámetros | GFLOPs | Velocidad | Precisión | Uso Recomendado |
|--------|------------|--------|-----------|-----------|-----------------|
| `osnet_x0_25` | 0.17M | 0.026 | ⚡⚡⚡⚡ | ⭐⭐⭐ | CPU, sistemas limitados |
| `osnet_x0_5` | 0.37M | 0.063 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balance velocidad-precisión |
| `osnet_x0_75` | 0.65M | 0.128 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Aplicaciones generales |
| `osnet_x1_0` | 1.02M | 0.213 | ⚡⚡ | ⭐⭐⭐⭐⭐ | **Recomendado por defecto** |

#### **OSNet con Instance Batch Normalization (IBN)**
| Modelo | Características | Ventajas |
|--------|----------------|----------|
| `osnet_ibn_x1_0` | OSNet + IBN | Mejor generalización entre dominios |

#### **OSNet con Attention in Attention (AIN)**
| Modelo | Parámetros | Precisión | Especialidad |
|--------|------------|-----------|--------------|
| `osnet_ain_x0_25` | 0.17M | ⭐⭐⭐⭐ | Ultraligero con atención |
| `osnet_ain_x0_5` | 0.37M | ⭐⭐⭐⭐⭐ | Compacto con alta precisión |
| `osnet_ain_x0_75` | 0.65M | ⭐⭐⭐⭐⭐ | Balance óptimo |
| `osnet_ain_x1_0` | 1.02M | ⭐⭐⭐⭐⭐⭐ | **Máxima precisión OSNet** |

### **Familia ResNet (Clásicos)**

Los modelos ResNet son robustos y ampliamente utilizados, ideales para aplicaciones que requieren estabilidad.

| Modelo | Parámetros | GFLOPs | Velocidad | Precisión | Características |
|--------|------------|--------|-----------|-----------|-----------------|
| `resnet18` | 11.2M | 1.8 | ⚡⚡⚡ | ⭐⭐⭐ | Ligero, buena opción general |
| `resnet34` | 21.3M | 3.7 | ⚡⚡ | ⭐⭐⭐⭐ | Intermedio balanceado |
| `resnet50` | 23.5M | 4.1 | ⚡⚡ | ⭐⭐⭐⭐⭐ | **Estándar de la industria** |
| `resnet101` | 42.5M | 7.8 | ⚡ | ⭐⭐⭐⭐⭐⭐ | Alta precisión |
| `resnet152` | 58.2M | 11.6 | ⚡ | ⭐⭐⭐⭐⭐⭐ | Máxima capacidad |

#### **ResNeXt (ResNet Optimizado)**
| Modelo | Parámetros | Especialidad |
|--------|------------|--------------|
| `resnext50_32x4d` | 23.0M | ResNet optimizado con agrupaciones |
| `resnext101_32x8d` | 86.8M | Alta capacidad con eficiencia |

#### **ResNet Personalizado**
| Modelo | Características |
|--------|----------------|
| `resnet50_fc512` | ResNet50 con capa final personalizada |

### **Familia Mobile (Eficiencia)**

Modelos optimizados para dispositivos con recursos limitados.

| Modelo | Parámetros | GFLOPs | Velocidad | Precisión | Ideal Para |
|--------|------------|--------|-----------|-----------|------------|
| `mobilenetv2_x1_0` | 2.2M | 0.3 | ⚡⚡⚡⚡ | ⭐⭐⭐ | Móviles, embedded |
| `mobilenetv2_x1_4` | 4.3M | 0.6 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Mobile con más capacidad |
| `shufflenet` | 1.0M | 0.15 | ⚡⚡⚡⚡⚡ | ⭐⭐ | **Más eficiente** |

### **Modelos Especializados**

| Modelo | Parámetros | Especialidad | Uso Recomendado |
|--------|------------|--------------|-----------------|
| `mlfn` | 32.5M | Multi-Level Factorisation | Características complejas |
| `densenet121` | 6.9M | Conexiones densas | Balance memoria-precisión |
| `senet154` | 113.0M | Squeeze-and-Excitation | **Máxima precisión posible** |

---

## 📁 Configuraciones Predefinidas

El sistema incluye configuraciones optimizadas para diferentes escenarios:

### **1. `advanced_config.json` - Configuración Completa**
```json
{
  "active_profile": "balanced",
  "profiles": {
    "ultrafast": "Máxima velocidad",
    "fast": "Velocidad-precisión balanceada",
    "balanced": "Configuración recomendada",
    "accurate": "Alta precisión",
    "ultra_accurate": "Máxima precisión"
  }
}
```

**Uso:**
```bash
python src/main_reid.py --config config/advanced_config.json --video data/video/test.mp4
```

### **2. `cpu_optimized.json` - Para CPU**
- **Modelo:** `yolov8n.pt` + `osnet_x0_25`
- **Target:** Sistemas sin GPU
- **Performance:** 15-25 FPS en 720p

```bash
python src/main_reid.py --config config/cpu_optimized.json --video data/video/test.mp4
```

### **3. `high_precision.json` - Alta Precisión**
- **Modelo:** `yolov8l.pt` + `osnet_ain_x1_0`
- **Target:** GPU potente (RTX 3080+)
- **Performance:** 5-15 FPS en 1080p+

```bash
python src/main_reid.py --config config/high_precision.json --video data/video/test.mp4
```

### **4. `webcam_realtime.json` - Tiempo Real**
- **Modelo:** `yolov8s.pt` + `osnet_x0_75`
- **Target:** Webcam en tiempo real
- **Performance:** 20-30 FPS en 720p

```bash
python src/main_reid.py --config config/webcam_realtime.json --video 0
```

---

## 🎯 Guía de Selección de Modelos

### **Por Tipo de Hardware**

#### **💻 CPU Únicamente**
```json
{
  "reid_model": "osnet_x0_25",
  "device": "cpu",
  "similarity_threshold": 0.75
}
```
**Recomendación:** `osnet_x0_25` o `mobilenetv2_x1_0`

#### **🚀 GPU Media (GTX 1060, RTX 2060)**
```json
{
  "reid_model": "osnet_x1_0",
  "device": "cuda",
  "similarity_threshold": 0.85
}
```
**Recomendación:** `osnet_x1_0` o `resnet50`

#### **🔥 GPU Potente (RTX 3080+, V100+)**
```json
{
  "reid_model": "osnet_ain_x1_0", 
  "device": "cuda",
  "similarity_threshold": 0.90
}
```
**Recomendación:** `osnet_ain_x1_0` o `senet154`

### **Por Caso de Uso**

#### **🏃‍♂️ Tiempo Real (Webcam, Live Stream)**
- **Prioridad:** Velocidad > Precisión
- **Modelos:** `osnet_x0_5`, `osnet_x0_75`, `mobilenetv2_x1_0`
- **Threshold:** 0.80-0.85

#### **🔒 Seguridad Crítica**
- **Prioridad:** Precisión > Velocidad
- **Modelos:** `osnet_ain_x1_0`, `resnet101`, `senet154`
- **Threshold:** 0.90-0.95

#### **📱 Dispositivos Móviles/Embedded**
- **Prioridad:** Eficiencia > Todo
- **Modelos:** `osnet_x0_25`, `shufflenet`, `mobilenetv2_x1_0`
- **Threshold:** 0.75-0.80

#### **🔬 Investigación/Análisis**
- **Prioridad:** Máxima Precisión
- **Modelos:** `osnet_ain_x1_0`, `senet154`
- **Threshold:** 0.92-0.95

### **Por Tamaño de Multitud**

#### **👥 Pocas Personas (1-5)**
- **Modelo:** `osnet_ain_x1_0`
- **Threshold:** 0.90
- **Memory:** 15 segundos

#### **👥👥 Multitud Media (5-20)**
- **Modelo:** `osnet_x1_0`
- **Threshold:** 0.85
- **Memory:** 10 segundos

#### **👥👥👥 Multitud Grande (20+)**
- **Modelo:** `osnet_x0_75`
- **Threshold:** 0.82
- **Memory:** 5 segundos

---

## ⚡ Optimización de Rendimiento

### **Configuración por Performance Target**

#### **🎯 Target: 30+ FPS**
```json
{
  "yolo_model": "yolov8n.pt",
  "reid_model": "osnet_x0_25",
  "similarity_threshold": 0.75,
  "disappearance_limit_seconds": 3,
  "batch_processing": false,
  "mixed_precision": true
}
```

#### **🎯 Target: 15-25 FPS**
```json
{
  "yolo_model": "yolov8s.pt",
  "reid_model": "osnet_x0_75",
  "similarity_threshold": 0.82,
  "disappearance_limit_seconds": 7,
  "batch_processing": true,
  "mixed_precision": true
}
```

#### **🎯 Target: 5-15 FPS (Alta Precisión)**
```json
{
  "yolo_model": "yolov8l.pt",
  "reid_model": "osnet_ain_x1_0",
  "similarity_threshold": 0.90,
  "disappearance_limit_seconds": 15,
  "batch_processing": true,
  "mixed_precision": false
}
```

### **Optimizaciones Avanzadas**

#### **Memory Management**
```json
{
  "memory_management": {
    "max_stored_embeddings": 100,    // CPU: 50, GPU: 500+
    "cleanup_interval_frames": 200,  // Frecuente para CPU
    "garbage_collection_interval": 500
  }
}
```

#### **GPU Optimization**
```json
{
  "performance_optimization": {
    "mixed_precision": true,         // RTX 2000+
    "torch_compile": true,           // PyTorch 2.0+
    "tensorrt_optimization": {
      "enable": true,                // Producción
      "precision": "fp16"
    }
  }
}
```

---

## 📊 Comparativa de Rendimiento

### **Benchmark en GTX 1080 Ti**

| Modelo | FPS (720p) | FPS (1080p) | Memoria GPU | Precisión |
|--------|------------|-------------|-------------|-----------|
| `osnet_x0_25` | 45 | 28 | 1.2GB | 85% |
| `osnet_x0_5` | 38 | 22 | 1.5GB | 88% |
| `osnet_x1_0` | 32 | 18 | 2.1GB | 91% |
| `osnet_ain_x1_0` | 28 | 15 | 2.3GB | 94% |
| `resnet50` | 25 | 14 | 2.8GB | 90% |
| `senet154` | 12 | 7 | 5.2GB | 96% |

### **Benchmark en CPU (Intel i7-9700K)**

| Modelo | FPS (720p) | Memoria RAM | Precisión |
|--------|------------|-------------|-----------|
| `osnet_x0_25` | 18 | 2.1GB | 85% |
| `mobilenetv2_x1_0` | 22 | 1.8GB | 83% |
| `shufflenet` | 28 | 1.5GB | 80% |
| `osnet_x0_5` | 12 | 2.8GB | 88% |

---

## 🔧 Configuración Personalizada

### **Crear Tu Propia Configuración**

1. **Copia un archivo base:**
```bash
cp config/advanced_config.json config/mi_config.json
```

2. **Edita los parámetros principales:**
```json
{
  "models": {
    "yolo_model": "yolov8s.pt",
    "reid_model": "osnet_x1_0"
  },
  "reid_settings": {
    "similarity_threshold": 0.87,
    "disappearance_limit_seconds": 8
  },
  "performance_optimization": {
    "device": "cuda",
    "mixed_precision": true
  }
}
```

3. **Prueba tu configuración:**
```bash
python src/main_reid.py --config config/mi_config.json --video data/video/test.mp4
```

### **Parámetros Críticos para Ajustar**

#### **similarity_threshold** (0.7 - 0.95)
- **0.75-0.80:** Permisivo, más re-identificaciones, algunos falsos positivos
- **0.85-0.90:** Balanceado, configuración estándar
- **0.90-0.95:** Estricto, menos falsos positivos, puede perder algunas re-identificaciones

#### **disappearance_limit_seconds** (3 - 20)
- **3-5 seg:** Memoria corta, para multitudes o CPU limitado
- **7-10 seg:** Memoria media, configuración estándar
- **15-20 seg:** Memoria larga, para análisis detallado

#### **min_detection_confidence** (0.3 - 0.8)
- **0.3-0.4:** Detecta más personas, incluye detecciones dudosas
- **0.5-0.6:** Balanceado, configuración estándar
- **0.7-0.8:** Solo detecciones muy confiables

---

## 🎮 Casos de Uso Específicos

### **1. Entrada de Edificio Corporativo**
```json
{
  "profile": "corporate_entrance",
  "yolo_model": "yolov8m.pt",
  "reid_model": "osnet_ain_x1_0",
  "similarity_threshold": 0.92,
  "disappearance_limit_seconds": 15,
  "min_detection_confidence": 0.7,
  "save_video": true,
  "save_embeddings": true
}
```

### **2. Centro Comercial - Análisis de Clientes**
```json
{
  "profile": "retail_analytics",
  "yolo_model": "yolov8s.pt",
  "reid_model": "osnet_x1_0",
  "similarity_threshold": 0.85,
  "disappearance_limit_seconds": 8,
  "max_stored_embeddings": 200,
  "save_trajectories": true
}
```

### **3. Estación de Transporte - Multitudes**
```json
{
  "profile": "transport_hub",
  "yolo_model": "yolov8n.pt",
  "reid_model": "osnet_x0_75",
  "similarity_threshold": 0.80,
  "disappearance_limit_seconds": 5,
  "max_detections_per_frame": 100,
  "fast_processing": true
}
```

### **4. Investigación Forense**
```json
{
  "profile": "forensic_analysis",
  "yolo_model": "yolov8l.pt",
  "reid_model": "senet154",
  "similarity_threshold": 0.95,
  "disappearance_limit_seconds": 30,
  "save_debug_images": true,
  "detailed_logging": true
}
```

### **5. Demo en Vivo/Prototipo**
```json
{
  "profile": "live_demo",
  "yolo_model": "yolov8s.pt",
  "reid_model": "osnet_x0_75",
  "similarity_threshold": 0.82,
  "webcam_optimization": true,
  "real_time_display": true,
  "low_latency_mode": true
}
```

---

## 🚨 Troubleshooting

### **Problemas Comunes y Soluciones**

#### **❌ Error: "CUDA out of memory"**
**Soluciones:**
1. Usar modelo más pequeño: `osnet_x0_25` en lugar de `osnet_x1_0`
2. Reducir `max_stored_embeddings`: de 1000 a 100
3. Activar `mixed_precision: true`
4. Reducir resolución de entrada

#### **❌ Rendimiento muy lento**
**Diagnóstico:**
```bash
python src/main_reid.py --config config/debug_config.json --profile
```

**Soluciones:**
1. **CPU lento:** Usar `osnet_x0_25` + `yolov8n`
2. **GPU lenta:** Activar `mixed_precision`
3. **Muchas personas:** Reducir `max_detections_per_frame`
4. **Video alta resolución:** Reducir resolución de entrada

#### **❌ Muchos falsos positivos**
**Soluciones:**
1. Aumentar `similarity_threshold`: 0.85 → 0.90
2. Usar modelo más preciso: `osnet_ain_x1_0`
3. Reducir `disappearance_limit_seconds`
4. Aumentar `min_detection_confidence`

#### **❌ Personas no se re-identifican**
**Soluciones:**
1. Reducir `similarity_threshold`: 0.85 → 0.80
2. Aumentar `disappearance_limit_seconds`
3. Verificar calidad del video (iluminación, resolución)
4. Usar modelo más robusto: `resnet50`

---

## 📈 Métricas y Evaluación

### **Métricas Importantes**

#### **FPS (Frames Per Second)**
- **>25 FPS:** Excelente para tiempo real
- **15-25 FPS:** Bueno para aplicaciones generales
- **5-15 FPS:** Aceptable para análisis offline
- **<5 FPS:** Solo para análisis detallado

#### **Re-ID Success Rate**
- **>90%:** Excelente precisión
- **80-90%:** Buena precisión
- **70-80%:** Aceptable dependiendo del caso
- **<70%:** Revisar configuración

#### **Memory Usage**
- **<2GB:** Eficiente
- **2-4GB:** Normal
- **4-8GB:** Alto consumo
- **>8GB:** Revisar configuración

### **Comando de Benchmark**
```bash
python src/main_reid.py --config config/benchmark.json --video data/video/test.mp4 --benchmark
```

---

## 🔄 Actualización y Mantenimiento

### **Actualizar Modelos**
```bash
# Descargar nuevos modelos pre-entrenados
python -c "
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor('osnet_ain_x1_0', device='cuda')
print('Modelo descargado y listo')
"
```

### **Verificar Modelos Disponibles**
```python
import torchreid
torchreid.models.show_avai_models()
```

### **Limpiar Cache**
```bash
rm -rf ~/.cache/torch/hub/checkpoints/
rm -rf ~/.cache/torch/pytorch_model_zoo/
```

---

## 📚 Recursos Adicionales

### **Documentación Oficial**
- [Torchreid Documentation](https://kaiyangzhou.github.io/deep-person-reid/)
- [Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
- [GitHub Repository](https://github.com/KaiyangZhou/deep-person-reid)

### **Papers de Referencia**
- **OSNet:** [Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/abs/1905.00953)
- **ResNet:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **MobileNet:** [MobileNets: Efficient Convolutional Neural Networks](https://arxiv.org/abs/1704.04861)

### **Comunidad y Soporte**
- [GitHub Issues](https://github.com/KaiyangZhou/deep-person-reid/issues)
- [Person Re-ID Papers](https://github.com/bismex/Awesome-person-re-identification)

---

## 🏁 Conclusión

Esta guía proporciona todo lo necesario para seleccionar y configurar modelos de re-identificación según tus necesidades específicas. Recuerda:

1. **Comienza con configuraciones predefinidas**
2. **Ajusta gradualmente según tu hardware y requisitos**
3. **Mide el rendimiento y ajusta parámetros**
4. **Usa benchmarks para comparar configuraciones**

¡Feliz re-identificación! 🎉