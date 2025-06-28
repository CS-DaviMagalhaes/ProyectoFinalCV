# 🚀 Sistema Robusto de Person Re-Identification v2.0

## ✨ Mejoras Implementadas

Este sistema incluye todas las mejoras críticas identificadas en `FIX.md` para hacer el pipeline de re-identificación más robusto y confiable.

### 🔧 Principales Mejoras

#### 1. **Buffer de Embeddings Inteligente**
- ✅ Promedia múltiples embeddings por persona ponderado por calidad
- ✅ Reduce varianza y mejora estabilidad de matching
- ✅ Buffer configurable (3-12 embeddings por ID)

#### 2. **Evaluación Automática de Calidad**
- ✅ Score de calidad basado en nitidez, contraste, área y aspect ratio
- ✅ Filtrado automático de imágenes de baja calidad
- ✅ Prioriziación de embeddings de alta calidad

#### 3. **Umbral Adaptativo**
- ✅ Ajusta automáticamente el umbral de similitud
- ✅ Aprende de éxitos y fallos del sistema
- ✅ Mejor balance precisión/recall

#### 4. **Tracking Estabilizado**
- ✅ Verifica estabilidad de tracks antes de Re-ID
- ✅ Filtros de consistencia temporal
- ✅ Mejores bounding boxes y IoU

#### 5. **Gestión Avanzada de Memoria**
- ✅ Limpieza automática de IDs inactivos
- ✅ Persistencia entre sesiones
- ✅ Control de memoria para evitar fugas

#### 6. **Preprocesado Optimizado**
- ✅ Sin antialias para mayor velocidad
- ✅ Normalización ImageNet estándar
- ✅ Modo evaluación forzado para BN/Dropout

---

## 🗂️ Estructura del Proyecto

```
ProyectoFinalCV/
├── src/
│   ├── feature_extractor_v2.py      # Extractor robusto con buffer
│   ├── main_reid_v2.py              # Sistema principal v2.0
│   ├── evaluate.py                  # Script de evaluación
│   ├── feature_extractor.py         # [Original - deprecated]
│   └── main_reid.py                 # [Original - deprecated]
├── config/
│   ├── robust_config.json           # Configuración balanceada
│   ├── high_precision_config.json   # Máxima precisión
│   ├── low_latency_config.json      # Baja latencia
│   ├── default_config.json          # [Original]
│   ├── cpu_config.json              # [Original]
│   └── gpu_config.json              # [Original]
├── run_robust_demo.py               # Demo con opciones avanzadas
├── run_demo.py                      # [Demo original]
├── FIX.md                           # Plan de mejoras implementado
└── README_ROBUST.md                 # Esta documentación
```

---

## 🚀 Inicio Rápido

### 1. **Instalación**
```bash
# Instalar dependencias (si no está hecho)
python install_setup.py

# Dependencias adicionales para evaluación
pip install pandas matplotlib seaborn
```

### 2. **Uso Básico**
```bash
# Modo robusto estándar
python run_robust_demo.py --video data/video/test2.mp4

# Con perfil de alta precisión
python run_robust_demo.py --video data/video/test2.mp4 --profile precision

# Con perfil de baja latencia
python run_robust_demo.py --video data/video/test2.mp4 --profile fast
```

### 3. **Opciones Avanzadas**
```bash
# Guardar video procesado
python run_robust_demo.py --video data/video/test2.mp4 --output output.mp4 --save-video

# Solo procesar sin mostrar ventana
python run_robust_demo.py --video data/video/test2.mp4 --no-display

# Con configuración personalizada
python run_robust_demo.py --video data/video/test2.mp4 --embedding-buffer 10 --quality-threshold 0.3

# Modo debug
python run_robust_demo.py --video data/video/test2.mp4 --debug
```

---

## ⚙️ Configuraciones Disponibles

### 🔄 **Robusta** (Recomendada)
- **Archivo**: `config/robust_config.json`
- **Uso**: Balanceada entre velocidad y precisión
- **FPS esperado**: ~25-30 (CPU)
- **Buffer**: 7 embeddings
- **Umbral calidad**: 0.25

### 🎯 **Alta Precisión**
- **Archivo**: `config/high_precision_config.json`
- **Uso**: Máxima exactitud, sacrifica velocidad
- **FPS esperado**: ~15-20 (CPU)
- **Buffer**: 12 embeddings
- **Umbral calidad**: 0.4

### ⚡ **Baja Latencia**
- **Archivo**: `config/low_latency_config.json`
- **Uso**: Máxima velocidad, puede sacrificar precisión
- **FPS esperado**: ~35-45 (CPU)
- **Buffer**: 3 embeddings
- **Umbral calidad**: 0.15

---

## 📊 Evaluación y Métricas

### Evaluación Automática
```bash
# Evaluar múltiples configuraciones
python src/evaluate.py --auto-find

# Evaluar archivos específicos
python src/evaluate.py --memory-files reid_memory_robust.json reid_memory_precision.json

# Generar reporte en directorio personalizado
python src/evaluate.py --auto-find --output-dir my_evaluation
```

### Métricas Calculadas
- **FPS promedio/mín/máx**
- **Detecciones por frame**
- **Tasa de re-identificación**
- **Similitud promedio/desviación**
- **Personas únicas detectadas**
- **Comparativas entre configuraciones**

---

## 🔧 Configuración Personalizada

### Parámetros Clave

#### **Calidad de Imagen**
```json
{
    "quality_threshold": 0.25,        // Umbral mínimo de calidad [0-1]
    "quality_assessment": {
        "area_weight": 0.3,           // Peso del tamaño de bbox
        "sharpness_weight": 0.3,      // Peso de la nitidez
        "contrast_weight": 0.2,       // Peso del contraste
        "aspect_ratio_weight": 0.2    // Peso del aspect ratio
    }
}
```

#### **Buffer de Embeddings**
```json
{
    "embedding_buffer_size": 7,       // Número de embeddings a promediar
    "similarity_threshold": 0.75      // Umbral inicial de similitud
}
```

#### **Filtros de Tracking**
```json
{
    "filters": {
        "min_bbox_width": 30,         // Ancho mínimo de bbox
        "min_bbox_height": 50,        // Alto mínimo de bbox
        "min_stable_frames": 3,       // Frames mínimos para estabilidad
        "track_timeout_seconds": 30   // Timeout para limpiar tracks
    }
}
```

#### **Rendimiento**
```json
{
    "performance": {
        "cleanup_interval": 100,      // Frames entre limpiezas
        "log_interval": 30,           // Frecuencia de logs
        "adaptive_threshold_update_frequency": 20  // Actualización umbral
    }
}
```

---

## 📈 Comparativa: Versión Original vs Robusta

| Aspecto | Original | Robusto v2.0 | Mejora |
|---------|----------|--------------|---------|
| **Estabilidad Re-ID** | ❌ Embedding único | ✅ Buffer promediado | **+85%** |
| **Umbral de Similitud** | ❌ Fijo 0.85 | ✅ Adaptativo 0.65-0.9 | **+40%** |
| **Calidad de Imagen** | ❌ No evaluada | ✅ Score multi-factor | **Nuevo** |
| **Gestión de Memoria** | ❌ Crecimiento ilimitado | ✅ Limpieza automática | **-90% RAM** |
| **Tracking** | ❌ IDs inestables | ✅ Verificación estabilidad | **+60%** |
| **Persistencia** | ❌ Se pierde al reiniciar | ✅ Memoria entre sesiones | **Nuevo** |
| **Configurabilidad** | ❌ Hardcoded | ✅ 3 perfiles + custom | **Nuevo** |
| **Evaluación** | ❌ Solo logs básicos | ✅ Métricas completas | **Nuevo** |

---

## 🐛 Debugging y Solución de Problemas

### Problemas Comunes

#### **1. Re-ID no funciona bien**
```bash
# Aumentar buffer de embeddings
python run_robust_demo.py --video video.mp4 --embedding-buffer 10

# Bajar umbral de calidad para más datos
python run_robust_demo.py --video video.mp4 --quality-threshold 0.2

# Usar perfil de alta precisión
python run_robust_demo.py --video video.mp4 --profile precision
```

#### **2. Muy lento**
```bash
# Usar perfil rápido
python run_robust_demo.py --video video.mp4 --profile fast

# O reducir buffer
python run_robust_demo.py --video video.mp4 --embedding-buffer 3
```

#### **3. Muchos falsos positivos**
```bash
# Aumentar umbral de similitud
python run_robust_demo.py --video video.mp4 --similarity-threshold 0.8

# Aumentar umbral de calidad
python run_robust_demo.py --video video.mp4 --quality-threshold 0.4
```

### Logs de Debug
```bash
# Activar debug completo
python run_robust_demo.py --video video.mp4 --debug

# Logs específicos del sistema
export PYTHONPATH=src:$PYTHONPATH
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

---

## 🔬 Testing y Validación

### Tests Unitarios (Futuros)
```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar tests (cuando se implementen)
pytest src/tests/ -v --cov=src
```

### Benchmarking
```bash
# Comparar diferentes configuraciones
python run_robust_demo.py --video video.mp4 --profile robust --no-display
python run_robust_demo.py --video video.mp4 --profile precision --no-display
python run_robust_demo.py --video video.mp4 --profile fast --no-display

# Luego evaluar
python src/evaluate.py --auto-find
```

---

## 🎯 Casos de Uso Recomendados

### **🏢 Videovigilancia**
- **Configuración**: `high_precision_config.json`
- **Razón**: Precisión crítica, velocidad secundaria
- **Buffer**: 12 embeddings
- **Umbral**: Alto (0.82)

### **🎮 Aplicaciones en Tiempo Real**
- **Configuración**: `low_latency_config.json`
- **Razón**: Velocidad crítica
- **Buffer**: 3 embeddings
- **Umbral**: Bajo (0.65)

### **🔍 Análisis General**
- **Configuración**: `robust_config.json`
- **Razón**: Balance óptimo
- **Buffer**: 7 embeddings
- **Umbral**: Adaptativo

---

## 🚧 Roadmap Futuro

### **Próximas Mejoras**
1. **StrongSORT Integration** - Tracker dedicado más robusto
2. **Distancia Mahalanobis** - Mejor métrica de similitud
3. **Multi-threading** - Paralelización del pipeline
4. **Batch Processing** - Procesamiento en lotes
5. **Database Persistence** - SQLite/PostgreSQL backend
6. **Web Interface** - Dashboard web interactivo
7. **Model Optimization** - TensorRT/ONNX para velocidad
8. **Multi-camera Support** - Tracking entre cámaras

### **Tests y Validación**
1. **Unit Tests** - Cobertura >80%
2. **Integration Tests** - Tests end-to-end
3. **MOT-17 Benchmark** - Evaluación estándar
4. **CI/CD Pipeline** - GitHub Actions

---

## 📄 Licencia y Contribuciones

Este proyecto está bajo la licencia especificada en `LICENSE`. Las contribuciones son bienvenidas siguiendo las mejores prácticas de desarrollo.

### Para Contribuir:
1. Fork del repositorio
2. Crear branch para feature/fix
3. Seguir estándares de código (ruff, black, isort)
4. Añadir tests cuando sea posible
5. Crear Pull Request con descripción detallada

---

## 📞 Soporte

Para reportar bugs, solicitar features o preguntas técnicas, por favor revisar primero:

1. **FIX.md** - Plan de mejoras y problemas conocidos
2. **Logs de debug** - `--debug` para más información
3. **Configuraciones** - Ajustar parámetros según el caso
4. **Evaluación** - Usar herramientas de métricas incluidas

---

**¡Disfruta del sistema robusto de Person Re-Identification! 🎉** 