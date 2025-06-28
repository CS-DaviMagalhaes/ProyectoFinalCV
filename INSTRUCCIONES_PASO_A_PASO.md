# 🚀 Instrucciones Paso a Paso - Sistema de Re-identificación de Personas

Esta guía te llevará desde la instalación hasta obtener tus primeros resultados de re-identificación de personas.

## 📋 Tabla de Contenidos
1. [Preparación Inicial](#preparación-inicial)
2. [Instalación](#instalación)
3. [Primer Uso](#primer-uso)
4. [Configuración Avanzada](#configuración-avanzada)
5. [Solución de Problemas](#solución-de-problemas)
6. [Experimentación](#experimentación)

## 1. Preparación Inicial

### ✅ Verificar Requisitos del Sistema

```bash
# Verificar Python (necesario 3.8+)
python --version

# Verificar si tienes GPU (opcional pero recomendado)
nvidia-smi
```

### 📁 Preparar el Espacio de Trabajo

```bash
# Crear carpeta para el proyecto
mkdir ~/re_identification_project
cd ~/re_identification_project

# Clonar el repositorio (reemplaza con la URL real)
git clone [URL_DEL_REPOSITORIO] .
```

## 2. Instalación

### Opción A: Instalación Automática (Recomendada) 🌟

```bash
# Una sola línea para instalar todo
python install_setup.py
```

**¿Qué hace este comando?**
- Verifica tu versión de Python
- Detecta si tienes GPU/CUDA
- Instala PyTorch (GPU o CPU según tu sistema)
- Instala todas las dependencias
- Clona e instala Torchreid
- Ejecuta pruebas para verificar que todo funciona

### Opción B: Instalación Manual (Si tienes problemas con la automática)

```bash
# 1. Instalar dependencias base
pip install -r requirements.txt

# 2. Instalar PyTorch
# Para GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar Torchreid
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
cd ..
```

### ✅ Verificar Instalación

```bash
# Probar que todo funciona
python -c "import torch; import cv2; from ultralytics import YOLO; print('✅ Todo instalado correctamente')"
```

## 3. Primer Uso

### Paso 1: Preparar un Video de Prueba

```bash
# Opción A: Usar tu propio video
cp /ruta/a/tu/video.mp4 data/video/mi_video.mp4

# Opción B: Descargar un video de prueba (ejemplo)
# wget https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4 -O data/video/test.mp4
```

### Paso 2: Primera Ejecución - Demo Básico

```bash
# Ejecutar demo automático
python run_demo.py --mode basic
```

**¿Qué verás?**
- El sistema detectará automáticamente tu video
- Se abrirá una ventana mostrando el video procesado
- Verás cajas delimitadoras alrededor de las personas
- Cada persona tendrá un ID único
- En la consola verás estadísticas en tiempo real

### Paso 3: Entender los Resultados

**En la pantalla verás:**
- 🟢 **Cajas verdes**: Personas nuevas detectadas
- 🟡 **Cajas amarillas**: Personas re-identificadas
- **ID numbers**: Identificador único para cada persona
- **Estadísticas**: Frame actual, personas activas, re-identificaciones

**En la consola verás:**
```
INFO - Nueva persona: Track 1 -> ID Global 0
INFO - Re-identificación: Track 3 -> ID Global 0 (similitud: 0.892)
INFO - Progreso: 100/500 frames - Tiempo/frame: 0.156s
```

### Paso 4: Guardar Resultados

```bash
# Ejecutar con guardado de video
python src/main_reid.py --video data/video/mi_video.mp4 --output output/resultado.mp4
```

## 4. Configuración Avanzada

### Ajustar Parámetros para Mejor Rendimiento

#### Para Sistema con GPU Potente:
```bash
python src/main_reid.py --video data/video/mi_video.mp4 --config config/gpu_config.json
```

#### Para Sistema Solo CPU:
```bash
python src/main_reid.py --video data/video/mi_video.mp4 --config config/cpu_config.json
```

### Personalizar Configuración

Edita `config/default_config.json`:

```json
{
  "similarity_threshold": 0.85,     // ⬆️ Más estricto, ⬇️ más permisivo
  "disappearance_limit_seconds": 5, // Tiempo de memoria
  "yolo_model": "yolov8n.pt",      // n=rápido, l=preciso
  "reid_model": "osnet_x1_0"       // x0_25=rápido, x1_0=preciso
}
```

### Casos de Uso Específicos

#### Escenario 1: Entrada de Edificio (Alta Precisión)
```json
{
  "similarity_threshold": 0.90,
  "disappearance_limit_seconds": 10,
  "yolo_model": "yolov8m.pt",
  "reid_model": "osnet_x1_0"
}
```

#### Escenario 2: Multitudes (Velocidad)
```json
{
  "similarity_threshold": 0.80,
  "disappearance_limit_seconds": 3,
  "yolo_model": "yolov8n.pt",
  "reid_model": "osnet_x0_25"
}
```

## 5. Solución de Problemas

### Problema: "CUDA not available"
```bash
# Verificar CUDA
nvidia-smi

# Si no tienes CUDA, usar configuración CPU
python src/main_reid.py --video data/video/test.mp4 --config config/cpu_config.json
```

### Problema: "No module named 'torchreid'"
```bash
# Reinstalar Torchreid
rm -rf deep-person-reid
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -e .
cd ..
```

### Problema: Video muy lento
```bash
# Usar modelo más ligero
python src/main_reid.py --video data/video/test.mp4 --config config/cpu_config.json

# O reducir resolución del video
ffmpeg -i video_original.mp4 -vf scale=640:480 data/video/video_small.mp4
```

### Problema: Muchos falsos positivos
1. **Incrementar umbral de similitud**:
   - Cambiar `similarity_threshold` de 0.85 a 0.90+
2. **Usar modelo Re-ID más preciso**:
   - Cambiar `reid_model` a `"osnet_x1_0"`
3. **Reducir tiempo de memoria**:
   - Cambiar `disappearance_limit_seconds` a 3

## 6. Experimentación

### Probar Diferentes Modelos

```bash
# Modelo ultrarrápido (para prototipos)
python src/main_reid.py --video data/video/test.mp4 \
  --config config/cpu_config.json

# Modelo balanceado (recomendado)
python src/main_reid.py --video data/video/test.mp4 \
  --config config/default_config.json

# Modelo de alta precisión (para producción)
python src/main_reid.py --video data/video/test.mp4 \
  --config config/gpu_config.json
```

### Usar Webcam en Tiempo Real

```bash
# Demo con webcam
python run_demo.py --mode webcam

# O directamente
python src/main_reid.py --video 0
```

### Analizar Múltiples Videos

```bash
# Procesar todos los videos en la carpeta
for video in data/video/*.mp4; do
    echo "Procesando: $video"
    python src/main_reid.py --video "$video" --output "output/$(basename "$video")"
done
```

## 🎯 Checklist de Éxito

Después de seguir esta guía, deberías poder:

- [ ] ✅ Instalar el sistema sin errores
- [ ] ✅ Ejecutar el demo básico
- [ ] ✅ Ver personas detectadas con IDs únicos
- [ ] ✅ Observar re-identificaciones cuando personas salen y vuelven
- [ ] ✅ Guardar video procesado en la carpeta output/
- [ ] ✅ Ajustar configuración según tu hardware
- [ ] ✅ Entender las métricas mostradas en consola

## 🚀 Próximos Pasos

Una vez que domines lo básico:

1. **Experimenta con diferentes videos** y escenarios
2. **Ajusta parámetros** para optimizar según tu caso de uso
3. **Mide el rendimiento** y documenta los mejores ajustes
4. **Contribuye al proyecto** reportando bugs o mejoras
5. **Explora características avanzadas** en desarrollo

## 📚 Recursos Adicionales

- **README.md**: Documentación completa del proyecto
- **docs/PROYECTO_PROGRESO.md**: Estado actual y roadmap
- **config/**: Ejemplos de configuración para diferentes escenarios
- **GitHub Issues**: Para reportar problemas o sugerir mejoras

---

**¿Necesitas ayuda?** 
- Revisa la sección de solución de problemas
- Consulta los archivos de configuración de ejemplo
- Abre un issue en GitHub con detalles específicos

¡Disfruta explorando la re-identificación de personas! 🎉 