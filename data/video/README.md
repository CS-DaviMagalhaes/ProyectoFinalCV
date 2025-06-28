# Carpeta de Videos de Prueba

## 📹 Instrucciones para Colocar Videos

Esta carpeta debe contener los videos que quieres procesar con el sistema de re-identificación de personas.

### Formatos Soportados
- **MP4** (.mp4)
- **AVI** (.avi)
- **MOV** (.mov)
- **MKV** (.mkv)
- **WMV** (.wmv)

### Cómo Usar

1. **Copiar tu video aquí:**
   ```bash
   cp /ruta/a/tu/video.mp4 data/video/
   ```

2. **Ejecutar el sistema:**
   ```bash
   python src/main_reid.py --video data/video/tu_video.mp4
   ```

### Recomendaciones para Videos

#### ✅ Videos Ideales para Re-identificación:
- **Resolución**: 720p o superior (1280x720, 1920x1080)
- **FPS**: 25-30 fps
- **Duración**: 30 segundos a 5 minutos para pruebas
- **Escena**: Personas caminando, entrando y saliendo del frame
- **Iluminación**: Buena iluminación, evitar contraluces severos
- **Ángulo**: Cámara a nivel de calle o ligeramente elevada

#### ⚠️ Evitar:
- Videos muy oscuros o con poca iluminación
- Ángulos cenitales (desde el techo)
- Resoluciones muy bajas (menos de 480p)
- Videos con muchas oclusiones
- Escenas con multitudes muy densas

### Ejemplos de Escenarios Ideales

1. **Entrada de edificio**: Personas entrando y saliendo
2. **Pasillo**: Personas caminando en ambas direcciones
3. **Plaza o patio**: Personas cruzando el área
4. **Estación de transporte**: Flujo de personas con pausas
5. **Tienda o centro comercial**: Clientes entrando y saliendo

### Videos de Prueba Sugeridos

Si no tienes videos propios, puedes usar:

1. **Webcam en vivo**: Usa `--video 0` para cámara web
2. **Videos de demostración**: Busca videos de seguridad en YouTube
3. **Datasets públicos**: MOT Challenge, Market-1501 (para investigación)

### Estructura Esperada

```
data/video/
├── README.md                 # Este archivo
├── mi_video_prueba.mp4      # Tu video de prueba
├── camara_entrada.avi       # Otro video
└── demo_pasillo.mov         # Más videos...
```

### Verificar que el Video Funciona

Antes de procesar, verifica que el video se reproduce correctamente:

```bash
# Con OpenCV (Python)
python -c "import cv2; cap = cv2.VideoCapture('data/video/tu_video.mp4'); print('Video OK' if cap.isOpened() else 'Error')"

# Con VLC o reproductor de video
vlc data/video/tu_video.mp4
```

### Problemas Comunes

#### "Video no encontrado"
- Verifica que el archivo está en esta carpeta
- Revisa que el nombre del archivo sea correcto
- Asegúrate que el formato es soportado

#### "No se puede abrir el video"
- El archivo podría estar corrupto
- Codec no soportado por OpenCV
- Intenta convertir a MP4 con formato H.264

#### Rendimiento lento
- Reduce la resolución del video
- Usa un fragmento más corto del video
- Cambia a configuración CPU si la GPU es lenta

### Convertir Videos (si es necesario)

```bash
# Convertir a MP4 con FFmpeg
ffmpeg -i video_original.avi -c:v libx264 -c:a aac data/video/video_convertido.mp4

# Reducir resolución
ffmpeg -i video_grande.mp4 -vf scale=1280:720 data/video/video_720p.mp4

# Extraer fragmento (primeros 60 segundos)
ffmpeg -i video_largo.mp4 -t 60 data/video/video_corto.mp4
```

## 🚀 Listo para Usar

Una vez que tengas tu video en esta carpeta, ejecuta:

```bash
# Demo básico
python run_demo.py --mode basic

# O directamente
python src/main_reid.py --video data/video/TU_VIDEO.mp4
```

¡El sistema detectará automáticamente el video y comenzará el procesamiento! 