# Guía Detallada para Re-identificación de Personas con Torchreid como Baseline

**Objetivo:** Desarrollar un pipeline en Python que utilice **YOLOv8** para la detección y seguimiento de personas en un video, y **Torchreid** para extraer *embeddings* (características) que permitan re-identificar a las personas cuando salen y reingresan al campo de visión de la cámara.

**Audiencia:** Agente AI de desarrollo (ej. Cursor) y desarrolladores de visión por computadora.

## Diagrama de Flujo del Proyecto

```mermaid
graph TD
    subgraph "Entrada y Procesamiento"
        A[Video de Cámara de Seguridad] --> B{Lectura de Video (Frame a Frame)};
    end

    subgraph "Fase 1: Detección y Tracking"
        B --> C[Detector de Personas (YOLOv8)];
        C -- Detecciones (Bounding Boxes) --> D[Tracker (ByteTrack/StrongSORT)];
        D -- IDs Temporales y Coordenadas --> E[Tracks Activos en el Frame];
    end

    subgraph "Fase 2: Extracción de Features (Torchreid)"
        E --> F{Recortar Imagen de Persona};
        F --> G[Pre-procesar Imagen (Resize, Normalize)];
        G --> H(Modelo Torchreid Pre-entrenado);
        H --> I[Vector de Características (Embedding)];
    end

    subgraph "Fase 3: Lógica de Re-identificación"
        I --> J{Base de Datos de Embeddings};
        J <--> K(Lógica de Asociación);
        K -- Comparación de Similitud (Coseno) --> L{Asignar ID Global};
    end
    
    subgraph "Salida"
        E -- Coordenadas --> M[Visualización];
        L -- ID Global --> M;
        M --> N[Video Final con Re-ID Persistente];
    end
```

-----

## Fase 1: Configuración del Entorno de Desarrollo

El primer paso es configurar el entorno con todas las herramientas y librerías necesarias.

**1.1. Prerrequisitos:**

  * Python 3.8 o superior.
  * GPU NVIDIA con **CUDA** y **cuDNN** instalados para un rendimiento óptimo.

**1.2. Clonar Repositorios Necesarios:**
`torchreid` es el núcleo para la Re-ID, pero no realiza detección. Usaremos `ultralytics` para la detección y tracking inicial.

```bash
# Clonar el repositorio de Torchreid para acceder a sus herramientas y modelos
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid

# Instalar Torchreid y sus dependencias
pip install -e .

# Regresar al directorio principal
cd ..

# Instalar las librerías principales para el pipeline
pip install ultralytics opencv-python "numpy<2.0"
```

**Nota:** Se especifica `numpy<2.0` porque algunas librerías de visión por computadora aún pueden tener conflictos con la versión 2.0.

## Fase 2: Preparar el Modelo de Re-ID (Usando Torchreid)

No necesitamos re-entrenar un modelo. El poder de `torchreid` reside en su zoológico de modelos pre-entrenados de alto rendimiento. Nuestra tarea es cargar uno de estos modelos y prepararlo para la inferencia.

**2.1. Script para Cargar el Modelo:**
Crea un archivo `feature_extractor.py`. Este módulo se encargará de inicializar y utilizar el modelo de Re-ID.

```python
# file: feature_extractor.py

import torch
import torchreid
from torchreid.utils import FeatureExtractor
from torchvision.transforms import functional as F
import numpy as np

class ReIDFeatureExtractor:
    def __init__(self, model_name='osnet_x1_0', model_path='', device='cuda'):
        """
        Inicializa el extractor de características de Re-ID.

        Args:
            model_name (str): Nombre del modelo a usar de Torchreid.
            model_path (str): Ruta a los pesos pre-entrenados. Si está vacío, usa los de Torchreid.
            device (str): Dispositivo para correr el modelo ('cuda' o 'cpu').
        """
        self.device = device
        # Inicializa el extractor de características de Torchreid
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        print(f"Modelo Re-ID '{model_name}' cargado en el dispositivo '{device}'.")

    def preprocess_image(self, image_np):
        """
        Pre-procesa una imagen (numpy array) para el modelo Re-ID.
        
        Args:
            image_np (np.array): Imagen recortada de la persona en formato BGR (de OpenCV).
        
        Returns:
            torch.Tensor: Tensor de la imagen listo para el modelo.
        """
        # Convertir BGR a RGB
        image_rgb = image_np[:, :, ::-1].copy()
        # Convertir a Tensor
        image_tensor = F.to_tensor(image_rgb)
        # Redimensionar a las dimensiones esperadas por el modelo (usualmente 256x128)
        image_resized = F.resize(image_tensor, (256, 128), antialias=True)
        # Normalizar
        image_normalized = F.normalize(image_resized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Añadir dimensión de batch
        return image_normalized.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(self, image_np):
        """
        Extrae el vector de características (embedding) de una imagen.

        Args:
            image_np (np.array): Imagen recortada de la persona.

        Returns:
            np.array: Vector de características normalizado.
        """
        preprocessed_image = self.preprocess_image(image_np)
        features = self.extractor(preprocessed_image)
        # Normalizar el vector de características es crucial para la comparación de coseno
        features = features.cpu().numpy().flatten()
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

```

## Fase 3: Implementación del Pipeline de Inferencia Completo

Ahora, crearemos el script principal que une todo: detección, tracking, extracción y re-identificación.

**3.1. Script Principal `main_reid.py`:**

```python
# file: main_reid.py

import cv2
from ultralytics import YOLO
import numpy as np
from feature_extractor import ReIDFeatureExtractor
from collections import defaultdict

# --- PARÁMETROS CONFIGURABLES ---
VIDEO_PATH = 'path/to/your/video.mp4'  # O 0 para webcam
YOLO_MODEL = 'yolov8n.pt'
REID_MODEL = 'osnet_x1_0' # Modelo de Re-ID a usar
DEVICE = 'cuda' # 'cuda' o 'cpu'
SIMILARITY_THRESHOLD = 0.85 # Umbral para considerar que es la misma persona
DISAPPEARANCE_LIMIT_SECONDS = 5 # Segundos para mantener a alguien en memoria

# --- INICIALIZACIÓN DE MODELOS ---
print("Inicializando modelos...")
# Modelo de detección y tracking
yolo = YOLO(YOLO_MODEL)
# Modelo de extracción de características
reid_extractor = ReIDFeatureExtractor(model_name=REID_MODEL, device=DEVICE)
print("Inicialización completa.")

# --- ESTRUCTURAS DE DATOS ---
next_global_id = 0
# Diccionario para objetos trackeados: {track_id: global_id}
tracked_objects = {}
# Base de datos de embeddings: {global_id: [embedding, last_seen_frame]}
embeddings_db = defaultdict(list)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
disappearance_limit_frames = int(fps * DISAPPEARANCE_LIMIT_SECONDS)

# --- BUCLE PRINCIPAL ---
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detección y Tracking con YOLOv8
    results = yolo.track(frame, persist=True, classes=0, verbose=False) # classes=0 para personas
    
    if results[0].boxes.id is None:
        cv2.imshow("Re-ID Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # IDs y coordenadas de los tracks en el frame actual
    current_track_ids = results[0].boxes.id.int().cpu().tolist()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # --- LÓGICA DE RE-IDENTIFICACIÓN ---
    for i, track_id in enumerate(current_track_ids):
        x1, y1, x2, y2 = map(int, boxes[i])
        
        # Si el track_id es nuevo en este frame...
        if track_id not in tracked_objects:
            # Recortar la imagen de la persona
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0: continue

            # Extraer su embedding
            current_embedding = reid_extractor.extract_features(person_crop)
            
            # Comparar con los embeddings de personas desaparecidas
            best_match_id = -1
            max_similarity = -1

            # Limpiar la DB de embeddings viejos
            ids_to_check = list(embeddings_db.keys())
            for global_id in ids_to_check:
                if frame_count - embeddings_db[global_id][1] > disappearance_limit_frames:
                    del embeddings_db[global_id]
                    continue
                
                # Cálculo de la similitud del coseno
                similarity = np.dot(current_embedding, embeddings_db[global_id][0])
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = global_id
            
            # Asignar ID
            if max_similarity > SIMILARITY_THRESHOLD:
                # Re-identificación exitosa
                global_id = best_match_id
            else:
                # Persona nueva
                global_id = next_global_id
                next_global_id += 1
            
            tracked_objects[track_id] = global_id
            embeddings_db[global_id] = [current_embedding, frame_count]

        # Actualizar la DB con la última aparición
        global_id_to_update = tracked_objects[track_id]
        embeddings_db[global_id_to_update][1] = frame_count

        # Visualización
        assigned_id = tracked_objects.get(track_id, -1)
        label = f"ID: {assigned_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Re-ID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

-----

## Puntos Clave para Modificación y Evaluación

Si los resultados no son los esperados, aquí es donde los desarrolladores deben experimentar.

**1. En `main_reid.py` (Lógica de alto nivel):**

  * **`SIMILARITY_THRESHOLD`**: Este es el parámetro **más importante**.
      * **Si es muy alto (ej. 0.95):** El sistema será muy estricto. Habrá menos re-identificaciones incorrectas (falsos positivos), pero podrías fallar en re-identificar a la misma persona si su apariencia cambia ligeramente.
      * **Si es muy bajo (ej. 0.70):** El sistema será más permisivo. Re-identificarás a más personas correctamente, pero corres el riesgo de asignar el mismo ID a dos personas diferentes que visten de forma parecida.
  * **`DISAPPEARANCE_LIMIT_SECONDS`**: ¿Cuánto tiempo recuerdas a una persona?
      * **Valor alto:** Bueno para escenarios donde la gente puede salir por largos periodos (ej. una tienda grande con múltiples salidas fuera de cámara). Aumenta el tamaño de la base de datos de `embeddings_db`.
      * **Valor bajo:** Bueno para pasillos o áreas contenidas donde una ausencia larga significa que la persona probablemente ya se fue del lugar.
  * **`YOLO_MODEL`**: Cambia `yolov8n.pt` (nano) por `yolov8s.pt` (small) o `yolov8m.pt` (medium) para mejorar la precisión de la detección a costa de la velocidad. Una mala detección inicial arruina todo el pipeline.

**2. En `feature_extractor.py` (Corazón de la Re-ID):**

  * **`REID_MODEL` / `model_name`**: Este es el segundo parámetro más importante.
      * **`'osnet_x1_0'`**: Modelo estándar, muy preciso. Es un excelente punto de partida.
      * **`'osnet_x0_25'`**: Versión mucho más ligera y rápida. Ideal para sistemas con hardware limitado o que necesitan un FPS muy alto. Su precisión es menor, pero a menudo suficiente.
      * **`'osnet_ain_x1_0'`**: Versión "Attentive Instance Normalization", que puede ser más robusta a cambios de apariencia.
      * **Experimentar con otros modelos:** `torchreid` ofrece muchos. Revisa su documentación para ver modelos entrenados en diferentes datasets (ej. `dukemtmcreid`) que podrían funcionar mejor dependiendo del dominio de tu cámara.
  * **Dimensiones en `preprocess_image`**: El tamaño `(256, 128)` es estándar para muchos modelos de Re-ID. Si usas un modelo diferente, asegúrate de que las dimensiones de entrada coincidan con las que se usaron en su entrenamiento.

## Consideraciones Finales para los Desarrolladores

1.  **Rendimiento (FPS):** La Re-ID es computacionalmente cara. La extracción de *embeddings* es el cuello de botella. Para optimizar:
      * Usa un modelo Re-ID más ligero (`osnet_x0_25`).
      * Reduce la resolución del video de entrada.
      * No extraigas *embeddings* en cada frame para una persona ya identificada. Puedes hacerlo cada N frames o cuando el tracker indique baja confianza.
2.  **Robustez del Modelo:** Los modelos de Re-ID son sensibles a:
      * **Cambios drásticos de apariencia:** Si alguien se quita o pone una chaqueta, el sistema probablemente le asignará un nuevo ID.
      * **Iluminación:** Cambios severos de luz pueden alterar los colores y afectar el *embedding*.
      * **Oclusión severa:** Si el tracker pierde a una persona por mucho tiempo o solo se ve una pequeña parte de ella, el *embedding* será de baja calidad.
3.  **Calidad del Tracking:** El sistema depende del `track_id` que provee YOLO. Si el tracker es inestable y cambia de `track_id` para la misma persona constantemente, forzará a la lógica de Re-ID a trabajar de más y podría cometer errores. Usar un tracker más robusto como **StrongSORT** (que se puede integrar con YOLO) podría mejorar la estabilidad.
4.  **Ángulo de la Cámara:** Los modelos de Re-ID funcionan mejor con vistas de cuerpo completo o tres cuartos. Vistas cenitales (desde el techo) o muy bajas pueden degradar significativamente la precisión, ya que fueron entrenados mayormente con vistas a nivel de calle.