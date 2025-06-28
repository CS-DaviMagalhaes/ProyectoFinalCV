# 🛠️ FIX.md – Plan de Mejoras para un Sistema de Re-Identificación Robusto

Este archivo resume los **problemas detectados** y propone **acciones correctivas** para robustecer el pipeline YOLOv8 + Torchreid, de acuerdo a los requisitos originales (`REQUIRE.MD`).

---
## 1. Detección & Tracking  
> *La detección inestable desencadena errores de Re-ID.*

| Problema | Impacto | Solución Propuesta |
|----------|---------|--------------------|
| **YOLOv8n demasiado ligero** | Falsos negativos / bounding boxes pobres | Permitir elegir modelos `s/m/m-seg` según FPS objetivo.
| **Sin tracker dedicado** | Cambio frecuente de `track_id` → falla Re-ID | Integrar **StrongSORT** o **OC-SORT** (mantenemos embeddings del tracker para coherencia). |
| **Sin filtrado de IoU/confianza** | Detecciones ruidosas | Aplicar NMS por clase y un filtro de `conf > 0.4` + `iou_thr 0.5`. |

### Acciones
1. Añadir `tracker_type` en `config` (`bytetrack`, `strongsort`, `ocsrt`).
2. Usar `ultralytics.track` sólo para detección; delegar tracking a la librería elegida.

---
## 2. Preprocesado de Imagen  
> *La entrada al modelo de Re-ID debe replicar el pipeline de entrenamiento.*

| Problema | Impacto | Solución |
|----------|---------|----------|
| **Resize con antialias =True (Torch 2.7)**  | Lento; distinto de Torchreid | Cambiar a `interpolation=InterpolationMode.BILINEAR` y desactivar antialias. |
| **Falta `model.eval()`** | Lotes con BN/Dropout inestables | Añadir `self.extractor.model.eval()` tras cargar el modelo. |
| **No normaliza por lotes** | Ruido en embeddings | Crear `torch.no_grad()` en `preprocess_image`. |

---
## 3. Embedding & Matching  
> *La comparación directa de un único embedding es frágil.*

| Problema | Impacto | Mejora |
|----------|---------|--------|
| Embedding de **1 solo frame** | Alta varianza → falsos negativos | Promediar los últimos *k* embeddings por `global_id` (EMA). |
| **Umbral fijo** (0.85) | No se adapta a escenas | Calibrar online: usar distancia mínima histórica + margen adaptativo. |
| **Solo coseno** | No captura magnitud | Usar distancia de Mahalanobis con matriz de covarianza estimada online. |
| **Sin detección de outliers** | IDs mezclados | Aplicar DBSCAN sobre vectores para detectar clusters inconsistentes. |

### Acciones
1. Añadir `embedding_buffer_size` en config.  
2. Mantener `deque(maxlen=k)` de embeddings por persona.  
3. Implementar **quality score** para cada embedding (ej. score = confianza · area_bbox).

---
## 4. Gestión de Memoria / Base de Datos  

| Problema | Impacto | Solución |
|----------|---------|----------|
| `embeddings_db` crece indefinidamente | Fugas de RAM | Persistir en disco por bloques; GC por LRU + TTL. |
| No se guarda **timestamp real** | Difícil buscar eventos | Almacenar `datetime` además de `frame`. |
| Falta **persistencia** entre ejecuciones | Re-ID se pierde | Guardar en `embeddings.sqlite` o `parquet`.

---
## 5. Performance & Paralelismo  

- Usar `torch.compile()` (PyTorch > 2.0) para optimizar la red.
- Paralelizar etapas:
  1. Thread 1 → lectura/decodificación de video.
  2. Thread 2 → detección YOLO.
  3. Thread 3 → extracción Re-ID.
- Batch processing: extraer embeddings cada *N* frames o cuando el tracker indique "new state".

---
## 6. Configuración / Experimentos  

1. Añadir **config/experiments/** con YAMLs listos:  _high-precision_, _low-latency_, _multi-camera_.
2. Incluir script `evaluate.py` para métricas MOTA, IDF1 usando dataset MOT-17.

---
## 7. Código & Testing  

- Cobertura de **unit-tests** (>80%) con `pytest` para:
  - Preprocesado
  - Matching logic
  - Tracker interface
- Linter: `ruff` + `black` + `isort`.
- CI workflow (GitHub Actions) con matrix {cpu, mps}.

---
## 8. Roadmap de Implementación  

1. **Semana 1**: Refactor tracking (StrongSORT) + buffer de embeddings.  
2. **Semana 2**: Adaptación de umbral dinámico + tests unitarios.  
3. **Semana 3**: Gestión de memoria & persistencia.  
4. **Semana 4**: Optimización y benchmark (MOT-17, PRW).  

---
## 9. Recursos Útiles  
- StrongSORT: https://github.com/dyhBUPT/strongsort  
- Torchreid docs: https://kaiyangzhou.github.io/deep-person-reid/  
- MOT Challenge metrics: https://motchallenge.net/  

---
**Autor:** AI Assistant  
**Fecha:** 2025-06-28 