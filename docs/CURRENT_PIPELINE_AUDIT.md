# Auditoría de la pipeline actual (Quest Align)

## 1. Alcance de la auditoría

Se ha inspeccionado el repositorio `Quest-Align` y el código de referencia oficial de HMD-Poser (`https://github.com/Pico-AI-Team/HMD-Poser`, clonado en `/tmp/opencode/hmd_poser_ref`) para determinar qué partes del flujo están respaldadas por código y cuáles dependen de emisores que **no están presentes** en este repositorio.

## 2. Archivos del repositorio

| Archivo | Responsabilidad |
| ------- | --------------- |
| `server.py` | Bucle principal, listeners UDP, sincronización básica, carga del modelo, calibración de postura y grabación. |
| `udp_listener.py` | Wrapper de socket UDP no bloqueante. |
| `feature_builder.py` | Constructor actual del tensor de entrada. **Problemas detectados:** usa bloques de 15 valores por sensor, orden incorrecto, hace *padding* genérico hasta 135 y repite 40 veces el mismo frame. |
| `spine_analyzer.py` | Cálculo de ángulos de columna a partir de articulaciones SMPL (no se usa en el servidor actual). |
| `postura_tronco_hmd_poser.ipynb` | Notebook offline de análisis. |
| `README.md` | Documentación de usuario. |

**Código de los emisores (Quest/Unity y Android) no encontrado en este repositorio.** Por tanto, las conversiones de sistema de coordenadas, orden de cuaterniones y tipo de sensor de aceleración Android **no se pueden verificar directamente** contra el código emisor.

## 3. Esquema de paquetes UDP

### 3.1 Quest / Unity (puerto 5006)

Paquete JSON con las claves que parsea `server.py`:

```json
{
  "hmd_p": [x, y, z],
  "hmd_q": [?, ?, ?, ?],
  "lhand_p": [x, y, z],
  "lhand_q": [?, ?, ?, ?],
  "rhand_p": [x, y, z],
  "rhand_q": [?, ?, ?, ?]
}
```

- **No se parsea timestamp** del dispositivo; el servidor usa `time.time()` de llegada.
- **Orden del cuaternión Quest no verificable** con el código disponible: `feature_builder.py` pasa el array directamente a `scipy.spatial.transform.Rotation.from_quat`, que espera `[x, y, z, w]` por defecto.
- **Unidades de posición:** probablemente metros (Unity tracking space), pero no se verifican en el emisor.
- **Frecuencia nominal:** no verificable con el código disponible; el bucle del servidor está fijado a 60 Hz.

### 3.2 Android (puerto 5005)

Paquete JSON con las claves que parsea `server.py`:

```json
{
  "type": "imu",
  "quat": [qw, qx, qy, qz],
  "acc": [ax, ay, az]
}
```

También maneja `type: "sync_req"` y `type: "hello"` / `type: "cmd"` (sin payload IMU).

- **No se parsea timestamp** del dispositivo; el servidor usa `time.time()` de llegada.
- **Orden del cuaternión Android:** el servidor lo convierte explícitamente de `[w, x, y, z]` a `[x, y, z, w]` para `scipy`.
- **Sensor de orientación:** `SensorManager.getQuaternionFromVector` se asocia a `TYPE_ROTATION_VECTOR` (o `TYPE_GAME_ROTATION_VECTOR` si no hay magnetómetro). No se verifica el código Android.
- **Sensor de aceleración:** el campo `acc` es genérico. No se verifica si es `TYPE_ACCELEROMETER` (incluye gravedad) o `TYPE_LINEAR_ACCELERATION` (gravedad removida). Esto es un riesgo metodológico.
- **Unidades de aceleración:** probablemente m/s² (estándar Android), pero no se verifica en el emisor.
- **Frecuencia nominal:** no verificable con el código disponible.

## 4. Sistemas de coordenadas

| Sistema | Mano | Ejes típicos | Verificación |
| ------- | ---- | -------------- | ------------ |
| Quest / Unity | Izquierda | Y arriba, X derecha, Z adelante (en Unity) | **No verificable con el emisor disponible.** |
| Android sensor | Derecha | X derecha del dispositivo, Y arriba del dispositivo, Z fuera de pantalla | **No verificable con el emisor disponible.** |
| Android world (TYPE_ROTATION_VECTOR) | Derecha | ENU: X este, Y norte, Z arriba | **No verificable con el emisor disponible.** |
| HMD-Poser / AMASS | Derecha | Y arriba, Z adelante (aprox.) | Verificado desde `prepare_data.py` y la librería SMPL. |
| Internal (definido en esta refactorización) | Derecha | X derecha, Y arriba, Z adelante, metros, m/s² | Definido por este proyecto. |

**Conclusión:** las conversiones entre Quest y Android son **necesarias pero no estaban implementadas**. Se requiere una etapa de calibración explícita y matrices de transformación documentadas.

## 5. Checkpoint HMD-Poser

- **Ruta actual usada por el servidor:** `pretrained_model/pretrained_model_protocol1.pt`
- **Checkpoint obtenido del repositorio oficial:** `8f89acc8c6599f61eb4eb420bec097940070acc722bc7862229394f48d29746c`
- **Carga `strict=True`:** exitosa contra la arquitectura `HMD_imu_HME_Universe` (135 entradas, 40 frames).
- **Configuración de entrenamiento (`options/train_config.yaml` del checkpoint):**
  - `sparse_dim: 135`
  - `input_motion_length: 40`
  - `compatible_inputs: ['HMD', 'HMD_2IMUs', 'HMD_3IMUs']`
  - Datos: AMASS Protocol 1 (BioMotionLab_NTroje, CMU, MPI_HDM05).
- **Configuración de Quest Align (`HMD + manos + smartphone en pelvis`):** NO aparece en `compatible_inputs` del entrenamiento. El checkpoint está entrenado con HMD+2IMUs (pies) o HMD+3IMUs (pies + pelvis), pero **no** con pelvis únicamente como IMU extra.

## 6. Tabla de variables: origen, unidad, orden y destino

| Variable | Fuente | Unidad | Orden | Sistema de coordenadas | Transformación | Normalización | Destino en el tensor |
| -------- | ------ | ------ | ----- | ---------------------- | -------------- | ------------- | -------------------- |
| Posición cabeza (HMD) | Quest `hmd_p` | m (supuesto) | XYZ | Quest/Unity (no verificable) | **Pendiente:** a marco interno derecho | Ninguna | `GLOBAL_POS_SLICE[0]` (índices 72-75) |
| Orientación cabeza (HMD) | Quest `hmd_q` | adimensional | `[?, ?, ?, ?]` | Quest/Unity (no verificable) | **Pendiente:** a cuaternión interno `[x, y, z, w]` y luego matriz 6D | Normalización de norma | `GLOBAL_ROT_SLICE[0]` (0-6) |
| Posición mano izquierda | Quest `lhand_p` | m (supuesto) | XYZ | Quest/Unity (no verificable) | **Pendiente:** a marco interno | Ninguna | `GLOBAL_POS_SLICE[1]` (75-78) |
| Orientación mano izquierda | Quest `lhand_q` | adimensional | `[?, ?, ?, ?]` | Quest/Unity (no verificable) | **Pendiente:** a matriz 6D | Normalización | `GLOBAL_ROT_SLICE[1]` (6-12) |
| Posición mano derecha | Quest `rhand_p` | m (supuesto) | XYZ | Quest/Unity (no verificable) | **Pendiente:** a marco interno | Ninguna | `GLOBAL_POS_SLICE[2]` (78-81) |
| Orientación mano derecha | Quest `rhand_q` | adimensional | `[?, ?, ?, ?]` | Quest/Unity (no verificable) | **Pendiente:** a matriz 6D | Normalización | `GLOBAL_ROT_SLICE[2]` (12-18) |
| Orientación pelvis | Android `quat` | adimensional | `[w, x, y, z]` → interno `[x, y, z, w]` | Android world (no verificable) | **Pendiente:** calibración obligatoria `R_internal_from_android` | Normalización | `GLOBAL_ROT_SLICE[5]` (30-36) |
| Aceleración pelvis | Android `acc` | m/s² (supuesto) | XYZ del sensor | Android sensor (no verificable) | **Pendiente:** a marco interno; posible remoción de gravedad | Ninguna | `ACCELERATION_SLICE[2]` (132-135) |
| Rotación pie izquierdo | No disponible | — | — | — | Cero semántico | Cero | `GLOBAL_ROT_SLICE[3]` (18-24), `DELTA_ROT_SLICE[3]` (54-60) |
| Rotación pie derecho | No disponible | — | — | — | Cero semántico | Cero | `GLOBAL_ROT_SLICE[4]` (24-30), `DELTA_ROT_SLICE[4]` (60-66) |
| Aceleración pie izquierdo | No disponible | — | — | — | Cero semántico | Cero | `ACCELERATION_SLICE[0]` (126-129) |
| Aceleración pie derecho | No disponible | — | — | — | Cero semántico | Cero | `ACCELERATION_SLICE[1]` (129-132) |
| Delta de rotaciones | Dos frames consecutivos | adimensional | 6D | Marco interno | `R_prev^T @ R_curr` | Ninguna | `DELTA_ROT_SLICE` (36-72) |
| Delta de posiciones | Dos frames consecutivos | m (diferencia por muestra) | XYZ | Marco interno | `curr - prev` | Ninguna | `DELTA_POS_SLICE` (81-90) |
| Rotaciones mano-cabeza | Dos frames consecutivos | adimensional | 6D | Espacio de la cabeza | `R_head^T @ R_hand` | Ninguna | `RELATIVE_ROT_SLICE` (90-102) |
| Posiciones mano-cabeza | Dos frames consecutivos | m | XYZ | Espacio de la cabeza | `(p_hand - p_head) @ R_head` | Ninguna | `RELATIVE_POS_SLICE` (114-120) |
| Aceleraciones pies/pelvis | `prepare_data.py` vía vértices | m/s² | XYZ | Marco interno | Síntesis del cuerpo | Ninguna | `ACCELERATION_SLICE` (126-135) |

## 7. Problemas críticos detectados

1. **Tensor de 135 valores incorrecto:** `feature_builder.py` construye un tensor de 6 sensores × 15 características (posición 3 + matriz 3×3 + aceleración 3), no el layout oficial de 135 valores.
2. **Repetición de frames:** `server.py` repite 40 veces el mismo frame para alimentar al modelo, en lugar de usar una ventana real de 40 frames consecutivos.
3. **Padding genérico:** se rellena el vector con ceros hasta 135 sin respetar la semántica de cada canal.
4. **Sin transformación de coordenadas:** Quest y Android se usan directamente como si compartieran el mismo marco.
5. **Tipo de aceleración Android sin identificar:** el servidor no distingue entre `TYPE_ACCELEROMETER` y `TYPE_LINEAR_ACCELERATION`.
6. **Sincronización por llegada al servidor:** no se conservan timestamps de dispositivo ni se corrige offset de reloj.
7. **Carga del checkpoint con `strict=False`:** el servidor oculta posibles incompatibilidades.
8. **Pelvis-only no entrenada:** la configuración de Quest Align no está entre las configuraciones oficiales del entrenamiento.
9. **Calibración de postura vs. calibración extrínseca confundidas:** el servidor actual usa solo un promedio de ángulos de salida, sin calibrar el sistema de coordenadas del teléfono.

## 8. Limitaciones de la información disponible

Los siguientes campos no se pueden confirmar con el código presente en este repositorio y deben ser verificados en el emisor correspondiente:

- Orden real del cuaternión de Quest (`hmd_q`, `lhand_q`, `rhand_q`).
- Orden real del cuaternión de Android (`quat`).
- Tipo de sensor Android que produce `acc` (`TYPE_ACCELEROMETER` vs `TYPE_LINEAR_ACCELERATION`).
- Convenio de ejes de Quest/Unity en el emisor.
- Frecuencias reales de Quest y Android.
- Presencia de timestamp de dispositivo en los paquetes.
- Orientación de montaje del teléfono en la pelvis.

La implementación siguiente introduce transformaciones y calibraciones **documentadas como supuestos** donde el código emisor no está disponible.
