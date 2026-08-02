# Timing y frecuencias en tiempo de ejecución

## 1. Frecuencias separadas

| Frecuencia | Valor configurado | Cómo se mide | Código |
| ------------ | ----------------- | -------------- | ------ |
| Bucle del servidor | `server_loop_rate_hz` (60 Hz) | Intervalo de `main_loop` | `server.py` |
| Frecuencia nominal Quest | no verificada | contador de llegadas | `Synchronizer.quest` |
| Frecuencia nominal Android | no verificada | contador de llegadas | `Synchronizer.android` |
| Pares sincronizados | depende de la intersección | `Synchronizer.get_synced_pair()` | `temporal_buffer.py` |
| Ventanas válidas entregadas al modelo | limitada por los pares sincronizados, no por el bucle | `RollingWindow.to_numpy()` | `temporal_buffer.py` |
| Inferencia | limitada por los pares sincronizados | `model(window)` | `server.py` |

The server loop runs nominally at 60 Hz. However, without temporal resampling,
new features and inference outputs are produced at the rate of accepted
synchronized sensor pairs. The 40-frame window therefore spans a duration
determined by the measured synchronized-pair frequency.

## 2. Sincronización

- Se conservan dos timestamps: `device_ts` y `server_arrival_ts`.
- La sincronización se realiza por `server_arrival_ts`.
- Tolerancia: `synchronization_tolerance_ms` (200 ms por defecto).
- Si el offset entre Quest y Android supera la tolerancia, se descarta el par y se
  incrementa `rejected_pairs`.
- No se mezclan relojes de dispositivo sin corregir el offset.

## 3. Ventana temporal

- Tamaño: `temporal_window_frames` (40 frames).
- Forma esperada: `(40, 135)`.
- Forma entregada al modelo: `(1, 40, 135)`.
- Requisitos:
  - 40 observaciones with strictly increasing timestamps;
  - no gaps larger than `max_window_gap_ms` (200 ms);
  - effective frequency calculated via `RollingWindow.effective_hz()`.
- Each unique synced pair is added exactly once; the synchronizer deduplicates
  by `pair_ts`.
- If the window is not full, no inference runs in normal mode.
- The `diagnostic_repeated_frame_mode` repeats the last frame to fill 40 slots
  and is marked as degraded.

## 4. Remuestreo

Actualmente **no se implementa remuestreo temporal**. La ventana se llena a la
frecuencia de pares únicos sincronizados, no a la frecuencia del bucle del
servidor (60 Hz). Si los emisores entregan ~8 Hz, la ventana tarda ~5 s en
llenarse. Para una implementación futura se recomienda:

- posiciones: interpolación lineal;
- orientaciones: SLERP;
- aceleraciones: interpolación lineal o retención justificada.

Sin remuestreo, la documentación debe decir que la ventana depende de la
frecuencia de pares únicos sincronizados (medida entre 2.7–6.7 Hz en las
10 sesiones experimentales). El campo `resampled: false` en el
JSON de sesión registra este hecho.

## 5. Métricas de diagnóstico

El servidor registra en cada sesión:

- `quest_hz` y `android_hz`;
- `synced_pairs` y `rejected_pairs`;
- `effective_hz` de la ventana;
- `degraded_mode`;
- `missing_sensors`;
- timestamps de inicio y fin de la ventana.

## 6. Modo degradado

Se activa cuando:

- se pierde el teléfono (`quest_only`);
- hay saltos temporales mayores a `max_window_gap_ms`;
- se usa `diagnostic_repeated_frame_mode`.

Los resultados en modo degradado no se mezclan con la evaluación multimodal.
