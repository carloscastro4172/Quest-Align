# Timing y frecuencias en tiempo de ejecución

## 1. Frecuencias separadas

| Frecuencia | Valor configurado | Cómo se mide | Código |
| ------------ | ----------------- | -------------- | ------ |
| Bucle del servidor | `server_loop_rate_hz` (60 Hz) | Intervalo de `main_loop` | `server.py` |
| Frecuencia nominal Quest | no verificada | contador de llegadas | `Synchronizer.quest` |
| Frecuencia nominal Android | no verificada | contador de llegadas | `Synchronizer.android` |
| Pares sincronizados | depende de la intersección | `Synchronizer.get_synced_pair()` | `temporal_buffer.py` |
| Ventanas válidas entregadas al modelo | 60 Hz una vez llena | `RollingWindow.to_numpy()` | `temporal_buffer.py` |
| Inferencia | 60 Hz cuando la ventana está llena | `model(window)` | `server.py` |

## 2. Sincronización

- Se conservan dos timestamps: `device_ts` y `server_arrival_ts`.
- La sincronización se realiza por `server_arrival_ts`.
- Tolerancia: `synchronization_tolerance_ms` (100 ms por defecto).
- Si el offset entre Quest y Android supera la tolerancia, se descarta el par y se
  incrementa `rejected_pairs`.
- No se mezclan relojes de dispositivo sin corregir el offset.

## 3. Ventana temporal

- Tamaño: `temporal_window_frames` (40 frames).
- Forma esperada: `(40, 135)`.
- Forma entregada al modelo: `(1, 40, 135)`.
- Requisitos:
  - 40 observaciones con timestamps crecientes;
  - sin saltos mayores a `max_window_gap_ms` (100 ms);
  - frecuencia efectiva calculada por `RollingWindow.effective_hz()`.
- Si la ventana no está llena, no se ejecuta inferencia en modo normal.
- El modo `diagnostic_repeated_frame_mode` repite el último frame para completar 40
  y se marca como degradado.

## 4. Remuestreo

Actualmente **no se implementa remuestreo temporal**. Si los emisores no entregan
60 Hz estables, la ventana de 40 frames no representa exactamente el mismo horizonte
temporal del entrenamiento. Para una implementación futura se recomienda:

- posiciones: interpolación lineal;
- orientaciones: SLERP;
- aceleraciones: interpolación lineal o retención justificada.

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
