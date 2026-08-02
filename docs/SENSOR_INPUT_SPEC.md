# Especificación de entrada de sensores

## 1. Paquete Quest / Unity (UDP puerto 5006)

### 1.1 Esquema JSON

```json
{
  "hmd_p": [x, y, z],
  "hmd_q": [q0, q1, q2, q3],
  "lhand_p": [x, y, z],
  "lhand_q": [q0, q1, q2, q3],
  "rhand_p": [x, y, z],
  "rhand_q": [q0, q1, q2, q3]
}
```

### 1.2 Orden de cuaterniones

**No verificado con el código del emisor.** El servidor actual interpreta `hmd_q`,
`lhand_q` y `rhand_q` como cuaterniones en formato `[x, y, z, w]`, que es el que
espera `scipy.spatial.transform.Rotation.from_quat`.

Para que la implementación sea robusta, se recomienda incluir un campo de metadatos:

```json
{ "quat_order": "xyzw" }
```

### 1.3 Unidades

- Posiciones: **metros** (supuesto, a confirmar en el emisor).
- Cuaterniones: adimensional.

### 1.4 Timestamp

El servidor acepta opcionalmente un campo `ts` o `timestamp` en el JSON, expresado
en segundos con precisión de punto flotante. Si no está presente, se usa el
timestamp de llegada al servidor (`time.time()`).

### 1.5 Ejemplo de paquete real (sintético, sin datos personales)

```json
{
  "ts": 1234567890.123,
  "hmd_p": [0.0, 1.6, 0.0],
  "hmd_q": [0.0, 0.0, 0.0, 1.0],
  "lhand_p": [0.2, 1.0, 0.3],
  "lhand_q": [0.0, 0.0, 0.0, 1.0],
  "rhand_p": [-0.2, 1.0, 0.3],
  "rhand_q": [0.0, 0.0, 0.0, 1.0]
}
```

## 2. Paquete Android (UDP puerto 5005)

### 2.1 Esquema JSON

```json
{
  "type": "imu",
  "ts": 1234567890123,
  "quat": [qw, qx, qy, qz],
  "acc": [ax, ay, az]
}
```

### 2.2 Orden de cuaterniones

El servidor convierte el cuaternión de Android de `[w, x, y, z]` a `[x, y, z, w]`
antes de normalizarlo y transformarlo. Esto coincide con el formato que devuelve
`SensorManager.getQuaternionFromVector()` de Android.

### 2.3 Sensor de orientación

El campo `quat` se asocia a `TYPE_ROTATION_VECTOR` (o `GAME_ROTATION_VECTOR` si no
hay magnetómetro). Esta identificación debe verificarse en el emisor Android.

### 2.4 Sensor de aceleración

El campo `acc` es genérico. Es **crítico** identificar si el emisor envía:

- `TYPE_ACCELEROMETER`: incluye gravedad. El servidor debe restar el vector de
  gravedad en el marco interno (`[0, -9.80665, 0]` m/s²) después de transformar.
- `TYPE_LINEAR_ACCELERATION`: la gravedad ya ha sido removida por Android.
  No se debe restar gravedad.

La configuración `android_acceleration_type` de `config.yaml` controla este
comportamiento. El valor por defecto es `null` (no verificado), y el código NO
aplica ninguna corrección de gravedad en ese caso.

### 2.5 Unidades

- `acc`: **m/s²** (supuesto, a confirmar en el emisor).
- `quat`: adimensional.

### 2.6 Timestamp

El servidor acepta un campo `ts` o `timestamp` en milisegundos o segundos. Si
no está presente, se usa el timestamp de llegada al servidor.

### 2.7 Ejemplo de paquete real (sintético, sin datos personales)

```json
{
  "type": "imu",
  "ts": 1234567890123,
  "quat": [1.0, 0.0, 0.0, 0.0],
  "acc": [0.0, 0.0, 0.0]
}
```

## 3. Paquetes de control Android

El servidor reconoce los siguientes tipos de paquete (`type`) del Android:

- `hello`: mensaje de inicio, se ignora.
- `cmd`: comando, se ignora.
- `sync_req`: petición de sincronización de reloj. El servidor responde con
  `sync_resp` usando `t1` y `t2` en nanosegundos. No se devuelven datos IMU.

## 4. Sincronización

- Se conservan dos timestamps: `device_ts` (del paquete) y `server_arrival_ts`.
- Para emparejar Quest y Android se usa **únicamente** `server_arrival_ts`, porque
  los relojes de los dispositivos no son comparables sin una calibración de offset.
- El `device_ts` se mantiene solo para diagnóstico y análisis de jitter.

## 5. Frecuencias

- Frecuencia nominal del bucle del servidor: `server_loop_rate_hz` (60 Hz).
- Frecuencia nominal del modelo: `expected_model_rate_hz` (60 Hz).
- Frecuencias reales de Quest y Android: **no verificables con el código de este
  repositorio**. Deben medirse con el emisor correspondiente.
