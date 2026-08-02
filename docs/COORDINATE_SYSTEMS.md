# Sistemas de coordenadas

## 1. Marco interno de Quest Align

Definido explícitamente en `config.yaml` y en `src/coordinate_frames.py`:

- **Sistema:** derecho (right-handed).
- **Ejes:** X derecha, Y arriba, Z adelante (positivo hacia delante).
- **Unidades:**
  - posiciones: metros (m);
  - aceleraciones: m/s²;
  - rotaciones: cuaterniones `[x, y, z, w]` y matrices 3×3 con determinante +1.

Este marco intenta coincidir con el de AMASS/SMPL usado por HMD-Poser.

## 2. Marco Quest / Unity

### 2.1 Supuesto (sin verificar el emisor)

Unity usa un sistema **izquierdo** con:

- X derecha;
- Y arriba;
- Z adelante.

### 2.2 Transformación al marco interno

La posición se transforma reflejando el eje Z:

```
p_internal = [ p_x, p_y, -p_z ]
```

Para una rotación, se pasa a matriz 3×3 y se aplica la transformación de semejanza
con la matriz de reflexión `S = diag(1, 1, -1)`:

```
M_internal = S @ M_unity @ S
```

`S @ M @ S` conserva el determinante +1, por lo que el resultado es una rotación
válida en el marco derecho.

### 2.3 Ejemplo numérico

Cuaternión de Unity identidad `[0, 0, 0, 1]`:

```python
M_unity = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
M_internal = S @ M_unity @ S
            = [[1, 0,  0],
               [0, 1,  0],
               [0, 0, -1]] @ M_unity @ [[1, 0,  0],
                                        [0, 1,  0],
                                        [0, 0, -1]]
            = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]
```

Para una rotación de 90° alrededor de Y, la transformación produce la matriz de
rotación correspondiente en el marco derecho.

## 3. Marco Android

### 3.1 Marco del sensor

Android define el marco del dispositivo como derecho:

- X: derecha del dispositivo;
- Y: arriba del dispositivo;
- Z: fuera de la pantalla (hacia el usuario).

### 3.2 Marco world (TYPE_ROTATION_VECTOR)

`SensorManager.getQuaternionFromVector` devuelve una orientación relativa al
marco ENU (East-North-Up):

- X: este;
- Y: norte;
- Z: arriba.

### 3.3 Transformación al marco interno

La transformación se calcula en la **calibración obligatoria**:

```
R_internal_from_android = R_neutral_pelvis_internal @ (R_phone_world_neutral @ R_body_from_phone)^T
```

Donde:

- `R_neutral_pelvis_internal`: orientación de la pelvis en el marco interno durante
  la calibración. Dado que Quest Align no mide la pelvis directamente, se supone
  que está alineada con la cabeza en postura neutral.
- `R_phone_world_neutral`: orientación del teléfono en el marco Android world durante
  la calibración.
- `R_body_from_phone`: orientación de montaje del teléfono respecto a la pelvis. Si
  no se conoce, se usa la identidad y se documenta el supuesto.

### 3.4 Transformación de la aceleración

```
acc_internal = R_internal_from_android @ R_body_from_phone @ acc_device
```

Si `android_acceleration_type == "accelerometer"`, se resta la gravedad interna:

```
acc_internal = acc_internal - [0, -9.80665, 0]
```

Si `android_acceleration_type == "linear"`, no se resta gravedad.

## 4. Calibración

La calibración se realiza en dos etapas separadas:

1. **Calibración de sensores:** en el primer frame sincronizado se computan las
   matrices `R_internal_from_android` y `R_body_from_phone`. El usuario debe estar
   en postura neutral.
2. **Calibración de postura neutral:** los primeros `posture_calib_frames` frames
   se usan para calcular el offset de pitch/roll de salida del modelo.

Las matrices de calibración se almacenan en el objeto `Calibration` y se guardan en
el JSON de cada sesión.

## 5. Limitaciones conocidas

- El orden de los cuaterniones de Quest no está verificado con el emisor.
- El tipo de aceleración Android (`TYPE_ACCELEROMETER` vs `TYPE_LINEAR_ACCELERATION`)
  no está verificado con el emisor.
- El montaje exacto del teléfono en la pelvis (`R_body_from_phone`) no se mide de
  forma independiente; se usa la identidad por defecto.
- La alineación entre el marco Android world y el Quest tracking space se basa en el
  supuesto de que la pelvis está alineada con la cabeza en postura neutral.
