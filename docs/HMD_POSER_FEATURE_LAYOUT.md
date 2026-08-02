# Layout del vector de 135 características de HMD-Poser

Este layout se extrae directamente del preprocesamiento oficial de HMD-Poser:
`prepare_data.py` en el repositorio oficial.

La suma se verifica automáticamente:

```
36 + 36 + 9 + 9 + 12 + 12 + 6 + 6 + 9 = 135
```

## Tabla de índices

| Índices | Dimensión | Variable | Orden de sensores | Unidad | Marco |
| ------- | --------- | -------- | ----------------- | ------ | ----- |
| 0:6 | 6 | Rotación global 6D | head | adimensional | Interno |
| 6:12 | 6 | Rotación global 6D | left_hand | adimensional | Interno |
| 12:18 | 6 | Rotación global 6D | right_hand | adimensional | Interno |
| 18:24 | 6 | Rotación global 6D | left_foot | adimensional | Interno |
| 24:30 | 6 | Rotación global 6D | right_foot | adimensional | Interno |
| 30:36 | 6 | Rotación global 6D | pelvis | adimensional | Interno |
| 36:42 | 6 | Delta rotación global 6D | head | adimensional | Interno |
| 42:48 | 6 | Delta rotación global 6D | left_hand | adimensional | Interno |
| 48:54 | 6 | Delta rotación global 6D | right_hand | adimensional | Interno |
| 54:60 | 6 | Delta rotación global 6D | left_foot | adimensional | Interno |
| 60:66 | 6 | Delta rotación global 6D | right_foot | adimensional | Interno |
| 66:72 | 6 | Delta rotación global 6D | pelvis | adimensional | Interno |
| 72:75 | 3 | Posición global XYZ | head | m | Interno |
| 75:78 | 3 | Posición global XYZ | left_hand | m | Interno |
| 78:81 | 3 | Posición global XYZ | right_hand | m | Interno |
| 81:84 | 3 | Delta posición global XYZ | head | m | Interno |
| 84:87 | 3 | Delta posición global XYZ | left_hand | m | Interno |
| 87:90 | 3 | Delta posición global XYZ | right_hand | m | Interno |
| 90:96 | 6 | Rotación 6D relativa a head | left_hand | adimensional | Espacio de head |
| 96:102 | 6 | Rotación 6D relativa a head | right_hand | adimensional | Espacio de head |
| 102:108 | 6 | Delta rotación relativa a head | left_hand | adimensional | Espacio de head |
| 108:114 | 6 | Delta rotación relativa a head | right_hand | adimensional | Espacio de head |
| 114:117 | 3 | Posición relativa a head | left_hand | m | Espacio de head |
| 117:120 | 3 | Posición relativa a head | right_hand | m | Espacio de head |
| 120:123 | 3 | Delta posición relativa a head | left_hand | m | Espacio de head |
| 123:126 | 3 | Delta posición relativa a head | right_hand | m | Espacio de head |
| 126:129 | 3 | Aceleración | left_foot | m/s² | Interno |
| 129:132 | 3 | Aceleración | right_foot | m/s² | Interno |
| 132:135 | 3 | Aceleración | pelvis | m/s² | Interno |

## Operaciones de las variables temporales

### Delta de rotación global

```
R_delta = R_prev^T @ R_curr
```

### Delta de posición global

```
p_delta = p_curr - p_prev
```

### Rotación relativa a la cabeza

```
R_rel = R_head^T @ R_hand
```

### Posición relativa a la cabeza

```
p_rel = (p_hand - p_head) @ R_head
```

### Delta de rotación relativa a la cabeza

```
R_rel_delta = R_rel_prev^T @ R_rel_curr
```

### Delta de posición relativa a la cabeza

```
p_rel_delta = p_rel_curr - p_rel_prev
```

## Representación 6D

La conversión a 6D sigue exactamente la implementación de HMD-Poser:

```python
pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
```

Es decir, las dos primeras columnas de la matriz de rotación concatenadas:

```
[R[:,0] | R[:,1]] = [R00, R10, R20, R01, R11, R21]
```

## Configuraciones de sensores y canales anulados

Las máscaras se extraen de `dataset/dataloader.py` del repositorio oficial:

- `HMD_3IMUs`: todos los canales presentes.
- `HMD_2IMUs`: se anulan los canales de la pelvis (índices 30:36, 66:72, 132:135).
- `HMD`: se anulan pies (rotaciones y deltas) y todas las aceleraciones.

Para Quest Align (`quest_plus_pelvis`):

- Sensores presentes: head, left_hand, right_hand, pelvis.
- Sensores ausentes: left_foot, right_foot.
- Canales anulados: índices 18:24, 24:30, 54:60, 60:66, 126:129, 129:132.

Si se pierde el teléfono, se añaden a los anteriores los canales de pelvis:
30:36, 66:72, 132:135.

## Nombres de las constantes en el código

```python
GLOBAL_ROT_SLICE          = slice(0, 36)
DELTA_ROT_SLICE           = slice(36, 72)
GLOBAL_POS_SLICE          = slice(72, 81)
DELTA_POS_SLICE           = slice(81, 90)
RELATIVE_ROT_SLICE        = slice(90, 102)
DELTA_RELATIVE_ROT_SLICE  = slice(102, 114)
RELATIVE_POS_SLICE        = slice(114, 120)
DELTA_RELATIVE_POS_SLICE  = slice(120, 126)
ACCELERATION_SLICE        = slice(126, 135)
```

## Esquema de features

```yaml
feature_schema_version: "hmd_poser_135_v1"
```
