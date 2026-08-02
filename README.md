# HMD-Poser based Spine Posture Analysis (Quest Align)

Este proyecto utiliza componentes e ideas de [HMD-Poser](https://github.com/Pico-AI-Team/HMD-Poser) para realizar análisis de postura de la columna en tiempo real con datos de un Meta Quest y un smartphone Android.

## Estado actual

La pipeline ha sido refactorizada para corregir la construcción del tensor de entrada,
la sincronización temporal y la validación del checkpoint. Ver `docs/CURRENT_PIPELINE_AUDIT.md`
para el listado completo de problemas corregidos.

## Estructura del repositorio

```
Quest-Align/
├── server.py                         # Servidor UDP principal
├── udp_listener.py                   # Wrapper de sockets UDP
├── spine_analyzer.py                 # Análisis offline de la columna (legacy)
├── postura_tronco_hmd_poser.ipynb    # Notebook offline
├── config.yaml                       # Configuración centralizada
├── src/
│   ├── config.py                     # Loader de config.yaml
│   ├── coordinate_frames.py          # Transformaciones de coordenadas y calibración
│   ├── feature_constants.py          # Constantes del layout 135-D
│   ├── feature_builder.py            # Constructor del vector 135-D
│   ├── temporal_buffer.py            # Sincronización y ventana deslizante
│   ├── checkpoint_validator.py       # Validación del checkpoint
│   ├── hmd_poser_network.py          # Arquitectura del modelo (del repo oficial)
│   └── posture_extraction.py         # Extracción de ángulos de columna
├── tests/                            # Tests pytest
├── docs/                             # Documentación metodológica
└── artifacts/                        # Reportes generados
```

## Dependencias

- Python >= 3.9
- PyTorch >= 2.0.1
- NumPy
- SciPy
- PyYAML
- pytest

Para ejecutar el servidor también se necesitan los modelos corporales SMPL+H si se
quiere usar la forward kinematics completa. El smoke test de la red neuronal no los
requiere.

## Configuración

Edita `config.yaml` antes de ejecutar. Parámetros importantes:

- `checkpoint_path`: ruta al checkpoint HMD-Poser.
- `strict_checkpoint_loading`: `true` por defecto.
- `android_acceleration_type`: `null`, `accelerometer` o `linear`. Debe coincidir con
  el sensor real del emisor Android.
- `quest_quaternion_order`: `null`, `xyzw` o `wxyz`. Debe coincidir con el emisor Quest.
- `synchronization_tolerance_ms`: tolerancia de sincronización Quest/Android.
- `temporal_window_frames`: 40 frames.

## Cómo ejecutar

### Tests

```bash
python3 -m pytest tests -v
```

### Validación del checkpoint

```bash
python3 -c "from src.checkpoint_validator import run_checkpoint_validation; run_checkpoint_validation()"
```

Se genera `artifacts/checkpoint_compatibility.json`.

### Servidor en tiempo real

```bash
python3 server.py
```

El servidor escucha:

- Quest / Unity en el puerto configurado (`quest_port`, 5006 por defecto).
- Android en el puerto configurado (`android_port`, 5005 por defecto).

### Demo end-to-end sintética

```bash
python3 -m pytest tests/test_end_to_end.py -v
```

## Documentación

- `docs/CURRENT_PIPELINE_AUDIT.md`: auditoría inicial y tabla de variables.
- `docs/SENSOR_INPUT_SPEC.md`: especificación de los paquetes UDP.
- `docs/COORDINATE_SYSTEMS.md`: sistemas de coordenadas y transformaciones.
- `docs/HMD_POSER_FEATURE_LAYOUT.md`: layout exacto de los 135 valores.
- `docs/CHECKPOINT_COMPATIBILITY.md`: resultado de la validación del checkpoint.
- `docs/RUNTIME_TIMING.md`: frecuencias, sincronización y ventana temporal.
- `docs/PAPER_METHODS_UPDATE.md`: texto listo para actualizar el paper.

## Limitaciones importantes

- El orden de los cuaterniones de Quest y el tipo de aceleración Android no se pueden
  verificar sin el código de los emisores. El repositorio actual no incluye las
  aplicaciones Quest/Unity ni Android.
- La configuración de hardware de Quest Align (HMD + manos + smartphone en pelvis)
  **no** aparece en las configuraciones de entrenamiento del checkpoint HMD-Poser.
  El checkpoint es arquitectónicamente compatible, pero no se puede afirmar que la
  configuración pelvis-only fue entrenada.
- El full body model SMPL+H no está incluido; la extracción de postura usa una
  cinemática directa simplificada con la jerarquía SMPL estándar.

## Agradecimientos

Este trabajo se basa en **HMD-Poser** (PICO-AI-Team, CVPR 2024).

```
@inproceedings{daip2024hmdposer,
  title={HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations},
  author={Dai, Peng and Zhang, Yang and Liu, Tao and Fan, Zhen and Du, Tianyuan and Su, Zhuo and Zheng, Xiaozheng and Li, Zeming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
