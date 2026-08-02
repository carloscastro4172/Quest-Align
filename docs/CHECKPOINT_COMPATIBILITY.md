# Compatibilidad del checkpoint HMD-Poser

## 1. Checkpoint utilizado

- **Ruta:** `pretrained_model/pretrained_model_protocol1.pt`
- **SHA-256:** `8f89acc8c6599f61eb4eb420bec097940070acc722bc7862229394f48d29746c`
- **Origen:** repositorio oficial HMD-Poser (`Pico-AI-Team/HMD-Poser`).
- **Configuración de entrenamiento:** `options/train_config.yaml` del checkpoint.

## 2. Resultado de la carga `strict=True`

```
strict_load: true
architecturally_compatible: true
forward_pass_success: true
```

La arquitectura `HMD_imu_HME_Universe` acepta el `state_dict` del checkpoint sin
claves faltantes ni inesperadas.

## 3. Separación de compatibilidades

### A. Compatibilidad arquitectónica

El checkpoint es **arquitectónicamente compatible** con el tensor de entrada de
Quest Align:

- Dimensión de entrada: `sparse_dim == 135`.
- Longitud temporal: `input_motion_length == 40`.
- Forma del tensor de entrada: `(1, 40, 135)`.
- Forma de salida: `pred_pose (1, 40, 132)`, `pred_shapes (1, 40, 16)`.
- `strict=True` no produce errores.
- El forward pass no produce NaN ni dimensiones inesperadas.

### B. Compatibilidad de configuración / distribución

**El checkpoint NO fue entrenado con la configuración de hardware de Quest Align.**

Las configuraciones originales del entrenamiento son:

- `HMD`: solo HMD + manos (todos los canales de pies y pelvis anulados).
- `HMD_2IMUs`: HMD + manos + dos IMUs en los pies.
- `HMD_3IMUs`: HMD + manos + dos IMUs en los pies + un IMU en la pelvis.

Quest Align usa:

- HMD + manos + smartphone en la pelvis.
- **Sin IMUs en los pies.**

Por tanto, los canales de pies se anulan con ceros, pero el canal de pelvis se
alimenta con un sensor real que no estaba presente en la misma configuración en
el entrenamiento. La caja de texto adecuada para el paper es:

> "The checkpoint is architecturally compatible with the 135-dimensional tensor,
> but the pelvis-only inertial configuration was not explicitly validated as an
> original training configuration."

## 4. Limitaciones del forward pass

El smoke test ejecuta únicamente la red neuronal (`HMD_imu_HME_Universe`). No se
ha ejecutado el módulo completo de forward kinematics con el body model SMPL+H,
porque el repositorio de Quest Align no incluye los archivos de body models
(`smplh` / `dmpls`). Para obtener salidas de articulaciones 3D con FK se requiere
instalar `human_body_prior` y descargar los modelos corporales correspondientes.

## 5. Reporte automático

Cada vez que se ejecuta el servidor o el test de checkpoint se genera
`artifacts/checkpoint_compatibility.json` con los campos:

- `checkpoint_path`
- `sha256`
- `strict_load`
- `missing_keys`
- `unexpected_keys`
- `forward_pass_success`
- `output_shape`
- `quest_plus_pelvis_seen_during_training`
- `architecturally_compatible`
- `configuration_validated`
- `notes`

## 6. Política de `strict`

- Por defecto se usa `strict=True`.
- Si `strict=True` falla, el servidor se detiene y reporta el error.
- Solo se permite `strict=False` si `strict_checkpoint_loading: false` en
  `config.yaml`, y en ese caso se marca el resultado como
  `checkpoint_partial_load: true`.

## 7. Conclusión

- El checkpoint se carga correctamente y la red acepta el tensor de 135 valores.
- La compatibilidad arquitectónica está verificada.
- La compatibilidad estadística/biomecánica de la configuración pelvis-only no está
  verificada porque no aparece en las configuraciones de entrenamiento del
  checkpoint.
