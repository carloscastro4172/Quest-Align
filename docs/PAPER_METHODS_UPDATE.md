# Actualización del apartado metodológico del paper

Este documento contiene texto listo para sustituir las secciones **Feature
Construction** y **HMD-Poser Inference** del paper, limitado a afirmaciones que
están respaldadas por la implementación actual de Quest Align.

---

## Verified implementation facts

### Feature construction

The input feature vector for HMD-Poser has exactly **135 values** and follows the
layout of the official pre-processing script `prepare_data.py` from the
HMD-Poser repository:

```
[global rotation 6D (6 segments × 6)]                → 36
[global rotation delta 6D (6 segments × 6)]          → 36
[global position XYZ (head, lhand, rhand)]           →  9
[global position delta XYZ (head, lhand, rhand)]     →  9
[relative hand-in-head rotation 6D (lhand, rhand)]   → 12
[relative hand-in-head rotation delta 6D]            → 12
[relative hand-in-head position XYZ (lhand, rhand)]  →  6
[relative hand-in-head position delta XYZ]           →  6
[acceleration XYZ (left_foot, right_foot, pelvis)]    →  9
Total                                                135
```

The order of the six segments is: head, left hand, right hand, left foot, right
foot, pelvis.

All quaternions are normalized before conversion to rotation matrices. Zero-norm
quaternions are rejected. The 6D rotation representation is computed as the first
two columns of the rotation matrix, concatenated, exactly as in the official HMD-Poser
implementation.

Temporal differences are computed from two real consecutive frames. Rotation deltas
use the relative rotation `R_prev^T @ R_curr`; position deltas are simple frame
differences `p_curr - p_prev`. Hand-in-head rotations are `R_head^T @ R_hand`, and
hand-in-head positions are `(p_hand - p_head) @ R_head`.

### Sensor setup and missing channels

The Quest Align hardware provides head pose, left hand pose, right hand pose,
pelvis orientation from the Android smartphone, and pelvis acceleration from the
Android smartphone. Left foot and right foot sensors are absent. Their semantic
channels in the 135-D vector are zeroed explicitly:

- left foot global rotation (18:24) and delta (54:60);
- right foot global rotation (24:30) and delta (60:66);
- left foot acceleration (126:129);
- right foot acceleration (129:132).

When the smartphone is lost, the server switches to `quest_only` mode and also
zeros the pelvis channels (30:36, 66:72, 132:135). It does not insert a fake
gravity vector.

### Coordinate systems

An internal right-handed frame is defined with X right, Y up, Z forward, meters
for positions, and m/s² for accelerations. Quest/Unity data are transformed into
this frame. Android data are transformed through a calibration matrix
`R_internal_from_android` computed during a mandatory neutral-pose calibration.

### Checkpoint loading

The HMD-Poser checkpoint `pretrained_model_protocol1.pt` is loaded with
`strict=True`. The `state_dict` matches the `HMD_imu_HME_Universe` architecture.
A smoke test with input shape `(1, 40, 135)` completes without NaN or dimensional
errors.

### Timing

The server runs at a nominal 60 Hz loop. Synchronization uses the server arrival
time of each UDP packet with a tolerance of 100 ms. The rolling window contains
40 real frames; the last frame is not repeated 40 times in normal mode.

---

## Claims that remain unsupported

### Pelvis-only training configuration

The HMD-Poser checkpoint was trained with three sensor configurations:
`HMD`, `HMD_2IMUs` (feet IMUs), and `HMD_3IMUs` (feet + pelvis IMUs). The
Quest Align configuration (`HMD + hands + pelvis smartphone`) is **not** one of
the original training configurations. Therefore, the following statement must not
be made in the paper:

> "The model was trained on the exact HMD + hands + pelvis smartphone setup used
> by Quest Align."

The correct statement is:

> "The HMD-Poser checkpoint is architecturally compatible with the 135-dimensional
> input tensor, but the pelvis-only inertial configuration was not explicitly
> validated as an original training configuration."

### Android sensor type

The Android `acc` field has not been verified against the emitter code to be
`TYPE_ACCELEROMETER` (gravity included) or `TYPE_LINEAR_ACCELERATION` (gravity
removed). The implementation supports both via `config.yaml`, but the default is
`null` (unverified). The paper should state that the sensor type was configured
as linear acceleration only after verification, or that the value is unverified
if the emitter code is not available.

### Quest quaternion order

The order of the Quest quaternion fields (`hmd_q`, `lhand_q`, `rhand_q`) has not
been verified against the emitter. The server assumes `[x, y, z, w]`, which is the
`scipy` default. The paper should either include an emitter-side verification or
state this assumption.

### Biomechanical validity

A forward pass of the neural network does not prove that the output pose is
biomechanically correct for the Quest Align hardware configuration. The full body
model (SMPL+H) and its body models were not available in the repository, so the
complete forward-kinematics path was not validated.

### Old experimental results

Any previous experimental results (e.g., pitch/roll measurements) were obtained
with the old pipeline that padded the 15-D per-sensor tensor and repeated a single
frame 40 times. Those results **must not** be attributed to the new 135-D pipeline.

---

## Texto sugerido para reemplazar las secciones del paper

### Feature Construction

> We construct a 135-D feature vector per frame that exactly matches the input
> pre-processing of the public HMD-Poser model. The vector is partitioned into
> nine semantic blocks: (i) global 6D rotations for the six body segments head,
> left hand, right hand, left foot, right foot, and pelvis; (ii) global 6D
> rotation deltas; (iii) global positions for head, left hand, and right hand;
> (iv) position deltas; (v) left/right hand rotations relative to the head;
> (vi) deltas of those relative rotations; (vii) left/right hand positions
> relative to the head; (viii) deltas of those relative positions; and (ix)
> accelerations for left foot, right foot, and pelvis. Because Quest Align does
> not have foot sensors, the corresponding foot channels are explicitly zeroed.
> When the smartphone is unavailable, the pelvis channels are also zeroed and
> the system switches to a degraded `quest_only` mode. All quaternions are
> normalized, and the 6D representation is computed as the first two columns of
> the rotation matrix, as in the original HMD-Poser code.

### HMD-Poser Inference

> We feed the network a sliding window of 40 real 135-D frames. The checkpoint
> `pretrained_model_protocol1.pt` is loaded with `strict=True`, and the
> architecture accepts an input of shape `(1, 40, 135)`. The checkpoint is
> architecturally compatible with our 135-D tensor, but it was originally trained
> on HMD, HMD+2IMUs, and HMD+3IMUs configurations; the Quest Align setup
> (HMD + hands + pelvis smartphone) is not one of those original training
> configurations. Therefore, the inference results are reported as the network
> output, without claiming biomechanical validation for the pelvis-only sensor
> layout.
