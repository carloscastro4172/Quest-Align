# Android sensor provenance

Verified from the decompiled APK in `App/` (May 2026 build).

## Acceleration sensor

| Field | Value |
|-------|-------|
| Primary sensor | `Sensor.TYPE_ACCELEROMETER_UNCALIBRATED` (constant 35) |
| Fallback | `Sensor.TYPE_ACCELEROMETER` (constant 1) |
| NOT used | `Sensor.TYPE_LINEAR_ACCELERATION` (constant 10) |
| Unit | m/s² |
| Gravity | Included (accelerometer-type signal) |
| Components transmitted | First 3 (ax, ay, az); bias estimates discarded |
| Registration delay | `SensorManager.SENSOR_DELAY_GAME` |

## Orientation sensor

| Field | Value |
|-------|-------|
| Sensor | `Sensor.TYPE_ROTATION_VECTOR` (constant 11) |
| Conversion | `SensorManager.getQuaternionFromVector()` |
| Quaternion order | `[w, x, y, z]` |
| Normalization | Yes (by Android API) |
| Invalid indicator | `quat_valid: false`, identity quaternion `[1, 0, 0, 0]` |

## Gyroscope sensor

| Field | Value |
|-------|-------|
| Primary sensor | `Sensor.TYPE_GYROSCOPE_UNCALIBRATED` (constant 16) |
| Fallback | `Sensor.TYPE_GYROSCOPE` (constant 4) |
| Used in feature tensor | No (logged only) |
| Unit | rad/s |
| Registration delay | `SensorManager.SENSOR_DELAY_GAME` |

## UDP transmission

| Field | Value |
|-------|-------|
| Scheduling interval | 16 ms (≈ 62.5 Hz nominal) |
| Advertised rate (`imu_rate_hz`) | 60 Hz |
| Destination port | Configurable (set to 5005 during experiments) |
| Destination IP | Configurable (default 192.168.1.10) |
| Packet IDs | `pelvis`, `left_shin`, `right_shin` (default: `pelvis`) |
| Timestamp source | `SystemClock.elapsedRealtimeNanos()` at packet construction |
| SensorEvent.timestamp | NOT transmitted |

## JSON packet schema

```json
{
  "type": "imu",
  "id": "pelvis",
  "seq": 0,
  "t_sensor_ns": 0,
  "t_server_est_ns": 0,
  "sync_ok": true,
  "sync_delay_ns": 0,
  "sync_offset_ns": 0,
  "quat_valid": true,
  "sensors": {
    "acc": true,
    "gyro": true,
    "quat": true
  },
  "acc": [0.0, 0.0, 0.0],
  "gyro": [0.0, 0.0, 0.0],
  "quat": [1.0, 0.0, 0.0, 0.0]
}
```

## Clock synchronization

The app sends `sync_req` with `t0_ns` and computes:
- `offset = ((t1 - t0) + (t2 - t3)) / 2`
- `delay = (t3 - t0) - (t2 - t1)`

Accepts samples only when `0 < delay < 50 ms`. Updates offset via EMA with α = 0.1.

## Limits of provenance

- The historical packets do not identify which of the two acceleration sensors
  (35 or 1) was ultimately selected by each experimental smartphone.
- The APK confirms the selection policy but not the per-device runtime outcome.
