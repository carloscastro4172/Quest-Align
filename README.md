# HMD-Poser based Spine Posture Analysis (Quest Align)

This project uses components and ideas from [HMD-Poser](https://github.com/Pico-AI-Team/HMD-Poser) to perform real-time spine posture analysis using data from a Meta Quest HMD and an Android smartphone.

## Current State

The pipeline has been refactored to fix the input tensor construction, temporal synchronization, and checkpoint validation. See `docs/CURRENT_PIPELINE_AUDIT.md` for the full list of corrected issues.

## Repository Structure

```
Quest-Align/
├── server.py                         # Main UDP server
├── udp_listener.py                   # UDP socket wrapper
├── spine_analyzer.py                 # Offline spine analysis (legacy)
├── postura_tronco_hmd_poser.ipynb    # Offline notebook
├── config.yaml                       # Centralized configuration
├── RECORDS/                          # 18 experimental sessions (May 2026)
├── src/
│   ├── config.py                     # config.yaml loader
│   ├── coordinate_frames.py          # Coordinate transforms and calibration
│   ├── feature_constants.py          # 135-D layout constants
│   ├── feature_builder.py            # 135-D feature vector constructor
│   ├── temporal_buffer.py            # Synchronization and sliding window
│   ├── checkpoint_validator.py       # Checkpoint validation
│   ├── hmd_poser_network.py          # Model architecture (from official repo)
│   └── posture_extraction.py         # Spine angle extraction
├── tests/                            # Pytest tests
├── docs/                             # Methodological documentation
├── artifacts/                        # Generated reports
└── pretrained_model/                 # HMD-Poser checkpoint (SHA-256 verified)
```

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0.1
- NumPy
- SciPy
- PyYAML
- pytest

The SMPL+H body models are also needed to run the full forward kinematics.
The neural network smoke test does not require them.

## Configuration

Edit `config.yaml` before running. Important parameters:

- `checkpoint_path`: path to the HMD-Poser checkpoint.
- `strict_checkpoint_loading`: `true` by default.
- `android_acceleration_type`: `null`, `accelerometer`, or `linear`. Must match the actual Android emitter sensor.
- `quest_quaternion_order`: `null`, `xyzw`, or `wxyz`. Must match the Quest emitter.
- `synchronization_tolerance_ms`: Quest/Android pairing tolerance.
- `temporal_window_frames`: 40 frames.

## How to Run

### Tests

```bash
python3 -m pytest tests -v
```

### Checkpoint Validation

```bash
python3 -c "from src.checkpoint_validator import run_checkpoint_validation; run_checkpoint_validation()"
```

Generates `artifacts/checkpoint_compatibility.json`.

### Real-time Server

```bash
python3 server.py
```

The server listens on:

- Quest / Unity on the configured port (`quest_port`, 5006 by default).
- Android on the configured port (`android_port`, 5005 by default).

### Synthetic End-to-end Demo

```bash
python3 -m pytest tests/test_end_to_end.py -v
```

## Documentation

- `docs/CURRENT_PIPELINE_AUDIT.md`: initial audit and variable table.
- `docs/SENSOR_INPUT_SPEC.md`: UDP packet specification.
- `docs/COORDINATE_SYSTEMS.md`: coordinate systems and transforms.
- `docs/HMD_POSER_FEATURE_LAYOUT.md`: exact 135-value layout.
- `docs/CHECKPOINT_COMPATIBILITY.md`: checkpoint validation results.
- `docs/RUNTIME_TIMING.md`: frequencies, synchronization, and temporal window.
- `docs/PAPER_METHODS_UPDATE.md`: paper-ready text for the methods section.

## Measured Performance (10 participants, May 2026)

The `RECORDS/` directory contains 10 experimental sessions from real users.
Each session includes anthropometric data (`data_users.txt`) and per-frame
CSV files (`cm_session_*.csv`) with timestamp, ground-truth label, predicted
label, and pitch/roll angles in degrees.

### Participants

| S | Gender | Age | Height (m) | Weight (kg) |
|---|--------|-----|------------|-------------|
| 1 | F | 24 | 1.56 | 55.0 |
| 2 | M | 23 | 1.67 | 69.0 |
| 3 | M | 22 | 1.73 | 63.0 |
| 4 | M | 24 | 1.70 | 60.3 |
| 5 | F | 25 | 1.60 | 57.0 |
| 6 | M | 23 | 1.66 | 53.0 |
| 7 | M | 24 | 1.75 | 63.0 |
| 8 | M | 23 | 1.70 | 62.5 |
| 9 | M | 23 | 1.68 | 67.2 |
| 10 | M | 25 | 1.65 | 66.8 |

### Session stats

| S | Frames | Duration (s) | Hz | Steps | Pitch range (°) | Roll range (°) |
|---|--------|-------------|-----|-------|-----------------|----------------|
| 1 | 366 | 58.3 | 6.3 | 6 | [-31, +19] | [-20, +67] |
| 2 | 408 | 60.6 | 6.7 | 6 | [-18, +42] | [-25, +69] |
| 3 | 252 | 40.2 | 6.2 | 4 | [-34, -6] | [-15, +37] |
| 4 | 252 | 40.2 | 6.2 | 4 | [-35, -4] | [-16, +39] |
| 5 | 248 | 56.5 | 4.4 | 6 | [-80, +7] | [-19, +84] |
| 6 | 140 | 35.5 | 3.9 | 4 | [-34, +56] | [-76, +9] |
| 7 | 120 | 44.9 | 2.7 | 4 | [-57, +27] | [-20, +45] |
| 8 | 140 | 35.5 | 3.9 | 4 | [-35, +57] | [-77, +10] |
| 9 | 408 | 60.6 | 6.7 | 6 | [-18, +43] | [-26, +71] |
| 10 | 120 | 44.9 | 2.7 | 4 | [-57, +28] | [-22, +46] |

**Total:** 2,454 frames across 10 sessions (477.3 s).  
**Mean Hz:** 5.0 (range 2.7–6.7).

### Protocol steps

Each session followed a fixed sequence of posture steps:

- `CALIBRACION` – neutral standing calibration
- `CENTRO_1` / `CENTRO_2` – neutral posture validation
- `INCLINACION_ADELANTE` – forward hunch
- `LATERAL_DER` – right lateral deviation
- `LATERAL_IZQ` – left lateral deviation

Sessions 3, 4, 6, 7, 8, 10 omit `CENTRO_2` and one lateral direction.

### Methodology notes

The server loop runs at 60 Hz, but the actual synchronized-pair rate is
**~3–7 Hz** depending on the session. The HMD-Poser model was trained on
AMASS data subsampled to 60 Hz. The 40-frame window at 5 Hz spans ~8 s
of real time versus ~0.67 s at the training rate. Without temporal
resampling, this mismatch must be reported.

The old 18-session JSON dataset (also tracked in previous commits) was
recorded with a preliminary pipeline that padded a single frame 40 times.
Those results are superseded by this 10-participant CSV dataset.

## Important Limitations

- The Quest quaternion order and Android acceleration type cannot be verified without the emitter source code. This repository does not include the Quest/Unity or Android applications.
- The effective synchronized-pair frequency measured experimentally is **~3–7 Hz**, not 60 Hz. The 40-frame window therefore spans up to ~8 s of real time, not 0.67 s. The HMD-Poser checkpoint was trained at 60 Hz; without temporal resampling, the inference temporal horizon differs from the training distribution.
- The full SMPL+H body model is not included; posture extraction uses a simplified forward kinematics routine with the standard SMPL hierarchy.

## Acknowledgements

This work is based on **HMD-Poser** (PICO-AI-Team, CVPR 2024).

```
@inproceedings{daip2024hmdposer,
  title={HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations},
  author={Dai, Peng and Zhang, Yang and Liu, Tao and Fan, Zhen and Du, Tianyuan and Su, Zhuo and Zheng, Xiaozheng and Li, Zeming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
