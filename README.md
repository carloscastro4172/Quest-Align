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
└── artifacts/                        # Generated reports
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

## Important Limitations

- The Quest quaternion order and Android acceleration type cannot be verified without the emitter source code. This repository does not include the Quest/Unity or Android applications.
- The Quest Align hardware configuration (HMD + hands + smartphone on pelvis) **does not** appear among the training configurations of the HMD-Poser checkpoint. The checkpoint is architecturally compatible, but the pelvis-only configuration cannot be claimed as a trained setup.
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
