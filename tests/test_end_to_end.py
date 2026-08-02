"""
Test end-to-end: paquetes simulados Quest + Android
→ sincronización
→ transformación de coordenadas
→ rolling buffer
→ tensor (1, 40, 135)
→ checkpoint
→ salida SMPL
→ extracción de postura
"""

import time
import numpy as np
import torch

from src.config import load_config
from src.coordinate_frames import build_calibration_from_neutral
from src.sensor_availability import quest_plus_pelvis_availability
from src.feature_builder import build_hmd_poser_features
from src.temporal_buffer import Synchronizer, RollingWindow
from src.checkpoint_validator import CheckpointValidator
from src.posture_extraction import extract_last_frame_pose_6d, extract_spine_angles


def _quest_packet(head_pos=(0,0,0), left_pos=(0.2,-0.1,0.3), right_pos=(-0.2,-0.1,0.3)):
    import json
    return json.dumps({
        'hmd_p': list(head_pos), 'hmd_q': [0, 0, 0, 1],
        'lhand_p': list(left_pos), 'lhand_q': [0, 0, 0, 1],
        'rhand_p': list(right_pos), 'rhand_q': [0, 0, 0, 1],
    }).encode('utf-8')


def _android_packet(pelvis_rot=(0,0,0,1), pelvis_accel=(0,0,0)):
    import json
    return json.dumps({
        'type': 'imu',
        'quat': list(pelvis_rot),
        'acc': list(pelvis_accel),
    }).encode('utf-8')


def test_end_to_end_simulated_packets():
    cfg = load_config('config.yaml')
    validator = CheckpointValidator(cfg)
    validator.validate()
    model = validator.get_model()
    assert model is not None
    model.eval()

    # Sincronizador
    sync = Synchronizer(tolerance_ms=cfg.synchronization_tolerance_ms)
    rw = RollingWindow(size=cfg.temporal_window_frames)
    availability = quest_plus_pelvis_availability()

    # Calibración con un frame neutral sintético
    calibration = build_calibration_from_neutral(
        quest_head_rot=np.array([0, 0, 0, 1], dtype=np.float32),
        quest_quaternion_order="xyzw",
        android_phone_rot=np.array([1, 0, 0, 0], dtype=np.float32),
        android_quaternion_order="wxyz",
        timestamp=time.time(),
        acceleration_includes_gravity=False,
    )

    # Enviar 40 pares de paquetes simulados
    prev_frame = None
    for i in range(cfg.temporal_window_frames):
        t = time.time() + i * 0.016
        left_pos = (0.2 + i * 0.001, -0.1, 0.3)
        q = _quest_packet(left_pos=left_pos)
        a = _android_packet(pelvis_accel=(0.0, 0.0, 1.0))

        # Parseo manual similar a server.py
        import json
        qj = json.loads(q.decode('utf-8'))
        aj = json.loads(a.decode('utf-8'))
        quest_frame = {
            'head': {'pos': np.array(qj['hmd_p'], dtype=np.float32),
                     'rot': np.array(qj['hmd_q'], dtype=np.float32)},
            'left_hand': {'pos': np.array(qj['lhand_p'], dtype=np.float32),
                          'rot': np.array(qj['lhand_q'], dtype=np.float32)},
            'right_hand': {'pos': np.array(qj['rhand_p'], dtype=np.float32),
                           'rot': np.array(qj['rhand_q'], dtype=np.float32)},
        }
        android_frame = {
            'pelvis': {'rot': np.array(aj['quat'], dtype=np.float32),
                       'accel': np.array(aj['acc'], dtype=np.float32)},
        }
        sync.add_quest(quest_frame, t)
        sync.add_android(android_frame, t)

        pair = sync.get_synced_pair()
        assert pair is not None, "los paquetes deberían emparejarse"
        qd, ad, _, _ = pair
        current_frame = {**qd, **ad}
        feature = build_hmd_poser_features(current_frame, prev_frame, calibration, availability)
        rw.add(feature, t)
        prev_frame = current_frame

    window = rw.to_numpy()
    assert window is not None
    assert window.shape == (cfg.temporal_window_frames, cfg.sparse_dim)
    assert not rw.degraded_mode

    # Inferencia
    input_tensor = torch.from_numpy(window).unsqueeze(0).float()
    with torch.no_grad():
        pred_pose, pred_shapes = model(input_tensor)
    assert pred_pose.shape == (1, cfg.input_motion_length, 132)
    assert torch.all(torch.isfinite(pred_pose))

    # Postura
    pred_6d = extract_last_frame_pose_6d((pred_pose, pred_shapes))
    angles = extract_spine_angles(pred_6d)
    assert 'pitch_deg' in angles
    assert 'roll_deg' in angles
    assert np.isfinite(angles['pitch_deg'])
    assert np.isfinite(angles['roll_deg'])


def test_end_to_end_degraded_without_phone():
    cfg = load_config('config.yaml')
    from src.sensor_availability import quest_only_availability
    availability = quest_only_availability()
    calibration = build_calibration_from_neutral(
        quest_head_rot=np.array([0, 0, 0, 1], dtype=np.float32),
        quest_quaternion_order="xyzw",
        android_phone_rot=np.array([1, 0, 0, 0], dtype=np.float32),
        android_quaternion_order="wxyz",
        timestamp=time.time(),
        acceleration_includes_gravity=False,
    )
    frame = {
        'head': {'pos': np.zeros(3), 'rot': np.array([0,0,0,1])},
        'left_hand': {'pos': np.array([0.2,-0.1,0.3]), 'rot': np.array([0,0,0,1])},
        'right_hand': {'pos': np.array([-0.2,-0.1,0.3]), 'rot': np.array([0,0,0,1])},
        'pelvis': {'rot': np.array([0,0,0,1]), 'accel': np.array([0,0,0])},
    }
    f = build_hmd_poser_features(frame, None, calibration, availability)
    from src.feature_constants import ACCELERATION_SLICE, GLOBAL_ROT_SLICE, DELTA_ROT_SLICE
    # Pelvis debe estar en cero
    assert np.allclose(f[GLOBAL_ROT_SLICE.start + 5*6: GLOBAL_ROT_SLICE.start + 6*6], 0)
    assert np.allclose(f[DELTA_ROT_SLICE.start + 5*6: DELTA_ROT_SLICE.start + 6*6], 0)
    assert np.allclose(f[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9], 0)
    # Pies también
    assert np.allclose(f[ACCELERATION_SLICE.start: ACCELERATION_SLICE.start + 6], 0)
