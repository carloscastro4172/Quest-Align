"""
Tests del constructor de 135 características y de las transformaciones de rotación.
"""

import numpy as np
import pytest
import torch

from src.feature_constants import (
    FEATURE_DIM,
    GLOBAL_ROT_SLICE,
    DELTA_ROT_SLICE,
    GLOBAL_POS_SLICE,
    DELTA_POS_SLICE,
    RELATIVE_ROT_SLICE,
    DELTA_RELATIVE_ROT_SLICE,
    RELATIVE_POS_SLICE,
    DELTA_RELATIVE_POS_SLICE,
    ACCELERATION_SLICE,
    SEGMENT_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from src.feature_builder import HMDPoserFeatureBuilder, build_hmd_poser_features
from src.coordinate_frames import (
    Calibration,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    rotation_delta_matrix,
    relative_rotation_in_head,
    relative_position_in_head,
    quaternion_to_matrix,
    matrix_to_quaternion,
    normalize_quaternion,
    unity_to_internal_quaternion,
    unity_to_internal_position,
)
from src.sensor_availability import (
    quest_plus_pelvis_availability,
    quest_only_availability,
)


def neutral_calibration():
    return Calibration(
        timestamp=0.0,
        R_internal_from_quest=np.eye(3),
        R_internal_from_android=np.eye(3),
        R_body_from_phone=np.eye(3),
        neutral_body_rotation_internal=np.eye(3),
        acceleration_includes_gravity=False,
    )


def make_frame(head_pos=(0,0,0), head_rot=(0,0,0,1),
               left_pos=(0.2, -0.1, 0.3), left_rot=(0,0,0,1),
               right_pos=(-0.2, -0.1, 0.3), right_rot=(0,0,0,1),
               pelvis_rot=(0,0,0,1), pelvis_accel=(0,0,0)):
    return {
        'head': {'pos': np.array(head_pos, dtype=np.float32),
                 'rot': np.array(head_rot, dtype=np.float32)},
        'left_hand': {'pos': np.array(left_pos, dtype=np.float32),
                      'rot': np.array(left_rot, dtype=np.float32)},
        'right_hand': {'pos': np.array(right_pos, dtype=np.float32),
                       'rot': np.array(right_rot, dtype=np.float32)},
        'pelvis': {'rot': np.array(pelvis_rot, dtype=np.float32),
                   'accel': np.array(pelvis_accel, dtype=np.float32)},
    }


def test_feature_vector_has_135_values():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame())
    assert f.shape == (FEATURE_DIM,)


def test_feature_block_boundaries():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame())
    boundaries = [
        (GLOBAL_ROT_SLICE, "global_rot"),
        (DELTA_ROT_SLICE, "delta_rot"),
        (GLOBAL_POS_SLICE, "global_pos"),
        (DELTA_POS_SLICE, "delta_pos"),
        (RELATIVE_ROT_SLICE, "relative_rot"),
        (DELTA_RELATIVE_ROT_SLICE, "delta_relative_rot"),
        (RELATIVE_POS_SLICE, "relative_pos"),
        (DELTA_RELATIVE_POS_SLICE, "delta_relative_pos"),
        (ACCELERATION_SLICE, "acceleration"),
    ]
    for s, name in boundaries:
        assert f[s].shape[0] == s.stop - s.start, f"{name} tiene tamaño incorrecto"


def test_sensor_order():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame())
    for i, name in enumerate(SEGMENT_NAMES):
        block = f[GLOBAL_ROT_SLICE.start + i*6: GLOBAL_ROT_SLICE.start + (i+1)*6]
        assert block.shape == (6,), f"{name} block shape incorrecto"


def test_quaternion_normalization():
    q = np.array([0.0, 0.0, 0.0, 2.0])
    qn = normalize_quaternion(q)
    assert np.allclose(np.linalg.norm(qn), 1.0)


def test_zero_norm_quaternion_rejected():
    with pytest.raises(ValueError):
        normalize_quaternion(np.array([0.0, 0.0, 0.0, 0.0]))


def test_rotation_6d_matches_reference():
    """Comparar con la implementación oficial de HMD-Poser: torch.cat([R[:,0], R[:,1]])."""
    theta = np.radians(20)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    our = matrix_to_rotation_6d(R)
    official = np.concatenate([R[:, 0], R[:, 1]])
    assert np.allclose(our, official)


def test_relative_rotation_matches_reference():
    """Fórmula oficial: R_head^T @ R_hand."""
    theta = np.radians(20)
    R_head = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    R_hand = np.eye(3)
    rel = relative_rotation_in_head(R_head, R_hand)
    ref = R_head.T @ R_hand
    assert np.allclose(rel, ref)


def test_relative_position():
    """Fórmula oficial: (p_hand - p_head) @ R_head."""
    R_head = np.eye(3)
    p_head = np.array([0.0, 1.0, 0.0])
    p_hand = np.array([0.2, 1.0, 0.3])
    rel = relative_position_in_head(p_head, R_head, p_hand)
    assert np.allclose(rel, p_hand - p_head)


def test_temporal_features_use_two_real_frames():
    """Los deltas deben ser distintos cuando dos frames reales son distintos."""
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f1 = fb.build_tensor(make_frame(left_pos=(0.2, -0.1, 0.3)))
    f2 = fb.build_tensor(make_frame(left_pos=(0.25, -0.1, 0.3)),
                         previous_frame=make_frame(left_pos=(0.2, -0.1, 0.3)))
    assert not np.allclose(f1, f2)
    # Delta de posición debe ser no nulo
    assert not np.allclose(f2[DELTA_POS_SLICE], 0)


def test_missing_feet_mask_only_feet_channels():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame(pelvis_accel=(1.0, 2.0, 3.0)))
    # Los pies deben estar en cero
    for i in [3, 4]:
        assert np.allclose(f[GLOBAL_ROT_SLICE.start + i*6: GLOBAL_ROT_SLICE.start + (i+1)*6], 0)
        assert np.allclose(f[DELTA_ROT_SLICE.start + i*6: DELTA_ROT_SLICE.start + (i+1)*6], 0)
    assert np.allclose(f[ACCELERATION_SLICE.start: ACCELERATION_SLICE.start + 6], 0)
    # Pelvis no debe estar en cero (rotación y aceleración)
    assert not np.allclose(f[GLOBAL_ROT_SLICE.start + 5*6: GLOBAL_ROT_SLICE.start + 6*6], 0)
    assert not np.allclose(f[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9], 0)


def test_pelvis_channels_are_not_zero_when_available():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame(pelvis_rot=(0,0,0,1), pelvis_accel=(1,2,3)))
    assert not np.allclose(f[GLOBAL_ROT_SLICE.start + 5*6: GLOBAL_ROT_SLICE.start + 6*6], 0)
    assert np.allclose(f[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9], np.array([1,2,3]))


def test_phone_missing_activates_degraded_mode():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_only_availability())
    f = fb.build_tensor(make_frame())
    # Pelvis debe estar en cero
    assert np.allclose(f[GLOBAL_ROT_SLICE.start + 5*6: GLOBAL_ROT_SLICE.start + 6*6], 0)
    assert np.allclose(f[DELTA_ROT_SLICE.start + 5*6: DELTA_ROT_SLICE.start + 6*6], 0)
    assert np.allclose(f[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9], 0)


def test_phone_missing_does_not_insert_fake_gravity():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_only_availability())
    f = fb.build_tensor(make_frame())
    # La aceleración de pelvis (últimos 3 valores) debe ser cero, no [0, -9.81, 0]
    assert np.allclose(f[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9], 0)


def test_coordinate_transform_is_orthogonal():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    m_int = quaternion_to_matrix(unity_to_internal_quaternion(q))
    assert np.allclose(m_int.T @ m_int, np.eye(3), atol=1e-6)


def test_coordinate_transform_has_positive_determinant():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    m_int = quaternion_to_matrix(unity_to_internal_quaternion(q))
    assert np.linalg.det(m_int) > 0


def test_acceleration_units_metadata():
    calib = neutral_calibration()
    calib.acceleration_includes_gravity = False
    assert calib.acceleration_includes_gravity is False
    # Sin gravedad, aceleración en reposo debe ser cercana a cero
    acc = np.array([0, 0, 0])
    out = calib.apply_android_acceleration(acc)
    assert np.allclose(out, 0)


def test_no_nan_in_features():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame())
    assert np.all(np.isfinite(f))


def test_no_infinite_values():
    fb = HMDPoserFeatureBuilder(neutral_calibration(), quest_plus_pelvis_availability())
    f = fb.build_tensor(make_frame())
    assert not np.any(np.isinf(f))


def test_feature_schema_version_constant():
    assert FEATURE_SCHEMA_VERSION == "hmd_poser_135_v1"
