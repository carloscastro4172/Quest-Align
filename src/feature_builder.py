"""
Constructor del vector de 135 características de HMD-Poser.

Implementa exactamente el layout del preprocesamiento oficial de HMD-Poser:
    https://github.com/Pico-AI-Team/HMD-Poser/blob/main/prepare_data.py

No se construye por bloques de 15 valores por sensor; se construye por bloques
semánticos (rotaciones globales, deltas, posiciones, etc.).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Any, Optional

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
)
from src.coordinate_frames import (
    Calibration,
    normalize_quaternion,
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    rotation_delta_matrix,
    relative_rotation_in_head,
    relative_position_in_head,
    is_finite_vector,
    reorder_quaternion_to_xyzw,
)
from src.sensor_availability import SensorAvailability


class FeatureBuilderError(ValueError):
    pass


def _get_segment_rotation_matrix(
    segment_name: str,
    frame: Dict[str, Any],
    calibration: Calibration,
    availability: SensorAvailability,
    quest_quaternion_order: str = "xyzw",
    android_quaternion_order: str = "wxyz",
) -> np.ndarray:
    """Obtain the internal rotation matrix of a segment, applying the configured
    quaternion-order reordering."""
    if not getattr(availability, segment_name, False):
        return np.zeros((3, 3), dtype=np.float64)

    if segment_name in ("head", "left_hand", "right_hand"):
        q = np.asarray(frame[segment_name]["rot"], dtype=np.float64)
        q = reorder_quaternion_to_xyzw(q, quest_quaternion_order)
        q = normalize_quaternion(q, label=f"{segment_name}_rot")
        m_unity = quaternion_to_matrix(q)
        return calibration.apply_quest_rotation_matrix(m_unity)

    if segment_name == "pelvis":
        q = np.asarray(frame["pelvis"]["rot"], dtype=np.float64)
        if q.shape != (4,):
            raise FeatureBuilderError(f"pelvis_rot debe tener shape (4,), tiene {q.shape}")
        q = reorder_quaternion_to_xyzw(q, android_quaternion_order)
        q = normalize_quaternion(q, label="pelvis_rot")
        m_phone = quaternion_to_matrix(q)
        return calibration.apply_android_rotation(m_phone)

    raise FeatureBuilderError(f"Segmento desconocido: {segment_name}")


def _get_segment_position(
    segment_name: str,
    frame: Dict[str, Any],
    calibration: Calibration,
    availability: SensorAvailability,
) -> np.ndarray:
    """Posición global en metros en el marco interno. Solo head, left_hand, right_hand."""
    if segment_name not in ("head", "left_hand", "right_hand"):
        raise FeatureBuilderError(f"No hay posición global para el segmento {segment_name}")
    if not getattr(availability, segment_name, False):
        return np.zeros(3, dtype=np.float64)

    pos = np.asarray(frame[segment_name]["pos"], dtype=np.float64)
    if pos.shape != (3,):
        raise FeatureBuilderError(f"{segment_name}_pos debe tener shape (3,), tiene {pos.shape}")
    if not is_finite_vector(pos):
        raise FeatureBuilderError(f"{segment_name}_pos contiene NaN/inf")
    return calibration.apply_quest_position(pos)


def _get_pelvis_acceleration(
    frame: Dict[str, Any],
    calibration: Calibration,
    availability: SensorAvailability,
) -> np.ndarray:
    if not availability.pelvis:
        return np.zeros(3, dtype=np.float64)
    acc = np.asarray(frame["pelvis"]["accel"], dtype=np.float64)
    if acc.shape != (3,):
        raise FeatureBuilderError(f"pelvis_accel debe tener shape (3,), tiene {acc.shape}")
    if not is_finite_vector(acc):
        raise FeatureBuilderError("pelvis_accel contiene NaN/inf")
    return calibration.apply_android_acceleration(acc)


def build_hmd_poser_features(
    current_frame: Dict[str, Any],
    previous_frame: Optional[Dict[str, Any]],
    calibration: Calibration,
    availability: SensorAvailability,
    quest_quaternion_order: str = "xyzw",
    android_quaternion_order: str = "wxyz",
) -> np.ndarray:
    """
    Construye el vector de 135 características de HMD-Poser a partir de dos frames
    sincronizados consecutivos.

    Args:
        current_frame: frame actual con datos de Quest y Android ya parseados.
        previous_frame: frame anterior. Si es None, los deltas se ponen a cero.
        calibration: instancia Calibration con las matrices de transformación.
        availability: instancia SensorAvailability indicando sensores presentes.

    Returns:
        np.ndarray de shape (135,) con el layout oficial de HMD-Poser.
    """
    if not calibration.is_valid():
        raise FeatureBuilderError("La calibración no contiene matrices de rotación válidas")

    feature = np.zeros(FEATURE_DIM, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Rotaciones globales 6D para los 6 segmentos
    # ------------------------------------------------------------------
    global_rot_mats = {}
    for i, segment in enumerate(SEGMENT_NAMES):
        M = _get_segment_rotation_matrix(segment, current_frame, calibration, availability,
                                          quest_quaternion_order, android_quaternion_order)
        global_rot_mats[segment] = M
        if np.allclose(M, 0):
            feature[GLOBAL_ROT_SLICE.start + i * 6: GLOBAL_ROT_SLICE.start + (i + 1) * 6] = 0.0
        else:
            feature[GLOBAL_ROT_SLICE.start + i * 6: GLOBAL_ROT_SLICE.start + (i + 1) * 6] = matrix_to_rotation_6d(M)

    # ------------------------------------------------------------------
    # 2. Delta de rotaciones globales
    # ------------------------------------------------------------------
    if previous_frame is None:
        for i, segment in enumerate(SEGMENT_NAMES):
            feature[DELTA_ROT_SLICE.start + i * 6: DELTA_ROT_SLICE.start + (i + 1) * 6] = 0.0
    else:
        for i, segment in enumerate(SEGMENT_NAMES):
            M_prev = _get_segment_rotation_matrix(segment, previous_frame, calibration, availability,
                                                   quest_quaternion_order, android_quaternion_order)
            M_curr = global_rot_mats[segment]
            if np.allclose(M_prev, 0) or np.allclose(M_curr, 0):
                feature[DELTA_ROT_SLICE.start + i * 6: DELTA_ROT_SLICE.start + (i + 1) * 6] = 0.0
            else:
                M_delta = rotation_delta_matrix(M_prev, M_curr)
                feature[DELTA_ROT_SLICE.start + i * 6: DELTA_ROT_SLICE.start + (i + 1) * 6] = matrix_to_rotation_6d(M_delta)

    # ------------------------------------------------------------------
    # 3. Posiciones globales XYZ (head, left_hand, right_hand)
    # ------------------------------------------------------------------
    position_segments = ["head", "left_hand", "right_hand"]
    pos_current = {}
    for i, segment in enumerate(position_segments):
        p = _get_segment_position(segment, current_frame, calibration, availability)
        pos_current[segment] = p
        feature[GLOBAL_POS_SLICE.start + i * 3: GLOBAL_POS_SLICE.start + (i + 1) * 3] = p

    # ------------------------------------------------------------------
    # 4. Delta de posiciones globales
    # ------------------------------------------------------------------
    if previous_frame is None:
        feature[DELTA_POS_SLICE] = 0.0
    else:
        for i, segment in enumerate(position_segments):
            p_prev = _get_segment_position(segment, previous_frame, calibration, availability)
            p_curr = pos_current[segment]
            feature[DELTA_POS_SLICE.start + i * 3: DELTA_POS_SLICE.start + (i + 1) * 3] = p_curr - p_prev

    # ------------------------------------------------------------------
    # 5. Rotaciones relativas mano-cabeza
    # ------------------------------------------------------------------
    M_head = global_rot_mats["head"]
    M_left_hand = global_rot_mats["left_hand"]
    M_right_hand = global_rot_mats["right_hand"]

    if np.allclose(M_head, 0) or np.allclose(M_left_hand, 0):
        feature[RELATIVE_ROT_SLICE.start: RELATIVE_ROT_SLICE.start + 6] = 0.0
    else:
        feature[RELATIVE_ROT_SLICE.start: RELATIVE_ROT_SLICE.start + 6] = matrix_to_rotation_6d(
            relative_rotation_in_head(M_head, M_left_hand)
        )

    if np.allclose(M_head, 0) or np.allclose(M_right_hand, 0):
        feature[RELATIVE_ROT_SLICE.start + 6: RELATIVE_ROT_SLICE.start + 12] = 0.0
    else:
        feature[RELATIVE_ROT_SLICE.start + 6: RELATIVE_ROT_SLICE.start + 12] = matrix_to_rotation_6d(
            relative_rotation_in_head(M_head, M_right_hand)
        )

    # ------------------------------------------------------------------
    # 6. Delta de rotaciones relativas mano-cabeza
    # ------------------------------------------------------------------
    if previous_frame is None:
        feature[DELTA_RELATIVE_ROT_SLICE] = 0.0
    else:
        M_head_prev = _get_segment_rotation_matrix("head", previous_frame, calibration, availability,
                                                     quest_quaternion_order, android_quaternion_order)
        M_left_hand_prev = _get_segment_rotation_matrix("left_hand", previous_frame, calibration, availability,
                                                         quest_quaternion_order, android_quaternion_order)
        M_right_hand_prev = _get_segment_rotation_matrix("right_hand", previous_frame, calibration, availability,
                                                          quest_quaternion_order, android_quaternion_order)

        if np.allclose(M_head_prev, 0) or np.allclose(M_left_hand_prev, 0):
            rel_left_prev = np.zeros((3, 3))
        else:
            rel_left_prev = relative_rotation_in_head(M_head_prev, M_left_hand_prev)

        if np.allclose(M_head, 0) or np.allclose(M_left_hand, 0):
            rel_left_curr = np.zeros((3, 3))
        else:
            rel_left_curr = relative_rotation_in_head(M_head, M_left_hand)

        if np.allclose(rel_left_prev, 0) or np.allclose(rel_left_curr, 0):
            feature[DELTA_RELATIVE_ROT_SLICE.start: DELTA_RELATIVE_ROT_SLICE.start + 6] = 0.0
        else:
            feature[DELTA_RELATIVE_ROT_SLICE.start: DELTA_RELATIVE_ROT_SLICE.start + 6] = matrix_to_rotation_6d(
                rotation_delta_matrix(rel_left_prev, rel_left_curr)
            )

        if np.allclose(M_head_prev, 0) or np.allclose(M_right_hand_prev, 0):
            rel_right_prev = np.zeros((3, 3))
        else:
            rel_right_prev = relative_rotation_in_head(M_head_prev, M_right_hand_prev)

        if np.allclose(M_head, 0) or np.allclose(M_right_hand, 0):
            rel_right_curr = np.zeros((3, 3))
        else:
            rel_right_curr = relative_rotation_in_head(M_head, M_right_hand)

        if np.allclose(rel_right_prev, 0) or np.allclose(rel_right_curr, 0):
            feature[DELTA_RELATIVE_ROT_SLICE.start + 6: DELTA_RELATIVE_ROT_SLICE.start + 12] = 0.0
        else:
            feature[DELTA_RELATIVE_ROT_SLICE.start + 6: DELTA_RELATIVE_ROT_SLICE.start + 12] = matrix_to_rotation_6d(
                rotation_delta_matrix(rel_right_prev, rel_right_curr)
            )

    # ------------------------------------------------------------------
    # 7. Posiciones relativas mano-cabeza
    # ------------------------------------------------------------------
    p_head = pos_current["head"]
    p_left_hand = pos_current["left_hand"]
    p_right_hand = pos_current["right_hand"]

    if np.allclose(M_head, 0):
        feature[RELATIVE_POS_SLICE] = 0.0
    else:
        feature[RELATIVE_POS_SLICE.start: RELATIVE_POS_SLICE.start + 3] = relative_position_in_head(
            p_head, M_head, p_left_hand
        )
        feature[RELATIVE_POS_SLICE.start + 3: RELATIVE_POS_SLICE.start + 6] = relative_position_in_head(
            p_head, M_head, p_right_hand
        )

    # ------------------------------------------------------------------
    # 8. Delta de posiciones relativas mano-cabeza
    # ------------------------------------------------------------------
    if previous_frame is None:
        feature[DELTA_RELATIVE_POS_SLICE] = 0.0
    else:
        p_head_prev = _get_segment_position("head", previous_frame, calibration, availability)
        M_head_prev = _get_segment_rotation_matrix("head", previous_frame, calibration, availability)
        p_left_hand_prev = _get_segment_position("left_hand", previous_frame, calibration, availability)
        p_right_hand_prev = _get_segment_position("right_hand", previous_frame, calibration, availability)

        if np.allclose(M_head, 0) or np.allclose(M_head_prev, 0):
            feature[DELTA_RELATIVE_POS_SLICE] = 0.0
        else:
            rel_left_curr = relative_position_in_head(p_head, M_head, p_left_hand)
            rel_left_prev = relative_position_in_head(p_head_prev, M_head_prev, p_left_hand_prev)
            feature[DELTA_RELATIVE_POS_SLICE.start: DELTA_RELATIVE_POS_SLICE.start + 3] = rel_left_curr - rel_left_prev

            rel_right_curr = relative_position_in_head(p_head, M_head, p_right_hand)
            rel_right_prev = relative_position_in_head(p_head_prev, M_head_prev, p_right_hand_prev)
            feature[DELTA_RELATIVE_POS_SLICE.start + 3: DELTA_RELATIVE_POS_SLICE.start + 6] = rel_right_curr - rel_right_prev

    # ------------------------------------------------------------------
    # 9. Aceleraciones: left_foot, right_foot, pelvis
    # ------------------------------------------------------------------
    pelvis_acc = _get_pelvis_acceleration(current_frame, calibration, availability)
    feature[ACCELERATION_SLICE.start: ACCELERATION_SLICE.start + 3] = 0.0           # left_foot
    feature[ACCELERATION_SLICE.start + 3: ACCELERATION_SLICE.start + 6] = 0.0       # right_foot
    feature[ACCELERATION_SLICE.start + 6: ACCELERATION_SLICE.start + 9] = pelvis_acc  # pelvis

    # ------------------------------------------------------------------
    # 10. Aplicar máscara semántica de sensores ausentes
    # ------------------------------------------------------------------
    for idx in availability.mask_indices():
        feature[idx] = 0.0

    if not np.all(np.isfinite(feature)):
        raise FeatureBuilderError("El vector de características contiene NaN o infinitos")

    return feature.astype(np.float32)


class HMDPoserFeatureBuilder:
    """Wrapper with the old interface to ease server migration."""

    def __init__(self, calibration: Calibration, availability: SensorAvailability,
                 quest_quaternion_order: str = "xyzw",
                 android_quaternion_order: str = "wxyz"):
        self.calibration = calibration
        self.availability = availability
        self.quest_quaternion_order = quest_quaternion_order
        self.android_quaternion_order = android_quaternion_order

    def build_tensor(self, current_frame: Dict[str, Any],
                     previous_frame: Optional[Dict[str, Any]] = None) -> np.ndarray:
        return build_hmd_poser_features(current_frame, previous_frame, self.calibration,
                                        self.availability,
                                        self.quest_quaternion_order,
                                        self.android_quaternion_order)

    def build_window(self, frames: list) -> np.ndarray:
        """Build a (N, 135) window from a list of consecutive frames."""
        features = []
        for i, frame in enumerate(frames):
            prev = frames[i - 1] if i > 0 else None
            features.append(self.build_tensor(frame, prev))
        return np.stack(features, axis=0)
