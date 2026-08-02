"""
Extracción de ángulos de columna a partir de la salida 6D de HMD-Poser.

El modelo devuelve `pred_pose` de shape (batch, seq, 22*6):
    - primeros 6 valores: orientación global de la pelvis (root_orient);
    - siguientes 21*6: rotaciones locales de las articulaciones 1..21.

Para obtener la orientación global del tronco se aplica una cinemática directa
simplificada con la jerarquía SMPL de 22 articulaciones.

ADVERTENCIA: la jerarquía SMPL usada aquí es la estándar de SMPL+H. Si se dispone
del body model, la tabla `bm.kintree_table[0][:22]` debe preferirse a la lista
incluida aquí.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.coordinate_frames import rotation_6d_to_matrix


# Jerarquía SMPL de 22 articulaciones (0=root, 21=right wrist).
# Fuente: SMPL+H kintree_table[0][:22].
SMPL_22_PARENTS = np.array([
    0,   # 0  pelvis
    0,   # 1  left_hip
    0,   # 2  right_hip
    0,   # 3  spine1
    1,   # 4  left_knee
    2,   # 5  right_knee
    3,   # 6  spine2
    4,   # 7  left_ankle
    5,   # 8  right_ankle
    6,   # 9  spine3
    7,   # 10 left_foot
    8,   # 11 right_foot
    9,   # 12 neck
    12,  # 13 left_collar
    12,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
])


def _norm180(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def sixd_to_matrix_batch(d6: np.ndarray) -> np.ndarray:
    """Convierte un array de 6D de shape (N, 6) a matrices (N, 3, 3)."""
    d6 = np.asarray(d6, dtype=np.float64).reshape(-1, 6)
    N = d6.shape[0]
    out = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        out[i] = rotation_6d_to_matrix(d6[i])
    return out


def forward_kinematics_global(local_rotations: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Calcula rotaciones globales a partir de rotaciones locales.

    Args:
        local_rotations: array de shape (N, 3, 3).
        parents: array de shape (N,) con el índice padre de cada articulación.

    Returns:
        global_rotations: array de shape (N, 3, 3).
    """
    N = local_rotations.shape[0]
    global_rot = np.zeros((N, 3, 3), dtype=np.float64)
    global_rot[0] = local_rotations[0]
    for i in range(1, N):
        global_rot[i] = global_rot[parents[i]] @ local_rotations[i]
    return global_rot


def extract_spine_angles(pred_pose_6d: np.ndarray,
                          parents: np.ndarray = SMPL_22_PARENTS,
                          degrees: bool = True) -> dict:
    """
    Extrae ángulos de inclinación sagital (pitch) y frontal (roll) del tronco.

    Args:
        pred_pose_6d: array de shape (22, 6) con la pose local 6D.
        parents: jerarquía articular.
        degrees: si True, devuelve grados; si False, radianes.

    Returns:
        dict con 'pitch_deg', 'roll_deg', 'trunk_global_matrix'.
    """
    local_mats = sixd_to_matrix_batch(pred_pose_6d)
    global_mats = forward_kinematics_global(local_mats, parents)

    # Orientación global del tronco: pelvis * spine1 * spine2 * spine3
    trunk = global_mats[0] @ global_mats[3] @ global_mats[6] @ global_mats[9]

    # Convención ZXY: euler[1] = flexión/extensión (pitch), euler[2] = lateral (roll)
    euler = R.from_matrix(trunk).as_euler('ZXY', degrees=True)
    pitch = _norm180(float(euler[1]))
    roll = _norm180(float(euler[2]))

    return {
        "pitch_deg": pitch,
        "roll_deg": roll,
        "trunk_global_matrix": trunk,
    }


def extract_last_frame_pose_6d(model_output: tuple) -> np.ndarray:
    """
    model_output es la tupla (pred_pose, pred_shapes) de HMD_imu_HME_Universe.
    Devuelve la pose 6D del último frame: shape (22, 6).
    """
    pred_pose = model_output[0]  # (batch, seq, 22*6)
    last = pred_pose[:, -1, :].reshape(-1, 6)
    return last.detach().cpu().numpy()
