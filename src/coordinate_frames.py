"""
Transformaciones de coordenadas entre Quest/Unity, Android y el marco interno común.

ADVERTENCIA: el código de los emisores (Quest/Unity y Android) no está incluido en
este repositorio. Las conversiones definidas aquí son las más razonables dadas las
convenciones documentadas de cada plataforma, pero deben verificarse con el emisor
real y, en el caso del teléfono, completarse con la calibración de montaje.

Marco interno (Quest Align):
    - Derecho.
    - Unidades: metros (m) para posición, m/s² para aceleración.
    - Ejes: X derecha, Y arriba, Z adelante (positivo hacia delante).
    - Cuaterniones en formato [x, y, z, w].
    - Compatible con el marco de AMASS/SMPL usado por HMD-Poser.

Quest / Unity (supuesto hasta verificar el emisor):
    - Izquierdo, Y arriba, X derecha, Z adelante.
    - Las posiciones parecen estar en metros.
    - El orden del cuaternión no está verificado.

Android (supuesto hasta verificar el emisor):
    - TYPE_ROTATION_VECTOR devuelve un cuaternión [w, x, y, z] relativo al
      marco ENU (East-North-Up): X este, Y norte, Z arriba.
    - El acelerómetro (TYPE_ACCELEROMETER o TYPE_LINEAR_ACCELERATION) devuelve
      m/s² según los ejes del dispositivo (X derecha, Y arriba, Z fuera de pantalla).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

# Matriz de reflexión que convierte el eje Z de Unity a nuestro eje Z interno.
# No es una rotación, sino una reflexión; la transformación de matrices de rotación
# usa S @ M @ S (S = S^{-1}) para conservar el determinante +1.
S_UNITY_TO_INTERNAL = np.diag([1.0, 1.0, -1.0])

# Gravedad en el marco interno (Y arriba).
GRAVITY_INTERNAL = np.array([0.0, -9.80665, 0.0], dtype=np.float64)

# Tolerancia para matrices de rotación
ROTATION_ORTHOGONALITY_TOL = 1e-4
ROTATION_DETERMINANT_TOL = 1e-4


class QuaternionError(ValueError):
    pass


class RotationMatrixError(ValueError):
    pass


def normalize_quaternion(q: np.ndarray, eps: float = 1e-8, label: str = "quaternion") -> np.ndarray:
    """Normaliza un cuaternión y rechaza normas cercanas a cero."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise QuaternionError(f"{label} debe tener 4 componentes, tiene {q.shape}")
    norm = np.linalg.norm(q)
    if norm < eps:
        raise QuaternionError(f"{label} tiene norma casi nula: {norm}")
    return q / norm


def _order_indices(order: str) -> Tuple[int, int, int, int]:
    """Mapea un orden de cuaternión a los índices [w,x,y,z] de un array [?, ?, ?, ?]."""
    order = order.lower().strip()
    if order == "xyzw":
        return 3, 0, 1, 2
    if order == "wxyz":
        return 0, 1, 2, 3
    raise QuaternionError(f"Orden de cuaternión no soportado: {order}")


def _to_xyzw(q: np.ndarray, order: str) -> np.ndarray:
    w, x, y, z = _order_indices(order)
    return np.array([q[x], q[y], q[z], q[w]], dtype=np.float64)


def _from_xyzw(q: np.ndarray, order: str) -> np.ndarray:
    w, x, y, z = _order_indices(order)
    out = np.zeros(4, dtype=np.float64)
    out[w], out[x], out[y], out[z] = q[3], q[0], q[1], q[2]
    return out


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Convierte un cuaternión [x, y, z, w] a matriz de rotación 3x3."""
    q = normalize_quaternion(q_xyzw, label="quaternion_to_matrix")
    return R.from_quat(q).as_matrix().astype(np.float64)


def matrix_to_quaternion(m: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Convierte una matriz de rotación 3x3 a cuaternión en el orden solicitado."""
    m = np.asarray(m, dtype=np.float64)
    check_rotation_matrix(m, label="matrix_to_quaternion")
    q = R.from_matrix(m).as_quat()  # scipy devuelve [x, y, z, w]
    return _from_xyzw(q, order)


def check_rotation_matrix(m: np.ndarray, tol_ortho: float = ROTATION_ORTHOGONALITY_TOL,
                          tol_det: float = ROTATION_DETERMINANT_TOL, label: str = "matriz") -> None:
    """Verifica que una matriz sea ortogonal con determinante +1."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise RotationMatrixError(f"{label} debe tener shape (3,3), tiene {m.shape}")
    eye_diff = np.abs(m.T @ m - np.eye(3))
    if np.max(eye_diff) > tol_ortho:
        raise RotationMatrixError(
            f"{label} no es ortogonal (max |M^T M - I| = {np.max(eye_diff):.2e})"
        )
    det = np.linalg.det(m)
    if det <= 0 or abs(det - 1.0) > tol_det:
        raise RotationMatrixError(f"{label} tiene determinante {det}, se esperaba +1")


def unity_to_internal_matrix(m_unity: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz de rotación en el marco de Unity (izquierdo) a una matriz
    de rotación en el marco interno derecho.

    Matematicamente: M_int = S @ M_unity @ S, con S = diag(1,1,-1).
    """
    m_unity = np.asarray(m_unity, dtype=np.float64)
    check_rotation_matrix(m_unity, label="unity_to_internal_matrix input")
    m_int = S_UNITY_TO_INTERNAL @ m_unity @ S_UNITY_TO_INTERNAL
    check_rotation_matrix(m_int, label="unity_to_internal_matrix output")
    return m_int


def unity_to_internal_quaternion(q_unity_xyzw: np.ndarray) -> np.ndarray:
    """Convierte un cuaternión de Unity (supuesto [x,y,z,w]) a cuaternión interno."""
    m_unity = quaternion_to_matrix(q_unity_xyzw)
    m_int = unity_to_internal_matrix(m_unity)
    return matrix_to_quaternion(m_int, order="xyzw")


def unity_to_internal_position(p_unity: np.ndarray) -> np.ndarray:
    """Convierte una posición de Unity (izquierdo) a posición interna (derecho)."""
    p = np.asarray(p_unity, dtype=np.float64).reshape(3)
    return S_UNITY_TO_INTERNAL @ p


def unity_to_internal_rotation_6d(m_unity: np.ndarray) -> np.ndarray:
    """Convierte una matriz de rotación Unity a representación 6D interna."""
    m_int = unity_to_internal_matrix(m_unity)
    return matrix_to_rotation_6d(m_int)


def matrix_to_rotation_6d(m: np.ndarray) -> np.ndarray:
    """
    Conversión 6D exacta del repositorio oficial:
    https://github.com/Pico-AI-Team/HMD-Poser/blob/main/utils/utils_transform.py

    Devuelve las dos primeras columnas de la matriz concatenadas:
        [R[:,0], R[:,1]] -> [R00, R10, R20, R01, R11, R21]
    """
    m = np.asarray(m, dtype=np.float64)
    check_rotation_matrix(m, label="matrix_to_rotation_6d")
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float64)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Convierte una representación 6D a matriz de rotación 3x3."""
    d6 = np.asarray(d6, dtype=np.float64).reshape(6)
    b1 = d6[:3]
    b2 = d6[3:6]
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 - np.dot(b1, b2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_delta_matrix(m_prev: np.ndarray, m_curr: np.ndarray) -> np.ndarray:
    """
    Delta de rotación entre dos frames: R_delta = R_prev^T @ R_curr.
    Esta es la misma operación usada en `prepare_data.py` de HMD-Poser.
    """
    check_rotation_matrix(m_prev, label="rotation_delta prev")
    check_rotation_matrix(m_curr, label="rotation_delta curr")
    return m_prev.T @ m_curr


def relative_rotation_in_head(m_head: np.ndarray, m_hand: np.ndarray) -> np.ndarray:
    """
    Rotación de la mano relativa a la cabeza: R_head^T @ R_hand.
    Igual que `hands_rotation_mat_in_head_space` en `prepare_data.py`.
    """
    check_rotation_matrix(m_head, label="relative_rotation_in_head head")
    check_rotation_matrix(m_hand, label="relative_rotation_in_head hand")
    return m_head.T @ m_hand


def relative_position_in_head(p_head: np.ndarray, m_head: np.ndarray,
                                p_hand: np.ndarray) -> np.ndarray:
    """
    Posición de la mano relativa a la cabeza, expresada en el espacio de la cabeza.
    Fórmula exacta del repositorio oficial:
        (p_hand - p_head) @ R_head
    """
    m_head = np.asarray(m_head, dtype=np.float64)
    diff = np.asarray(p_hand, dtype=np.float64) - np.asarray(p_head, dtype=np.float64)
    return diff @ m_head


def android_to_internal_rotation(m_phone_world: np.ndarray,
                                  R_internal_from_android: np.ndarray,
                                  R_body_from_phone: np.ndarray = np.eye(3)) -> np.ndarray:
    """
    Aplica la calibración al cuaternión/teléfono Android para obtener la orientación
    de la pelvis en el marco interno.
    """
    m = R_internal_from_android @ m_phone_world @ R_body_from_phone
    check_rotation_matrix(m, label="android_to_internal_rotation output")
    return m


def android_to_internal_acceleration(acc_device: np.ndarray,
                                        R_internal_from_android: np.ndarray,
                                        R_body_from_phone: np.ndarray = np.eye(3),
                                        includes_gravity: bool = False,
                                        gravity_vector: np.ndarray = GRAVITY_INTERNAL) -> np.ndarray:
    """
    Transforma la aceleración del teléfono al marco interno.

    Args:
        acc_device: aceleración en ejes del teléfono (m/s²).
        R_internal_from_android: rotación del marco Android world al marco interno.
        R_body_from_phone: rotación del cuerpo (pelvis) respecto al teléfono.
        includes_gravity: si True, se resta la gravedad interna después de transformar.
        gravity_vector: vector gravedad en el marco interno.

    Returns:
        Aceleración lineal en el marco interno (m/s²).
    """
    acc_device = np.asarray(acc_device, dtype=np.float64).reshape(3)
    acc_body = R_body_from_phone @ acc_device
    acc_internal = R_internal_from_android @ acc_body
    if includes_gravity:
        acc_internal = acc_internal - gravity_vector
    return acc_internal


def is_finite_vector(v: np.ndarray) -> bool:
    return np.all(np.isfinite(v))


@dataclass
class Calibration:
    """
    Calibración de un frame de captura. Contiene las matrices de transformación
    explícitas entre los marcos de los sensores y el marco interno.
    """
    # Momento en el que se calculó la calibración (timestamp de servidor, segundos).
    timestamp: float

    # Transformación del marco de Unity/Quest al marco interno.
    # Para el marco supuesto de Unity, es una reflexión de Z fija.
    R_internal_from_quest: np.ndarray = field(default_factory=lambda: S_UNITY_TO_INTERNAL.copy())

    # Transformación del marco Android world al marco interno.
    # Se calcula en la calibración obligatoria; no es una identidad silenciosa.
    R_internal_from_android: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Transformación del teléfono al segmento corporal (pelvis).
    # Si no se conoce el montaje, permanece como identidad y se documenta.
    R_body_from_phone: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Orientación del cuerpo (pelvis) en el marco interno durante la calibración.
    # Se usa para documentar el supuesto de postura neutral.
    neutral_body_rotation_internal: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Vector de gravedad esperado en el marco interno.
    gravity_internal: np.ndarray = field(default_factory=lambda: GRAVITY_INTERNAL.copy())

    # True si la aceleración Android incluye gravedad; False si es linear acceleration;
    # None si no se ha verificado.
    acceleration_includes_gravity: Optional[bool] = None

    def __post_init__(self):
        self.R_internal_from_quest = np.asarray(self.R_internal_from_quest, dtype=np.float64)
        self.R_internal_from_android = np.asarray(self.R_internal_from_android, dtype=np.float64)
        self.R_body_from_phone = np.asarray(self.R_body_from_phone, dtype=np.float64)
        self.neutral_body_rotation_internal = np.asarray(self.neutral_body_rotation_internal, dtype=np.float64)
        self.gravity_internal = np.asarray(self.gravity_internal, dtype=np.float64)

    def is_valid(self) -> bool:
        """
        Valida las matrices de rotación propiamente dichas.

        R_internal_from_quest NO es una matriz de rotación (es una reflexión S), así
        que no se somete a la comprobación de determinante +1.
        """
        try:
            check_rotation_matrix(self.R_internal_from_android, label="R_internal_from_android")
            check_rotation_matrix(self.R_body_from_phone, label="R_body_from_phone")
            check_rotation_matrix(self.neutral_body_rotation_internal, label="neutral_body_rotation_internal")
            return True
        except RotationMatrixError:
            return False

    def apply_quest_position(self, p_unity: np.ndarray) -> np.ndarray:
        return unity_to_internal_position(p_unity)

    def apply_quest_rotation(self, q_unity_xyzw: np.ndarray) -> np.ndarray:
        return unity_to_internal_quaternion(q_unity_xyzw)

    def apply_quest_rotation_matrix(self, m_unity: np.ndarray) -> np.ndarray:
        return unity_to_internal_matrix(m_unity)

    def apply_android_rotation(self, m_phone_world: np.ndarray) -> np.ndarray:
        return android_to_internal_rotation(m_phone_world, self.R_internal_from_android, self.R_body_from_phone)

    def apply_android_acceleration(self, acc_device: np.ndarray) -> np.ndarray:
        return android_to_internal_acceleration(
            acc_device,
            self.R_internal_from_android,
            self.R_body_from_phone,
            includes_gravity=bool(self.acceleration_includes_gravity) if self.acceleration_includes_gravity is not None else False,
            gravity_vector=self.gravity_internal,
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "R_internal_from_quest": self.R_internal_from_quest.tolist(),
            "R_internal_from_android": self.R_internal_from_android.tolist(),
            "R_body_from_phone": self.R_body_from_phone.tolist(),
            "neutral_body_rotation_internal": self.neutral_body_rotation_internal.tolist(),
            "gravity_internal": self.gravity_internal.tolist(),
            "acceleration_includes_gravity": self.acceleration_includes_gravity,
        }


def build_calibration_from_neutral(
    quest_head_rot_xyzw: np.ndarray,
    android_phone_rot_wxyz: np.ndarray,
    timestamp: float,
    assume_pelvis_aligned_with_head: bool = True,
    phone_mount_transform: Optional[np.ndarray] = None,
    acceleration_includes_gravity: Optional[bool] = None,
) -> Calibration:
    """
    Construye una calibración a partir de un frame de postura neutral.

    Args:
        quest_head_rot_xyzw: orientación de la cabeza en Quest (supuesto [x,y,z,w]).
        android_phone_rot_wxyz: orientación del teléfono en Android [w,x,y,z].
        timestamp: momento de la calibración.
        assume_pelvis_aligned_with_head: si True, se supone que la pelvis está alineada
            con la cabeza en la postura neutral (yaw/pitch/roll similares). Esto es un
            supuesto porque Quest Align no mide la pelvis directamente.
        phone_mount_transform: matriz de rotación del teléfono respecto a la pelvis.
            Si es None, se usa la identidad y se documenta que el montaje no se conoce.
        acceleration_includes_gravity: indica si el campo 'acc' de Android incluye
            gravedad. None si no se ha verificado.

    Returns:
        Objeto Calibration listo para transformar frames de captura.
    """
    # Convertir a matrices en el marco interno
    q_head_int = normalize_quaternion(quest_head_rot_xyzw, label="quest_head_rot")
    m_head_int = unity_to_internal_matrix(quaternion_to_matrix(q_head_int))

    q_phone_xyzw = _to_xyzw(android_phone_rot_wxyz, order="wxyz")
    q_phone_xyzw = normalize_quaternion(q_phone_xyzw, label="android_phone_rot")
    m_phone_world = quaternion_to_matrix(q_phone_xyzw)

    # R_internal_from_android: mapea el marco Android world al marco interno.
    # Si se asume pelvis alineada con la cabeza, R_body_internal_neutral = m_head_int.
    # R_body_from_phone = identity si no se conoce montaje.
    R_body_from_phone = np.eye(3) if phone_mount_transform is None else np.asarray(phone_mount_transform, dtype=np.float64)
    check_rotation_matrix(R_body_from_phone, label="phone_mount_transform")

    if assume_pelvis_aligned_with_head:
        neutral_body_rotation_internal = m_head_int.copy()
    else:
        neutral_body_rotation_internal = np.eye(3)

    # R_body_internal = R_internal_from_android @ m_phone_world @ R_body_from_phone
    # En neutral: R_body_internal = neutral_body_rotation_internal
    # => R_internal_from_android = neutral_body_rotation_internal @ (m_phone_world @ R_body_from_phone)^T
    R_internal_from_android = neutral_body_rotation_internal @ (
        m_phone_world @ R_body_from_phone
    ).T

    check_rotation_matrix(R_internal_from_android, label="R_internal_from_android")

    return Calibration(
        timestamp=timestamp,
        R_internal_from_quest=S_UNITY_TO_INTERNAL,
        R_internal_from_android=R_internal_from_android,
        R_body_from_phone=R_body_from_phone,
        neutral_body_rotation_internal=neutral_body_rotation_internal,
        acceleration_includes_gravity=acceleration_includes_gravity,
    )


def rotation_matrix_from_euler_xyz(degrees: Tuple[float, float, float]) -> np.ndarray:
    """Utilidad de test: matriz de rotación a partir de ángulos XYZ en grados."""
    rx, ry, rz = np.radians(degrees)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
