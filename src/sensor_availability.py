"""
Disponibilidad de sensores y máscaras semánticas para el vector de 135 características.

Las máscaras de HMD-Poser (HMD, HMD_2IMUs, HMD_3IMUs) se extraen de
`dataset/dataloader.py` del repositorio oficial.

Quest Align usa una configuración de hardware distinta:
    - HMD (cabeza) + manos: siempre presentes.
    - Pelvis: orientación y aceleración desde el smartphone Android.
    - Pies (left_foot, right_foot): siempre ausentes.
"""

from dataclasses import dataclass
from typing import List

from src.feature_constants import (
    QUEST_ALIGN_PELVIS_ONLY_MISSING_INDICES,
    QUEST_ALIGN_QUEST_ONLY_MISSING_INDICES,
    HMD_2IMU_MASK_INDICES,
    HMD_MASK_INDICES,
)


@dataclass
class SensorAvailability:
    head: bool = True
    left_hand: bool = True
    right_hand: bool = True
    left_foot: bool = False
    right_foot: bool = False
    pelvis: bool = True

    def missing_sensors(self) -> List[str]:
        return [name for name, present in self.__dict__.items() if not present]

    def as_hmd_poser_mode(self) -> str:
        """
        Devuelve el modo HMD-Poser más cercano a la disponibilidad actual.
        Quest Align nunca es exactamente HMD_3IMUs porque le faltan los pies.
        """
        if self.pelvis:
            return "quest_plus_pelvis"
        return "quest_only"

    def mask_indices(self) -> List[int]:
        """Índices del vector de 135 que deben ponerse a cero por sensores ausentes."""
        if self.pelvis:
            return list(QUEST_ALIGN_PELVIS_ONLY_MISSING_INDICES)
        else:
            return list(QUEST_ALIGN_QUEST_ONLY_MISSING_INDICES)

    def degraded(self) -> bool:
        return not (self.left_foot and self.right_foot and self.pelvis)


def quest_plus_pelvis_availability() -> SensorAvailability:
    return SensorAvailability(
        head=True, left_hand=True, right_hand=True,
        left_foot=False, right_foot=False, pelvis=True
    )


def quest_only_availability() -> SensorAvailability:
    return SensorAvailability(
        head=True, left_hand=True, right_hand=True,
        left_foot=False, right_foot=False, pelvis=False
    )


def hmd_poser_2imu_availability() -> SensorAvailability:
    """HMD + 2 IMUs en los pies. No coincide con el hardware de Quest Align."""
    return SensorAvailability(
        head=True, left_hand=True, right_hand=True,
        left_foot=True, right_foot=True, pelvis=False
    )


def hmd_poser_3imu_availability() -> SensorAvailability:
    """HMD + 3 IMUs: pies + pelvis. No coincide con el hardware de Quest Align."""
    return SensorAvailability(
        head=True, left_hand=True, right_hand=True,
        left_foot=True, right_foot=True, pelvis=True
    )


def apply_mask(feature_vector: List[float], availability: SensorAvailability) -> List[float]:
    """Pone a cero los canales semánticos que corresponden a sensores ausentes."""
    import numpy as np
    v = np.asarray(feature_vector, dtype=np.float64).copy()
    indices = availability.mask_indices()
    v[indices] = 0.0
    return v
