"""
Constantes del layout del vector de 135 características de HMD-Poser.

El orden y las dimensiones se derivan directamente del preprocesamiento oficial
`prepare_data.py` del repositorio HMD-Poser:
https://github.com/Pico-AI-Team/HMD-Poser/blob/main/prepare_data.py

No se deben usar números mágicos en el resto del código; usar estos nombres.
"""

# Número total de características y longitud temporal esperada
FEATURE_DIM = 135
SEQUENCE_LENGTH = 40

# Orden de los seis segmentos en los bloques de rotación global y delta
SEGMENT_NAMES = [
    "head",          # índice 0
    "left_hand",     # índice 1
    "right_hand",    # índice 2
    "left_foot",     # índice 3
    "right_foot",    # índice 4
    "pelvis",        # índice 5
]
N_SEGMENTS = len(SEGMENT_NAMES)

# Rotaciones globales 6D para los seis segmentos (36 valores)
GLOBAL_ROT_SLICE = slice(0, 36)

# Cambios de rotación entre frames para los seis segmentos (36 valores)
DELTA_ROT_SLICE = slice(36, 72)

# Posiciones globales XYZ: head, left_hand, right_hand (9 valores)
GLOBAL_POS_SLICE = slice(72, 81)

# Cambios de posición XYZ: head, left_hand, right_hand (9 valores)
DELTA_POS_SLICE = slice(81, 90)

# Rotaciones 6D de las manos relativas a la cabeza (12 valores)
#  - left_hand relative to head
#  - right_hand relative to head
RELATIVE_ROT_SLICE = slice(90, 102)

# Cambios temporales de las rotaciones relativas cabeza-manos (12 valores)
DELTA_RELATIVE_ROT_SLICE = slice(102, 114)

# Posiciones XYZ de las manos relativas a la cabeza (6 valores)
RELATIVE_POS_SLICE = slice(114, 120)

# Cambios temporales de las posiciones relativas cabeza-manos (6 valores)
DELTA_RELATIVE_POS_SLICE = slice(120, 126)

# Aceleraciones XYZ: left_foot, right_foot, pelvis (9 valores)
ACCELERATION_SLICE = slice(126, 135)

# Verificación de la suma de dimensiones
assert (
    (GLOBAL_ROT_SLICE.stop - GLOBAL_ROT_SLICE.start)
    + (DELTA_ROT_SLICE.stop - DELTA_ROT_SLICE.start)
    + (GLOBAL_POS_SLICE.stop - GLOBAL_POS_SLICE.start)
    + (DELTA_POS_SLICE.stop - DELTA_POS_SLICE.start)
    + (RELATIVE_ROT_SLICE.stop - RELATIVE_ROT_SLICE.start)
    + (DELTA_RELATIVE_ROT_SLICE.stop - DELTA_RELATIVE_ROT_SLICE.start)
    + (RELATIVE_POS_SLICE.stop - RELATIVE_POS_SLICE.start)
    + (DELTA_RELATIVE_POS_SLICE.stop - DELTA_RELATIVE_POS_SLICE.start)
    + (ACCELERATION_SLICE.stop - ACCELERATION_SLICE.start)
    == FEATURE_DIM
), "La suma de las slices no coincide con FEATURE_DIM"

# Sub-slices dentro de cada bloque de 6D (6 valores por segmento)
SEGMENT_6D_SLICES = {
    name: slice(GLOBAL_ROT_SLICE.start + i * 6, GLOBAL_ROT_SLICE.start + (i + 1) * 6)
    for i, name in enumerate(SEGMENT_NAMES)
}

SEGMENT_DELTA_6D_SLICES = {
    name: slice(DELTA_ROT_SLICE.start + i * 6, DELTA_ROT_SLICE.start + (i + 1) * 6)
    for i, name in enumerate(SEGMENT_NAMES)
}

# Sub-slices de posiciones globales (3 valores por segmento)
# Solo los tres primeros segmentos tienen posición global
GLOBAL_POS_SEGMENT_SLICES = {
    "head": slice(72, 75),
    "left_hand": slice(75, 78),
    "right_hand": slice(78, 81),
}

DELTA_POS_SEGMENT_SLICES = {
    "head": slice(81, 84),
    "left_hand": slice(84, 87),
    "right_hand": slice(87, 90),
}

# Sub-slices de rotaciones relativas a la cabeza
LEFT_HAND_RELATIVE_ROT_SLICE = slice(90, 96)
RIGHT_HAND_RELATIVE_ROT_SLICE = slice(96, 102)

LEFT_HAND_RELATIVE_DELTA_ROT_SLICE = slice(102, 108)
RIGHT_HAND_RELATIVE_DELTA_ROT_SLICE = slice(108, 114)

# Sub-slices de posiciones relativas a la cabeza
LEFT_HAND_RELATIVE_POS_SLICE = slice(114, 117)
RIGHT_HAND_RELATIVE_POS_SLICE = slice(117, 120)

LEFT_HAND_RELATIVE_DELTA_POS_SLICE = slice(120, 123)
RIGHT_HAND_RELATIVE_DELTA_POS_SLICE = slice(123, 126)

# Sub-slices de aceleración (3 valores por segmento)
LEFT_FOOT_ACCEL_SLICE = slice(126, 129)
RIGHT_FOOT_ACCEL_SLICE = slice(129, 132)
PELVIS_ACCEL_SLICE = slice(132, 135)

# Máscaras de sensores ausentes según las configuraciones oficiales de HMD-Poser
# (extraídas de dataset/dataloader.py)
#
# HMD_3IMUs: no se pone nada a cero (todos los canales presentes)
# HMD_2IMUs: se ocultan los canales de la pelvis (índices 30-35, 66-71, 132-134)
# HMD:      se ocultan pies (rotación + delta) y todas las aceleraciones

HMD_2IMU_MASK_INDICES = (
    list(range(30, 36))      # pelvis global rotation
    + list(range(66, 72))    # pelvis delta rotation
    + list(range(132, 135))  # pelvis acceleration
)

HMD_MASK_INDICES = (
    list(range(18, 36))      # left_foot + right_foot global rotation
    + list(range(54, 72))    # left_foot + right_foot delta rotation
    + list(range(126, 135)) # todas las aceleraciones
)

# Configuración hardware de Quest Align: HMD + manos + smartphone en pelvis
# Los pies (left_foot, right_foot) nunca están presentes.
QUEST_ALIGN_PELVIS_ONLY_MISSING_INDICES = (
    list(range(18, 24))      # left_foot global rotation
    + list(range(24, 30))    # right_foot global rotation
    + list(range(54, 60))    # left_foot delta rotation
    + list(range(60, 66))    # right_foot delta rotation
    + list(range(126, 129))  # left_foot acceleration
    + list(range(129, 132))  # right_foot acceleration
)

# Si se pierde el teléfono, también se ocultan los canales de la pelvis
QUEST_ALIGN_QUEST_ONLY_MISSING_INDICES = (
    QUEST_ALIGN_PELVIS_ONLY_MISSING_INDICES
    + list(range(30, 36))      # pelvis global rotation
    + list(range(66, 72))      # pelvis delta rotation
    + list(range(132, 135))    # pelvis acceleration
)

# Feature schema version
FEATURE_SCHEMA_VERSION = "hmd_poser_135_v1"
