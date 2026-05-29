"""
Módulo para construir el tensor de entrada para el modelo HMD-Poser.

Se encarga de la conversión matemática de cuaterniones y del ensamblaje
del tensor en el formato exacto que espera el modelo.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class HMDPoserFeatureBuilder:
    def __init__(self):
        """
        Prepara el ensamblador del vector de características para HMD-Poser.
        """
        # El orden de los sensores según el paper/código para "HMD + 3 IMUs" es:
        # Head, Left Hand, Right Hand, Left Foot, Right Foot, Pelvis
        # En nuestro caso, Left/Right Foot son ceros, y Pelvis viene de Android.
        # El paper también menciona que para HMD+3IMUs, usan:
        # HMD (cabeza), Mandos (manos) y un IMU en la pelvis.
        # Los otros dos IMUs (pies) se simulan con ceros.
        pass

    def _quaternion_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """
        Convierte un cuaternión (X, Y, Z, W) a una matriz de rotación 3x3.
        
        Args:
            quat (np.ndarray): Cuaternión en formato [x, y, z, w].
        
        Returns:
            np.ndarray: Matriz de rotación de 3x3.
        """
        if quat.shape != (4,):
            raise ValueError(f"El cuaternión debe tener shape (4,), pero tiene {quat.shape}")
        # Scipy espera cuaterniones en formato (x, y, z, w)
        return R.from_quat(quat).as_matrix()

    def build_tensor(self, quest_data: dict, android_data: dict) -> torch.Tensor:
        """
        Construye el tensor de entrada para HMD-Poser a partir de los datos
        de los sensores sincronizados.

        Args:
            quest_data (dict): Datos del Meta Quest 3S.
                               Formato: {'hmd': {'pos', 'rot'}, 
                                         'left_hand': {'pos', 'rot'},
                                         'right_hand': {'pos', 'rot'}}
            android_data (dict): Datos del smartphone Android.
                                 Formato: {'pelvis': {'rot', 'accel'}}

        Returns:
            torch.Tensor: Tensor de entrada para el modelo HMD-Poser.
                          Shape: (1, N_SENSORS, FEATURE_DIM)
        """
        # --- 1. Conversión de Cuaterniones a Matrices de Rotación ---
        hmd_rot_mat = self._quaternion_to_rotation_matrix(quest_data['hmd']['rot'])
        left_hand_rot_mat = self._quaternion_to_rotation_matrix(quest_data['left_hand']['rot'])
        right_hand_rot_mat = self._quaternion_to_rotation_matrix(quest_data['right_hand']['rot'])
        pelvis_rot_mat = self._quaternion_to_rotation_matrix(android_data['pelvis']['rot'])

        # --- 2. Preparación de los datos de cada sensor ---
        # El formato de características por sensor es [Pos(3), Rot(3x3), Acel(3)]
        
        # HMD (Cabeza)
        hmd_pos = quest_data['hmd']['pos']
        hmd_accel = np.zeros(3) # HMD no provee aceleración lineal en este setup
        hmd_feature = np.concatenate([hmd_pos, hmd_rot_mat.flatten(), hmd_accel])

        # Mano Izquierda
        left_hand_pos = quest_data['left_hand']['pos']
        left_hand_accel = np.zeros(3) # Mandos no proveen aceleración lineal
        left_hand_feature = np.concatenate([left_hand_pos, left_hand_rot_mat.flatten(), left_hand_accel])

        # Mano Derecha
        right_hand_pos = quest_data['right_hand']['pos']
        right_hand_accel = np.zeros(3)
        right_hand_feature = np.concatenate([right_hand_pos, right_hand_rot_mat.flatten(), right_hand_accel])

        # Pelvis (desde Android)
        pelvis_pos = np.zeros(3) # Android no provee posición
        pelvis_accel = android_data['pelvis']['accel']
        pelvis_feature = np.concatenate([pelvis_pos, pelvis_rot_mat.flatten(), pelvis_accel])

        # --- 3. Zero-Padding para los sensores faltantes (rodillas/pies) ---
        # El modelo "HMD + 3 IMUs" espera 5 sensores en total.
        # HMD, LHand, RHand, LPelvis, LFoot, RFoot.
        # En el código fuente, el orden es:
        # head, left_hand, right_hand, left_foot, right_foot
        # Y la pelvis se trata de forma especial o se asume fija.
        # Re-analizando el paper, la configuración "HMD+3IMUs" es:
        # HMD, LHand, RHand, Pelvis. Los otros dos (pies) son ceros.
        # El tensor de entrada del modelo parece esperar 6 sensores:
        # [HMD, LHand, RHand, Pelvis, LFoot, RFoot]
        
        # La dimensión de características es 3 (pos) + 9 (rot_mat) + 3 (acel) = 15
        feature_dim = 15
        zero_feature = np.zeros(feature_dim)

        # --- 4. Ensamblaje final del tensor ---
        # El orden es crucial y debe coincidir con el esperado por el modelo.
        # Basado en el código y paper, el orden lógico es:
        # HMD, Mano Izq, Mano Der, Pelvis, Pie Izq (cero), Pie Der (cero)
        # Sin embargo, el dataloader.py podría tener un orden específico.
        # Asumimos el orden más lógico: cabeza, manos, pelvis, pies.
        
        # Corrección: El paper dice "HMD+3IMUs" para HMD, L/R Hand, L/R Foot.
        # Y "HMD+Scalable IMUs" para la versión con pelvis.
        # Para "HMD+3IMUs", el input es [HMD_pos, HMD_rot, LH_rot, RH_rot, LF_rot, RF_rot]
        # Para nuestro caso "HMD + 2 Mandos + 1 Smartphone", el input es:
        # [HMD_pos, HMD_rot, LH_pos, LH_rot, RH_pos, RH_rot, Pelvis_rot, Pelvis_accel]
        # Y los pies son ceros.
        
        # El tensor final debe tener un shape (N, S, F) donde N=1 (batch), S=6 (sensores), F=15 (features)
        
        # Sensor 0: HMD
        # Sensor 1: Left Hand
        # Sensor 2: Right Hand
        # Sensor 3: Pelvis
        # Sensor 4: Left Foot (zero-padded)
        # Sensor 5: Right Foot (zero-padded)

        stacked_features = np.stack([
            hmd_feature,
            left_hand_feature,
            right_hand_feature,
            pelvis_feature,
            zero_feature,  # Left Foot
            zero_feature   # Right Foot
        ], axis=0)

        # Convertir a tensor de PyTorch y añadir la dimensión de batch
        input_tensor = torch.from_numpy(stacked_features).float().unsqueeze(0)
        
        # Shape final: (1, 6, 15)
        return input_tensor