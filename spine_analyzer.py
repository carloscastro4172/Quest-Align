"""
Módulo para el análisis biomecánico de la columna vertebral a partir
de las coordenadas de las articulaciones SMPL.
"""
import numpy as np

class SpineAnalyzer:
    def __init__(self):
        """
        Inicializa el analizador de la columna vertebral.
        """
        # Nombres de las articulaciones clave de la columna en el modelo SMPL
        self.spine_joints = ['pelvis', 'spine1', 'spine2', 'spine3', 'neck']

    def _calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula el ángulo en grados entre dos vectores.
        """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def calculate_spine_angles(self, smpl_joints: dict) -> dict:
        """
        Calcula los ángulos de desviación de la columna en los planos frontal y sagital.

        Args:
            smpl_joints (dict): Un diccionario con las coordenadas 3D de las
                                articulaciones SMPL.
                                Ej: {'pelvis': [x,y,z], 'spine1': [x,y,z], ...}

        Returns:
            dict: Un diccionario con los ángulos calculados en grados.
                  {'frontal_plane_deg': float, 'sagittal_plane_deg': float}
        """
        # --- Extracción de coordenadas de las articulaciones ---
        try:
            pelvis = smpl_joints['pelvis']
            spine1 = smpl_joints['spine1']
            spine2 = smpl_joints['spine2']
            spine3 = smpl_joints['spine3']
            neck = smpl_joints['neck']
        except KeyError as e:
            raise ValueError(f"La articulación {e} no se encontró en la salida del modelo.")

        # --- Cálculo de Vectores de la Columna ---
        # Vector desde la pelvis hasta el cuello, que representa la dirección general de la columna
        overall_spine_vector = neck - pelvis

        # Vector de referencia "ideal" (perfectamente vertical)
        # En el sistema de coordenadas de SMPL, el eje Y es generalmente el vertical.
        vertical_ref_vector = np.array([0, 1, 0])

        # --- Cálculo de Desviación en el Plano Frontal (Escoliosis) ---
        # Proyectamos el vector de la columna en el plano XY (frontal, visto desde adelante)
        # y medimos su desviación con respecto a la vertical.
        spine_vector_frontal_proj = np.array([overall_spine_vector[0], overall_spine_vector[1], 0])
        
        # El ángulo entre el vector proyectado y la vertical en este plano
        # nos da una medida de la inclinación lateral.
        frontal_angle_deg = self._calculate_angle_between_vectors(spine_vector_frontal_proj, vertical_ref_vector)
        
        # Determinar la dirección de la inclinación (izquierda/derecha)
        if spine_vector_frontal_proj[0] < 0:
            frontal_angle_deg *= -1 # Inclinación hacia la izquierda (negativo)

        # --- Cálculo de Desviación en el Plano Sagital (Cifosis/Lordosis) ---
        # Proyectamos el vector de la columna en el plano YZ (lateral)
        # y medimos su desviación.
        spine_vector_sagittal_proj = np.array([0, overall_spine_vector[1], overall_spine_vector[2]])
        
        sagittal_angle_deg = self._calculate_angle_between_vectors(spine_vector_sagittal_proj, vertical_ref_vector)

        # Determinar la dirección (hacia adelante/atrás)
        # Si la componente Z es positiva, es una inclinación hacia adelante (cifosis)
        if spine_vector_sagittal_proj[2] > 0:
             sagittal_angle_deg *= -1 # Inclinación hacia adelante (negativo)

        return {
            'frontal_plane_deg': frontal_angle_deg,
            'sagittal_plane_deg': sagittal_angle_deg
        }
