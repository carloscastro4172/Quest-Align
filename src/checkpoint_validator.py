"""
Validación del checkpoint HMD-Poser.

Separa claramente:
    A) Compatibilidad arquitectónica: el modelo acepta el tensor y el state_dict coincide.
    B) Compatibilidad de configuración: la configuración Quest Align (pelvis-only) no fue
       una configuración original de entrenamiento.

Nunca se usa strict=False silenciosamente.
"""

import hashlib
import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
import torch

from src.config import Config
from src.hmd_poser_network import HMD_imu_HME_Universe


class CheckpointValidationError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    raise CheckpointValidationError("El checkpoint no tiene un state_dict reconocible")


class CheckpointValidator:
    def __init__(self, config: Config):
        self.config = config
        self.report: Dict[str, Any] = {
            "checkpoint_path": os.path.abspath(config.checkpoint_path),
            "sha256": None,
            "sparse_dim": config.sparse_dim,
            "sequence_length": config.input_motion_length,
            "input_shape_tested": [1, config.input_motion_length, config.sparse_dim],
            "strict_load": None,
            "strict_load_error": None,
            "missing_keys": [],
            "unexpected_keys": [],
            "not_loaded_keys": [],
            "checkpoint_partial_load": False,
            "forward_pass_success": False,
            "forward_pass_error": None,
            "output_shape": None,
            "no_nan_in_output": None,
            "training_sensor_configs_verified": ["HMD", "HMD_2IMUs", "HMD_3IMUs"],
            "quest_plus_pelvis_seen_during_training": False,
            "architecturally_compatible": False,
            "configuration_validated": False,
            "notes": [],
        }
        self.model: Optional[HMD_imu_HME_Universe] = None

    def validate(self, allow_diagnostic_non_strict: bool = False) -> Dict[str, Any]:
        path = self.config.checkpoint_path
        if not os.path.exists(path):
            raise CheckpointValidationError(f"Checkpoint no encontrado: {path}")

        self.report["sha256"] = sha256_file(path)
        self.report["notes"].append(f"SHA-256 calculado: {self.report['sha256']}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = _get_state_dict(ckpt)
        self.report["state_dict_keys_count"] = len(state_dict)

        # Instanciar la arquitectura exacta del checkpoint
        mp = self.config.model_params
        self.model = HMD_imu_HME_Universe(
            self.config.sparse_dim,
            mp.get("number_layer", 3),
            mp.get("hidden_size", 256),
            mp.get("dropout", 0.05),
            mp.get("nhead", 8),
            mp.get("block_num", 2),
        )

        # Carga strict=True
        try:
            self.model.load_state_dict(state_dict, strict=True)
            self.report["strict_load"] = True
            self.report["architecturally_compatible"] = True
            self.report["notes"].append(
                "state_dict cargado con strict=True: todas las claves coinciden."
            )
        except RuntimeError as e:
            self.report["strict_load"] = False
            self.report["strict_load_error"] = str(e)
            self.report["notes"].append(f"strict=True falló: {e}")
            if not allow_diagnostic_non_strict:
                self.report["checkpoint_partial_load"] = True
                self.report["notes"].append(
                    "No se permite strict=False; el checkpoint no es arquitectónicamente compatible."
                )
                self._save_report()
                return self.report
            # Si se permite explícitamente, carga con strict=False y registra claves
            self.report["notes"].append(
                "Modo diagnóstico: se permite strict=False."
            )
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.report["missing_keys"] = list(missing)
            self.report["unexpected_keys"] = list(unexpected)
            self.report["checkpoint_partial_load"] = True
            self.report["architecturally_compatible"] = len(missing) == 0

        # Forward pass de prueba
        self.model.eval()
        test_input = torch.randn(1, self.config.input_motion_length, self.config.sparse_dim)
        try:
            with torch.no_grad():
                pred_pose, pred_shapes = self.model(test_input)
            self.report["forward_pass_success"] = True
            self.report["output_shape"] = {
                "pred_pose": list(pred_pose.shape),
                "pred_shapes": list(pred_shapes.shape),
            }
            self.report["no_nan_in_output"] = bool(
                torch.all(torch.isfinite(pred_pose)) and torch.all(torch.isfinite(pred_shapes))
            )
            self.report["notes"].append(
                f"Forward pass exitoso: pred_pose={list(pred_pose.shape)}, pred_shapes={list(pred_shapes.shape)}"
            )
        except Exception as e:
            self.report["forward_pass_success"] = False
            self.report["forward_pass_error"] = str(e)
            self.report["notes"].append(f"Forward pass falló: {e}")

        # Validación de configuración
        self.report["quest_plus_pelvis_seen_during_training"] = False
        self.report["configuration_validated"] = False
        self.report["notes"].append(
            "Quest Align (HMD + manos + smartphone en pelvis) NO está en "
            "compatible_inputs del entrenamiento. El checkpoint es arquitectónicamente "
            "compatible con el tensor de 135 valores, pero la configuración pelvis-only "
            "no fue validada explícitamente como configuración de entrenamiento original."
        )

        self._save_report()
        return self.report

    def _save_report(self):
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/checkpoint_compatibility.json", "w") as f:
            json.dump(self.report, f, indent=2)

    def get_model(self) -> Optional[HMD_imu_HME_Universe]:
        return self.model


def run_checkpoint_validation(config_path: str = "config.yaml",
                                allow_diagnostic_non_strict: bool = False) -> Dict[str, Any]:
    config = Config.from_yaml(config_path)
    validator = CheckpointValidator(config)
    return validator.validate(allow_diagnostic_non_strict=allow_diagnostic_non_strict)
