"""
Carga de configuración centralizada.

Garantiza que los parámetros de server.py, tests y documentación se lean del
mismo archivo.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    checkpoint_path: str
    strict_checkpoint_loading: bool
    model_name: str
    sparse_dim: int
    input_motion_length: int
    server_loop_rate_hz: int
    expected_model_rate_hz: int
    synchronization_tolerance_ms: float
    max_window_gap_ms: float
    sensor_timeout_s: float
    temporal_window_frames: int
    posture_calib_frames: int
    android_acceleration_type: Optional[str]
    quest_quaternion_order: Optional[str]
    android_quaternion_order: Optional[str]
    internal_coordinate_system: str
    phone_mount_transform: Optional[list]
    required_sensor_mode: str
    quest_port: int
    android_port: int
    records_dir: str
    feature_schema_version: str

    # Model params (dict)
    model_params: dict

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"El archivo de configuración {path} está vacío")
        return cls(
            checkpoint_path=data.get("checkpoint_path", "pretrained_model/pretrained_model_protocol1.pt"),
            strict_checkpoint_loading=data.get("strict_checkpoint_loading", True),
            model_name=data.get("model_name", "HMD_imu_HME_Universe"),
            sparse_dim=int(data.get("sparse_dim", 135)),
            input_motion_length=int(data.get("input_motion_length", 40)),
            server_loop_rate_hz=int(data.get("server_loop_rate_hz", 60)),
            expected_model_rate_hz=int(data.get("expected_model_rate_hz", 60)),
            synchronization_tolerance_ms=float(data.get("synchronization_tolerance_ms", 100.0)),
            max_window_gap_ms=float(data.get("max_window_gap_ms", 100.0)),
            sensor_timeout_s=float(data.get("sensor_timeout_s", 0.5)),
            temporal_window_frames=int(data.get("temporal_window_frames", 40)),
            posture_calib_frames=int(data.get("posture_calib_frames", 30)),
            android_acceleration_type=data.get("android_acceleration_type"),
            quest_quaternion_order=data.get("quest_quaternion_order"),
            android_quaternion_order=data.get("android_quaternion_order", "wxyz"),
            internal_coordinate_system=data.get("internal_coordinate_system", "right_handed_y_up_z_forward_meters"),
            phone_mount_transform=data.get("phone_mount_transform"),
            required_sensor_mode=data.get("required_sensor_mode", "quest_plus_pelvis"),
            quest_port=int(data.get("quest_port", 5006)),
            android_port=int(data.get("android_port", 5005)),
            records_dir=data.get("records_dir", "RECORDS"),
            feature_schema_version=data.get("feature_schema_version", "hmd_poser_135_v1"),
            model_params=data.get("model_params", {}),
        )

    def acceleration_includes_gravity(self) -> Optional[bool]:
        if self.android_acceleration_type is None:
            return None
        if self.android_acceleration_type == "accelerometer":
            return True
        if self.android_acceleration_type == "linear":
            return False
        raise ValueError(f"android_acceleration_type desconocido: {self.android_acceleration_type}")

    def records_dir_abs(self) -> str:
        return os.path.abspath(self.records_dir)

    def to_dict(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "strict_checkpoint_loading": self.strict_checkpoint_loading,
            "model_name": self.model_name,
            "sparse_dim": self.sparse_dim,
            "input_motion_length": self.input_motion_length,
            "server_loop_rate_hz": self.server_loop_rate_hz,
            "expected_model_rate_hz": self.expected_model_rate_hz,
            "synchronization_tolerance_ms": self.synchronization_tolerance_ms,
            "max_window_gap_ms": self.max_window_gap_ms,
            "sensor_timeout_s": self.sensor_timeout_s,
            "temporal_window_frames": self.temporal_window_frames,
            "posture_calib_frames": self.posture_calib_frames,
            "android_acceleration_type": self.android_acceleration_type,
            "quest_quaternion_order": self.quest_quaternion_order,
            "android_quaternion_order": self.android_quaternion_order,
            "internal_coordinate_system": self.internal_coordinate_system,
            "phone_mount_transform": self.phone_mount_transform,
            "required_sensor_mode": self.required_sensor_mode,
            "quest_port": self.quest_port,
            "android_port": self.android_port,
            "records_dir": self.records_dir,
            "feature_schema_version": self.feature_schema_version,
        }


def load_config(path: str = "config.yaml") -> Config:
    return Config.from_yaml(path)
