"""
Sincronización temporal y ventana deslizante de frames para HMD-Poser.

Separación de frecuencias:
    - frecuencia nominal del bucle del servidor;
    - frecuencia de llegada de paquetes Quest;
    - frecuencia de llegada de paquetes Android;
    - frecuencia de pares sincronizados;
    - frecuencia de ventanas válidas entregadas al modelo.

Reglas:
    - Se conservan dos timestamps: device_timestamp (si el paquete lo incluye) y
      server_arrival_timestamp.
    - La sincronización se hace por server_arrival_timestamp porque los relojes de
      Quest y Android no son comparables sin una calibración de offset.
    - El timestamp de dispositivo se conserva solo para diagnóstico.
    - No se mezclan relojes sin corregir el offset.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
import numpy as np


class SyncError(ValueError):
    pass


@dataclass
class SensorSample:
    server_arrival_ts: float
    device_ts: Optional[float]
    data: Any


@dataclass
class SyncStats:
    quest_packets: int = 0
    android_packets: int = 0
    synced_pairs: int = 0
    rejected_pairs: int = 0
    late_packets: int = 0
    duplicated_timestamps: int = 0

    # Frecuencias instantáneas (Hz)
    quest_hz: float = 0.0
    android_hz: float = 0.0
    synced_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quest_packets": self.quest_packets,
            "android_packets": self.android_packets,
            "synced_pairs": self.synced_pairs,
            "rejected_pairs": self.rejected_pairs,
            "late_packets": self.late_packets,
            "duplicated_timestamps": self.duplicated_timestamps,
            "quest_hz": self.quest_hz,
            "android_hz": self.android_hz,
            "synced_hz": self.synced_hz,
        }


class SensorBuffer:
    """Cola temporal para un sensor individual."""

    def __init__(self, name: str, maxsize: int = 120):
        self.name = name
        self.samples: deque = deque(maxlen=maxsize)
        self.last_device_ts: Optional[float] = None
        self._stats_late = 0
        self._stats_duplicated = 0

    def add(self, sample: SensorSample) -> None:
        if sample.device_ts is not None:
            if self.last_device_ts is not None and sample.device_ts <= self.last_device_ts:
                self._stats_duplicated += 1
                return
            self.last_device_ts = sample.device_ts
        self.samples.append(sample)

    def latest(self) -> Optional[SensorSample]:
        return self.samples[-1] if self.samples else None

    def arrival_hz(self, window_s: float = 1.0) -> float:
        if len(self.samples) < 2:
            return 0.0
        now = time.time()
        arrivals = [s.server_arrival_ts for s in self.samples if now - s.server_arrival_ts <= window_s]
        if len(arrivals) < 2:
            return 0.0
        span = max(arrivals) - min(arrivals)
        return (len(arrivals) - 1) / span if span > 0 else 0.0

    def clear(self):
        self.samples.clear()
        self.last_device_ts = None


class Synchronizer:
    """
    Empareja muestras de Quest y Android por server_arrival_timestamp.

    La tolerancia declarada en documentación y la utilizada por el código deben ser
    idénticas; se carga desde la configuración.
    """

    def __init__(self, tolerance_ms: float = 100.0):
        self.tolerance_ms = tolerance_ms
        self.tolerance_s = tolerance_ms / 1000.0
        self.quest = SensorBuffer("Quest")
        self.android = SensorBuffer("Android")
        self.stats = SyncStats()
        self._last_pair_ts: Optional[float] = None

    def add_quest(self, data: Any, server_arrival_ts: float, device_ts: Optional[float] = None):
        self.quest.add(SensorSample(server_arrival_ts, device_ts, data))
        self.stats.quest_packets += 1

    def add_android(self, data: Any, server_arrival_ts: float, device_ts: Optional[float] = None):
        self.android.add(SensorSample(server_arrival_ts, device_ts, data))
        self.stats.android_packets += 1

    def get_synced_pair(self) -> Optional[Tuple[Any, Any, float, float]]:
        """
        Devuelve (quest_data, android_data, quest_server_ts, android_server_ts) si
        las últimas muestras de cada buffer están dentro de la tolerancia.
        """
        q = self.quest.latest()
        a = self.android.latest()
        if q is None or a is None:
            return None

        offset_ms = abs(q.server_arrival_ts - a.server_arrival_ts) * 1000.0
        if offset_ms > self.tolerance_ms:
            self.stats.rejected_pairs += 1
            return None

        self.stats.synced_pairs += 1
        self._last_pair_ts = max(q.server_arrival_ts, a.server_arrival_ts)
        return q.data, a.data, q.server_arrival_ts, a.server_arrival_ts

    def update_hz(self):
        self.stats.quest_hz = self.quest.arrival_hz()
        self.stats.android_hz = self.android.arrival_hz()

    def reset(self):
        self.quest.clear()
        self.android.clear()
        self.stats = SyncStats()
        self._last_pair_ts = None


class RollingWindow:
    """
    Ventana deslizante real de frames. No repite el último frame para rellenar.
    """

    def __init__(self, size: int = 40, max_gap_ms: float = 100.0,
                 diagnostic_repeated_frame_mode: bool = False):
        self.size = size
        self.max_gap_ms = max_gap_ms
        self.frames: deque = deque(maxlen=size)
        self.timestamps: deque = deque(maxlen=size)
        self.diagnostic_repeated_frame_mode = diagnostic_repeated_frame_mode
        self.degraded_mode = False

    def add(self, frame: np.ndarray, ts: float) -> bool:
        """
        Añade un frame a la ventana. Rechaza timestamps que no sean crecientes o
        saltos mayores que max_gap_ms.
        """
        if self.timestamps and ts <= self.timestamps[-1]:
            # Timestamp fuera de orden o duplicado
            return False
        if self.timestamps:
            gap_ms = (ts - self.timestamps[-1]) * 1000.0
            if gap_ms > self.max_gap_ms:
                # Resetear la ventana para evitar mezclar frames no consecutivos
                self.frames.clear()
                self.timestamps.clear()
                self.degraded_mode = True

        self.frames.append(frame)
        self.timestamps.append(ts)
        if self.is_full():
            self.degraded_mode = False
        return True

    def is_full(self) -> bool:
        return len(self.frames) >= self.size

    def to_numpy(self) -> Optional[np.ndarray]:
        if not self.is_full():
            if self.diagnostic_repeated_frame_mode and len(self.frames) > 0:
                # Modo diagnóstico explícito: repetir el último frame para completar.
                # Los resultados se marcan como degradados.
                self.degraded_mode = True
                last = np.asarray(self.frames[-1])
                out = np.repeat(last[np.newaxis, :], self.size, axis=0)
                return out
            return None
        return np.stack(list(self.frames), axis=0)

    def effective_hz(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        n = len(self.timestamps) - 1
        return n / span if span > 0 else 0.0

    def reset(self):
        self.frames.clear()
        self.timestamps.clear()
        self.degraded_mode = False

    def get_state(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "filled": len(self.frames),
            "effective_hz": self.effective_hz(),
            "degraded_mode": self.degraded_mode,
            "diagnostic_repeated_frame_mode": self.diagnostic_repeated_frame_mode,
            "first_ts": self.timestamps[0] if self.timestamps else None,
            "last_ts": self.timestamps[-1] if self.timestamps else None,
        }
