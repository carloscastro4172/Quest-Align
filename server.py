# server.py
# Versión refactorizada de Quest Align.
# Ver docs/CURRENT_PIPELINE_AUDIT.md para el listado de problemas corregidos.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import threading
import json
import os
import socket as sock_module
from typing import Optional
import numpy as np
import torch
from collections import deque
from datetime import datetime
from enum import Enum, auto

from udp_listener import UDPListener

from src.config import load_config
from src.coordinate_frames import Calibration, build_calibration_from_neutral
from src.sensor_availability import (
    SensorAvailability,
    quest_plus_pelvis_availability,
    quest_only_availability,
)
from src.feature_builder import HMDPoserFeatureBuilder
from src.temporal_buffer import Synchronizer, RollingWindow
from src.checkpoint_validator import CheckpointValidator
from src.posture_extraction import extract_last_frame_pose_6d, extract_spine_angles


class SyncState(Enum):
    WAITING_SENSORS = auto()
    WARMING_UP = auto()
    CALIBRATING = auto()
    RECORDING = auto()
    QUEST_ONLY = auto()


class BiomechanicsServer:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networking
        self.quest_port = self.cfg.quest_port
        self.android_port = self.cfg.android_port
        self._last_android_addr = None

        # Sensor data
        self.latest_quest_data = None
        self.latest_android_data = None
        self.quest_lock = threading.Lock()
        self.android_lock = threading.Lock()
        self.last_quest_ts = 0.0
        self.last_android_ts = 0.0

        # Synchronization
        self.sync = Synchronizer(tolerance_ms=self.cfg.synchronization_tolerance_ms)

        # Calibration
        self.calibration: Calibration = None
        self.sensor_calib_done = False
        self.posture_calib_done = False
        self._calib_pitches = []
        self._calib_rolls = []
        self._pitch_offset = 0.0
        self._roll_offset = 0.0

        # Window
        self.rolling_window = RollingWindow(
            size=self.cfg.temporal_window_frames,
            max_gap_ms=self.cfg.max_window_gap_ms,
            diagnostic_repeated_frame_mode=False,
        )

        # Feature builder
        self._availability = quest_plus_pelvis_availability()
        self.feature_builder: HMDPoserFeatureBuilder = None

        # Model
        self.model = None
        self._load_model()

        # Session
        self.session_active = False
        self.session_number = 0
        self.session_frames = []
        self.inference_results = []
        self.session_start_ts = 0.0
        self.session_lock = threading.Lock()

        self.sync_state = SyncState.WAITING_SENSORS
        self.warmup_start_ts = 0.0
        self.warmup_duration = 3.0

        self.running = False
        self.quest_listener = UDPListener("Quest", self.quest_port, self._quest_handler)
        self.android_listener = UDPListener("Android", self.android_port, self._android_handler)

        # Stats
        self._quest_count = 0
        self._android_count = 0
        self._stats_ts = time.time()

        # Rate limiting for prints
        self._last_sync_warn = 0.0
        self._last_missing_warn = 0.0

        print(f"Dispositivo activo: {self.device}")
        print(f"Configuración: {config_path}")
        print(f"Modo requerido: {self.cfg.required_sensor_mode}")
        print(f"Feature schema: {self.cfg.feature_schema_version}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            validator = CheckpointValidator(self.cfg)
            report = validator.validate(allow_diagnostic_non_strict=not self.cfg.strict_checkpoint_loading)
            print(f"[CHECKPOINT] strict_load={report['strict_load']}, "
                  f"arquitectonicamente_compatible={report['architecturally_compatible']}, "
                  f"forward_pass_success={report['forward_pass_success']}")
            if not report['strict_load'] and self.cfg.strict_checkpoint_loading:
                print("[CHECKPOINT] ERROR: strict=True falló y la configuración no permite carga parcial.")
                self.model = None
                return
            self.model = validator.get_model()
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                print("Modelo cargado correctamente.")
        except Exception as e:
            print(f"[CHECKPOINT] Error al cargar el modelo: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # UDP handlers
    # ------------------------------------------------------------------
    def _quest_handler(self, data: bytes, addr):
        ts = time.time()
        with self.quest_lock:
            self.latest_quest_data = data
            self.last_quest_ts = ts
            self._quest_count += 1
        parsed = self._parse_quest_raw(data)
        if parsed:
            self.sync.add_quest(parsed, ts, device_ts=parsed.get('_device_ts'))

    def _android_handler(self, data: bytes, addr):
        self._last_android_addr = addr
        arrival_ts = time.time()
        with self.android_lock:
            self.latest_android_data = data
            self.last_android_ts = arrival_ts
            self._android_count += 1
        parsed = self._parse_android_raw(data)
        if parsed:
            self.sync.add_android(parsed, arrival_ts, device_ts=parsed.get('_device_ts'))

    def _send_sync_response(self, req: dict):
        if self._last_android_addr is None:
            return
        try:
            t1 = int(time.time() * 1e9)
            t2 = int(time.time() * 1e9)
            resp = {
                'type': 'sync_resp',
                'id': req.get('id', ''),
                't0_ns': req.get('t0_ns', 0),
                't1_ns': t1,
                't2_ns': t2,
            }
            msg = json.dumps(resp).encode('utf-8')
            sock = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM)
            sock.sendto(msg, self._last_android_addr)
            sock.close()
        except Exception as e:
            print(f"[Sync] Error respondiendo sync: {e}")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _extract_device_ts(self, j: dict) -> Optional[float]:
        for key in ('ts', 'timestamp', 'device_ts', 't', 'time'):
            if key in j:
                val = j[key]
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def _parse_quest_raw(self, data: bytes):
        try:
            j = json.loads(data.decode('utf-8'))
            device_ts = self._extract_device_ts(j)
            out = {
                '_device_ts': device_ts,
                '_server_arrival_ts': time.time(),
                'head': {
                    'pos': np.array(j['hmd_p'], dtype=np.float32),
                    'rot': np.array(j['hmd_q'], dtype=np.float32),
                },
                'left_hand': {
                    'pos': np.array(j['lhand_p'], dtype=np.float32),
                    'rot': np.array(j['lhand_q'], dtype=np.float32),
                },
                'right_hand': {
                    'pos': np.array(j['rhand_p'], dtype=np.float32),
                    'rot': np.array(j['rhand_q'], dtype=np.float32),
                },
            }
            # Note: quaternion order is NOT verified against the emitter.
            return out
        except Exception as e:
            print(f"[Parse Quest] Error: {e}")
            return None

    def _parse_android_raw(self, data: bytes):
        try:
            j = json.loads(data.decode('utf-8'))
            tipo = j.get('type', '')
            if tipo in ('hello', 'cmd'):
                return None
            if tipo == 'sync_req':
                self._send_sync_response(j)
                return None
            if tipo != 'imu':
                return None

            device_ts = self._extract_device_ts(j)
            quat = j['quat']
            qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            rot = np.array([qw, qx, qy, qz], dtype=np.float32)
            acc = np.array([float(j['acc'][0]), float(j['acc'][1]), float(j['acc'][2])], dtype=np.float32)

            return {
                '_device_ts': device_ts,
                '_server_arrival_ts': time.time(),
                'pelvis': {
                    'rot': rot,
                    'accel': acc,
                },
            }
        except Exception as e:
            print(f"[Parse Android] Error: {e}")
            return None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _update_sync_state(self):
        now = time.time()
        quest_alive = (now - self.last_quest_ts) <= self.cfg.sensor_timeout_s
        android_alive = (now - self.last_android_ts) <= self.cfg.sensor_timeout_s

        if self.sync_state == SyncState.WAITING_SENSORS:
            if quest_alive and android_alive:
                self.warmup_start_ts = now
                self.sync_state = SyncState.WARMING_UP
                print("\n[SYNC] Ambos sensores detectados.")
                print(f"[SYNC] Calentando {self.warmup_duration:.0f}s...")
            elif quest_alive:
                self.sync_state = SyncState.QUEST_ONLY
                self._availability = quest_only_availability()
                self._rebuild_feature_builder()
                print("[SYNC] Quest detectado. Iniciando sin teléfono (modo degradado).")
                if not self.session_active:
                    self._start_auto_session()

        elif self.sync_state == SyncState.WARMING_UP:
            if not quest_alive:
                self.sync_state = SyncState.WAITING_SENSORS
                print("[SYNC] Quest perdido durante warmup.")
                return
            if not android_alive:
                self.sync_state = SyncState.QUEST_ONLY
                self._availability = quest_only_availability()
                self._rebuild_feature_builder()
                self.rolling_window.reset()
                print("[SYNC] Android perdido durante warmup. Continuando sin teléfono.")
                if not self.session_active:
                    self._start_auto_session()
                return
            elapsed = now - self.warmup_start_ts
            if elapsed >= self.warmup_duration:
                self.sync_state = SyncState.CALIBRATING
                print("[SYNC] Warmup completado. Iniciando calibración de sensores...")
                self._attempt_sensor_calibration()

        elif self.sync_state == SyncState.CALIBRATING:
            if not quest_alive:
                self._reset_to_waiting()
                return
            if not android_alive:
                self._switch_to_quest_only()
                return
            if self.sensor_calib_done:
                self.sync_state = SyncState.RECORDING
                self._availability = quest_plus_pelvis_availability()
                self._rebuild_feature_builder()
                if not self.session_active:
                    self._start_auto_session()
                print("[SYNC] Calibración completada. Iniciando grabación.")

        elif self.sync_state == SyncState.RECORDING:
            if not quest_alive:
                print("\n[SYNC] Quest desconectado. Cerrando sesión.")
                self._close_session_and_process()
                self.running = False
            elif not android_alive:
                self._switch_to_quest_only()
                print("[SYNC] Android perdido. Continuando sin teléfono (modo degradado).")

        elif self.sync_state == SyncState.QUEST_ONLY:
            if not quest_alive:
                print("\n[SYNC] Quest desconectado. Cerrando sesión.")
                self._close_session_and_process()
                self.running = False
            elif android_alive:
                self.warmup_start_ts = now
                self.sync_state = SyncState.WARMING_UP
                self.rolling_window.reset()
                print("[SYNC] Android reconectado. Resincronizando...")

    def _reset_to_waiting(self):
        self.sync_state = SyncState.WAITING_SENSORS
        self.sensor_calib_done = False
        self.calibration = None
        self.rolling_window.reset()

    def _switch_to_quest_only(self):
        self.sync_state = SyncState.QUEST_ONLY
        self._availability = quest_only_availability()
        self._rebuild_feature_builder()
        self.rolling_window.reset()
        self._warn_missing_sensors()

    def _warn_missing_sensors(self):
        now = time.time()
        if now - self._last_missing_warn >= 1.0:
            missing = self._availability.missing_sensors()
            print(f"[DEGRADED] Sensores ausentes: {missing}")
            self._last_missing_warn = now

    def _rebuild_feature_builder(self):
        if self.calibration is None:
            # Sin calibración no se puede construir; el builder se creará tras calibrar.
            self.feature_builder = None
        else:
            self.feature_builder = HMDPoserFeatureBuilder(self.calibration, self._availability)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def _attempt_sensor_calibration(self):
        pair = self.sync.get_synced_pair()
        if pair is None:
            return
        quest_data, android_data, _, _ = pair
        try:
            # Calibración de sensores: se usa el primer frame sincronizado como neutral.
            # Esto es un supuesto porque no se mide la pelvis directamente.
            self.calibration = build_calibration_from_neutral(
                quest_head_rot_xyzw=quest_data['head']['rot'],
                android_phone_rot_wxyz=android_data['pelvis']['rot'],
                timestamp=time.time(),
                assume_pelvis_aligned_with_head=True,
                phone_mount_transform=self.cfg.phone_mount_transform,
                acceleration_includes_gravity=self.cfg.acceleration_includes_gravity(),
            )
            self.sensor_calib_done = True
            self._rebuild_feature_builder()
            print(f"[CALIB] Calibración de sensores calculada en t={self.calibration.timestamp:.3f}")
            print(f"[CALIB] Aclaración: se asume pelvis alineada con la cabeza en postura neutral.")
        except Exception as e:
            print(f"[CALIB] Error en calibración: {e}")

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def _start_session(self):
        with self.session_lock:
            self.session_frames = []
            self.inference_results = []
            self.session_active = True
            self.session_number += 1
            self.session_start_ts = time.time()
        self.posture_calib_done = False
        self._calib_pitches = []
        self._calib_rolls = []
        self._pitch_offset = 0.0
        self._roll_offset = 0.0
        print(f"\n[SESSION #{self.session_number}] Grabación iniciada.")
        print(f"[CALIB] Mantén postura recta durante los primeros {self.cfg.posture_calib_frames} frames...")

    def _start_auto_session(self):
        self._start_session()
        print("[SESSION] Grabando hasta que el Quest se desconecte.")
        print("[SESSION] Detén la transmisión desde el Meta para terminar.")

        def _progress_printer():
            tick = 0
            while self.running and self.session_active:
                time.sleep(1)
                tick += 1
                if tick % 10 == 0:
                    results = list(self.inference_results)
                    n = len(results)
                    elapsed = int(time.time() - self.session_start_ts)
                    if results:
                        pitches = [r['pitch'] for r in results]
                        rolls = [r['roll'] for r in results]
                        avg_p = sum(pitches) / n
                        avg_r = sum(rolls) / n
                        max_p = max(pitches)
                        min_p = min(pitches)
                        print(f"  ── t={elapsed}s | {n} frames ──────────────────────")
                        print(f"     avg pitch: {avg_p:+.1f}°  avg roll: {avg_r:.1f}°")
                        print(f"     max pitch: {max_p:+.1f}°  min pitch: {min_p:+.1f}°")

        t = threading.Thread(target=_progress_printer, daemon=True)
        t.start()

    def _record_frame(self, quest_data: dict, android_data: dict, feature: np.ndarray,
                        inference_result: dict = None):
        ts = time.time()
        result = inference_result

        pelvis = android_data.get('pelvis', {})
        entry = {
            'ts': ts,
            'frame_idx': len(self.session_frames),
            'sensors': {
                'hmd_pos': quest_data['head']['pos'].tolist(),
                'hmd_rot': quest_data['head']['rot'].tolist(),
                'lhand_pos': quest_data['left_hand']['pos'].tolist(),
                'lhand_rot': quest_data['left_hand']['rot'].tolist(),
                'rhand_pos': quest_data['right_hand']['pos'].tolist(),
                'rhand_rot': quest_data['right_hand']['rot'].tolist(),
                'pelvis_rot': pelvis.get('rot', [0, 0, 0, 1]).tolist() if isinstance(pelvis.get('rot'), np.ndarray) else [0, 0, 0, 1],
                'pelvis_accel': pelvis.get('accel', [0, 0, 0]).tolist() if isinstance(pelvis.get('accel'), np.ndarray) else [0, 0, 0],
            },
            'feature_vector': feature.tolist(),
            'availability': {k: v for k, v in self._availability.__dict__.items()},
            'inference': {
                'pitch_deg': round(result['total_pitch'], 3),
                'roll_deg': round(result['total_roll'], 3),
            } if result else None,
        }

        with self.session_lock:
            self.session_frames.append(entry)
            if result:
                self.inference_results.append({
                    'pitch': result['total_pitch'],
                    'roll': result['total_roll'],
                })

        if result and not self.posture_calib_done:
            self._calib_pitches.append(result['total_pitch'])
            self._calib_rolls.append(result['total_roll'])
            if len(self._calib_pitches) >= self.cfg.posture_calib_frames:
                self._pitch_offset = sum(self._calib_pitches) / len(self._calib_pitches)
                self._roll_offset = sum(self._calib_rolls) / len(self._calib_rolls)
                self.posture_calib_done = True
                print(f"\n[CALIB] Offset de postura calculado → pitch={self._pitch_offset:+.1f}°  roll={self._roll_offset:.1f}°")
                print("[CALIB] Calibración de postura completada.\n")
            else:
                n = len(self._calib_pitches)
                print(f"[CALIB] {n:>2}/{self.cfg.posture_calib_frames}  pitch_raw={result['total_pitch']:+.1f}°")

        if result and self.posture_calib_done:
            pitch = result['total_pitch'] - self._pitch_offset
            roll = result['total_roll'] - self._roll_offset
            if pitch > 15.0:
                estado = "ENCORV >"
                icono = "!"
            elif pitch < -15.0:
                estado = "ENCORV <"
                icono = "!"
            elif abs(roll) > 15.0:
                dir_roll = ">" if roll > 0 else "<"
                estado = f"LATERAL{dir_roll}"
                icono = "~"
            else:
                estado = "OK"
                icono = " "
            n = len(self.inference_results)
            print(f"[{icono}] frame={n:>4}  Pitch: {pitch:+7.1f}°  Roll: {roll:+6.1f}°  [{estado}]")

    def _run_inference(self, feature_window: np.ndarray):
        if self.model is None:
            return None
        try:
            window = np.asarray(feature_window, dtype=np.float32)
            if window.shape != (self.cfg.temporal_window_frames, self.cfg.sparse_dim):
                raise ValueError(f"Shape incorrecto: {window.shape}")
            input_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                outputs = self.model(input_tensor)
            pred_pose_6d = extract_last_frame_pose_6d(outputs)
            angles = extract_spine_angles(pred_pose_6d)
            return {
                'total_pitch': angles['pitch_deg'],
                'total_roll': abs(angles['roll_deg']),
            }
        except Exception as e:
            print(f"[Inference] Error: {e}")
            return None

    def _close_session_and_process(self):
        with self.session_lock:
            self.session_active = False
            frames = list(self.session_frames)
            results = list(self.inference_results)
            sess_num = self.session_number

        n_frames = len(frames)
        n_inference = len(results)

        print(f"\n{'='*52}")
        print(f"  SESION #{sess_num} COMPLETADA")
        print(f"  Frames grabados:     {n_frames}")
        print(f"  Inferencias válidas: {n_inference}")
        print(f"{'='*52}")

        if n_inference < 10:
            print(f"  Sesión demasiado corta ({n_inference} frames).")
            print(f"{'='*52}\n")
            return

        pitches = [r['pitch'] for r in results]
        rolls = [r['roll'] for r in results]
        avg_pitch = sum(pitches) / n_inference
        avg_roll = sum(rolls) / n_inference
        max_pitch = max(pitches)
        max_roll = max(rolls)
        min_pitch = min(pitches)
        duracion_s = frames[-1]['ts'] - frames[0]['ts'] if n_frames > 1 else 0

        print(f"  Duración sesión:     {duracion_s:.1f}s")
        if duracion_s > 0:
            print(f"  Hz promedio:         {n_frames / duracion_s:.1f}")
        print(f"{'─'*52}")
        print(f"  ENCORVAMIENTO (Pitch)")
        print(f"    Promedio: {avg_pitch:+.2f}°")
        print(f"    Máximo:   {max_pitch:+.2f}°")
        print(f"    Mínimo:   {min_pitch:+.2f}°")
        print(f"{'─'*52}")
        print(f"  DESVIACIÓN LATERAL (Roll)")
        print(f"    Promedio: {avg_roll:.2f}°")
        print(f"    Máximo:   {max_roll:.2f}°")
        print(f"{'─'*52}")

        if avg_pitch > 15.0:
            print(f"  ALERTA: Encorvamiento promedio excesivo ({avg_pitch:.1f}°)")
        elif avg_roll > 15.0:
            print(f"  ALERTA: Inclinación lateral promedio ({avg_roll:.1f}°)")
        else:
            print("  Postura promedio saludable durante la sesión.")
        print(f"{'='*52}\n")

        self._save_session_json(sess_num, frames, results, {
            'avg_pitch': avg_pitch,
            'avg_roll': avg_roll,
            'max_pitch': max_pitch,
            'max_roll': max_roll,
            'min_pitch': min_pitch,
            'n_frames': n_frames,
            'duration': duracion_s,
        })

    def _save_session_json(self, sess_num, frames, results, summary):
        os.makedirs(self.cfg.records_dir_abs(), exist_ok=True)
        fmt = "%Y-%m-%d_%H-%M-%S"
        start_str = datetime.fromtimestamp(self.session_start_ts).strftime(fmt)
        end_str = datetime.fromtimestamp(time.time()).strftime(fmt)
        filename = os.path.join(self.cfg.records_dir_abs(), f"{start_str}__{end_str}.json")

        try:
            output = {
                'meta': {
                    'session': sess_num,
                    'start_time': start_str,
                    'end_time': end_str,
                    'duration_s': round(time.time() - self.session_start_ts, 2),
                    'total_frames': len(frames),
                    'valid_inferences': len(results),
                    'hz_avg': round(len(frames) / max(time.time() - self.session_start_ts, 1), 1),
                    'calib_pitch_offset': round(float(self._pitch_offset), 2),
                    'calib_roll_offset': round(float(self._roll_offset), 2),
                    'sensor_mode': self._availability.as_hmd_poser_mode(),
                    'missing_sensors': self._availability.missing_sensors(),
                    'feature_schema_version': self.cfg.feature_schema_version,
                    'checkpoint_sha256': self._checkpoint_sha256(),
                    'strict_checkpoint_loading': self.cfg.strict_checkpoint_loading,
                },
                'summary': {
                    'avg_pitch_deg': round(float(summary.get('avg_pitch', 0)), 2),
                    'avg_roll_deg': round(float(summary.get('avg_roll', 0)), 2),
                    'max_pitch_deg': round(float(summary.get('max_pitch', 0)), 2),
                    'min_pitch_deg': round(float(summary.get('min_pitch', 0)), 2),
                    'max_roll_deg': round(float(summary.get('max_roll', 0)), 2),
                    'alert_hunchback': bool(summary.get('avg_pitch', 0) > 15.0),
                    'alert_lateral': bool(summary.get('avg_roll', 0) > 15.0),
                },
                'frames': frames,
            }

            class _Encoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    return super().default(obj)

            with open(filename, 'w') as fp:
                json.dump(output, fp, indent=2, cls=_Encoder)
            print(f"[SESSION] Guardado en: {filename}")
            print(f"[SESSION] {len(results)} inferencias | avg_pitch={summary.get('avg_pitch',0):+.1f}° | avg_roll={summary.get('avg_roll',0):.1f}°")
        except Exception as e:
            print(f"[SESSION] Error al guardar JSON: {e}")

    def _checkpoint_sha256(self) -> str:
        try:
            import hashlib
            h = hashlib.sha256()
            with open(self.cfg.checkpoint_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _update_hz_stats(self):
        now = time.time()
        elapsed = now - self._stats_ts
        if elapsed >= 1.0:
            self.sync.update_hz()
            self._stats_ts = now

    def main_loop(self):
        interval = 1.0 / self.cfg.server_loop_rate_hz
        while self.running:
            t0 = time.time()
            self._update_hz_stats()
            self._update_sync_state()

            if self.sync_state in (SyncState.WAITING_SENSORS, SyncState.WARMING_UP):
                self._sleep_remaining(t0, interval)
                continue

            if self.feature_builder is None:
                self._sleep_remaining(t0, interval)
                continue

            # Obtener par sincronizado según modo
            if self.sync_state == SyncState.RECORDING:
                pair = self.sync.get_synced_pair()
                if pair is None:
                    self._sleep_remaining(t0, interval)
                    continue
                quest_data, android_data, _, _ = pair
            elif self.sync_state == SyncState.QUEST_ONLY:
                with self.quest_lock:
                    raw = self.latest_quest_data
                if raw is None:
                    self._sleep_remaining(t0, interval)
                    continue
                quest_data = self._parse_quest_raw(raw)
                android_data = {'pelvis': {'rot': np.array([0, 0, 0, 1], dtype=np.float32),
                                           'accel': np.array([0, 0, 0], dtype=np.float32)}}
                if quest_data is None:
                    self._sleep_remaining(t0, interval)
                    continue
            else:
                self._sleep_remaining(t0, interval)
                continue

            current_combined = {**quest_data, **android_data}
            prev_combined = getattr(self, '_last_combined_frame', None)

            try:
                feature = self.feature_builder.build_tensor(current_combined, prev_combined)
            except Exception as e:
                print(f"[Feature] Error: {e}")
                self._sleep_remaining(t0, interval)
                continue

            self._last_combined_frame = current_combined
            accepted = self.rolling_window.add(feature, time.time())

            window = self.rolling_window.to_numpy()
            if window is not None:
                result = self._run_inference(window)
                if accepted:
                    self._record_frame(quest_data, android_data, feature, result)
            else:
                if accepted:
                    self._record_frame(quest_data, android_data, feature, None)

            self._sleep_remaining(t0, interval)

    def _sleep_remaining(self, t0: float, interval: float):
        elapsed = time.time() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    def start(self):
        self.running = True
        self.quest_listener.start()
        self.android_listener.start()
        print(f"[SERVER] Quest   → puerto {self.quest_port}")
        print(f"[SERVER] Android → puerto {self.android_port}")
        print(f"[SERVER] Warmup  → {self.warmup_duration}s")
        print(f"[SERVER] Ventana → {self.cfg.synchronization_tolerance_ms}ms")
        print("[SERVER] Esperando sensores...\n")
        self.main_loop()

    def stop(self):
        if self.session_active:
            print("\n[SERVER] Cerrando servidor. Procesando sesión activa...")
            self._close_session_and_process()
        self.running = False
        self.quest_listener.stop()
        self.android_listener.stop()
        print("Servidor detenido.")


if __name__ == "__main__":
    server = BiomechanicsServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        if server.running:
            server.stop()
