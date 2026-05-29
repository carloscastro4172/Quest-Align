# server.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import threading
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from collections import deque
from enum import Enum, auto

from udp_listener import UDPListener
from feature_builder import HMDPoserFeatureBuilder

# ── Configuración ──────────────────────────────
QUEST_PORT           = 5006
ANDROID_PORT         = 5005
MAIN_LOOP_FREQ       = 60
SENSOR_TIMEOUT       = 0.5

UMBRAL_ENCORVAMIENTO = 15.0   # pitch > 15° = encorvamiento (calibrado: recta=~+6°)
UMBRAL_DESVIACION    = 15.0   # |roll| > 15° = inclinación lateral (calibrado: recta=~0°)
FILTRO_VENTANA       = 7      # frames para filtro de mediana anti-outliers

SYNC_WARMUP_SECS     = 3.0
SYNC_WINDOW_MS       = 250
SYNC_BUFFER_SIZE     = 60
MIN_SESSION_FRAMES   = 10
CALIB_FRAMES         = 30     # frames iniciales usados para calibrar el offset de postura recta
SESSION_DURATION_SECS = 60     # duración fija de grabación en segundos

RECORDS_DIR = "/home/carlos/Documents/Desarrollo_movil_/FINAL/HMD-Poser/RECORDS"


PELVIS_ROT_FALLBACK   = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
PELVIS_ACCEL_FALLBACK = np.array([0.0, -9.81, 0.0],     dtype=np.float32)


class SyncState(Enum):
    WAITING_SENSORS = auto()
    WARMING_UP      = auto()
    RECORDING       = auto()
    QUEST_ONLY      = auto()


class BiomechanicsServer:
    def __init__(self):
        self.latest_quest_data   = None
        self.latest_android_data = None
        self.quest_lock          = threading.Lock()
        self.android_lock        = threading.Lock()
        self.last_quest_ts       = 0.0
        self.last_android_ts     = 0.0

        self.quest_buffer   = deque(maxlen=SYNC_BUFFER_SIZE)
        self.android_buffer = deque(maxlen=SYNC_BUFFER_SIZE)
        self.buffer_lock    = threading.Lock()

        self.session_frames    = []
        self.session_lock      = threading.Lock()
        self.session_active    = False
        self.session_number    = 0
        self.inference_results = []

        self.sync_state      = SyncState.WAITING_SENSORS
        self.warmup_start_ts = 0.0

        self.sync_stats = {
            'quest_hz':       0,
            'android_hz':     0,
            'session_frames': 0,
        }
        self._quest_count   = 0
        self._android_count = 0
        self._stats_ts      = time.time()

        self.feature_builder = HMDPoserFeatureBuilder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._last_android_addr = None   # se actualiza en _android_handler

        # Calibración: offset que se resta a cada inferencia
        self._calib_pitches   = []
        self._calib_rolls     = []
        self._calib_done      = False
        self._pitch_offset    = 0.0
        self._roll_offset     = 0.0

        # Filtro de mediana: ventana deslizante para eliminar outliers del modelo
        self._pitch_window    = []
        self._roll_window     = []
        print(f"Dispositivo activo: {self.device}")

        try:
            from model.hmd_imu_model import HMDIMUModel
            from utils.utils_config import load_config

            config     = load_config("options/test_config.yaml")
            self.model = HMDIMUModel(config, self.device)
            self.model.load_network(
                "pretrained_model/pretrained_model_protocol1.pt",
                self.model.netG,
                strict=False
            )
            self.model.to(self.device)
            self.model.eval()
            print("Modelo cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None

        self.quest_listener   = UDPListener("Quest",   QUEST_PORT,   self._quest_handler)
        self.android_listener = UDPListener("Android", ANDROID_PORT, self._android_handler)
        self.running = False

    # ────────────────────────────────────────────
    #  Handlers UDP
    # ────────────────────────────────────────────
    def _quest_handler(self, data: bytes, addr):
        ts = time.time()
        with self.quest_lock:
            self.latest_quest_data = data
            self.last_quest_ts     = ts
            self._quest_count     += 1
        parsed = self._parse_quest_raw(data)
        if parsed:
            with self.buffer_lock:
                self.quest_buffer.append((ts, parsed))

    def _android_handler(self, data: bytes, addr):
        # Guardar dirección para poder responder sync_resp
        self._last_android_addr = addr
        arrival_ts = time.time()
        with self.android_lock:
            self.latest_android_data = data
            self.last_android_ts     = arrival_ts
            self._android_count     += 1
        parsed = self._parse_android_raw(data)
        if parsed:
            # Usar siempre el tiempo de llegada al servidor (time.time()).
            # El t_server_est del Android usa elapsedRealtimeNanos (desde boot del teléfono),
            # que es incompatible con time.time() (Unix epoch). El offset NTP nunca converge
            # porque la diferencia es ~décadas en nanosegundos.
            with self.buffer_lock:
                self.android_buffer.append((arrival_ts, parsed))

    # ────────────────────────────────────────────
    #  Parseo
    # ────────────────────────────────────────────


    def _send_sync_response(self, req: dict):
        """
        Responde el sync_req del Android con t1 y t2
        para que el Android pueda calcular el offset de reloj.
        """
        if self._last_android_addr is None:
            return
        try:
            import socket as sock_module
            t1 = int(time.time() * 1e9)
            t2 = int(time.time() * 1e9)

            resp = {
                'type':   'sync_resp',
                'id':     req.get('id', ''),
                't0_ns':  req.get('t0_ns', 0),
                't1_ns':  t1,
                't2_ns':  t2
            }

            msg  = json.dumps(resp).encode('utf-8')
            sock = sock_module.socket(sock_module.AF_INET, sock_module.SOCK_DGRAM)
            sock.sendto(msg, self._last_android_addr)
            sock.close()
        except Exception as e:
            print(f"[Sync] Error respondiendo sync: {e}")
            
    def _parse_quest_raw(self, data: bytes):
        try:
            j = json.loads(data.decode('utf-8'))
            return {
                'hmd': {
                    'pos': np.array(j['hmd_p'],   dtype=np.float32),
                    'rot': np.array(j['hmd_q'],   dtype=np.float32)
                },
                'left_hand': {
                    'pos': np.array(j['lhand_p'], dtype=np.float32),
                    'rot': np.array(j['lhand_q'], dtype=np.float32)
                },
                'right_hand': {
                    'pos': np.array(j['rhand_p'], dtype=np.float32),
                    'rot': np.array(j['rhand_q'], dtype=np.float32)
                }
            }
        except Exception as e:
            print(f"[Parse Quest] Error: {e}")
            return None

    def _parse_android_raw(self, data: bytes):
        try:
            j    = json.loads(data.decode('utf-8'))
            tipo = j.get('type', '')

            # Paquetes de control: hello y cmd se ignoran silenciosamente
            if tipo in ('hello', 'cmd'):
                return None

            # Responder sync_req con sync_resp (y no devolver datos IMU)
            if tipo == 'sync_req':
                self._send_sync_response(j)
                return None

            # Solo procesar paquetes tipo 'imu'
            if tipo != 'imu':
                return None

            # El cuaternión llega como [qw, qx, qy, qz] desde Android
            # (SensorManager.getQuaternionFromVector devuelve w primero)
            quat  = j['quat']
            qw_in, qx_in, qy_in, qz_in = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

            # Scipy/feature_builder espera formato [x, y, z, w]
            rot   = np.array([qx_in, qy_in, qz_in, qw_in], dtype=np.float32)

            # El acelerómetro viene como array [ax, ay, az]
            acc   = j['acc']
            accel = np.array([float(acc[0]), float(acc[1]), float(acc[2])], dtype=np.float32)

            # Normalizar cuaternión por seguridad
            norm = np.linalg.norm(rot)
            if norm > 1e-6:
                rot = rot / norm

            return {
                'pelvis': {
                    'rot':   rot,
                    'accel': accel
                }
            }
        except Exception as e:
            print(f"[Parse Android] Error: {e}")
            return None
    
    def _get_android_fallback(self):
        return {
            'pelvis': {
                'rot':   PELVIS_ROT_FALLBACK.copy(),
                'accel': PELVIS_ACCEL_FALLBACK.copy()
            }
        }

    # ────────────────────────────────────────────
    #  Sincronización temporal
    # ────────────────────────────────────────────
    def _update_sync_state(self):
        now           = time.time()
        quest_alive   = (now - self.last_quest_ts)   <= SENSOR_TIMEOUT
        android_alive = (now - self.last_android_ts) <= SENSOR_TIMEOUT

        if self.sync_state == SyncState.WAITING_SENSORS:
            if quest_alive and android_alive:
                self.warmup_start_ts = now
                self.sync_state      = SyncState.WARMING_UP
                print(f"\n[SYNC] Ambos sensores detectados.")
                print(f"[SYNC] Calentando {SYNC_WARMUP_SECS:.0f}s...")
            elif quest_alive:
                self.sync_state = SyncState.QUEST_ONLY
                print("[SYNC] Quest detectado. Iniciando con fallback de pelvis.")
                # Arrancar la única sesión global si aún no existe
                if not self.session_active:
                    self._start_auto_session()

        elif self.sync_state == SyncState.WARMING_UP:
            elapsed   = now - self.warmup_start_ts
            remaining = SYNC_WARMUP_SECS - elapsed

            if not quest_alive:
                self.sync_state = SyncState.WAITING_SENSORS
                print("[SYNC] Quest perdido durante warmup.")
                return
            if not android_alive:
                self.sync_state = SyncState.QUEST_ONLY
                print("[SYNC] Android perdido durante warmup. Continuando sin él.")
                return
            if remaining > 0:
                if int(remaining) != int(remaining + 0.016):
                    print(f"[SYNC] Sincronizando... {remaining:.1f}s")
                return

            self.sync_state = SyncState.RECORDING
            if not self.session_active:
                self._start_auto_session()

        elif self.sync_state == SyncState.RECORDING:
            if not quest_alive:
                # Quest apagado/detenido → cerrar sesión y parar servidor
                print("\n[SYNC] Quest desconectado. Cerrando sesión y deteniendo servidor.")
                self._close_session_and_process()
                self.running = False
            elif not android_alive:
                # Solo Android perdido → continuar con fallback
                print("[SYNC] Android perdido. Continuando con fallback de pelvis.")
                self.sync_state = SyncState.QUEST_ONLY

        elif self.sync_state == SyncState.QUEST_ONLY:
            if not quest_alive:
                # Quest apagado/detenido → cerrar sesión y parar servidor
                print("\n[SYNC] Quest desconectado. Cerrando sesión y deteniendo servidor.")
                self._close_session_and_process()
                self.running = False
            elif android_alive:
                self.warmup_start_ts = now
                self.sync_state      = SyncState.WARMING_UP
                print("[SYNC] Android reconectado. Resincronizando...")

    # ────────────────────────────────────────────
    #  Sesión
    # ────────────────────────────────────────────
    def _start_session(self):
        """Inicia sesión de grabación (llamada interna base)."""
        with self.session_lock:
            self.session_frames    = []
            self.inference_results = []
            self.session_active    = True
            self.session_number   += 1
            self.session_start_ts  = time.time()
        # Resetear calibración para la nueva sesión
        self._calib_pitches = []
        self._calib_rolls   = []
        self._calib_done    = False
        self._pitch_offset  = 0.0
        self._roll_offset   = 0.0
        self._pitch_window  = []
        self._roll_window   = []
        print(f"\n[SESSION #{self.session_number}] Grabación iniciada.")
        print(f"[CALIB] Mantén postura recta durante los primeros {CALIB_FRAMES} frames...")

    def _start_auto_session(self):
        """Inicia la única sesión global. El timer corre exactamente una vez
        durante SESSION_DURATION_SECS de reloj de pared, sin importar
        cuántas veces el Quest se desconecte y reconecte."""
        self._start_session()
        print(f"[SESSION] Grabando hasta que el Quest se desconecte.")
        print(f"[SESSION] Detén la transmisión desde el Meta para terminar.")

        def _progress_printer():
            """Imprime resumen cada 10s mientras el servidor corre."""
            tick = 0
            while self.running and self.session_active:
                time.sleep(1)
                tick += 1
                if tick % 10 == 0:
                    results = list(self.inference_results)
                    n       = len(results)
                    elapsed = int(time.time() - self.session_start_ts)
                    if results:
                        pitches = [r['pitch'] for r in results]
                        rolls   = [r['roll']  for r in results]
                        avg_p   = sum(pitches) / n
                        avg_r   = sum(rolls)   / n
                        max_p   = max(pitches)
                        min_p   = min(pitches)
                        print(f"  ── t={elapsed}s | {n} frames ──────────────────────")
                        print(f"     avg pitch: {avg_p:+.1f}°  avg roll: {avg_r:.1f}°")
                        print(f"     max pitch: {max_p:+.1f}°  min pitch: {min_p:+.1f}°")

        t = threading.Thread(target=_progress_printer, daemon=True)
        t.start()

    def _record_frame(self, quest_data: dict, android_data: dict):
        ts     = time.time()
        result = self._run_inference(quest_data, android_data)

        # Construir entrada limpia del frame (sin claves internas _sync_ok/_t_server_est)
        pelvis = android_data.get('pelvis', {})
        entry = {
            'ts':        ts,
            'frame_idx': len(self.session_frames),
            'sensors': {
                'hmd_pos':    quest_data['hmd']['pos'].tolist(),
                'hmd_rot':    quest_data['hmd']['rot'].tolist(),
                'lhand_pos':  quest_data['left_hand']['pos'].tolist(),
                'lhand_rot':  quest_data['left_hand']['rot'].tolist(),
                'rhand_pos':  quest_data['right_hand']['pos'].tolist(),
                'rhand_rot':  quest_data['right_hand']['rot'].tolist(),
                'pelvis_rot':   pelvis.get('rot',   [0,0,0,1]).tolist() if hasattr(pelvis.get('rot', None), 'tolist') else [0,0,0,1],
                'pelvis_accel': pelvis.get('accel', [0,0,0]).tolist()   if hasattr(pelvis.get('accel', None), 'tolist') else [0,0,0],
            },
            'inference': {
                'pitch_deg': round(result['total_pitch'], 3),
                'roll_deg':  round(result['total_roll'],  3),
            } if result else None
        }

        with self.session_lock:
            self.session_frames.append(entry)
            if result:
                self.inference_results.append({
                    'pitch': result['total_pitch'],
                    'roll':  result['total_roll']
                })

        # Calibración automática: primeros CALIB_FRAMES frames definen el offset
        if result and not self._calib_done:
            self._calib_pitches.append(result['total_pitch'])
            self._calib_rolls.append(result['total_roll'])
            if len(self._calib_pitches) >= CALIB_FRAMES:
                self._pitch_offset = sum(self._calib_pitches) / len(self._calib_pitches)
                self._roll_offset  = sum(self._calib_rolls)   / len(self._calib_rolls)
                self._calib_done   = True
                print(f"\n[CALIB] Offset calculado → pitch={self._pitch_offset:+.1f}°  roll={self._roll_offset:.1f}°")
                print(f"[CALIB] Calibración completada. Valores normalizados desde ahora.\n")
            else:
                n = len(self._calib_pitches)
                print(f"[CALIB] {n:>2}/{CALIB_FRAMES}  pitch_raw={result['total_pitch']:+.1f}°")

        # Aplicar offset de calibración y filtro de mediana
        if result and self._calib_done:
            raw_pitch = result['total_pitch'] - self._pitch_offset
            raw_roll  = result['total_roll']  - self._roll_offset

            # Filtro de mediana: acumular en ventana y tomar la mediana
            self._pitch_window.append(raw_pitch)
            self._roll_window.append(raw_roll)
            if len(self._pitch_window) > FILTRO_VENTANA:
                self._pitch_window.pop(0)
                self._roll_window.pop(0)

            sorted_p = sorted(self._pitch_window)
            sorted_r = sorted(self._roll_window)
            mid = len(sorted_p) // 2
            pitch_filtered = sorted_p[mid]
            roll_filtered  = sorted_r[mid]

            result = {
                'total_pitch': pitch_filtered,
                'total_roll':  roll_filtered,
            }

        # Mostrar inferencia en tiempo real en consola
        if result and self._calib_done:
            pitch = result['total_pitch']
            roll  = result['total_roll']

            # Indicador visual de postura
            if pitch > UMBRAL_ENCORVAMIENTO:
                estado = "ENCORV >"  # hacia adelante
                icono  = "!"
            elif pitch < -UMBRAL_ENCORVAMIENTO:
                estado = "ENCORV <"  # hacia atrás (hiperlordosis)
                icono  = "!"
            elif abs(roll) > UMBRAL_DESVIACION:
                dir_roll = ">" if roll > 0 else "<"
                estado = f"LATERAL{dir_roll}"
                icono  = "~"
            else:
                estado = "OK"
                icono  = " "

            n = len(self.inference_results)
            print(f"[{icono}] frame={n:>4}  Pitch: {pitch:+7.1f}°  Roll: {roll:+6.1f}°  [{estado}]")

    def _close_session_and_process(self):
        with self.session_lock:
            self.session_active = False
            frames   = list(self.session_frames)
            results  = list(self.inference_results)
            sess_num = self.session_number

        n_frames    = len(frames)
        n_inference = len(results)

        print(f"\n{'='*52}")
        print(f"  SESION #{sess_num} COMPLETADA")
        print(f"  Frames grabados:     {n_frames}")
        print(f"  Inferencias válidas: {n_inference}")
        print(f"{'='*52}")

        if n_inference < MIN_SESSION_FRAMES:
            print(f"  Sesión demasiado corta ({n_inference} frames).")
            print(f"  Se necesitan al menos {MIN_SESSION_FRAMES} frames.")
            print(f"{'='*52}\n")
            return

        pitches      = [r['pitch'] for r in results]
        rolls        = [r['roll']  for r in results]
        avg_pitch    = sum(pitches) / n_inference
        avg_roll     = sum(rolls)   / n_inference
        max_pitch    = max(pitches)
        max_roll     = max(rolls)
        min_pitch    = min(pitches)
        duracion_s   = frames[-1]['ts'] - frames[0]['ts'] if n_frames > 1 else 0

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

        if avg_pitch > UMBRAL_ENCORVAMIENTO:
            print(f"  ALERTA: Encorvamiento promedio excesivo ({avg_pitch:.1f}°)")
            print(f"  CONSEJO: Levanta la cabeza y retrae los hombros.")
        elif avg_roll > UMBRAL_DESVIACION:
            print(f"  ALERTA: Inclinación lateral promedio ({avg_roll:.1f}°)")
            print(f"  CONSEJO: No te apoyes solo en un lado.")
        else:
            print(f"  Postura promedio saludable durante la sesión.")
        print(f"{'='*52}\n")

        end_ts = time.time()
        with self.session_lock:
            start_ts = getattr(self, 'session_start_ts', end_ts - duracion_s)

        self._save_session_json(sess_num, frames, results, {
            'avg_pitch': avg_pitch,
            'avg_roll':  avg_roll,
            'max_pitch': max_pitch,
            'max_roll':  max_roll,
            'min_pitch': min_pitch,
            'n_frames':  n_frames,
            'duration':  duracion_s
        }, start_ts=start_ts, end_ts=end_ts)

    def _save_session_json(self, sess_num, frames, results, summary,
                           start_ts: float = 0.0, end_ts: float = 0.0):
        import os
        from datetime import datetime

        os.makedirs(RECORDS_DIR, exist_ok=True)

        fmt       = "%Y-%m-%d_%H-%M-%S"
        start_str = datetime.fromtimestamp(start_ts).strftime(fmt) if start_ts else "unknown"
        end_str   = datetime.fromtimestamp(end_ts).strftime(fmt)   if end_ts   else "unknown"
        filename  = os.path.join(RECORDS_DIR, f"{start_str}__{end_str}.json")

        # Calcular lista de inferencias válidas para el bloque 'predictions'
        valid = [f for f in frames if f.get('inference') is not None]
        pitches = [f['inference']['pitch_deg'] for f in valid]
        rolls   = [f['inference']['roll_deg']  for f in valid]

        try:
            output = {
                'meta': {
                    'session':          sess_num,
                    'start_time':       start_str,
                    'end_time':         end_str,
                    'duration_s':       round(end_ts - start_ts, 2),
                    'total_frames':     len(frames),
                    'valid_inferences': len(valid),
                    'hz_avg':           round(len(frames) / max(end_ts - start_ts, 1), 1),
                    'calib_pitch_offset': round(float(self._pitch_offset), 2),
                    'calib_roll_offset':  round(float(self._roll_offset),  2),
                },
                'summary': {
                    'avg_pitch_deg':   round(float(summary.get('avg_pitch', 0)), 2),
                    'avg_roll_deg':    round(float(summary.get('avg_roll',  0)), 2),
                    'max_pitch_deg':   round(float(summary.get('max_pitch', 0)), 2),
                    'min_pitch_deg':   round(float(summary.get('min_pitch', 0)), 2),
                    'max_roll_deg':    round(float(summary.get('max_roll',  0)), 2),
                    'alert_hunchback': bool(summary.get('avg_pitch', 0) > UMBRAL_ENCORVAMIENTO),
                    'alert_lateral':   bool(summary.get('avg_roll',  0) > UMBRAL_DESVIACION),
                },
                # Cada frame tiene sus datos de sensores + la inferencia del modelo
                'frames': frames,
            }
            import numpy as np
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
            print(f"[SESSION] {len(valid)} inferencias | avg_pitch={summary.get('avg_pitch',0):+.1f}° | avg_roll={summary.get('avg_roll',0):.1f}°")
        except Exception as e:
            print(f"[SESSION] Error al guardar JSON: {e}")

    # ────────────────────────────────────────────
    #  Sincronización por timestamp
    # ────────────────────────────────────────────
    def _get_synced_pair(self):
        with self.buffer_lock:
            if not self.quest_buffer or not self.android_buffer:
                return None

            quest_ts,   quest_data   = self.quest_buffer[-1]
            android_ts, android_data = self.android_buffer[-1]

            # Ambos timestamps son time.time() del servidor (tiempo de llegada del paquete).
            # La diferencia real entre sensores a 60Hz es <17ms, la ventana de 100ms es holgada.
            offset_ms = abs(quest_ts - android_ts) * 1000.0

            if offset_ms <= SYNC_WINDOW_MS:
                return quest_data, android_data

            # Fuera de ventana: loguear para diagnóstico (máx 1 vez por segundo)
            now = time.time()
            if now - getattr(self, '_last_sync_warn', 0) >= 1.0:
                print(f"[SYNC] Par rechazado: offset={offset_ms:.1f}ms "
                      f"(ventana={SYNC_WINDOW_MS}ms) — jitter de red elevado")
                self._last_sync_warn = now
            return None

    # ────────────────────────────────────────────
    #  Inferencia — alineada con notebook
    # ────────────────────────────────────────────
    def _run_inference(self, quest_data: dict, android_data: dict):
        if self.model is None:
            return None
        try:
            single_frame = self.feature_builder.build_tensor(quest_data, android_data)

            target_dim = 135
            flat = single_frame.view(-1)
            if flat.shape[0] < target_dim:
                padding = torch.zeros(target_dim - flat.shape[0])
                flat    = torch.cat([flat, padding], dim=0)
            else:
                flat = flat[:target_dim]

            input_tensor = flat.view(1, 1, -1).repeat(1, 40, 1).to(self.device).float()

            with torch.no_grad():
                outputs      = self.model(input_tensor, do_fk=True)
                pred_pose_6d = outputs[0][:, -1, :].reshape(22, 6)

            from utils import utils_transform
            theta_aa = utils_transform.sixd2aa(pred_pose_6d.reshape(-1, 6))

            def get_rot(idx):
                """Retorna objeto Rotation para una articulación SMPL."""
                return R.from_rotvec(theta_aa[idx].cpu().numpy())

            # Índices SMPL del tronco:
            # 3=spine1 (lumbar), 6=spine2 (torácica media), 9=spine3 (torácica alta)
            # Composición encadenada: R_total = R_spine1 * R_spine2 * R_spine3
            r_trunk = get_rot(3) * get_rot(6) * get_rot(9)

            # Ángulos de Euler ZXY — convención biomecánica:
            # X = flexión/extensión (pitch), Z = inclinación lateral (roll)
            euler = r_trunk.as_euler('ZXY', degrees=True)
            pitch_raw = euler[1]   # flexión/extensión
            roll_raw  = euler[2]   # inclinación lateral

            # Normalizar al rango [-180, 180]
            def norm180(a):
                return ((a + 180) % 360) - 180

            pitch = norm180(pitch_raw)
            roll  = norm180(roll_raw)

            return {
                'total_pitch': pitch,
                'total_roll':  abs(roll)
            }
        except Exception as e:
            print(f"[Inference] Error: {e}")
            return None

    # ────────────────────────────────────────────
    #  Estadísticas Hz
    # ────────────────────────────────────────────
    def _update_stats(self):
        now     = time.time()
        elapsed = now - self._stats_ts
        if elapsed >= 1.0:
            self.sync_stats['quest_hz']   = int(self._quest_count   / elapsed)
            self.sync_stats['android_hz'] = int(self._android_count / elapsed)
            self._quest_count   = 0
            self._android_count = 0
            self._stats_ts      = now

    # ────────────────────────────────────────────
    #  Bucle principal
    # ────────────────────────────────────────────
    def start(self):
        self.running = True
        self.quest_listener.start()
        self.android_listener.start()
        print(f"[SERVER] Quest   → puerto {QUEST_PORT}")
        print(f"[SERVER] Android → puerto {ANDROID_PORT}")
        print(f"[SERVER] Warmup  → {SYNC_WARMUP_SECS}s")
        print(f"[SERVER] Ventana → {SYNC_WINDOW_MS}ms")
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

    def main_loop(self):
        interval = 1.0 / MAIN_LOOP_FREQ

        while self.running:
            t0 = time.time()

            self._update_stats()
            self._update_sync_state()

            if self.sync_state in (SyncState.WAITING_SENSORS, SyncState.WARMING_UP):
                time.sleep(interval)
                continue

            try:
                # Obtener par de datos según modo de sensores
                if self.sync_state == SyncState.RECORDING:
                    pair = self._get_synced_pair()
                    if pair is None:
                        time.sleep(interval)
                        continue
                    quest_data, android_data = pair

                else:  # QUEST_ONLY
                    with self.quest_lock:
                        raw = self.latest_quest_data
                    if raw is None:
                        time.sleep(interval)
                        continue
                    quest_data   = self._parse_quest_raw(raw)
                    android_data = self._get_android_fallback()
                    if not quest_data:
                        time.sleep(interval)
                        continue

                # Acumular frame si hay sesión activa
                if self.session_active:
                    self._record_frame(quest_data, android_data)

            except Exception as e:
                print(f"[Loop] Error: {e}")

            elapsed = time.time() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)



if __name__ == "__main__":
    server = BiomechanicsServer()
    try:
        server.start()          # bloquea en main_loop hasta que stop() lo libera
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        if server.running:
            server.stop()