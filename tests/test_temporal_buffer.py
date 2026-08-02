"""
Tests de sincronización temporal y ventana deslizante.
"""

import time
import numpy as np
import pytest

from src.temporal_buffer import Synchronizer, RollingWindow
from src.config import load_config


def test_rolling_buffer_contains_40_distinct_timestamps():
    rw = RollingWindow(size=40)
    for i in range(40):
        rw.add(np.ones(135, dtype=np.float32), time.time() + i * 0.016)
    w = rw.to_numpy()
    assert w is not None
    assert w.shape == (40, 135)
    assert len(set(rw.timestamps)) == 40


def test_rolling_buffer_resets_on_gap():
    rw = RollingWindow(size=40, max_gap_ms=50.0)
    for i in range(20):
        rw.add(np.ones(135), time.time() + i * 0.016)
    # Salto mayor que 50ms
    rw.add(np.ones(135), time.time() + 20 * 0.016 + 0.1)
    assert rw.degraded_mode
    assert len(rw.frames) == 1


def test_out_of_order_timestamp_handling():
    rw = RollingWindow(size=40)
    now = time.time()
    rw.add(np.ones(135), now)
    assert rw.add(np.ones(135), now - 0.01) is False
    assert len(rw.frames) == 1


def test_synchronization_tolerance_matches_config():
    cfg = load_config('config.yaml')
    sync = Synchronizer(tolerance_ms=cfg.synchronization_tolerance_ms)
    now = time.time()
    sync.add_quest({'data': 1}, now)
    sync.add_android({'data': 2}, now + cfg.synchronization_tolerance_ms / 1000.0 * 0.5)
    assert sync.get_synced_pair() is not None

    sync2 = Synchronizer(tolerance_ms=cfg.synchronization_tolerance_ms)
    sync2.add_quest({'data': 1}, now)
    sync2.add_android({'data': 2}, now + cfg.synchronization_tolerance_ms / 1000.0 * 2.0)
    assert sync2.get_synced_pair() is None


def test_diagnostic_repeated_frame_mode():
    rw = RollingWindow(size=40, diagnostic_repeated_frame_mode=True)
    rw.add(np.ones(135), time.time())
    w = rw.to_numpy()
    assert w is not None
    assert w.shape == (40, 135)
    assert rw.degraded_mode
