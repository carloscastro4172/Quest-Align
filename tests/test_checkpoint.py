"""
Tests de validación del checkpoint HMD-Poser.
"""

import os
import json
import numpy as np
import pytest
import torch

from src.config import load_config
from src.checkpoint_validator import CheckpointValidator, run_checkpoint_validation


def test_checkpoint_strict_loading():
    cfg = load_config('config.yaml')
    validator = CheckpointValidator(cfg)
    report = validator.validate()
    assert report['strict_load'] is True
    assert report['checkpoint_partial_load'] is False


def test_checkpoint_forward_pass():
    cfg = load_config('config.yaml')
    validator = CheckpointValidator(cfg)
    report = validator.validate()
    assert report['forward_pass_success'] is True
    assert report['no_nan_in_output'] is True
    assert report['output_shape']['pred_pose'] == [1, cfg.input_motion_length, 132]
    assert report['output_shape']['pred_shapes'] == [1, cfg.input_motion_length, 16]


def test_model_input_shape():
    cfg = load_config('config.yaml')
    validator = CheckpointValidator(cfg)
    report = validator.validate()
    model = validator.get_model()
    assert model is not None
    x = torch.randn(1, cfg.input_motion_length, cfg.sparse_dim)
    model.eval()
    with torch.no_grad():
        pred_pose, pred_shapes = model(x)
    assert pred_pose.shape == (1, cfg.input_motion_length, 22 * 6)
    assert pred_shapes.shape == (1, cfg.input_motion_length, 16)


def test_architectural_compatibility_report():
    report = run_checkpoint_validation('config.yaml')
    assert report['architecturally_compatible'] is True
    assert report['quest_plus_pelvis_seen_during_training'] is False
    assert report['configuration_validated'] is False
    assert os.path.exists('artifacts/checkpoint_compatibility.json')


def test_checkpoint_sha256_matches_file():
    cfg = load_config('config.yaml')
    validator = CheckpointValidator(cfg)
    report = validator.validate()
    # Recalcular SHA256 manualmente
    import hashlib
    h = hashlib.sha256()
    with open(cfg.checkpoint_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    assert report['sha256'] == h.hexdigest()
