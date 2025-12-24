import json
import subprocess
from pathlib import Path


def test_validate_and_register_registers(monkeypatch, tmp_path):
    # create metrics file with high PR-AUC
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    mfile = metrics_dir / "metrics.json"
    mfile.write_text(json.dumps({"val_pr_auc": 0.5}))

    calls = {}

    def fake_check_call(cmd):
        calls['cmd'] = cmd

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    from importlib.machinery import SourceFileLoader
    V = SourceFileLoader("validate_and_register", "scripts/validate_and_register.py").load_module()

    ok = V.validate_and_register(
        "fake_run",
        run_dir=str(tmp_path),
        min_pr_auc=0.1,
        artifact_path="model/ctr_model_hashing.joblib",
    )
    assert ok is True
    assert 'cmd' in calls


def test_validate_and_register_does_not_register(monkeypatch, tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    mfile = metrics_dir / "metrics.json"
    mfile.write_text(json.dumps({"val_pr_auc": 0.01}))

    called = False

    def fake_check_call(cmd):
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    from importlib.machinery import SourceFileLoader
    V = SourceFileLoader("validate_and_register", "scripts/validate_and_register.py").load_module()

    ok = V.validate_and_register(
        "fake_run",
        run_dir=str(tmp_path),
        min_pr_auc=0.1,
        artifact_path="model/ctr_model_hashing.joblib",
    )
    assert ok is False
    assert called is False
