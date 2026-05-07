from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epuck2_sim_real_control.session_manifest import SessionManifest, load_manifest, save_manifest


def test_save_and_load_manifest_roundtrip(tmp_path: Path):
    path = tmp_path / 'session.yaml'
    manifest = SessionManifest(mode='real', enable_real_robot=True)
    saved = save_manifest(path, manifest)
    loaded = load_manifest(saved)
    assert loaded.mode == 'real'
    assert loaded.enable_real_robot is True
    assert loaded.project_name == 'epuck2_sim_real_control'


def test_load_manifest_uses_defaults_for_missing_keys(tmp_path: Path):
    path = tmp_path / 'session.yaml'
    path.write_text('mode: sim\n')
    loaded = load_manifest(path)
    assert loaded.mode == 'sim'
    assert loaded.sim_backend == 'webots'
    assert loaded.require_operator_confirm is True
