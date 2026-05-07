from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SessionManifest:
    project_name: str = 'epuck2_sim_real_control'
    robot_name: str = 'epuck2'
    mode: str = 'sim'
    sim_backend: str = 'webots'
    real_backend: str = 'epuck2_driver'
    policy_topic: str = '/epuck2/cmd_vel'
    observation_topic: str = '/epuck2/odom'
    enable_real_robot: bool = False
    require_operator_confirm: bool = True
    notes: list[str] = field(default_factory=lambda: [
        'Start in simulation first.',
        'Gate real-robot motion behind operator confirmation.',
        'Keep sim and real topic contracts aligned before transfer.',
    ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MANIFEST = SessionManifest()


def load_manifest(path: str | Path) -> SessionManifest:
    resolved = Path(path).expanduser().resolve()
    data = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError('session manifest must be a YAML mapping')
    merged = DEFAULT_MANIFEST.to_dict() | data
    return SessionManifest(**merged)


def save_manifest(path: str | Path, manifest: SessionManifest) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    return resolved
