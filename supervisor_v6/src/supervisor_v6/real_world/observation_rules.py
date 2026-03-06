from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class MotionRule:
    name: str
    description: str
    predicate: Callable[[np.ndarray, np.ndarray, Dict], bool]
    critical: bool = True


class ObservationRulesEngine:
    """Registry of motion observation rules; validates episodes before storing."""

    DEFAULT_RULES: List[MotionRule] = [
        MotionRule(
            name="min_motion_amplitude",
            description="Episode must show ≥ 0.05 rad joint displacement.",
            predicate=lambda cmd, js, _: float(np.max(np.abs(js[-1] - js[0]))) >= 0.05,
            critical=True,
        ),
        MotionRule(
            name="no_joint_limit_violation",
            description="No joint must exceed ±2.9 rad during episode.",
            predicate=lambda cmd, js, _: bool(np.all(np.abs(js) < 2.9)),
            critical=True,
        ),
        MotionRule(
            name="command_motion_correlation",
            description="Commanded direction must match observed motion direction.",
            predicate=lambda cmd, js, _: bool(np.dot(cmd, js[-1] - js[0]) > 0),
            critical=False,
        ),
        MotionRule(
            name="all_cameras_present",
            description="At least 2 cameras must have valid frames in episode.",
            predicate=lambda cmd, js, frames: sum(v is not None for v in frames.values()) >= 2,
            critical=True,
        ),
        MotionRule(
            name="velocity_bounded",
            description="Joint velocity must stay below 3.0 rad/s (safety filter).",
            predicate=lambda cmd, js, _: bool(np.all(np.abs(np.diff(js, axis=0)) < 3.0 * 0.033)),
            critical=True,
        ),
    ]

    def __init__(self, extra_rules: List[MotionRule] | None = None):
        self.rules: List[MotionRule] = list(self.DEFAULT_RULES)
        if extra_rules:
            self.rules.extend(extra_rules)

    def add_rule(self, rule: MotionRule):
        self.rules.append(rule)

    def remove_rule(self, name: str):
        self.rules = [r for r in self.rules if r.name != name]

    def validate(self, command_vec: np.ndarray, joint_states: np.ndarray, frames: Dict) -> Tuple[bool, List[str]]:
        failed: List[str] = []
        critical_fail = False
        for rule in self.rules:
            try:
                ok = rule.predicate(command_vec, joint_states, frames)
            except Exception:
                ok = False
            if not ok:
                failed.append(rule.name)
                if rule.critical:
                    critical_fail = True
        return (not critical_fail), failed

    def print_rules(self):
        for r in self.rules:
            flag = "(critical)" if r.critical else "(soft)"
            print(f"  {flag:10s} {r.name:32s} — {r.description}")
