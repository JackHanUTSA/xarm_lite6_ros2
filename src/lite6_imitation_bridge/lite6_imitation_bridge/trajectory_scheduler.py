from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


class SchedulerStop(RuntimeError):
    pass


@dataclass(frozen=True)
class JointTarget:
    frame_index: int
    joint_positions: list[float]
    source_record: dict[str, Any]


@dataclass(frozen=True)
class ScheduledStep:
    sequence_index: int
    frame_index: int
    joint_positions: list[float]
    scheduled_time_sec: float
    source_record: dict[str, Any]


@dataclass(frozen=True)
class LoadedJointTargetFile:
    source_path: str
    executable_targets: list[JointTarget]
    rejections: list[dict[str, Any]]


@dataclass(frozen=True)
class ScheduleRunResult:
    completed_steps: int
    stop_reason: str
    last_completed_frame_index: int | None


def _coerce_joint_positions(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 6:
        return None
    joint_positions = [float(item) for item in value]
    if not all(math.isfinite(item) for item in joint_positions):
        return None
    return joint_positions


def _rejection_reason(record: dict[str, Any]) -> str:
    reasons = [str(reason) for reason in record.get("rejection_reasons", []) if str(reason).strip()]
    if reasons:
        return f"record marked invalid: {'; '.join(reasons)}"
    return "record marked invalid"


def load_joint_target_file(source_path: str | Path) -> LoadedJointTargetFile:
    resolved_path = Path(source_path).expanduser().resolve()
    executable_targets: list[JointTarget] = []
    rejections: list[dict[str, Any]] = []

    for line_number, line in enumerate(resolved_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        frame_index = int(record.get("frame_index", len(executable_targets) + len(rejections)))
        if not bool(record.get("valid", False)):
            rejections.append({"frame_index": frame_index, "reason": _rejection_reason(record)})
            continue
        joint_positions = _coerce_joint_positions(record.get("joint_positions"))
        if joint_positions is None:
            rejections.append(
                {"frame_index": frame_index, "reason": "unsupported target representation: joint_positions required"}
            )
            continue
        executable_targets.append(
            JointTarget(frame_index=frame_index, joint_positions=joint_positions, source_record=dict(record))
        )

    return LoadedJointTargetFile(
        source_path=str(resolved_path), executable_targets=executable_targets, rejections=rejections
    )


def schedule_joint_targets(targets: Iterable[JointTarget | dict[str, Any]], rate_hz: float) -> list[ScheduledStep]:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be > 0")
    period_sec = 1.0 / float(rate_hz)
    scheduled: list[ScheduledStep] = []
    for sequence_index, target in enumerate(targets):
        if isinstance(target, JointTarget):
            frame_index = target.frame_index
            joint_positions = list(target.joint_positions)
            source_record = dict(target.source_record)
        else:
            frame_index = int(target["frame_index"])
            joint_positions = [float(value) for value in target["joint_positions"]]
            source_record = dict(target)
        scheduled.append(
            ScheduledStep(
                sequence_index=sequence_index,
                frame_index=frame_index,
                joint_positions=joint_positions,
                scheduled_time_sec=sequence_index * period_sec,
                source_record=source_record,
            )
        )
    return scheduled


def run_scheduled_steps(
    scheduled_steps: Sequence[ScheduledStep],
    publish_step: Callable[[ScheduledStep], None],
    health_check: Callable[[], dict[str, Any]],
    stop_requested: Callable[[], bool],
    monotonic_time: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ScheduleRunResult:
    start_time = monotonic_time()
    completed_steps = 0
    last_completed_frame_index: int | None = None

    for step in scheduled_steps:
        if stop_requested():
            return ScheduleRunResult(
                completed_steps=completed_steps,
                stop_reason="stop requested",
                last_completed_frame_index=last_completed_frame_index,
            )

        status = dict(health_check() or {})
        if not bool(status.get("ready", False)):
            return ScheduleRunResult(
                completed_steps=completed_steps,
                stop_reason=str(status.get("reason", "motion status not ready")),
                last_completed_frame_index=last_completed_frame_index,
            )

        target_time = start_time + step.scheduled_time_sec
        now = monotonic_time()
        if target_time > now:
            sleep_fn(target_time - now)

        if stop_requested():
            return ScheduleRunResult(
                completed_steps=completed_steps,
                stop_reason="stop requested",
                last_completed_frame_index=last_completed_frame_index,
            )

        status = dict(health_check() or {})
        if not bool(status.get("ready", False)):
            return ScheduleRunResult(
                completed_steps=completed_steps,
                stop_reason=str(status.get("reason", "motion status not ready")),
                last_completed_frame_index=last_completed_frame_index,
            )

        publish_step(step)
        completed_steps += 1
        last_completed_frame_index = step.frame_index

    return ScheduleRunResult(
        completed_steps=completed_steps,
        stop_reason="complete",
        last_completed_frame_index=last_completed_frame_index,
    )
