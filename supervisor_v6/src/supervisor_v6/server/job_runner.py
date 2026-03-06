from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .event_bus import EventSink
from .models import CreateJobRequest, JobStatus, Artifact


@dataclass
class JobContext:
    job_id: str
    request: CreateJobRequest
    sink: EventSink
    status: JobStatus
    stop_flag: threading.Event


class JobRunner:
    """Runs jobs in background threads and emits structured events."""

    def __init__(self):
        self._jobs: dict[str, JobContext] = {}
        self._lock = threading.Lock()

    def create_job(self, req: CreateJobRequest) -> JobContext:
        job_id = time.strftime("job_%Y%m%d_%H%M%S_") + hex(int(time.time() * 1e6))[-6:]
        out_dir = str(Path(req.out_dir))
        sink = EventSink(job_id=job_id, out_dir=out_dir)

        # Persist request for reproducibility
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "job.json").write_text(req.model_dump_json(indent=2), encoding="utf-8")

        status = JobStatus(job_id=job_id, state="queued", stage="init", message="queued", out_dir=out_dir)
        ctx = JobContext(job_id=job_id, request=req, sink=sink, status=status, stop_flag=threading.Event())

        with self._lock:
            self._jobs[job_id] = ctx

        t = threading.Thread(target=self._run_job, args=(ctx,), daemon=True)
        t.start()
        return ctx

    def get(self, job_id: str) -> Optional[JobContext]:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self, job_id: str) -> bool:
        ctx = self.get(job_id)
        if not ctx:
            return False
        ctx.stop_flag.set()
        ctx.sink.emit(stage=ctx.status.stage, kind="warning", message="stop requested")
        ctx.status.state = "stopped"
        return True

    def _artifact(self, ctx: JobContext, name: str, path: str, mime: str | None = None, meta: dict | None = None):
        art = Artifact(name=name, path=path, mime=mime, meta=meta or {})
        ctx.status.artifacts.append(art)
        ctx.sink.emit(stage=ctx.status.stage, kind="artifact", message=name, data=art.model_dump())

    def _run_job(self, ctx: JobContext):
        ctx.status.state = "running"
        ctx.status.stage = "init"
        ctx.status.message = "starting"
        ctx.sink.emit(stage="init", kind="stage_started", message="job started")

        try:
            # NOTE: This is a UI-first runner. The actual RL/robot pipeline stays in supervisor_v6.* modules.
            # We intentionally keep the job runner thin and stage-oriented.

            # Stage: validate
            if ctx.stop_flag.is_set():
                return
            ctx.status.stage = "validate"
            ctx.sink.emit(stage="validate", kind="stage_started", message="validating request")
            time.sleep(0.1)
            ctx.sink.emit(stage="validate", kind="stage_finished", message="ok")

            # Stage: placeholder pipeline
            # Wire this to SupervisorAgentV6 + RealWorldIntegrator next.
            for stage, msg in [
                ("connect_robot", "connecting robot interface"),
                ("discover_cameras", "discovering cameras"),
                ("record_motion", "recording motion dataset"),
                ("reconstruct_world", "reconstructing scene"),
                ("build_urdf", "building URDF"),
                ("convert_usd", "converting to USD"),
            ]:
                if ctx.stop_flag.is_set():
                    ctx.sink.emit(stage=stage, kind="warning", message="stopped")
                    return
                ctx.status.stage = stage
                ctx.sink.emit(stage=stage, kind="stage_started", message=msg)
                # Progress ticks
                for i in range(5):
                    if ctx.stop_flag.is_set():
                        break
                    ctx.sink.emit(stage=stage, kind="progress", pct=(i + 1) * 20, message=f"{msg} ({i+1}/5)")
                    time.sleep(0.1)
                ctx.sink.emit(stage=stage, kind="stage_finished", message="done")

            # Final
            ctx.status.stage = "done"
            ctx.status.state = "finished"
            ctx.status.message = "finished"
            ctx.sink.emit(stage="done", kind="job_finished", message="job finished")

        except Exception as e:
            ctx.status.state = "error"
            ctx.status.error = str(e)
            tb = traceback.format_exc()
            ctx.sink.emit(stage=ctx.status.stage, kind="error", message=str(e), data={"trace": tb})
