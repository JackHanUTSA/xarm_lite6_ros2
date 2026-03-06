from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .job_runner import JobRunner
from .models import CreateJobRequest, JobStatus, StopJobResponse


app = FastAPI(title="Supervisor Agent V6", version="0.0.1")
runner = JobRunner()


@app.post("/v6/jobs", response_model=JobStatus)
def create_job(req: CreateJobRequest):
    ctx = runner.create_job(req)
    return ctx.status


@app.get("/v6/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    ctx = runner.get(job_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="job not found")
    return ctx.status


@app.post("/v6/jobs/{job_id}/stop", response_model=StopJobResponse)
def stop_job(job_id: str):
    ok = runner.stop(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return StopJobResponse(job_id=job_id, ok=True)


@app.get("/v6/jobs/{job_id}/events")
def sse_events(job_id: str):
    ctx = runner.get(job_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="job not found")

    def gen() -> Iterator[bytes]:
        # Basic SSE loop.
        # We also send a keepalive comment every ~15s so proxies don't close idle streams.
        last_ping = time.time()
        for ev in ctx.sink.subscribe():
            now = time.time()
            if now - last_ping > 15:
                yield b": keepalive\n\n"
                last_ping = now
            payload = json.dumps(ev.to_dict(), ensure_ascii=False)
            yield ("data: " + payload + "\n\n").encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")
