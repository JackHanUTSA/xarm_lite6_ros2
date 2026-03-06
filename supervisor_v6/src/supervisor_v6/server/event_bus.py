from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


@dataclass
class Event:
    ts: float
    job_id: str
    stage: str
    kind: str  # stage_started|progress|log|warning|artifact|stage_finished|error|job_finished
    message: str = ""
    pct: Optional[float] = None
    data: Dict[str, Any] | None = None

    def to_dict(self) -> dict:
        d = {
            "ts": self.ts,
            "job_id": self.job_id,
            "stage": self.stage,
            "kind": self.kind,
            "message": self.message,
        }
        if self.pct is not None:
            d["pct"] = self.pct
        if self.data is not None:
            d["data"] = self.data
        return d


class EventSink:
    """Per-job event sink.

    - Keeps an in-memory queue for SSE subscribers
    - Appends to out_dir/events.jsonl for persistence
    """

    def __init__(self, job_id: str, out_dir: str):
        self.job_id = job_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.out_dir / "events.jsonl"
        self._q: queue.Queue[Event] = queue.Queue(maxsize=10_000)
        self._lock = threading.Lock()

    def emit(self, stage: str, kind: str, message: str = "", pct: float | None = None, data: dict | None = None):
        ev = Event(ts=time.time(), job_id=self.job_id, stage=stage, kind=kind, message=message, pct=pct, data=data)
        line = json.dumps(ev.to_dict(), ensure_ascii=False)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        try:
            self._q.put_nowait(ev)
        except queue.Full:
            # Drop if overwhelmed; persistence file still has the event.
            pass

    def subscribe(self) -> Iterator[Event]:
        """Blocking iterator of new events."""
        while True:
            ev = self._q.get()
            yield ev
