# supervisor_v6 scaffold

This is an auto-generated scaffold based on `Supervisor Agent V6` spec (docx -> text).

Entry points:

### Python
```python
from supervisor_v6.supervisor_agent_v6 import SupervisorAgentV6
sup = SupervisorAgentV6()

# Use skip flags to avoid hanging during early wiring.
spec = sup.setup_from_real_world("ros2:/xarm", skip_cameras=True, skip_recording=True)
print(spec)
```

### UI/Dashboard server (FastAPI + SSE)
Run:
```bash
python3 -m uvicorn supervisor_v6.server.api:app --host 0.0.0.0 --port 8000
```

API:
- POST /v6/jobs
- GET  /v6/jobs/{job_id}
- GET  /v6/jobs/{job_id}/events   (Server-Sent Events)
- POST /v6/jobs/{job_id}/stop

Notes:
- Uses ROS2 by default (`ROS2RobotInterface`).
- Many functions are stubs/placeholders: fill in robot-specific details, Isaac integration, reconstruction.
