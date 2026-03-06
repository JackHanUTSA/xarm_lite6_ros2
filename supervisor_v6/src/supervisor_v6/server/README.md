# supervisor_v6.server

UI-facing surface area.

## Run

```bash
python3 -m uvicorn supervisor_v6.server.api:app --host 0.0.0.0 --port 8000
```

## Try

Create job:

```bash
curl -X POST http://localhost:8000/v6/jobs \
  -H 'content-type: application/json' \
  -d '{"robot":"ros2:/xarm","name":"lite6","mode":"full","out_dir":"robot_assets/lite6"}'
```

Then stream events:

```bash
curl -N http://localhost:8000/v6/jobs/<job_id>/events
```
