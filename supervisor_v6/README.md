# supervisor_v6 scaffold

This is an auto-generated scaffold based on `Supervisor Agent V6` spec (docx -> text).

Entry point:

```python
from supervisor_v6.supervisor_agent_v6 import SupervisorAgentV6
sup = SupervisorAgentV6()
spec = sup.setup_from_real_world("ros2:/xarm")
print(spec)
```

Notes:
- Uses ROS2 by default (`ROS2RobotInterface`).
- Many functions are stubs/placeholders: fill in robot-specific details, Isaac integration, reconstruction.
