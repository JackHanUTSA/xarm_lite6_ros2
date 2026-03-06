from supervisor_v6.supervisor_agent_v6 import SupervisorAgentV6

if __name__ == "__main__":
    sup = SupervisorAgentV6()
    # Use skip flags so the demo doesn't hang on camera discovery unless you want it.
    spec = sup.setup_from_real_world("ros2:/xarm", robot_name="reconstructed_arm", skip_cameras=True, skip_recording=True)
    print(spec)
