import importlib
import sys
import types


def _install_gui_stubs():
    pyqt5 = types.ModuleType("PyQt5")
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtcore.QThread = type("QThread", (), {})
    qtcore.pyqtSignal = lambda *args, **kwargs: None
    qtcore.Qt = types.SimpleNamespace(Horizontal=1)
    qtwidgets.QWidget = type("QWidget", (), {})

    sys.modules["PyQt5"] = pyqt5
    sys.modules["PyQt5.QtCore"] = qtcore
    sys.modules["PyQt5.QtWidgets"] = qtwidgets

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.ok = lambda: False
    rclpy.try_shutdown = lambda: None
    rclpy.spin_until_future_complete = lambda *args, **kwargs: None
    rclpy.executors = types.SimpleNamespace(SingleThreadedExecutor=type("SingleThreadedExecutor", (), {}))
    sys.modules["rclpy"] = rclpy

    node_module = types.ModuleType("rclpy.node")
    node_module.Node = type("Node", (), {})
    sys.modules["rclpy.node"] = node_module

    qos_module = types.ModuleType("rclpy.qos")
    qos_module.QoSProfile = type("QoSProfile", (), {"__init__": lambda self, depth=1: None})
    qos_module.QoSReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    qos_module.QoSDurabilityPolicy = types.SimpleNamespace(VOLATILE=1)
    sys.modules["rclpy.qos"] = qos_module

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = type("JointState", (), {})
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {})
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

    std_srvs = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.Trigger = type("Trigger", (), {"Request": type("Request", (), {})})
    sys.modules["std_srvs"] = std_srvs
    sys.modules["std_srvs.srv"] = std_srvs_srv


def test_format_preview_summary_label_mentions_preview_only():
    _install_gui_stubs()
    gui = importlib.import_module("lite6_gui.gui")

    text = gui.format_preview_summary_label(
        {
            "frame_count": 3,
            "valid_frame_count": 2,
            "invalid_frame_count": 1,
            "clamped_frame_count": 1,
            "frame_index_range": [10, 12],
        }
    )

    assert "Preview only" in text
    assert "frames=3" in text
    assert "valid=2" in text
    assert "range=10-12" in text
