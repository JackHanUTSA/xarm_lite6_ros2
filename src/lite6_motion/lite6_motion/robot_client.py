from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class VendorServiceCall:
    service_name: str
    payload: Dict[str, object]


class Lite6RobotClient:
    def __init__(self, default_speed: float = 0.25, default_acc: float = 0.5, default_timeout: float = 15.0):
        self.default_speed = default_speed
        self.default_acc = default_acc
        self.default_timeout = default_timeout

    def prepare_sequence(self) -> List[VendorServiceCall]:
        return [
            VendorServiceCall('/ufactory/clean_error', {}),
            VendorServiceCall('/ufactory/clean_warn', {}),
            VendorServiceCall('/ufactory/motion_enable', {'id': 8, 'data': 1}),
            VendorServiceCall('/ufactory/set_mode', {'data': 0}),
            VendorServiceCall('/ufactory/set_state', {'data': 0}),
        ]

    def build_move_joint_payload(
        self,
        angles: List[float],
        speed: float | None = None,
        acc: float | None = None,
        timeout: float | None = None,
    ) -> Dict[str, object]:
        if len(angles) != 6:
            raise ValueError('Lite6 move expects exactly 6 joint angles')
        return {
            'angles': [float(value) for value in angles],
            'speed': float(self.default_speed if speed is None else speed),
            'acc': float(self.default_acc if acc is None else acc),
            'mvtime': 0.0,
            'wait': False,
            'timeout': float(self.default_timeout if timeout is None else timeout),
            'radius': -1.0,
            'relative': False,
        }

    def build_home_payload(self) -> Dict[str, object]:
        return self.build_move_joint_payload([0.0] * 6)
