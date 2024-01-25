import torch


def quaternion_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        2.0 * (quat[:, -1] * quat[:, 2] + quat[:, 0] * quat[:, 1]),
        quat[:, -1] * quat[:, -1]
        + quat[:, 0] * quat[:, 0]
        - quat[:, 1] * quat[:, 1]
        - quat[:, 2] * quat[:, 2],
    )
