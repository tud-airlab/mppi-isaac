import torch

def omnidirectional_point_robot_dynamics(states: torch.Tensor, actions: torch.Tensor, t: int) -> torch.Tensor:
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]

    dt = 0.05

    new_x = x + actions[:, 0] * dt
    new_y = y + actions[:, 1] * dt
    new_theta = theta + actions[:, 2] * dt

    new_states = torch.stack([new_x, new_y, new_theta], dim=1)
    return new_states