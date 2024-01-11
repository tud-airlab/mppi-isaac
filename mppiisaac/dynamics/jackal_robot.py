import torch

def clip_actions(forward_velocity: torch.Tensor, rotational_velocity: torch.Tensor):
    max_linear_vel = 0.5
    max_rot_vel = 2
    
    # Clip forward and rotational velocities based on robot constraints
    forward_velocity = torch.clamp(forward_velocity, -max_linear_vel, max_linear_vel)
    rotational_velocity = torch.clamp(rotational_velocity, -max_rot_vel, max_rot_vel)
    return forward_velocity, rotational_velocity

def jackal_robot_dynamics(states: torch.Tensor, actions: torch.Tensor, dt: int) -> torch.Tensor:
    x, y, theta, vx, vy, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]

    actions[:,0], actions[:,1] = clip_actions(actions[:,0], actions[:,1])

    # Update velocity and position using bicycle model
    new_vx = actions[:,0] * torch.cos(theta)
    new_vy = actions[:,0] * torch.sin(theta)
    new_omega = actions[:,1]

    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_theta = theta + new_omega * dt

    new_states = torch.stack([new_x, new_y, new_theta, new_vx, new_vy, new_omega], dim=1)
    return new_states, actions