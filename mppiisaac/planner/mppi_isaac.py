from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass
import torch

from mppiisaac.planner.mppi import MPPIPlanner
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper



class MPPIisaacPlanner(MPPIPlanner):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg):
        super().__init__(cfg.mppi)

        self.sim = IsaacGymWrapper(cfg.isaacgym, num_envs=cfg.mppi.num_samples)

        self.actor_positions = self.sim.root_state[:, 0:2]  # [x, y]
        self.actor_velocities = self.sim.root_state[:, 3:5]  # [vx, vy]

        # nav_goal
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)


    def dynamics(self, state, u, t=None):
        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step()

        # return torch.cat((self.actor_positions, self.actor_velocities), axis=1), u
        return self.sim.dof_state.view(-1, 4), u

    def running_cost(self, state, u):
        state_pos = torch.cat((state[:, 0].unsqueeze(1), state[:, 2].unsqueeze(1)), 1)

        return self._get_navigation_cost(state_pos, self.nav_goal)

    @staticmethod
    def _get_navigation_cost(pos, goal_pos):
        return torch.clamp(
            torch.linalg.norm(pos - goal_pos, axis=1) - 0.05, min=0, max=1999
        )

    def compute_action(self, q, qdot, obst=None):
        if obst:
            # NOTE: for now this updates based on id in the list of obstacles
            self.sim.update_root_state_tensor_by_obstacles(obst)

        state = (
            torch.tensor([q[0], qdot[0], q[1], qdot[1]])
            .type(torch.float32)
            .to(self.cfg.device)
        )  # [x, vx, y, vy]
        state = state.repeat(self.K, 1)
        self.sim.set_dof_state_tensor(state)

        c = self.command(
            torch.cat((self.actor_positions, self.actor_velocities), axis=1)
        )
        return c.cpu()
