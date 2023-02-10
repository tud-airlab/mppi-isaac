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

    def __init__(self, cfg, objective):
        super().__init__(cfg.mppi, cfg.nx)

        self.sim = IsaacGymWrapper(cfg.isaacgym, cfg.urdf_file, num_envs=cfg.mppi.num_samples)

        self.actor_positions = self.sim.root_state[:, 0:2]  # [x, y]
        self.actor_velocities = self.sim.root_state[:, 3:5]  # [vx, vy]
        self.objective = objective


    def dynamics(self, state, u, t=None):
        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step()

        # return torch.cat((self.actor_positions, self.actor_velocities), axis=1), u
        num_rigid_bodies = int(self.sim.rigid_body_state.shape[0] / self.sim.num_envs)
        num_root_states = int(self.sim.root_state.shape[0] / self.sim.num_envs)
        return (
            self.sim.root_state.view(-1, num_root_states*13),
            self.sim.dof_state.view(-1, self.nx),
            self.sim.rigid_body_state.view(-1, num_rigid_bodies*13),
            u
        )

    def running_cost(self, root_state, dof_state, rigid_body_state):
        return self.objective.compute_cost(root_state, dof_state, rigid_body_state)

        return self._get_navigation_cost(state_pos, self.nav_goal)

    def compute_action(self, q, qdot, obst=None):
        if obst:
            # NOTE: for now this updates based on id in the list of obstacles
            self.sim.update_root_state_tensor_by_obstacles(obst)

        reordered_state = []
        for i in range(int(self.nx/2)):
            reordered_state.append(q[i])
            reordered_state.append(qdot[i])
        state = (
            torch.tensor(reordered_state)
            .type(torch.float32)
            .to(self.cfg.device)
        )  # [x, vx, y, vy]
        state = state.repeat(self.K, 1)
        self.sim.set_dof_state_tensor(state)

        c = self.command(
            torch.cat((self.actor_positions, self.actor_velocities), axis=1)
        )
        return c.cpu()
