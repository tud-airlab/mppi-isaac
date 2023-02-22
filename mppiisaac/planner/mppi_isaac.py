from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi import MPPIPlanner

import torch

torch.set_printoptions(precision=2, sci_mode=False)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective):
        self.cfg = cfg
        self.objective = objective

        self.mppi = MPPIPlanner(
            cfg.mppi, cfg.nx, dynamics=self.dynamics, running_cost=self.running_cost
        )

        self.sim = IsaacGymWrapper(
            cfg.isaacgym, cfg.urdf_file, cfg.fix_base, cfg.flip_visual, num_envs=cfg.mppi.num_samples
        )

        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

    def dynamics(self, _, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the state is already saved in the simulator itself, so we ignore it.
        # Note: t is an unused step dependent dynamics variable

        self.sim.set_dof_velocity_target_tensor(u)
        self.sim.step()

        return (self.state_place_holder, u)

    def running_cost(self, _):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the state is already saved and accesible in the simulator itself, so we ignore it and pass a handle to the simulator.
        return self.objective.compute_cost(self.sim)

    def compute_action(self, q, qdot, obst=None):
        if obst:
            # NOTE: for now this updates based on id in the list of obstacles
            self.sim.update_root_state_tensor_by_obstacles(obst)

        reordered_state = []
        for i in range(int(self.cfg.nx / 2)):
            reordered_state.append(q[i])
            reordered_state.append(qdot[i])
        state = (
            torch.tensor(reordered_state).type(torch.float32).to(self.cfg.mppi.device)
        )  # [x, vx, y, vy]
        state = state.repeat(self.sim.num_envs, 1)
        self.sim.set_dof_state_tensor(state)

        actions = self.mppi.command(self.state_place_holder).cpu()
        return actions
