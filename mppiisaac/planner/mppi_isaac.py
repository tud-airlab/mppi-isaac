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
            cfg.isaacgym,
            cfg.urdf_file,
            cfg.fix_base,
            cfg.flip_visual,
            num_envs=cfg.mppi.num_samples,
        )

        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

    def _ik(self, u):
        r = self.cfg.wheel_radius
        L = self.cfg.wheel_base
        wheel_sets = self.cfg.wheel_count // 2

        # Diff drive fk
        u_ik = u.clone()
        u_ik[:, 0] = (u[:, 0] / r) - ((L * u[:, 1]) / (2 * r))
        u_ik[:, 1] = (u[:, 0] / r) + ((L * u[:, 1]) / (2 * r))

        if wheel_sets > 1:
            u_ik = u_ik.repeat(1, wheel_sets)

        return u_ik

    def dynamics(self, _, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the state is already saved in the simulator itself, so we ignore it.
        # Note: t is an unused step dependent dynamics variable

        if self.cfg.differential_drive:
            u_ik = self._ik(u)
            self.sim.set_dof_velocity_target_tensor(u_ik)
        else:
            self.sim.set_dof_velocity_target_tensor(u)

        self.sim.step()

        return (self.state_place_holder, u)

    def running_cost(self, _):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the state is already saved and accesible in the simulator itself, so we ignore it and pass a handle to the simulator.
        return self.objective.compute_cost(self.sim)

    def compute_action(self, q, qdot, obst=None):
        self.sim.reset_root_state()

        if obst:
            # NOTE: for now this updates based on id in the list of obstacles
            self.sim.update_root_state_tensor_by_obstacles(obst)

        # Deal with non fixed base link robots, that we need to reset the root_state position / velocity of.
        # Currently only differential drive bases are non fixed. We also have to deal with the kinematic transforms.
        if self.cfg.differential_drive:
            pos = q[:3]
            vel = qdot[:3]
            self.sim.update_root_state_tensor_robot(pos, vel)

            if len(q) > 3:
                # Add the dof state of the wheels (assuming 2 wheel here)
                q = [0] * self.cfg.wheel_count + q[3:]
                # TODO: reset the dof_state velocity of the wheels based on the measured ground velocity
                qdot = [0] * self.cfg.wheel_count + q[3:]
            else:
                q = [0] * self.cfg.wheel_count
                qdot = [0] * self.cfg.wheel_count

            reordered_state = []
            for i in range(int(self.cfg.nx / 2)+(self.cfg.wheel_count-2)):
                reordered_state.append(q[i])
                reordered_state.append(qdot[i])
            state = (
                torch.tensor(reordered_state).type(torch.float32).to(self.cfg.mppi.device)
            )  # [x, vx, y, vy]

        else:
            reordered_state = []
            for i in range(int(self.cfg.nx / 2)):
                reordered_state.append(q[i])
                reordered_state.append(qdot[i])
            state = (
                torch.tensor(reordered_state).type(torch.float32).to(self.cfg.mppi.device)
            )  # [x, vx, y, vy]

        state = state.repeat(self.sim.num_envs, 1)
        self.sim.set_dof_state_tensor(state)

        self.sim.save_root_state()
        actions = self.mppi.command(self.state_place_holder).cpu()
        return actions
