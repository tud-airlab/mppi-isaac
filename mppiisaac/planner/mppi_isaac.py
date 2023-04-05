from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi import MPPIPlanner
from typing import Callable, Optional
import io

from isaacgym import gymtorch
import torch

torch.set_printoptions(precision=2, sci_mode=False)


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):
        self.cfg = cfg
        self.objective = objective

        self.sim = IsaacGymWrapper(
            cfg.isaacgym,
            cfg.urdf_file,
            cfg.fix_base,
            cfg.flip_visual,
            robot_init_pos=cfg.initial_position,
            num_envs=cfg.mppi.num_samples,
            ee_link=cfg.ee_link,
            disable_gravity=cfg.disable_gravity,
        )

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
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

    def compute_action(self, q, qdot, obst=None, obst_tensor=None):
        self.sim.reset_root_state()

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
            for i in range(int(self.cfg.nx / 2) + (self.cfg.wheel_count - 2)):
                reordered_state.append(q[i])
                reordered_state.append(qdot[i])
            state = (
                torch.tensor(reordered_state)
                .type(torch.float32)
                .to(self.cfg.mppi.device)
            )  # [x, vx, y, vy]

        else:
            reordered_state = []
            for i in range(int(self.cfg.nx / 2)):
                reordered_state.append(q[i])
                reordered_state.append(qdot[i])
            state = (
                torch.tensor(reordered_state)
                .type(torch.float32)
                .to(self.cfg.mppi.device)
            )  # [x, vx, y, vy]

        state = state.repeat(self.sim.num_envs, 1)
        self.sim.set_dof_state_tensor(state)

        # NOTE: There are two different ways of updating obstacle root_states
        # Both update based on id in the list of obstacles
        if obst:
            self.sim.update_root_state_tensor_by_obstacles(obst)

        if obst_tensor:
            self.sim.update_root_state_tensor_by_obstacles_tensor(obst_tensor)

        self.sim.save_root_state()
        actions = self.mppi.command(self.state_place_holder).cpu()
        return actions

    def reset_rollout_sim(
        self, dof_state_tensor, root_state_tensor, rigid_body_state_tensor
    ):
        self.sim.ee_positions_buffer = []
        self.sim.dof_state[:] = bytes_to_torch(dof_state_tensor)
        self.sim.root_state[:] = bytes_to_torch(root_state_tensor)
        self.sim.rigid_body_state[:] = bytes_to_torch(rigid_body_state_tensor)

        self.sim.gym.set_dof_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.sim.dof_state))
        self.sim.gym.set_actor_root_state_tensor(self.sim.sim, gymtorch.unwrap_tensor(self.sim.root_state))

    def command(self):
        return torch_to_bytes(self.mppi.command(self.state_place_holder))

    def add_to_env(self, env_cfg_additions):
        self.sim.add_to_envs(env_cfg_additions)

    def get_rollouts(self):
        return torch_to_bytes(torch.stack(self.sim.ee_positions_buffer))

