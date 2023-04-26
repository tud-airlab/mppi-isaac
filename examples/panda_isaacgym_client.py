
import gym
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import hydra
from omegaconf import OmegaConf
import os
import torch
from mppiisaac.priors.fabrics_panda import FabricsPandaPrior
import zerorpc

from mppiisaac.utils.config_store import ExampleConfig

class JointSpaceGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = torch.cat(
            (
                sim.dof_state[:, 0].unsqueeze(1),
                sim.dof_state[:, 2].unsqueeze(1),
                sim.dof_state[:, 4].unsqueeze(1),
                sim.dof_state[:, 6].unsqueeze(1),
                sim.dof_state[:, 8].unsqueeze(1),
                sim.dof_state[:, 10].unsqueeze(1),
                sim.dof_state[:, 12].unsqueeze(1),
            ),
            1,
        )
        # dof_states = gym.acquire_dof_state_tensor(sim)
        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )


class EndEffectorGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.ort_goal = torch.tensor([1, 0, 0, 0], device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, :3]
        ort = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, 3:7]
        # dof_states = gym.acquire_dof_state_tensor(sim)

        reach_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)
        align_cost = torch.linalg.norm(ort - self.ort_goal, axis=1)
        return 10 * reach_cost + align_cost
        # return torch.clamp(
        #     torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        # )


@hydra.main(version_base=None, config_path="../conf", config_name="config_panda")
def run_panda_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = EndEffectorGoalObjective(cfg, cfg.mppi.device)
    # objective = JointSpaceGoalObjective(cfg, cfg.mppi.device)
    if cfg.mppi.use_priors == True:
        prior = FabricsPandaPrior(cfg)
    else:
        prior = None
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__ == "__main__":
    run_panda_robot()
