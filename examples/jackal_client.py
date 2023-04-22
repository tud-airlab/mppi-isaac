import gym
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import os
import torch
import zerorpc

from mppiisaac.utils.config_store import ExampleConfig


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = sim.robot_positions[:, 0, :2]
#        print(pos[0], self.nav_goal)
        err = torch.linalg.norm(pos - self.nav_goal, axis=1)
        cost = torch.clamp(
            err - 0.05, min=0, max=1999
        )
        return err * 1.5


@hydra.main(version_base=None, config_path="../conf", config_name="config_jackal_robot")
def run_jackal_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
    prior = None
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_jackal_robot()
