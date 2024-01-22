from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
import pytorch3d.transforms

from mppiisaac.utils.config_store import ExampleConfig


class Objective(object):
    def __init__(self, cfg, device):
        self.device = device

    def compute_cost(self, sim):
        robot_pos = sim.robot_positions[:, 0, :2]
        rel_robot_pos = robot_pos - torch.tensor([2.0, 2.0], device=self.device)
        euclidean_cost = torch.norm(rel_robot_pos, dim=1)

        robot_height = sim.robot_positions[:, 0, 2]
        height_cost = torch.abs(robot_height - 1.0)

        return euclidean_cost + height_cost


@hydra.main(version_base=None, config_path="../conf", config_name="config_anymal")
def run_anymal(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
    prior = None
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_anymal()
