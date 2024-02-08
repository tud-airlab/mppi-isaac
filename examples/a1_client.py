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
        self.desired_dof_state = torch.tensor([0., 0.,  0.7,  0, -1.0,  0] * 4)

    def compute_cost(self, sim):
        robot_pos = sim.robot_positions[:, 0, :2]
        rel_robot_pos = robot_pos - torch.tensor([0.0, 0.0], device=self.device)
        euclidean_cost = torch.sum(torch.square(rel_robot_pos))

        upward_quat = torch.tensor([0., 0., 0., 1.], device=self.device)
        orientation_cost = torch.sum(torch.square(sim.root_state[:,0,3:7] - upward_quat.expand_as(sim.root_state[:,0,3:7])), dim=1)

        # only penalize the pose
        pose_indices = torch.arange(0, 24, 2, device=self.device)  # indices for position states
        pose_cost = torch.sum(torch.square(sim.dof_state[:, pose_indices] - self.desired_dof_state[pose_indices].expand_as(sim.dof_state[:, pose_indices])), dim=1)

        robot_height = sim.robot_positions[:, 0, 2]
        height_cost = torch.square(robot_height - 0.35)

        return 100.*euclidean_cost + 100.0*height_cost + 1.0*pose_cost + 100.0*orientation_cost
        # return 1.0*pose_cost 


@hydra.main(version_base=None, config_path="../conf", config_name="config_a1")
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
