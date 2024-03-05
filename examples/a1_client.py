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
        rel_robot_pos = robot_pos - torch.tensor([-1.0, 0.0], device=self.device)
        euclidean_cost = torch.sum(torch.square(rel_robot_pos))

        upward_quat = torch.tensor([0., 0., 0., 1.], device=self.device)
        orientation_cost = torch.sum(torch.square(sim.root_state[:,0,3:7] - upward_quat.expand_as(sim.root_state[:,0,3:7])), dim=1)

        vel_x_cost = torch.square(sim.root_state[:,0,7] + 0.2)

        # only penalize the pose
        pose_indices = torch.arange(0, 24, 2, device=self.device)  # indices for position states
        pose_cost = torch.sum(torch.square(sim.dof_state[:, pose_indices] - self.desired_dof_state[pose_indices].expand_as(sim.dof_state[:, pose_indices])), dim=1)

        robot_height = sim.robot_positions[:, 0, 2]
        height_cost = torch.square(robot_height - 0.35)

        # # Collision avoidance
        # contact_f = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        # coll = torch.sum(contact_f.reshape([sim.num_envs, int(contact_f.size(dim=0)/sim.num_envs)])[:, (sim.num_bodies - 1 - self.obst_number):sim.num_bodies], 1) # Consider obstacles

        # Collision avoidance
        contact_f = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        contact_f_reshaped = contact_f.reshape([sim.num_envs, int(contact_f.size(dim=0)/sim.num_envs)])

        # Indices for feet
        feet_indices = torch.tensor([6, 11, 16, 21], device=sim.device)

        # Consider all but feet
        non_feet_indices = torch.tensor([i for i in range(sim.num_bodies) if i not in feet_indices.tolist()], device=sim.device)
        coll = torch.sum(torch.index_select(contact_f_reshaped, 1, non_feet_indices), 1)

        return 100.*euclidean_cost + 10.0*height_cost + 0.5*pose_cost + 100.0*orientation_cost + 10.0*vel_x_cost + 10.0*coll
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
