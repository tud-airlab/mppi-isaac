
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
import pytorch3d.transforms

from mppiisaac.utils.config_store import ExampleConfig

class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.block_goal = torch.tensor([0.5, 0, 0.138], device=cfg.mppi.device)
        self.ort_goal = torch.tensor([1, 0, 0, 0], device=cfg.mppi.device)
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)
        #self.block_goal_ort = torch.tensor([-0.5, 0.5, -0.5, 0.5], device=cfg.mppi.device)
        self.block_goal_ort = torch.tensor([0, 0, 0, 1.0], device=cfg.mppi.device)
        self.block_goal_ort = torch.tensor([0, 0, 0.7071068, 0.7071068], device=cfg.mppi.device)

        self.w_robot_to_block_pos= 10#0.2
        self.w_block_to_goal_pos=  100#2.0 
        self.w_block_to_goal_ort=  10#0.2
        self.w_ee_hover=           60#5
        self.w_ee_align=           0#1
        self.w_ee_contact=         0.1#0.02
        self.w_push_align=         5#1
        self.ee_hover_height=      0.14#0.14

    def orientation_error(self, q1, q2_batch):
        """
        Computes the orientation error between a single quaternion and a batch of quaternions.
        
        Parameters:
        -----------
        q1 : torch.Tensor
            A tensor of shape (4,) representing the first quaternion.
        q2_batch : torch.Tensor
            An tensor of shape (batch_size, 4) representing the second set of quaternions.
            
        Returns:
        --------
        error_batch : torch.Tensor
            An tensor of shape (batch_size,) containing the orientation error between the first quaternion and each quaternion in the batch.
        """
        
        # Expand the first quaternion to match the batch size of the second quaternion
        q1_batch = q1.expand(q2_batch.shape[0], -1)
        
        # Normalize the quaternions
        q1_batch = q1_batch / torch.norm(q1_batch, dim=1, keepdim=True)
        q2_batch = q2_batch / torch.norm(q2_batch, dim=1, keepdim=True)
        
        # Compute the dot product between the quaternions in the batch
        dot_product_batch = torch.sum(q1_batch * q2_batch, dim=1)
        
        # Compute the angle between the quaternions in the batch
        # chatgpt
        #angle_batch = 2 * torch.acos(torch.abs(dot_product_batch))
        # method2
        angle_batch = torch.acos(2*torch.square(dot_product_batch)-1)
        # method 3
        # angle_batch = 1 - torch.square(dot_product_batch)
        # Return the orientation error for each quaternion in the batch
        error_batch = angle_batch
        
        return error_batch

    def compute_cost(self, sim):
        ee_index = sim.robot_rigid_body_ee_idx
        block_index = 2
        r_pos = sim.rigid_body_state[:, ee_index, :3]
        r_ort = sim.rigid_body_state[:, ee_index, 3:7]
        ee_height = r_pos[:, 2]
        block_pos = sim.root_state[:, block_index, :3]
        block_ort = sim.root_state[:, block_index, 3:7]
        robot_to_block = r_pos - block_pos
        block_to_goal = self.block_goal[0:2] - block_pos[:, 0:2]

        block_yaws = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ort), "ZYX")[:, -1]
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")

        # Costs
        robot_to_block_cost = torch.linalg.norm(robot_to_block[:, 0:2], axis=1)
        block_to_goal_pos_cost = torch.linalg.norm(block_to_goal, axis=1)
        block_to_goal_ort_cost = self.orientation_error(self.block_goal_ort, block_ort)
        ee_hover_cost = torch.abs(ee_height - self.ee_hover_height)
        push_align_cost = torch.sum(robot_to_block[:, 0:2] * block_to_goal, 1) / (
            robot_to_block_cost * block_to_goal_pos_cost
        )
        #ee_align_cost = torch.linalg.norm(r_ort - self.ort_goal, axis=1)
        #ee_align_cost = torch.pow(block_yaws, 2)
        ee_align_cost = torch.linalg.norm(robot_euler - self.ort_goal_euler, axis=1)

        # minimize contact
        # Only z component should be added here
        net_cf = torch.sum(
            torch.abs(
                torch.cat(
                    (
                        sim.net_cf[:, 0].unsqueeze(1),
                        sim.net_cf[:, 1].unsqueeze(1),
                        sim.net_cf[:, 2].unsqueeze(1),
                    ),
                    1,
                )
            ),
            1,
        )
        contact_cost = torch.abs(
            net_cf.reshape([block_pos.size()[0], -1])[:, block_index]
        )

        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_cost
            + self.w_block_to_goal_pos * block_to_goal_pos_cost
            + self.w_block_to_goal_ort * block_to_goal_ort_cost
            + self.w_ee_hover * ee_hover_cost 
            + self.w_ee_align * ee_align_cost
            + self.w_ee_contact * contact_cost
            + self.w_push_align * push_align_cost
        )

        return total_cost

@hydra.main(version_base=None, config_path="../conf", config_name="config_panda_push")
def run_panda_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
    if cfg.mppi.use_priors == True:
        prior = FabricsPandaPrior(cfg)
    else:
        prior = None
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__ == "__main__":
    run_panda_robot()
