
import gym
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import hydra
from omegaconf import OmegaConf
import os
import torch
import zerorpc
import pytorch3d.transforms

from mppiisaac.utils.config_store import ExampleConfig

class Objective(object):
    def __init__(self, cfg, device):
        
        # Tuning of the weights for baseline 2
        self.w_robot_to_block_pos=  20
        self.w_block_to_goal_pos=   20
        self.w_ee_align=            1 #1
        self.w_collision=           0
        # Task configration for comparison with baselines
        self.ee_index = 12
        self.block_index = 1
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)

        self.block_goal_pose = torch.tensor([-1, -1, 0.6], device=cfg.mppi.device)

        self.obst_number = 2
        self.success = False
        self.count = 0

    def compute_metrics(self, block_pos):
        Ex = torch.abs(self.block_goal_pose[0]-block_pos[-1,0])
        Ey = torch.abs(self.block_goal_pose[1]-block_pos[-1,1])
        return Ex, Ey
    
    def compute_cost(self, sim):
        r_pos = sim.rigid_body_state[:, self.ee_index, :3]
        r_ort = sim.rigid_body_state[:, self.ee_index, 3:7]
        block_pos = sim.root_state[:, self.block_index, :3]

        # Distances robot
        robot_to_block = r_pos - block_pos
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")   

        # Distances block
        block_to_goal =  self.block_goal_pose - block_pos
        
        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        block_to_goal_pos = torch.linalg.norm(block_to_goal, axis = 1)

        # Posture costs
        ee_align = torch.linalg.norm(robot_euler[:,0:2] - self.ort_goal_euler[0:2], axis=1)
       
         # Collision avoidance
        contact_f = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        coll = torch.sum(contact_f.reshape([sim.num_envs, int(contact_f.size(dim=0)/sim.num_envs)])[:, (sim.num_bodies - self.obst_number):sim.num_bodies], 1)

        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_dist
            + self.w_block_to_goal_pos * block_to_goal_pos
            + self.w_ee_align * ee_align
            + self.w_collision * coll
        )

        return total_cost

@hydra.main(version_base=None, config_path="../conf", config_name="config_omnipanda")
def run_omnipanda_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
   
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__ == "__main__":
    run_omnipanda_robot()
