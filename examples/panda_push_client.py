
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
import time

from mppiisaac.utils.config_store import ExampleConfig

class Objective(object):
    def __init__(self, cfg, device):
        
        # Tuning of the weights for baseline 1 nd eal experiments
        # self.w_robot_to_block_pos= 1#2
        # self.w_block_to_goal_pos=  6#6.0 
        # self.w_block_to_goal_ort=  2#2.0
        # self.w_ee_hover=           8#5
        # self.w_ee_align=           0.5#0.5
        # self.w_push_align=         0.3#0.4
        # self.w_collision=          0.0

        # self.w_robot_to_block_pos= 5#2
        # self.w_block_to_goal_pos=  25#12.0 
        # self.w_block_to_goal_ort=  21#10.0
        # self.w_ee_hover=           30#5
        # self.w_ee_align=           .3#0.2
        # self.w_push_align=         45#4.2
        # self.w_collision=          0.0

        self.w_robot_to_block_pos= 5#2
        self.w_block_to_goal_pos=  25#12.0 
        self.w_block_to_goal_ort=  21#10.0
        self.w_ee_hover=           30#5
        self.w_ee_align=           .3#0.2
        self.w_push_align=         45#4.2
        self.w_collision=          0.0

    
        # Task configration for comparison with baselines
        self.ee_index = 9
        self.block_index = 1
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)
        self.ee_hover_height = 0.14

        self.block_goal_pose_emdn_0 = torch.tensor([0.5, 0.3, 0.5, 0.0, 0.0, 0.0, 1.0], device=cfg.mppi.device)
        self.block_goal_pose_emdn_1 = torch.tensor([0.4, 0.3, 0.5, 0, 0, -0.7071068, 0.7071068], device=cfg.mppi.device) # Rotation 90 deg

        self.block_goal_pose_ur5_c = torch.tensor([0.65, 0, 0.5, 0, 0, 0, 1], device=cfg.mppi.device)
        self.block_goal_pose_ur5_l= torch.tensor([0.7, 0.2, 0.5,  0, 0, 0.258819, 0.9659258 ], device=cfg.mppi.device) # Rotation 30 deg
        self.block_goal_pose_ur5_r= torch.tensor([0.7, -0.2, 0.5,  0, 0, -0.258819, 0.9659258 ], device=cfg.mppi.device) # Rotation -30 deg

        # Select goal according to test
        self.block_goal_pose = torch.clone(self.block_goal_pose_ur5_r)
        self.block_ort_goal = torch.clone(self.block_goal_pose[3:7])
        self.goal_yaw = torch.atan2(2.0 * (self.block_ort_goal[-1] * self.block_ort_goal[2] + self.block_ort_goal[0] * self.block_ort_goal[1]), self.block_ort_goal[-1] * self.block_ort_goal[-1] + self.block_ort_goal[0] * self.block_ort_goal[0] - self.block_ort_goal[1] * self.block_ort_goal[1] - self.block_ort_goal[2] * self.block_ort_goal[2])

        self.success = False
        self.ee_celebration = 0.25
        self.count = 0
        self.obst_index = 11

    def compute_metrics(self, block_pos, block_ort):

        block_yaws = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])
        Ex = torch.abs(self.block_goal_pose[0]-block_pos[-1,0])
        Ey = torch.abs(self.block_goal_pose[1]-block_pos[-1,1])
        Etheta = torch.abs(block_yaws[-1] - self.goal_yaw)
        return Ex, Ey, Etheta
    
    def compute_cost(self, sim):
        r_pos = sim.rigid_body_state[:, self.ee_index, :3]
        r_ort = sim.rigid_body_state[:, self.ee_index, 3:7]
        ee_height = r_pos[:, 2]
        block_pos = sim.root_state[:, self.block_index, :3]
        block_ort = sim.root_state[:, self.block_index, 3:7]

        # print(block_pos[-1])

        # Distances robot
        robot_to_block = r_pos - block_pos
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")   

        # Distances block
        block_to_goal =  self.block_goal_pose[0:2] - block_pos[:,0:2]
        # Compute yaw from quaternion with formula directly
        block_yaws = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])
        #self.block_yaws = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ort), "ZYX")[:, -1]


        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis = 1)
        block_to_goal_pos = torch.linalg.norm(block_to_goal, axis = 1)
        block_to_goal_ort = torch.abs(block_yaws - self.goal_yaw)

        # Posture costs
        ee_align = torch.linalg.norm(robot_euler[:,0:2] - self.ort_goal_euler[0:2], axis=1)
        ee_hover_dist= torch.abs(ee_height - self.ee_hover_height) 
        push_align = torch.sum(robot_to_block[:,0:2]*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_pos)+1
        
        # print(push_align[-1])
        # Collision avoidance
        xyz_contatcs = torch.reshape((torch.sum(torch.abs(sim.net_cf), dim=1)), (sim.num_envs, sim.num_bodies))
        # print(len(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])))
        # coll = torch.sum(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])[:, (sim.num_bodies - self.obst_number):sim.num_bodies], 1)
        coll = xyz_contatcs[:,self.obst_index]

        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_dist 
            + self.w_block_to_goal_pos * block_to_goal_pos
            + self.w_block_to_goal_ort * block_to_goal_ort
            + self.w_ee_hover * ee_hover_dist
            + self.w_ee_align * ee_align
            + self.w_push_align * push_align
            + self.w_collision * coll
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
