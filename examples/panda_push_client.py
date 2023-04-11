
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
        
        # Tuning of the weights
        self.w_robot_to_block_pos= 10#0.2
        self.w_block_to_goal_pos=  220#2.0 
        self.w_block_to_goal_ort=  190#0.2
        self.w_ee_hover=           540#5
        self.w_ee_align=           3#1
        self.w_push_align=         450#1

        # Task configration for comparison with baselines
        self.ee_index = 11
        self.block_index = 4
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)
        self.ee_hover_height = 0.135

        self.block_goal_pose_emdn_1 = torch.tensor([0.5, 0.3, 0.5, 0.0, 0.0, 0.0, 1.0], device=cfg.mppi.device)
        self.block_goal_pose_emdn_2 = torch.tensor([0.5, 0.3, 0.5, 0, 0, 0.7071068, 0.7071068], device=cfg.mppi.device) # Rotation 90 deg

        self.block_goal_pose_ur5_c = torch.tensor([0.65, 0, 0.5, 0, 0, 0, 1], device=cfg.mppi.device)
        self.block_goal_pose_ur5_l= torch.tensor([0.7, 0.2, 0.5,  0, 0, 0.258819, 0.9659258 ], device=cfg.mppi.device) # Rotation 30 deg
        self.block_goal_pose_ur5_r= torch.tensor([0.7, -0.2, 0.5,  0, 0, -0.258819, 0.9659258 ], device=cfg.mppi.device) # Rotation -30 deg

        # Select goal according to test
        self.block_goal_pose = torch.clone(self.block_goal_pose_ur5_l)
        self.block_ort_goal = torch.clone(self.block_goal_pose[3:7])

        self.success = False
        self.ee_celebration = 0.25
        self.count = 0

    def compute_cost(self, sim):
        r_pos = sim.rigid_body_state[:, self.ee_index, :3]
        r_ort = sim.rigid_body_state[:, self.ee_index, 3:7]
        ee_height = r_pos[:, 2]
        block_pos = sim.root_state[:, self.block_index, :3]
        block_ort = sim.root_state[:, self.block_index, 3:7]

        # Distances robot
        robot_to_block = r_pos - block_pos
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")   

        # Distances block
        block_to_goal = self.block_goal_pose[0:2] - block_pos[:,0:2]
        # Compute yaw from quaternion with formula directly
        block_yaws = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])
        goal_yaw = torch.atan2(2.0 * (self.block_ort_goal[-1] * self.block_ort_goal[2] + self.block_ort_goal[0] * self.block_ort_goal[1]), self.block_ort_goal[-1] * self.block_ort_goal[-1] + self.block_ort_goal[0] * self.block_ort_goal[0] - self.block_ort_goal[1] * self.block_ort_goal[1] - self.block_ort_goal[2] * self.block_ort_goal[2])
        #block_yaws = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ort), "ZYX")[:, -1]

        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis = 1)
        block_to_goal_pos = torch.linalg.norm(block_to_goal, axis = 1)
        block_to_goal_ort = torch.abs(block_yaws - goal_yaw)

        # Posture costs
        ee_align = torch.linalg.norm(robot_euler[:,0:2] - self.ort_goal_euler[0:2], axis=1)
        ee_hover_dist= torch.abs(ee_height - self.ee_hover_height) 
        push_align = torch.sum(robot_to_block[:,0:2]*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_pos)+1

        # Evaluation metrics 
        # ------------------------------------------------------------------------
        if self.count > 300:
            Ex = torch.abs(self.block_goal_pose[0]-block_pos[-1,0])
            Ey = torch.abs(self.block_goal_pose[1]-block_pos[-1,1])
            Etheta = torch.abs(block_yaws[-1] - goal_yaw)
            
            metric_1 = 1.5*(Ex+Ey)+0.01*Etheta
            print("Metric Baxter", metric_1)
            print("Ex", Ex)
            print("Ey", Ey)
            print("Angle", Etheta)

            # Ex < 0.025 and Ey < 0.01 and Etheta < 0.052:   # Stricter metric
            if Ex < 0.05 and Ey < 0.025 and Etheta < 0.17:
                print("Success")
                self.success = True

            self.count = 0
        else:
            self.count +=1
        # ---------------------------------------------------------------------------
        
        # Move to cartesian pose after succesful pushing, otherwise push
        if self.success == True:
            total_cost =  5*torch.abs(ee_height - self.ee_celebration) + 0.5*ee_align
        else:
            total_cost = (
                self.w_robot_to_block_pos * robot_to_block_dist
                + self.w_block_to_goal_pos * block_to_goal_pos
                + self.w_block_to_goal_ort * block_to_goal_ort
                + self.w_ee_hover * ee_hover_dist
                + self.w_ee_align * ee_align
                + self.w_push_align * push_align
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
