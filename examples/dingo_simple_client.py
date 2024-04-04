
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
        # self.w_robot_to_block_pos=  8
        # self.w_block_to_goal_pos=   3
        self.w_ee_to_goal_pos=      8.0
        self.w_ee_align=            0.5
        self.w_collision=           1.0
        self.w_neutral =            100.
        # Task configration for comparison with baselines
        self.ee_index = 25
        self.obstacle_idx = 2

        self.ee_goal_pose = torch.tensor([-4, 0, 0.6], device=cfg.mppi.device)
        self.ort_goal_euler = torch.tensor([3.1415/2, 0, 0], device=cfg.mppi.device)
        self.ort_goal_quat = pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(self.ort_goal_euler, "ZYX"))
        self.joints_neutral = torch.tensor([1., 0., 0., 0., 0., 0.], device=cfg.mppi.device) #same neutral position as controller Jelmer: [0.0, 0.35, 1.67, -1.69, -4.94, -1.37]

        self.obst_number = 2
        self.success = False
        self.count = 0

    def compute_metrics(self, ee_pose):
        # usually to check if it has been successful
        ee_loc = ee_pose[-1,:3]
        ee_rot = ee_pose[-1,3:7]
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(ee_rot), "ZYX")   
        robot_to_goal =  self.ee_goal_pose - ee_loc
        print("robot_to_goal", robot_to_goal)
        Ex = torch.linalg.norm(robot_to_goal)
        Eq = torch.linalg.norm((self.ort_goal_euler - robot_euler)[[1,2]])
        # Eq = torch.abs((robot_euler - self.ort_goal_euler)[2])
        return Ex, Eq
    
    def compute_cost(self, sim):
        r_pos = sim.rigid_body_state[:, self.ee_index, :3]
        r_ort = sim.rigid_body_state[:, self.ee_index, 3:7]
        joint_angles = sim.dof_state[:, 3:9] #number correct?
        print("joint_angles_1:", joint_angles[1, :])
        # block_pos = sim.root_state[:, self.block_index, :3]
        # root_state of shape (num_envs, num_bodies, 7) where the last 13 x y z, quaternion, vx vy vz, wx wy wz

        # Distances robot
        # robot_to_block = r_pos - block_pos
        robot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(r_ort), "ZYX")   
        robot_to_goal =  self.ee_goal_pose - r_pos
        
        # Distance costs
        # robot_to_block_dist = torch.linalg.norm(robot_to_block, axis = 1)
        robot_to_goal_pos = torch.linalg.norm(robot_to_goal, axis = 1)

        # Posture costs
        ee_align = torch.linalg.norm((robot_euler - self.ort_goal_euler)[:,[1,2]], axis=1)
        # ee_align = torch.abs((robot_euler - self.ort_goal_euler)[:, 2])

        # neutral position costs, kinova
        joint_distance = joint_angles - self.joints_neutral
        cost_default_joints = torch.linalg.norm(joint_distance, axis=1)

        # Collision avoidance
        contact_f = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        coll = torch.sum(contact_f.reshape([sim.num_envs, int(contact_f.size(dim=0)/sim.num_envs)])[:, (sim.num_bodies - 1 - self.obst_number):sim.num_bodies], 1) # Consider obstacles

        total_cost = (
            self.w_neutral * cost_default_joints #self.w_ee_to_goal_pos * robot_to_goal_pos + self.w_ee_align * ee_align + self.w_collision * coll + self.w_neutral * cost_default_joints
        )
        return total_cost

@hydra.main(version_base=None, config_path="../conf", config_name="config_dingo")
def run_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
   
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__ == "__main__":
    run_robot()
