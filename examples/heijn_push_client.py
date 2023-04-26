from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
import pytorch3d.transforms

from mppiisaac.utils.config_store import ExampleConfig

class Objective(object):
    def __init__(self, cfg, device):
        
        # Tuning of the weights for box
        self.w_robot_to_block_pos=  .2
        self.w_block_to_goal_pos=   2.
        self.w_block_to_goal_ort=   3.0
        self.w_push_align=          0.6
        self.w_collision=           10
        self.w_vel=                 0.

        # Tuning for Sphere
        # self.w_robot_to_block_pos=  0.2
        # self.w_block_to_goal_pos=   1.
        # self.w_block_to_goal_ort=   0
        # self.w_push_align=          0.6
        # self.w_collision=           .01
        # self.w_vel=                 .0
        
        # Task configration for comparison with baselines
        self.ee_index = 4
        self.block_index = 1
        self.ort_goal_euler = torch.tensor([0, 0, 0], device=cfg.mppi.device)
        self.ee_hover_height = 0.14

        self.block_goal_box = torch.tensor([0., 0., 0.5, 0.0, 0.0, 0.0, 1.0], device=cfg.mppi.device)
        self.block_goal_sphere = torch.tensor([0.5, 1., 0.5, 0, 0, -0.7071068, 0.7071068], device=cfg.mppi.device) # Rotation 90 deg

        # Select goal according to test
        self.block_goal_pose = torch.clone(self.block_goal_box)
        self.block_ort_goal = torch.clone(self.block_goal_pose[3:7])
        self.goal_yaw = torch.atan2(2.0 * (self.block_ort_goal[-1] * self.block_ort_goal[2] + self.block_ort_goal[0] * self.block_ort_goal[1]), self.block_ort_goal[-1] * self.block_ort_goal[-1] + self.block_ort_goal[0] * self.block_ort_goal[0] - self.block_ort_goal[1] * self.block_ort_goal[1] - self.block_ort_goal[2] * self.block_ort_goal[2])

        self.success = False
        self.count = 0

        # Number of obstacles
        self.obst_number = 3        # By convention, obstacles are the last actors

    def compute_metrics(self, block_pos, block_ort):

        block_yaws = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])
        Ex = torch.abs(self.block_goal_pose[0]-block_pos[-1,0])
        Ey = torch.abs(self.block_goal_pose[1]-block_pos[-1,1])
        Etheta = torch.abs(block_yaws[-1] - self.goal_yaw)
        return Ex, Ey, Etheta
    
    def compute_cost(self, sim):
        r_pos = sim.rigid_body_state[:, self.ee_index, :2]
       
        block_pos = sim.root_state[:, self.block_index, :3]
        block_ort = sim.root_state[:, self.block_index, 3:7]
        block_vel = sim.root_state[:, self.block_index, 7:10]
        
        # Distances robot
        robot_to_block = r_pos - block_pos[:,0:2]

        # Distances block
        block_to_goal =  self.block_goal_pose[0:2] - block_pos[:,0:2]
        # Compute yaw from quaternion with formula directly
        block_yaws = torch.atan2(2.0 * (block_ort[:,-1] * block_ort[:,2] + block_ort[:,0] * block_ort[:,1]), block_ort[:,-1] * block_ort[:,-1] + block_ort[:,0] * block_ort[:,0] - block_ort[:,1] * block_ort[:,1] - block_ort[:,2] * block_ort[:,2])
        self.block_yaws = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(block_ort), "ZYX")[:, -1]

        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis = 1)
        block_to_goal_pos = torch.linalg.norm(block_to_goal, axis = 1)
        block_to_goal_ort = torch.abs(block_yaws - self.goal_yaw)

        push_align = torch.sum(robot_to_block[:,0:2]*block_to_goal, 1)/(robot_to_block_dist*block_to_goal_pos)+1
        
        # Velocity cost
        vel = torch.linalg.norm(block_vel, axis = 1)

        # Collision avoidance
        xy_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)),1)
        coll = torch.sum(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])[:, (sim.num_bodies - self.obst_number):sim.num_bodies], 1)

        total_cost = (
            self.w_robot_to_block_pos * robot_to_block_dist
            + self.w_block_to_goal_pos * block_to_goal_pos
            + self.w_block_to_goal_ort * block_to_goal_ort
            + self.w_push_align * push_align
            + self.w_collision * coll
            + self.w_vel * vel
        )

        return total_cost

@hydra.main(version_base=None, config_path="../conf", config_name="config_heijn_push")
def run_heijn_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    objective = Objective(cfg, cfg.mppi.device)
    prior = None
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()

if __name__ == "__main__":
    run_heijn_robot()
