from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
import hydra
import torch
import pytorch3d.transforms
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        # Tuning of the weights for box
        self.weights = {
            "robot_to_block": 8.0,
            "block_to_goal": 4.0,
            "collision": 0.1,
            "robot_ori": 1.0,
            "base_vel": 2.0,
            "arm_vel": 0.1,
            "comfy_gripper_state": 200.0,
            "height_cost": 10000.0,
        }
        self.comfy_gripper_state = torch.tensor([0.025, 0.025], device=cfg.mppi.device)
        self.reset()

    def reset(self):
        self.prev_block_to_goal_dist = 1
        self.prev_robot_to_block_dist = 1

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name("omnipanda", "panda_hand")
        block_pos = sim.get_actor_position_by_name("panda_pick_block")
        goal_pos = sim.get_actor_position_by_name("goal")
        table_forces = sim.get_actor_contact_forces_by_name("table", "box")
        actor_dof = sim.get_dof_state()

        # Extract velocities from actor_dof
        actor_dof_velocities = actor_dof[:, 1::2]  # Assuming velocities are at odd indices
        base_vel = actor_dof_velocities[:, 0:3]
        arm_vel = actor_dof_velocities[:, 3:11]

        robot_to_block = r_pos[:, 0:3] - block_pos[:, 0:3]
        block_to_goal = block_pos[:, 0:3] - goal_pos[:, 0:3]

        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block, axis=1)
        block_to_goal_dist = torch.linalg.norm(block_to_goal, axis=1)
        robot_rpy = pytorch3d.transforms.matrix_to_euler_angles(
            pytorch3d.transforms.quaternion_to_matrix(r_pos[:, 3:7]), "ZYX"
        )[:, 0:2]
        robot_rpy_dist = torch.linalg.norm(robot_rpy, axis=1)

        # Force costs
        forces = torch.sum(torch.abs(table_forces[:, 0:3]), axis=1)

        # Velocity costs
        base_vel_cost = torch.sum(torch.square(base_vel), dim=1)
        arm_vel_cost = torch.sum(torch.square(arm_vel), dim=1)

        # Extract Gripper state
        actor_dof_positions = actor_dof[:, ::2] 
        gripper_state = actor_dof_positions[:, -2:]
        # Gripper cost
        gripper_cost = torch.sum(torch.square(gripper_state - self.comfy_gripper_state), dim=1)

        # Penalty if r_pos[2] is lower than 0.12
        height_cost = torch.clamp(0.12 - r_pos[:, 2], min=0)

        total_cost = (
            self.weights["robot_to_block"] * robot_to_block_dist
            + self.weights["block_to_goal"] * block_to_goal_dist
            + self.weights["collision"] * forces
            + self.weights["robot_ori"] * robot_rpy_dist
            + self.weights["base_vel"] * base_vel_cost
            + self.weights["arm_vel"] * arm_vel_cost
            + self.weights["comfy_gripper_state"] * gripper_cost
            + self.weights["height_cost"] * height_cost
        )

        self.prev_block_to_goal_dist = block_to_goal_dist

        return total_cost


@hydra.main(version_base=None, config_path=".", config_name="omni_panda_pick")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
