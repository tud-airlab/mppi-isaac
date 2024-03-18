from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.conversions import quaternion_to_yaw
import hydra
import torch
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        # Tuning of the weights for box
        self.weights = {
            "robot_to_block": 0.1,
            "block_to_goal": 2.0,
            "block_to_goal_ort": 3.0,
            "push_align": 0.6,
            "collision": 100,
            "velocity": 0.0,
        }

        self.goal_yaw = 0.0

    def reset(self):
        pass

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name(actor_name="boxer", link_name="ee_link")
        block_pos = sim.get_actor_position_by_name("block")
        block_vel = sim.get_actor_velocity_by_name("block")
        block_ort = sim.get_actor_orientation_by_name("block")
        block_goal = sim.get_actor_position_by_name("goal")

        # Distances robot
        robot_to_block = r_pos[:, 0:2] - block_pos[:, 0:2]
        block_to_goal = block_goal[:, 0:2] - block_pos[:, 0:2]
        block_yaws = quaternion_to_yaw(block_ort)

        # Distance costs
        robot_to_block_dist = torch.linalg.norm(robot_to_block[:, 0:2], axis=1)
        block_to_pos_dist = torch.linalg.norm(block_to_goal, axis=1)
        block_to_ort_dist = torch.abs(block_yaws - self.goal_yaw)

        # Push align cost
        push_align = (
            torch.sum(robot_to_block[:, 0:2] * block_to_goal, 1)
            / (robot_to_block_dist * block_to_pos_dist)
            + 1
        )

        # Collision avoidance
        obst1_forces = sim.get_actor_contact_forces_by_name(actor_name="paper_obst1", link_name="box")
        obst2_forces = sim.get_actor_contact_forces_by_name(actor_name="paper_obst2", link_name="box")
        coll = torch.sum(torch.abs(obst1_forces[:, 0:2]), axis=1) + torch.sum(torch.abs(obst2_forces[:, 0:2]), axis=1)

        # Velocity cost
        vel = torch.linalg.norm(block_vel[:, 0:2], axis=1)

        total_cost = (
            self.weights["robot_to_block"] * robot_to_block_dist
            + self.weights["block_to_goal"] * block_to_pos_dist
            + self.weights["block_to_goal_ort"] * block_to_ort_dist
            + self.weights["push_align"] * push_align
            + self.weights["velocity"] * vel
            + self.weights["collision"] * coll
        )

        return total_cost


@hydra.main(version_base=None, config_path=".", config_name="config_boxer_push")
def run_boxer_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_boxer_robot()
