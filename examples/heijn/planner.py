from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.mppi_utils import quaternion_to_yaw
import hydra
import torch
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        # Tuning of the weights for box
        self.weights = {
            "robot_to_block": 0.2,
            "block_to_pos": 2.0,
            "block_to_ort": 2.0,
            "push_align": 0.2,
            "velocity": 1.0,
        }

        self.goal_yaw = 0.0
    
    def reset(self):
        pass

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name("heijn", "front_link")
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

        # Velocity cost
        vel = torch.linalg.norm(block_vel[:, 0:2], axis=1)

        total_cost = (
            self.weights["robot_to_block"] * robot_to_block_dist
            + self.weights["block_to_pos"] * block_to_pos_dist
            + self.weights["block_to_ort"] * block_to_ort_dist
            + self.weights["push_align"] * push_align
            + self.weights["velocity"] * vel
        )

        return total_cost


@hydra.main(version_base=None, config_path="../../conf", config_name="config_heijn_push")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
