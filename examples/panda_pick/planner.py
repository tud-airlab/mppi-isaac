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
            "robot_to_block": 40.0,
            "block_to_goal": 10.0,
            "collision": 26.0,
            "robot_ori": 2.0,
        }
        self.reset()

    def reset(self):
        self.prev_block_to_goal_dist = 1
        self.prev_robot_to_block_dist = 1

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name("panda", "panda_ee")
        block_pos = sim.get_actor_position_by_name("panda_pick_block")
        goal_pos = sim.get_actor_position_by_name("goal")
        table_forces = sim.get_actor_contact_forces_by_name("table", "box")

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

        total_cost = (
            self.weights["robot_to_block"] * robot_to_block_dist
            + self.weights["block_to_goal"] * block_to_goal_dist
            + self.weights["collision"] * forces
            + self.weights["robot_ori"] * robot_rpy_dist
        )

        self.prev_block_to_goal_dist = block_to_goal_dist

        return total_cost


@hydra.main(version_base=None, config_path="../../conf", config_name="panda_pick")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
