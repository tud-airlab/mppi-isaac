from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.conversions import quaternion_to_yaw
import hydra
import torch
import pytorch3d.transforms
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        # Tuning of the weights for box
        self.weights = {
            "robot_to_goal": 1.0,
            "robot_ori": 0.5,
        }
        self.reset()
    
    def reset(self):
        pass

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name("panda", "panda_link7")
        goal_pos = sim.get_actor_position_by_name("goal")

        robot_to_goal = r_pos[:, 0:3] - goal_pos[:, 0:3]

        # Distance costs
        robot_to_goal_dist = torch.linalg.norm(robot_to_goal, axis=1)
        robot_rpy = pytorch3d.transforms.matrix_to_euler_angles(
            pytorch3d.transforms.quaternion_to_matrix(r_pos[:, 3:7]), "ZYX"
        )[:, 0:2]
        robot_rpy_dist = torch.linalg.norm(robot_rpy, axis=1)

        total_cost = (
            self.weights["robot_to_goal"] * robot_to_goal_dist
            + self.weights["robot_ori"] * robot_rpy_dist
        )

        return total_cost


@hydra.main(version_base=None, config_path=".", config_name="config_panda")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
