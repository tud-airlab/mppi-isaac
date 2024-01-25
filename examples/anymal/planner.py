from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
from mppiisaac.utils.mppi_utils import quaternion_to_yaw
import hydra
import torch
import pytorch3d.transforms
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        # Tuning of the weights for box
        self.weights = {
            "robot_to_goal": 1.0,
            "robot_off_ground": 5.0,
            "knees_off_ground": 5.0,
        }
        self.reset()

    def reset(self):
        self.prev_block_to_goal_dist = 1
        self.prev_robot_to_block_dist = 1

    def compute_cost(self, sim):
        body_pos = sim.get_actor_link_by_name("anymal", "base")
        goal_pos = sim.get_actor_position_by_name("goal")

        body_front_pos = sim.get_actor_link_by_name("anymal", "face_front")
        body_rear_pos = sim.get_actor_link_by_name("anymal", "face_rear")

        # Force costs
        body_to_goal = torch.linalg.norm(body_pos[:, 0:3] - goal_pos[:, 0:3], axis=1)
        body_height = 0.65
        body_off_ground = torch.abs(body_pos[:, 2] - body_height) + torch.abs(body_front_pos[:, 2] - body_height) + torch.abs(body_rear_pos[:, 2] - body_height)

        # high knee costs
        knee_pos_1 = sim.get_actor_link_by_name("anymal", "LF_KFE")
        knee_pos_2 = sim.get_actor_link_by_name("anymal", "LH_KFE")
        knee_pos_3 = sim.get_actor_link_by_name("anymal", "RH_KFE")
        knee_pos_4 = sim.get_actor_link_by_name("anymal", "RF_KFE")
        knee_height = 0.35
        knee_off_ground = torch.abs(knee_pos_1[:, 2] - knee_height) + torch.abs(knee_pos_2[:, 2] - knee_height) + torch.abs(knee_pos_3[:, 2] - knee_height) + torch.abs(knee_pos_4[:, 2] - knee_height)
        
        total_cost = (
            self.weights["robot_to_goal"] * body_to_goal +
            self.weights["robot_off_ground"] * body_off_ground +
            self.weights["knees_off_ground"] * knee_off_ground
        )

        return total_cost


@hydra.main(version_base=None, config_path="../../conf", config_name="config_anymal")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
