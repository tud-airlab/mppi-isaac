from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
import hydra
import torch
import zerorpc


class Objective(object):
    def __init__(self, cfg):
        pass
    
    def reset(self):
        pass

    def compute_cost(self, sim):
        r_pos = sim.get_actor_link_by_name(actor_name="boxer", link_name="ee_link")
        block_goal = sim.get_actor_position_by_name("goal")
        robot_to_goal = block_goal[:, 0:2] - r_pos[:, 0:2]
        robot_to_goal_dist = torch.linalg.norm(robot_to_goal, axis=1)

        wall_forces = sim.get_actor_contact_forces_by_name("wall", "box")
        forces = torch.sum(torch.abs(wall_forces[:, 0:3]), axis=1)

        return robot_to_goal_dist + forces


@hydra.main(version_base=None, config_path="../../conf", config_name="config_boxer_reach")
def run_heijn_robot(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_heijn_robot()
