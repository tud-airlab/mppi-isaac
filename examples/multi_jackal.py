import gym
import numpy as np
from urdfenvs.robots.jackal import JackalRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import os
import torch
from mpscenes.goals.static_sub_goal import StaticSubGoal

from mppiisaac.utils.config_store import ExampleConfig

# MPPI to navigate a simple robot to a goal position

class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = torch.cat((sim.root_state[:, 0, 0:2], sim.root_state[:, 1, 0:2]), axis=1)
        print(pos[0])
        print(self.nav_goal)


        cost = torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )
        cost += -0.2*torch.linalg.norm(pos[:, :2] - pos[:, 2:], axis=1)
        return cost * 1.5


def initalize_environment(cfg) -> UrdfEnv:
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    # urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../assets/urdf/" + cfg.urdf_file
    robots = [
        JackalRobot(mode="vel"),
        JackalRobot(mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.02, robots=robots, render=cfg.render)
    # Set the initial position and velocity of the jackal robot
    print(cfg.initial_actor_positions)
    env.reset(pos=np.array(cfg.initial_actor_positions))
    #goal_dict = {
    #    "weight": 1.0,
    #    "is_primary_goal": True,
    #    "indices": [0, 1],
    #    "parent_link": 0,
    #    "child_link": 1,
    #    "desired_position": cfg.goal,
    #    "epsilon": 0.05,
    #    "type": "staticSubGoal",
    #}
    #goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
   # env.add_goal(goal)
    return env


def set_planner(cfg):
    """
    Initializes the mppi planner for jackal robot.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    objective = Objective(cfg, cfg.mppi.device)
    planner = MPPIisaacPlanner(cfg, objective)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config_multi_jackal")
def run_jackal_robot(cfg: ExampleConfig):
    """
    Set the gym environment, the planner and run jackal robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)


    env = initalize_environment(cfg)
    planner = set_planner(cfg)

    action = np.zeros(8)
    ob, *_ = env.step(action)
    
    
    for _ in range(cfg.n_steps):
        #Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.

        #Todo fix joint with zero friction

        ob_robot0 = ob["robot_0"]
        ob_robot1 = ob["robot_1"]
        action = planner.compute_action(
            q=list(ob_robot0["joint_state"]["position"]) + list(ob_robot1["joint_state"]["position"]),
            qdot=list(ob_robot0["joint_state"]["velocity"]) + list(ob_robot1["joint_state"]["velocity"]),
        )
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_jackal_robot()
