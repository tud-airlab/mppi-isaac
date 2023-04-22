import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import mppiisaac
import hydra
import yaml
from yaml import SafeLoader
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
        pos = torch.cat(
            (sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1), sim.dof_state[:, 6].unsqueeze(1), sim.dof_state[:, 8].unsqueeze(1)), 1
        )

        goal_cost = torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )

        avoidance_cost = 1 / torch.clamp(
            torch.linalg.norm(pos[:, :2] - pos[:, 2:], axis=1) - 0.05, min=0, max=1999
        )

        return goal_cost #+ avoidance_cost



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
    with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/point_robot.yaml') as f:
        robot_cfg = yaml.load(f, Loader=SafeLoader)
    urdf_file = f'{os.path.dirname(mppiisaac.__file__)}/../assets/urdf/' + robot_cfg['urdf_file']
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.05, robots=robots, render=cfg.render)

    env.reset(pos=np.array(cfg.initial_actor_positions))
    # Set the initial position and velocity of the point mass.
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
    #env.add_goal(goal)
    return env


def set_planner(cfg):
    """
    Initializes the mppi planner for the point robot.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    # urdf = "../assets/point_robot.urdf"
    objective = Objective(cfg, cfg.mppi.device)
    prior = None
    planner = MPPIisaacPlanner(cfg, objective, prior)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config_multi_point_robot")
def run_point_robot(cfg: ExampleConfig):
    """
    Set the gym environment, the planner and run point robot example.
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

    action = np.zeros(int(cfg.nx / 2))
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
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
    res = run_point_robot()
