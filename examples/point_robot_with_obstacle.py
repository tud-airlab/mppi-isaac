import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import mppiisaac
import hydra
import yaml
from yaml import SafeLoader
from omegaconf import OmegaConf
import os
import torch
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mppiisaac.priors.fabrics_point import FabricsPointPrior

from mppiisaac.utils.config_store import ExampleConfig

# MPPI to navigate a simple robot to a goal position


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

        self.w_nav = 1.0 # 5.0
        self.w_obs = 1.0  
        self.w_coll = 0.1 # 0.01 
        
    def compute_cost(self, sim: IsaacGymWrapper):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)

        # Coll avoidance with distance
        obs_cost = torch.sum(
            1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),
            axis=1,
        )

        # Collision avoidance with contact forces
        xy_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)),1)
        coll = torch.sum(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])[:, 1:sim.num_bodies], 1) # skip the first, it is the robot

        return nav_cost * self.w_nav + coll * self.w_coll # + obs_cost * self.w_obs

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
        heijn_cfg = yaml.load(f, Loader=SafeLoader)
    urdf_file = f'{os.path.dirname(mppiisaac.__file__)}/../assets/urdf/' + heijn_cfg['urdf_file']
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.05, robots=robots, render=cfg.render, observation_checking=False)

    # Set the initial position and velocity of the point mass.
    env.reset()

    # add obstacle
    obst1Dict = {
        "type": "sphere",
        "geometry": {"position": [1.0, 1.0, 0.0], "radius": 0.5},
    }
    sphereObst = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
    env.add_obstacle(sphereObst)

    # TODO: Allow for non-sphere obstacles. Wait for update of urdfenvs.
    # obst2Dict = {
        # "type": "box",
        # "geometry": {
            # "position": [1.0, 2.0, 0.0],
            # "width": 0.3,
            # "height": 0.2,
            # "length": 1.0,
        # },
    # }
    # boxObst = BoxObstacle(name="simpleBox", content_dict=obst2Dict)
    # env.add_obstacle(boxObst)
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": cfg.goal,
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)

    # sense both
    sensor = FullSensor(
        goal_mask=["position"],
        obstacle_mask=["position", "velocity", "size"],
        variance=0.0,
    )
    env.add_sensor(sensor, [0])
    env.set_spaces()

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
    if cfg.mppi.use_priors == True:
        prior = FabricsPointPrior(cfg)
    else:
        prior = None
    planner = MPPIisaacPlanner(cfg, objective, prior)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config_point_robot")
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
    # OmegaConf.save(configs=cfg, f='test.yaml')

    env = initalize_environment(cfg)
    planner = set_planner(cfg)

    action = np.zeros(int(cfg.nx / 2))
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]["obstacles"]
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            obst=obst,
        )
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot()
