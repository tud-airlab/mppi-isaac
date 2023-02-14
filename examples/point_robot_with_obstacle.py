import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import os
import torch
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal

from mppiisaac.utils.config_store import ExampleConfig

# MPPI to navigate a simple robot to a goal position


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

        self.w_nav = 1.0
        self.w_obs = 0.05

    def compute_cost(self, sim: IsaacGymWrapper):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )

        #sim.gym.refresh_net_contact_force_tensor(sim.sim)
        #sim.net_cf

        obs_cost = torch.sum(
            1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),
            axis=1,
        )

        return nav_cost * self.w_nav + obs_cost * self.w_obs


def initalize_environment(urdf_file: str, render: bool, dt: float = 0.05) -> UrdfEnv:
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    urdf_file = (
        os.path.dirname(os.path.abspath(__file__)) + "/../assets/urdf/" + urdf_file
    )
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=dt, robots=robots, render=render)

    # Set the initial position and velocity of the point mass.
    env.reset()

    # add obstacle
    obst1Dict = {
        "type": "sphere",
        "geometry": {"position": [1.0, 1.0, 0.0], "radius": 0.5},
    }
    sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
    env.add_obstacle(sphereObst1)

    obst2Dict = {
        "type": "sphere",
        "geometry": {"position": [1.0, 2.0, 0.0], "radius": 0.3},
    }
    sphereObst2 = SphereObstacle(name="simpleSphere", content_dict=obst2Dict)
    env.add_obstacle(sphereObst2)

    # sense both
    sensor = FullSensor(
        goal_mask=["position"], obstacle_mask=["position", "velocity", "radius"]
    )
    env.add_sensor(sensor, [0])

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
    planner = MPPIisaacPlanner(cfg, objective)

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

    env = initalize_environment(cfg.urdf_file, cfg.render)
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
