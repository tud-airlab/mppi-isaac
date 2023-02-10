import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
from urdfenvs.sensors.full_sensor import FullSensor
import os
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal

from mppiisaac.utils.config_store import ExampleConfig

# MPPI to navigate a simple robot to a goal position

urdf_file = (
    os.path.dirname(os.path.abspath(__file__)) + "/../assets/urdf/point_robot.urdf"
)


def initalize_environment(render, dt):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
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
    planner = MPPIisaacPlanner(cfg)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config")
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

    env = initalize_environment(cfg.render, cfg.isaacgym.dt)
    planner = set_planner(cfg)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]['obstacles']
        action[0:2] = planner.compute_action(
            q=ob_robot["joint_state"]["position"][0:2],
            qdot=ob_robot["joint_state"]["velocity"][0:2],
            obst=obst
        )
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot()
