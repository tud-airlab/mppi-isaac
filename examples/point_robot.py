import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mppiisaac.planner.mppi import MPPIPlanner
import os

# MPPI to navigate a simple robot to a goal position

urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../assets/point_robot.urdf"

def initalize_environment(render):
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
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    env.reset()
    return env


def set_planner(goal_position: np.ndarray):
    """
    Initializes the mppi planner for the point robot.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    urdf = "../assets/point_robot.urdf"
    planner = None
    # planner = MPPIPlanner(config=config)
    return planner


def run_point_robot(n_steps=10000, render=True):
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
    env = initalize_environment(render)
    goal_position = np.array([2.0, 3.0])
    planner = set_planner(goal_position)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        #action[0:2] = planner.compute_action(
        #    q=ob_robot["joint_state"]["position"][0:2],
        #    qdot=ob_robot["joint_state"]["velocity"][0:2],
        #)
        ob, *_, = env.step(action)
    return {}

if __name__ == "__main__":
    res = run_point_robot(n_steps=10000, render=True)
