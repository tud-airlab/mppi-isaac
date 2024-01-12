import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mppiisaac.planner.mppi_custom_dynamics import MPPICustomDynamicsPlanner
import mppiisaac
import hydra
import yaml
from yaml import SafeLoader
from omegaconf import OmegaConf
import os
import torch
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mppiisaac.utils.config_store import ExampleConfig
import time
from mppiisaac.dynamics.jackal_robot import jackal_robot_dynamics

# MPPI to navigate a simple robot to a goal position


class Objective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def goal_cost(self, pos, goal):
        return torch.linalg.norm(pos - goal, axis=1) * 1.0
    
    def rot_cost(self, theta):
        return torch.abs(theta) * 0.05
    
    def dynamic_collision(self, position1, position2, radius):
        collision_cost = 1.0*(torch.linalg.norm(position1 - position2, axis=1) < radius)
        # collision_cost = 1.0*torch.pow((1/torch.linalg.norm(position1 - position2, axis=1)), 2)
        return collision_cost * 5

    def compute_cost(self, state: torch.Tensor):
        goal_cost_robot_1 = self.goal_cost(state[:,0:2], self.nav_goal[:2])
        goal_cost_robot_2 = self.goal_cost(state[:,6:8], self.nav_goal[2:4])
        rot_cost_robot_1 = self.rot_cost(state[:,2])
        rot_cost_robot_2 = self.rot_cost(state[:,8])
        collision_cost = self.dynamic_collision(state[:,0:2], state[:,6:8], 0.5)
        return goal_cost_robot_1 + goal_cost_robot_2 + rot_cost_robot_1 + rot_cost_robot_2 + collision_cost

class Dynamics(object):
    def __init__(self, cfg):
        self.dt = cfg.dt

    def simulate(self, states, control, t):
        new_states_1, control_1 = jackal_robot_dynamics(states[:,0:6], control[:,0:2], self.dt)
        new_states_2, control_2 = jackal_robot_dynamics(states[:,6:12], control[:,2:4], self.dt)
        new_states = torch.cat([new_states_1, new_states_2], dim=1)
        control = torch.cat([control_1, control_2], dim=1)
        return (new_states, control)


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
    with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/jackal.yaml') as f:
        jackal_cfg = yaml.load(f, Loader=SafeLoader)
    urdf_file = f'{os.path.dirname(mppiisaac.__file__)}/../assets/urdf/' + jackal_cfg['urdf_file']
    robots = [
        GenericDiffDriveRobot(
            urdf=urdf_file,
            mode="vel",
            actuated_wheels=[
                "rear_right_wheel",
                "rear_left_wheel",
                "front_right_wheel",
                "front_left_wheel",
            ],
            castor_wheels=[],
            wheel_radius = jackal_cfg['wheel_radius'],
            wheel_distance = jackal_cfg['wheel_base'],
        ),
        GenericDiffDriveRobot(
            urdf=urdf_file,
            mode="vel",
            actuated_wheels=[
                "rear_right_wheel",
                "rear_left_wheel",
                "front_right_wheel",
                "front_left_wheel",
            ],
            castor_wheels=[],
            wheel_radius = jackal_cfg['wheel_radius'],
            wheel_distance = jackal_cfg['wheel_base'],
        ),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=cfg.dt, robots=robots, render=cfg.render)
    # Set the initial position and velocity of the jackals.
    env.reset(pos=np.array(cfg.initial_actor_positions))
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": cfg.goal[:2],
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": cfg.goal[2:4],
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)
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
    dynamics = Dynamics(cfg)
    planner = MPPICustomDynamicsPlanner(cfg, objective, dynamics.simulate)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config_multi_jackal_custom_dynamics.yaml")
def run_jackal_robot(cfg: ExampleConfig):
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
    env = initalize_environment(cfg)
    planner = set_planner(cfg)

    action = np.zeros(int(cfg.nx / 2))
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot0 = ob["robot_0"]
        ob_robot1 = ob["robot_1"]

        state = np.concatenate((ob_robot0["joint_state"]["position"], ob_robot0["joint_state"]["velocity"],
                                ob_robot1["joint_state"]["position"], ob_robot1["joint_state"]["velocity"]))
        
        t = time.time()

        action = planner.compute_action(state)

        print(f"Time: {(time.time() - t)} s")
        
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_jackal_robot()
