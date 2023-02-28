import gym
from mpscenes.goals.static_sub_goal import StaticSubGoal
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import hydra
from omegaconf import OmegaConf
import os
import torch
from mppiisaac.priors.fabrics_panda import FabricsPandaPrior

from mppiisaac.utils.config_store import ExampleConfig

# MPPI to navigate a simple robot to a goal position

urdf_file = (
    os.path.dirname(os.path.abspath(__file__))
    + "/../assets/urdf/panda_bullet/panda.urdf"
)


class JointSpaceGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = torch.cat(
            (
                sim.dof_state[:, 0].unsqueeze(1),
                sim.dof_state[:, 2].unsqueeze(1),
                sim.dof_state[:, 4].unsqueeze(1),
                sim.dof_state[:, 6].unsqueeze(1),
                sim.dof_state[:, 8].unsqueeze(1),
                sim.dof_state[:, 10].unsqueeze(1),
                sim.dof_state[:, 12].unsqueeze(1),
            ),
            1,
        )
        # dof_states = gym.acquire_dof_state_tensor(sim)
        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )


class EndEffectorGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.ort_goal = torch.tensor([1, 0, 0, 0], device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, :3]
        ort = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, 3:7]
        # dof_states = gym.acquire_dof_state_tensor(sim)

        reach_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)
        align_cost = torch.linalg.norm(ort - self.ort_goal, axis=1)
        return 10 * reach_cost + align_cost
        # return torch.clamp(
        #     torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        # )


def initalize_environment(cfg):
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
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=cfg.render)
    # Set the initial position and velocity of the panda arm.
    env.reset()
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1, 2],
        "parent_link": "panda_link0",
        "child_link": "panda_hand",
        "desired_position": cfg.goal,
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)
    return env


def set_planner(cfg):
    """
    Initializes the mppi planner for the panda arm.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    objective = EndEffectorGoalObjective(cfg, cfg.mppi.device)
    # objective = JointSpaceGoalObjective(cfg, cfg.mppi.device)
    prior = FabricsPandaPrior(cfg)
    planner = MPPIisaacPlanner(cfg, objective, prior)

    return planner


@hydra.main(version_base=None, config_path="../conf", config_name="config_panda")
def run_panda_robot(cfg: ExampleConfig):
    """
    Set the gym environment, the planner and run panda robot example.
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

    action = np.zeros(7)
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob["robot_0"]
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
        )
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_panda_robot()
