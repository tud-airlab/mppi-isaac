from mppiisaac.planner.mppi import MPPIPlanner
from typing import Callable, Optional
import io
import os
import yaml
from yaml.loader import SafeLoader
from mppiisaac.dynamics.point_robot import omnidirectional_point_robot_dynamics

import torch

torch.set_printoptions(precision=2, sci_mode=False)


class MPPICustomDynamicsPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable):
        self.cfg = cfg
        self.objective = objective

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost
        )
        self.current_state = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))
    
    def update_objective(self, objective):
        self.objective = objective

    def dynamics(self, states, control, t):
        new_states = omnidirectional_point_robot_dynamics(states, control, self.cfg.dt)
        return (new_states, control)

    def running_cost(self, state):
        return self.objective.compute_cost(state)

    def compute_action(self, q, qdot):
        q_tensor = torch.tensor([q], dtype=torch.float32, device=self.cfg.mppi.device)
        qdot_tensor = torch.tensor([qdot], dtype=torch.float32, device=self.cfg.mppi.device)
        self.current_state = torch.cat([q_tensor, qdot_tensor], dim=1)

        actions = self.mppi.command(self.current_state).cpu()
        return actions
