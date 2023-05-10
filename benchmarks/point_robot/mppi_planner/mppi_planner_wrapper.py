from omegaconf import OmegaConf
from plannerbenchmark.generic.planner import Planner
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner

import torch

class ObjectiveOld(object):
    def __init__(self, goal, device):
        self.nav_goal = torch.tensor(goal, device=device)

        self.w_nav = 2.0
        self.w_obs = 1

    def compute_cost(self, sim: IsaacGymWrapper):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1), min=0, max=1999
        )

        # sim.gym.refresh_net_contact_force_tensor(sim.sim)
        # sim.net_cf

        # This can cause steady state error if the goal is close to an obstacle, better use contact forces later on
        obs_cost = torch.sum(
            1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),
            axis=1,
        )

        return nav_cost * self.w_nav + obs_cost * self.w_obs

class Objective(object):
    def __init__(self, goal, device):
        self.nav_goal = torch.tensor(goal, device=device)

        self.w_nav = 1.0
        self.w_obs = 0.5

    def compute_cost(self, sim: IsaacGymWrapper):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )

        # sim.gym.refresh_net_contact_force_tensor(sim.sim)
        # sim.net_cf

        # This can cause steady state error if the goal is close to an obstacle, better use contact forces later on
        obs_cost = torch.sum(
            1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),
            axis=1,
        )

        return nav_cost * self.w_nav + obs_cost * self.w_obs




class MPPIPlanner(Planner):

    def __init__(self, exp, **kwargs):
        super().__init__(exp, **kwargs)
        self.cfg = kwargs['config']
        initial_actor_position = exp.initState()[0].tolist() 
        initial_actor_position[2] += 0.05
        self.cfg['initial_actor_positions'] = [initial_actor_position]
        self._config = OmegaConf.create(kwargs)

        self.reset()

    def setJointLimits(self, limits):
        self._limits = limits

    def setGoal(self, motionPlanningGoal):
        cfg = OmegaConf.create(self.cfg)
        goal_position = motionPlanningGoal.sub_goals()[0].position()
        objective = Objective(goal_position, cfg.mppi.device)
        if not hasattr(self, '_planner'):
            self._planner = MPPIisaacPlanner(cfg, objective)


    def setSelfCollisionAvoidance(self, r_body):
        pass

    def setObstacles(self, obstacles, r_body):
        pass

    def concretize(self):
        pass

    def save(self, folderPath):
        file_name = folderPath + "/planner.yaml"
        OmegaConf.save(config=self._config, f=file_name)

    def computeAction(self, **kwargs):
        ob = kwargs
        obst = ob["FullSensor"]["obstacles"]
        for o in obst.values():
            o['type'] = 'sphere'

        action = self._planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            obst=obst,
        )
        return action.numpy()

