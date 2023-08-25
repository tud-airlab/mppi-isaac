from omegaconf import OmegaConf
from plannerbenchmark.generic.planner import Planner
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner

import torch

class Objective(object):
    def __init__(self, goal, device):
        self.nav_goal = torch.tensor(goal, device=device)

        self.w_nav = 2.0 # 5.0
        self.w_obs = 1.
        self.w_coll = 0.0 # 0.01 

    def compute_cost(self, sim: IsaacGymWrapper):
        dof_state = sim.dof_state
        pos = torch.cat((dof_state[:, 0].unsqueeze(1), dof_state[:, 2].unsqueeze(1)), 1)
        obs_positions = sim.obstacle_positions

        nav_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)
        # print(nav_cost[-1])
        # Coll avoidance with distance
        obs_cost = torch.sum(
            1 / torch.linalg.norm(obs_positions[:, :, :2] - pos.unsqueeze(1), axis=2),
            axis=1,
        )

        # Collision avoidance with contact forces
        xy_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1)), 1)),1)
        coll = torch.sum(xy_contatcs.reshape([sim.num_envs, int(xy_contatcs.size(dim=0)/sim.num_envs)])[:, 1:sim.num_bodies], 1) # skip the first, it is the robot

        return nav_cost * self.w_nav + coll * self.w_coll + obs_cost * self.w_obs

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
        else:
            self._planner.update_objective(objective)


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

