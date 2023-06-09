from omegaconf import OmegaConf
from plannerbenchmark.generic.planner import Planner
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import time
import torch


class Objective(object):
    def __init__(self, goal, device):
        self.nav_goal = torch.tensor(goal, device=device)
        self.ort_goal = torch.tensor([1, 0, 0, 0], device=device)
        self.w_coll = 10.
        self.w_pos = 1.
        self.w_ort = 0.0

    def compute_cost(self, sim):
        pos = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, :3]
        ort = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, 3:7]

        reach_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)
        align_cost = torch.linalg.norm(ort - self.ort_goal, axis=1)

        # Collision avoidance with contact forces
        xyz_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        coll_cost = torch.sum(xyz_contatcs.reshape([sim.num_envs, int(xyz_contatcs.size(dim=0)/sim.num_envs)])[:, 1:sim.num_bodies], 1) # skip the first, it is the robot

        return reach_cost * self.w_pos + align_cost * self.w_ort + coll_cost * self.w_coll

class MPPIPlanner(Planner):

    def __init__(self, exp, **kwargs):
        super().__init__(exp, **kwargs)
        self.cfg = kwargs['config']
        initial_actor_position = exp.initState()[0].tolist() 
        initial_actor_position = [0.0, 0.0, 0.05]
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

