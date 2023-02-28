import mppiisaac
import os
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.utils.config_store import ExampleConfig
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import numpy as np
import torch
import hydra


torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class FabricsPandaPrior(object):
    def __init__(self, cfg):
        self.nav_goal = list(cfg.goal)
        assert len(self.nav_goal) == 3

        self.weight = 1.0
        self.dt = cfg.isaacgym.dt
        self.device = cfg.mppi.device
        self.env_id = -2

        # Convert cartesian goal to goal composition
        goal_dict = {
            "subgoal0": {
                "weight": self.weight,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_link7",
                "desired_position": self.nav_goal,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)

        self._fabrics_prior = fabrics_panda(goal, cfg.urdf_file)

    def compute_command(self, sim):
        dofs = sim.dof_state[self.env_id].cpu() 
        pos = np.array(dofs[::2])
        vel = np.array(dofs[1::2])

        acc_action = self._fabrics_prior.compute_action(
            q=pos,
            qdot=vel,
            x_goal_0=self.nav_goal,
            weight_goal_0=self.weight,
            radius_body_panda_link3=np.array([0.02]),
            radius_body_panda_link4=np.array([0.02]),
            radius_body_panda_link7=np.array([0.02]),
        )
        if any(np.isnan(acc_action)):
            acc_action = np.zeros_like(acc_action)
        vel_action = torch.tensor(
            vel + acc_action * self.dt, dtype=torch.float32, device=self.device
        )
        return vel_action

def fabrics_panda(goal, urdf_file):
    """
    Initializes the fabric planner for the panda robot.
    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.
    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.
    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    urdf_file = "/panda_bullet/panda.urdf"
    urdf_abs_path = os.path.dirname(mppiisaac.__file__) + "/../assets/urdf/" + urdf_file
    with open(urdf_abs_path, "r") as file:
        urdf = file.read()

    planner = ParameterizedFabricPlanner(
        7,
        'panda',
        urdf=urdf,
        root_link='panda_link0',
        end_link='panda_link7',
    )
    q = planner.variables.position_variable()
    collision_links = ['panda_link7', 'panda_link3', 'panda_link4']
    self_collision_pairs = {}
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        number_obstacles=0,
        limits=panda_limits,
    )
    planner.concretize()
    return planner


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_panda"
)
def test(cfg: ExampleConfig):

    cfg.isaacgym.viewer = True
    sim = IsaacGymWrapper(
        cfg.isaacgym,
        cfg.urdf_file,
        cfg.fix_base,
        cfg.flip_visual,
        num_envs=1,
    )

    prior = FabricsPandaPrior(cfg)
    prior.env_id = 0

    while True:
        # Compute fabrics action
        command = prior.compute_command(sim)

        # Apply action
        sim.set_dof_velocity_target_tensor(command)

        # Update sim
        sim.step()

        sim.gym.sync_frame_time(sim.sim)


if __name__ == "__main__":
    test()
