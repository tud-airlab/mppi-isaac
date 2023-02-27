from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.utils.config_store import ExampleConfig
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import numpy as np
import torch
import hydra


torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class FabricsPointPrior(object):
    def __init__(self, cfg):
        self.nav_goal = list(cfg.goal)
        self.weight = 5.0
        self.dt = cfg.isaacgym.dt
        self.device = cfg.mppi.device
        self.env_id = -2
        self._fabrics_prior = fabrics_point(self.nav_goal, self.weight)

    def compute_command(self, sim):
        dof_state = sim.dof_state[self.env_id].cpu()
        pos = np.array([dof_state[0], dof_state[2]])
        vel = np.array([dof_state[1], dof_state[3]])

        acc_action = self._fabrics_prior.compute_action(
            q=pos,
            qdot=vel,
            x_goal_0=self.nav_goal,
            weight_goal_0=self.weight,
            radius_body_1=np.array([0.2]),
        )
        if any(np.isnan(acc_action)):
            acc_action = np.zeros_like(acc_action)
        vel_action = torch.tensor(
            vel + acc_action * self.dt, dtype=torch.float32, device=self.device
        )
        out = torch.cat((vel_action, torch.tensor([0], device="cuda:0")), axis=0)
        return out


def fabrics_point(goal, weight=0.5):
    """
    Initializes the fabric planner for the point robot.
    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.
    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.
    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    goal_dict = {
        "subgoal0": {
            "weight": weight,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 1,
            "desired_position": goal,
            "epsilon": 0.1,
            "type": "staticSubGoal",
        }
    }
    goal_composition = GoalComposition(name="goal", content_dict=goal_dict)

    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        robot_type,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal_composition,
        number_obstacles=0,
    )
    planner.concretize()
    return planner


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_point_robot"
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

    prior = FabricsPointPrior(cfg)
    prior.env_id = 0

    while True:
        # Compute fabrics action
        command = prior.compute_command(sim)

        # Apply action
        sim.set_dof_velocity_target_tensor(command)

        # Update sim
        sim.step()


if __name__ == "__main__":
    test()
