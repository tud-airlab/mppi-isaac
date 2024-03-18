import mppiisaac
import os
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from mppiisaac.utils.config_store import ExampleConfig
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import numpy as np
import torch
import hydra
import yourdfpy
from isaacgym import gymapi


torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)


class FabricsPandaPrior(object):
    def __init__(self, cfg, max_num_obstacles=10):
        self.nav_goal = list(cfg.goal)
        assert len(self.nav_goal) == 3

        self.weight = 1.0
        self.dt = cfg.isaacgym.dt
        self.device = cfg.mppi.device
        self.env_id = -2
        self.max_num_obstacles = max_num_obstacles

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

        self._fabrics_prior = fabrics_panda(goal, cfg.urdf_file, self.max_num_obstacles)

    def compute_command(self, sim):
        dofs = sim.dof_state[self.env_id].cpu() 
        pos = np.array(dofs[::2])
        vel = np.array(dofs[1::2])

        obst_positions = np.array(sim.obstacle_positions[self.env_id].cpu())
        obst_indices = torch.tensor([i for i, a in enumerate(sim.env_cfg) if a.type in ["sphere", "box"]], device="cuda:0")

        x_obsts = []
        radius_obsts = []
        for i in range(self.max_num_obstacles):
            if i < len(obst_positions):
                x_obsts.append(obst_positions[i])
                if sim.env_cfg[obst_indices[i]].type == 'sphere':
                    radius_obsts.append(sim.env_cfg[obst_indices[i]].size[0])
                else:
                    radius_obsts.append(0.2)
            else:
                x_obsts.append(np.array([100, 100, 100]))
                radius_obsts.append(0.2)

        acc_action = self._fabrics_prior.compute_action(
            q=pos,
            qdot=vel,
            x_obsts=x_obsts,
            radius_obsts=radius_obsts,
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

def fabrics_panda(goal, urdf_file, max_num_obstacles=10):
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

    # NOTE: overwrite urdf, because it has to be the bullet version to parse correctly by urdfFK
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

    urdf_robot = yourdfpy.urdf.URDF.load(urdf_abs_path)
    panda_limits = [
        [joint.limit.lower, joint.limit.upper]
        for joint in urdf_robot.robot.joints
        if joint.type == "revolute"
    ]

    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        number_obstacles=max_num_obstacles,
        limits=panda_limits,
    )
    planner.concretize()
    return planner


@hydra.main(
    version_base=None, config_path=".", config_name="config_panda"
)
def test(cfg: ExampleConfig):

    cfg.isaacgym.viewer = True
    sim = IsaacGymWrapper(
        cfg.isaacgym,
        cfg.urdf_file,
        cfg.fix_base,
        cfg.flip_visual,
        num_envs=1,
        robot_init_pos=cfg.initial_position,
        visualize_link=cfg.visualize_link,
        disable_gravity=cfg.disable_gravity,
    )

    sim.add_to_envs([
        {
            "type": "sphere",
            "name": "sphere0",
            "handle": None,
            "size": [0.1],
            "fixed": True,
        }
    ])
    sim.stop_sim()
    sim.start_sim()

    sim.update_root_state_tensor_by_obstacles_tensor(
        torch.tensor([[0.6, 0.3, 0.9, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], device="cuda:0")
    )

    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0)
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
