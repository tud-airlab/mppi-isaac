from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass
import torch


@dataclass
class IsaacSimConfig(object):
    dt: float = 0.05
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_client_threads: int = 0
    viewer: bool = False
    num_obstacles: int = 1
    spacing: float = 6.0
    urdf_file: str = "urdf/pointRobot.urdf"


def parse_isaacsim_config(cfg: IsaacSimConfig) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
    sim_params.num_client_threads = cfg.num_client_threads

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.001

    # return the configured params
    return sim_params


class IsaacSimWrapper:
    def __init__(self, cfg: IsaacSimConfig, num_envs: int = 0):
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(
            compute_device=0,
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=parse_isaacsim_config(cfg),
        )
        self.cfg = cfg

        if cfg.viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        else:
            self.viewer = None

        self.add_ground_plane()

        self.robot_asset = self.load_robot_asset_from_urdf(
            asset_file=cfg.urdf_file, asset_root="../assets", fix_base_link=True
        )

        self.envs = self.create_envs(num_envs)

        # NOTE: for now there is a fixed number of obstacles that get instantiated at a far away place
        self.add_obstacles()

        self.gym.prepare_sim(self.sim)

        self.root_state = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self.dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def create_envs(self, num_envs):
        spacing = self.cfg.spacing
        num_per_row = int(num_envs**0.5)

        robot_init_pose = gymapi.Transform()
        robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05)

        envs = []

        for i in range(num_envs):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-spacing, 0.0, -spacing),
                gymapi.Vec3(spacing, spacing, spacing),
                num_per_row,
            )

            robot_handle = self.gym.create_actor(
                env=env,
                asset=self.robot_asset,
                pose=robot_init_pose,
                name="robot",
                group=i,
                filter=1,
            )

            # Update point bot dynamics / control mode
            props = self.gym.get_asset_dof_properties(self.robot_asset)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"].fill(0.0)
            props["damping"].fill(600.0)
            self.gym.set_actor_dof_properties(env, robot_handle, props)

            envs.append(env)
        return envs

    def add_obstacles(self):
        asset_options_objects = gymapi.AssetOptions()
        asset_options_objects.fix_base_link = True

        #object_asset = self.gym.create_box(
        #    sim=self.sim,
        #    width=0.5,
        #    height=0.2,
        #    depth=0.5,
        #    options=asset_options_objects,
        #)

        object_asset = self.gym.create_sphere(
            sim=self.sim,
            radius=1.0,
            options=asset_options_objects,
        )

        init_pose = gymapi.Transform()
        init_pose.p = gymapi.Vec3(1.0, 1.0, 0.05)

        obstacle_indexes = []

        for i, env in enumerate(self.envs):
            for j in range(self.cfg.num_obstacles):
                box_handle = self.gym.create_actor(
                    env=env,
                    asset=object_asset,
                    pose=init_pose,
                    name=f"obstacle{j}",
                    group=i,
                    filter=1,
                )

                if i == 0:
                    obstacle_indexes.append(self.gym.get_actor_index(env, box_handle, gymapi.IndexDomain.DOMAIN_ENV))
        self.obst_idxs = obstacle_indexes
            # self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    def add_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

    def load_robot_asset_from_urdf(
        self, asset_file, asset_root="../assets", fix_base_link=False
    ):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.armature = 0.01
        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def set_dof_state_tensor(self, state):
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(state))

    def set_dof_velocity_target_tensor(self, u):
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

    def update_root_state_tensor_by_obstacles(self, obst):

        num_actors = self.gym.get_sim_actor_count(self.sim)

        assert len(obst) == len(self.obst_idxs)
        for idx, obs in zip(self.obst_idxs, obst):
            obs_state = torch.tensor(
                [*obs[0], 0, 0, 0, 1, *obs[1], 0, 0, 0], device="cuda:0"
            )

            for i in range(100):
                self.root_state[i+100] = obs_state


        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))
