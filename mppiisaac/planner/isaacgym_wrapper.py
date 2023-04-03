from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass, field
import torch
import numpy as np
from enum import Enum
from typing import List, Optional


import pathlib

file_path = pathlib.Path(__file__).parent.resolve()


@dataclass
class IsaacGymConfig(object):
    dt: float = 0.05
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_client_threads: int = 0
    viewer: bool = False
    num_obstacles: int = 10
    spacing: float = 6.0


def parse_isaacgym_config(cfg: IsaacGymConfig) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.dt
    sim_params.substeps = cfg.substeps
    sim_params.use_gpu_pipeline = cfg.use_gpu_pipeline
    sim_params.num_client_threads = cfg.num_client_threads

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.001

    # return the configured params
    return sim_params


class SupportedActorTypes(Enum):
    Axis = 1
    Robot = 2
    Sphere = 3
    Box = 4


@dataclass
class ActorWrapper:
    type: SupportedActorTypes
    name: str
    init_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0  # kg
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fixed: bool = False
    collision: bool = True
    friction: float = 0.8
    handle: Optional[int] = None


class IsaacGymWrapper:
    def __init__(
        self,
        cfg: IsaacGymConfig,
        urdf_file: str,
        fix_base: bool,
        flip_visual: bool,
        num_envs: int = 0,
        ee_link: str = None,
        disable_gravity: bool = False,
        viewer: bool = False,
    ):
        self.gym = gymapi.acquire_gym()

        self.env_cfg = [
            {
                "type": "box",
                "name": "x",
                "handle": None,
                "fixed": True,
                "size": [0.5, 0.05, 0.01],
                "init_pos": [0, 0.25, 0.01],
                "color": [1, 0.0, 0.2],
                "collision": False
            },
            {
                "type": "box",
                "name": "y",
                "handle": None,
                "fixed": True,
                "size": [0.05, 0.5, 0.01],
                "init_pos": [0.25, 0, 0.01],
                "color": [0.0, 1, 0.2],
                "collision": False
            },
            {"type": "robot", "name": "main_robot", "handle": None, "fixed": True},
        ]
        self.env_cfg = [ActorWrapper(**a) for a in self.env_cfg]

        self.cfg = cfg
        if viewer:
            self.cfg.viewer = viewer
        self.num_envs = num_envs
        self._urdf_file = urdf_file
        self._fix_base = fix_base
        self._flip_visual = flip_visual
        self._ee_link = ee_link
        self._disable_gravity = disable_gravity

        self.start_sim()

    def start_sim(self):
        self.sim = self.gym.create_sim(
            compute_device=0,
            graphics_device=0,
            type=gymapi.SIM_PHYSX,
            params=parse_isaacgym_config(self.cfg),
        )

        if self.cfg.viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        else:
            self.viewer = None

        self.add_ground_plane()

        # Load / create assets for all actors in the envs
        self.env_actor_assets = []
        for actor_cfg in self.env_cfg:
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = actor_cfg.fixed
            if actor_cfg.type == "robot":
                asset_file = "urdf/" + self._urdf_file
                asset_options.flip_visual_attachments = self._flip_visual
                asset_options.disable_gravity = self._disable_gravity
                actor_asset = self.gym.load_asset(
                    sim=self.sim,
                    rootpath=f"{file_path}/../../assets",
                    filename=asset_file,
                    options=asset_options,
                )
            elif actor_cfg.type == "box":
                actor_asset = self.gym.create_box(
                    sim=self.sim,
                    width=actor_cfg.size[0],
                    height=actor_cfg.size[1],
                    depth=actor_cfg.size[2],
                    options=asset_options,
                )
            elif actor_cfg.type == "sphere":
                actor_asset = self.gym.create_sphere(
                    sim=self.sim,
                    radius=actor_cfg.size[0],
                    options=asset_options,
                )
            else:
                raise NotImplementedError(f"actor asset of type {actor_cfg.type} is not yet implemented!")
            self.env_actor_assets.append(actor_asset)

        # Create envs and fill with assets
        self.envs = []
        for env_idx in range(self.num_envs):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-self.cfg.spacing, 0.0, -self.cfg.spacing),
                gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing),
                int(self.num_envs**0.5),
            )

            for actor_asset, actor_cfg in zip(self.env_actor_assets, self.env_cfg):
                self.create_actor(env, env_idx, actor_asset, actor_cfg)
            self.envs.append(env)

        self.gym.prepare_sim(self.sim)

        self.root_state = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)
        self.saved_root_state = None
        self.dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        ).view(self.num_envs, -1)
        self.rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)

        self.net_cf = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )

        # save buffer of ee states
        if self._ee_link:
            self.ee_positions_buffer = []

        # helpfull slices
        self.robot_positions = self.root_state[:, 2, 0:3]  # [x, y, z]
        self.robot_velocities = self.root_state[:, 2, 7:10]  # [x, y, z]
        self.obstacle_positions = self.root_state[:, 3:, 0:3]  # [x, y, z]
        if self._ee_link:
            self.ee_positions = self.rigid_body_state[:, self.robot_rigid_body_ee_idx, 0:3] # [x, y, z]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def stop_sim(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    def add_to_envs(self, additions):
        for a in additions:
            self.env_cfg.append(ActorWrapper(**a))
        self.stop_sim()
        self.start_sim()

    def create_actor(self, env, env_idx, asset, actor: ActorWrapper) -> int:
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*actor.init_pos)
        pose.r = gymapi.Quat(*actor.init_ori)
        handle = self.gym.create_actor(
            env=env,
            asset=asset,
            pose=pose,
            name=actor.name,
            group=env_idx if actor.collision else -1,
        )
        self.gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*actor.color)
        )
        props = self.gym.get_actor_rigid_body_properties(env, handle)
        props[0].mass = actor.mass
        self.gym.set_actor_rigid_body_properties(env, handle, props)
        props = self.gym.get_actor_rigid_shape_properties(env, handle)
        props[0].friction = actor.friction
        props[0].torsion_friction = actor.friction
        props[0].rolling_friction = actor.friction
        self.gym.set_actor_rigid_shape_properties(env, handle, props)

        if actor.type == "robot":
            if self._ee_link:
                self.robot_rigid_body_ee_idx = self.gym.find_actor_rigid_body_index(
                    env, handle, self._ee_link, gymapi.IndexDomain.DOMAIN_ENV
                )

            props = self.gym.get_asset_dof_properties(asset)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"].fill(0.0)
            props["damping"].fill(1e7)
            self.gym.set_actor_dof_properties(env, handle, props)

        return handle

    def add_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

    def set_dof_state_tensor(self, state):
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(state))

    def set_dof_velocity_target_tensor(self, u):
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            # self.gym.sync_frame_time(self.sim)

        if self._ee_link:
            self.ee_positions_buffer.append(self.ee_positions.clone())

    def set_root_state_tensor_by_actor_idx(self, state_tensor, idx):
        for i in range(self.num_envs):
            self.root_state[i, idx] = state_tensor

    def save_root_state(self):
        self.saved_root_state = self.root_state.clone()

    def reset_root_state(self):
        if self.saved_root_state is not None:
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.saved_root_state)
            )

    def update_root_state_tensor_robot(self, pos, vel):
        roll = 0
        pitch = 0
        yaw = pos[2]
        orientation = [
            np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2)
            - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2),
            np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
            + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2),
            np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
            - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2),
            np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2),
        ]

        self.root_state[:, 2, :2] = torch.tensor(pos[:2], device="cuda:0")
        self.root_state[:, 2, 3:7] = torch.tensor(orientation, device="cuda:0")
        self.root_state[:, 2, 7:10] = torch.tensor(vel, device="cuda:0")

    def update_root_state_tensor_by_obstacles(self, obstacles):
        """
        Note: obstacles param should be a list of obstacles,
        where each obstacle is a list of the following order [position, velocity, type, size]
        """
        env_cfg_changed = False

        for i, obst in enumerate(obstacles):
            pos, vel, o_type, o_size = obst
            name = f"{o_type}{i}"
            try:
                obst_idx = [idx for idx, actor in enumerate(self.env_cfg) if actor.name == name]
            except:
                self.env_cfg.append(
                    ActorWrapper({
                        "type": o_type,
                        "name": name,
                        "handle": None,
                        "size": o_size,
                        "fixed": True,
                    })
                )
                env_cfg_changed = True
                continue

            obst_state = torch.tensor(
                [*pos, 0, 0, 0, 1, *vel, 0, 0, 0], device="cuda:0"
            )

            # Note: reset simulator if size changed, because this cannot be done at runtime.
            if not all(
                [a == b for a, b in zip(o_size, self.env_cfg[obst_idx].size)]
            ):
                env_cfg_changed = True
                self.env_cfg[obst_idx].size = o_size

            for j, env in enumerate(self.envs):
                self.root_state[j, obst_idx] = obst_state

        # restart sim for env changes
        if env_cfg_changed:
            self.stop_sim()
            self.start_sim()

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )

    def update_root_state_tensor_by_obstacles_tensor(self, obst_tensor):
        for i, o_tensor in enumerate(obst_tensor):
            if self.env_cfg[i + 3].fixed:
                continue
            self.root_state[:, i + 3] = o_tensor.repeat(self.num_envs, 1)

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )

    def draw_lines(self, lines, env_idx=0):
        # convert list of vertices into line segments
        line_segments = torch.concat((lines[:-1], lines[1:]), axis=-1).flatten(end_dim=-2).cpu().numpy().astype(np.float32)
        num_lines = line_segments.shape[0]
        colors = np.zeros((num_lines,3), dtype=np.float32)
        colors[:, 1] = 255
        self.gym.add_lines(self.viewer, self.envs[env_idx], num_lines, line_segments, colors)