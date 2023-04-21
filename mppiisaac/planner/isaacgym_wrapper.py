from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass, field
import torch
import numpy as np
from enum import Enum
from typing import List, Optional, Any


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
    randomize_envs: bool = False
    randomization_sigma: float = 0.1


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
    flip_visual: bool = False
    urdf_file: str = None
    ee_link: str = None
    gravity: bool = True
    differential_drive: bool = False
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None
    wheel_count: Optional[float] = None
    left_wheel_joints: Optional[List[int]] = None
    right_wheel_joints: Optional[List[int]] = None
    caster_links: Optional[List[str]] = None


class IsaacGymWrapper:
    def __init__(
        self,
        cfg: IsaacGymConfig,
        actors: List[ActorWrapper],
        init_positions: List[List[float]],
        num_envs: int,
        viewer: bool = False,
    ):
        self.gym = gymapi.acquire_gym()
        self.env_cfg = actors

        assert len([a for a in self.env_cfg if a.type == "robot"]) == len(
            init_positions
        )

        for init_pos, actor_cfg in zip(init_positions, self.env_cfg):
            actor_cfg.init_pos = init_pos

        self.cfg = cfg
        if viewer:
            self.cfg.viewer = viewer
        self.num_envs = num_envs

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
            asset = self.load_asset(actor_cfg)
            self.env_actor_assets.append(asset)

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
                actor_cfg.handle = self.create_actor(
                    env, env_idx, actor_asset, actor_cfg
                )
            self.envs.append(env)

        self.ee_link_present = any([a.ee_link for a in self.env_cfg])

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
        if self.ee_link_present:
            self.ee_positions_buffer = []

        # helpfull slices
        self.robot_indices = torch.tensor([i for i, a in enumerate(self.env_cfg) if a.type == "robot"], device="cuda:0")

        self.obstacle_indices = torch.tensor([i for i, a in enumerate(self.env_cfg) if a.type in ["sphere", "box"]], device="cuda:0")

        if self.ee_link_present:
            self.ee_positions = self.rigid_body_state[
                :, self.robot_rigid_body_ee_idx, 0:3
            ]  # [x, y, z]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    @property
    def robot_positions(self):
        return torch.index_select(self.root_state, 1, self.robot_indices)[:, :, 0:3]

    @property
    def robot_velocities(self):
        return torch.index_select(self.root_state, 1, self.robot_indices)[:, :, 7:10]

    @property
    def obstacle_positions(self):
        return torch.index_select(self.root_state, 1, self.obstacle_indices)[:, :, 0:3]

    @property
    def robot_velocities(self):
        return torch.index_select(self.root_state, 1, self.obstacle_indices)[:, :, 7:10]

    def stop_sim(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def add_to_envs(self, additions):
        for a in additions:
            self.env_cfg.append(ActorWrapper(**a))
        self.stop_sim()
        self.start_sim()

    def load_asset(self, actor_cfg):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = actor_cfg.fixed
        
        if actor_cfg.type == "robot":
            asset_file = "urdf/" + actor_cfg.urdf_file
            asset_options.flip_visual_attachments = actor_cfg.flip_visual
            asset_options.disable_gravity = not actor_cfg.gravity
            actor_asset = self.gym.load_asset(
                sim=self.sim,
                rootpath=f"{file_path}/../../assets",
                filename=asset_file,
                options=asset_options,
            )
        elif actor_cfg.type == "box":
            noise = (
                np.random.normal(loc=0, scale=self.cfg.randomization_sigma, size=3)
                * self.cfg.randomize_envs * (actor_cfg.name == "block_to_push")
            )
            actor_asset = self.gym.create_box(
                sim=self.sim,
                width=actor_cfg.size[0] + noise[0],
                height=actor_cfg.size[1] + noise[1],
                depth=actor_cfg.size[2], # + noise[2], 
                options=asset_options,
            )
        elif actor_cfg.type == "sphere":
            noise = (
                np.random.normal(loc=0, scale=self.cfg.randomization_sigma, size=1)
                * self.cfg.randomize_envs * (actor_cfg.name == "block_to_push")
            )
            print(noise)
            actor_asset = self.gym.create_sphere(
                sim=self.sim,
                radius=actor_cfg.size[0] + noise[0],
                options=asset_options,
            )
        else:
            raise NotImplementedError(
                f"actor asset of type {actor_cfg.type} is not yet implemented!"
            )

        return actor_asset

    def create_actor(self, env, env_idx, asset, actor: ActorWrapper) -> int:
        if self.cfg.randomize_envs and actor.type != 'robot':
            asset = self.load_asset(actor)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*actor.init_pos)
        pose.r = gymapi.Quat(*actor.init_ori)
        handle = self.gym.create_actor(
            env=env,
            asset=asset,
            pose=pose,
            name=actor.name,
            group=env_idx if actor.collision else env_idx+self.num_envs,
        )

        if self.cfg.randomize_envs and actor.name == 'obj_to_push':
            actor.color = np.random.rand(3)

        self.gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*actor.color)
        )
        props = self.gym.get_actor_rigid_body_properties(env, handle)
        props[0].mass =  actor.mass + (actor.name == "block_to_push") * np.random.uniform(-0.3*actor.mass, 0.3*actor.mass)
        self.gym.set_actor_rigid_body_properties(env, handle, props)

        body_names = self.gym.get_actor_rigid_body_names(env, handle)
        body_to_shape = self.gym.get_actor_rigid_body_shape_indices(env, handle)
        caster_shapes = [
            b.start
            for body_idx, b in enumerate(body_to_shape)
            if actor.caster_links is not None
            and body_names[body_idx] in actor.caster_links
        ]

        props = self.gym.get_actor_rigid_shape_properties(env, handle)
        for i, p in enumerate(props):
            p.friction = actor.friction + (actor.name == "block_to_push") * np.random.uniform(-0.3*actor.friction, 0.3*actor.friction)
            p.torsion_friction = np.random.uniform(0.001, 0.01) 
            p.rolling_friction = actor.friction + (actor.name == "block_to_push") * np.random.uniform(-0.3*actor.friction, 0.3*actor.friction)

            if i in caster_shapes:
                p.friction = 0
                p.torsion_friction = 0
                p.rolling_friction = 0

        self.gym.set_actor_rigid_shape_properties(env, handle, props)

        if actor.type == "robot":
            # TODO: Currently the robot_rigid_body_ee_idx is only supported for a single robot case.
            if actor.ee_link:
                self.robot_rigid_body_ee_idx = self.gym.find_actor_rigid_body_index(
                    env, handle, actor.ee_link, gymapi.IndexDomain.DOMAIN_ENV
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

    def _ik(self, actor, u):
        r = actor.wheel_radius
        L = actor.wheel_base
        wheel_sets = actor.wheel_count // 2

        # Diff drive fk
        u_ik = u.clone()
        u_ik[:, 0] = (u[:, 0] / r) - ((L * u[:, 1]) / (2 * r))
        u_ik[:, 1] = (u[:, 0] / r) + ((L * u[:, 1]) / (2 * r))

        if wheel_sets > 1:
            u_ik = u_ik.repeat(1, wheel_sets)

        return u_ik

    def apply_robot_cmd_velocity(self, u_desired):
        vel_dof_shape = list(self.dof_state.size())
        vel_dof_shape[1] = vel_dof_shape[1] // 2
        u = torch.zeros(vel_dof_shape, device="cuda:0")

        u_desired_idx = 0
        u_dof_idx = 0
        for actor in self.env_cfg:
            if actor.type != "robot":
                continue
            actor_dof_count = self.gym.get_actor_dof_count(self.envs[0], actor.handle)
            dof_dict = self.gym.get_actor_dof_dict(self.envs[0], actor.handle)
            for i in range(actor_dof_count):
                if (
                    actor.differential_drive
                    and i >= actor_dof_count - actor.wheel_count
                ):
                    u_ik = self._ik(
                        actor, u_desired[:, u_desired_idx : u_desired_idx + 2]
                    )
                    u[:, u_dof_idx : u_dof_idx + actor.wheel_count] = u_ik
                    u_desired_idx += 2
                    u_dof_idx += actor.wheel_count
                    break
                else:
                    u[:, u_dof_idx] = u_desired[:, u_desired_idx]
                    u_desired_idx += 1
                    u_dof_idx += 1

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(u))

    def reset_robot_state(self, q, qdot):
        """
        This function is mainly used for compatibility with gym_urdf_envs pybullet sim.
        """

        q_idx = 0

        dof_state = []
        for actor in self.env_cfg:
            if actor.type != "robot":
                continue

            actor_dof_count = self.gym.get_actor_dof_count(self.envs[0], actor.handle)

            if actor.differential_drive:
                actor_q_count = actor_dof_count - (actor.wheel_count - 3)
            else:
                actor_q_count = actor_dof_count

            actor_q = q[q_idx : q_idx + actor_q_count]
            actor_qdot = qdot[q_idx : q_idx + actor_q_count]

            if actor.differential_drive:
                pos = actor_q[:3]
                vel = actor_qdot[:3]

                self.set_state_tensor_by_pos_vel(actor.handle, pos, vel)

                # assuming wheels at the back of the dof tensor
                actor_q = list(actor_q[3:]) + [0] * actor.wheel_count
                actor_qdot = list(actor_qdot[3:]) + [0] * actor.wheel_count

            for _q, _qdot in zip(actor_q, actor_qdot):
                dof_state.append(_q)
                dof_state.append(_qdot)

            q_idx += actor_q_count

        dof_state_tensor = torch.tensor(dof_state).type(torch.float32).to("cuda:0")

        dof_state_tensor = dof_state_tensor.repeat(self.num_envs, 1)
        self.set_dof_state_tensor(dof_state_tensor)

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )

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

        if self.ee_link_present:
            self.ee_positions_buffer.append(self.ee_positions.clone())

    def set_root_state_tensor_by_actor_idx(self, state_tensor, idx):
        for i in range(self.num_envs):
            self.root_state[i, idx] = state_tensor

    def save_root_state(self):
        self.saved_root_state = self.root_state.clone()

    def reset_root_state(self):
        if self.ee_link_present:
            self.ee_positions_buffer = []

        if self.saved_root_state is not None:
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.saved_root_state)
            )

    def set_state_tensor_by_pos_vel(self, handle, pos, vel):
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

        self.root_state[:, handle, :2] = torch.tensor(pos[:2], device="cuda:0")
        self.root_state[:, handle, 3:7] = torch.tensor(orientation, device="cuda:0")
        self.root_state[:, handle, 7:10] = torch.tensor(vel, device="cuda:0")

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
                obst_idx = [
                    idx for idx, actor in enumerate(self.env_cfg) if actor.name == name
                ][0]
            except:
                self.env_cfg.append(
                    ActorWrapper(
                        **{
                            "type": o_type,
                            "name": name,
                            "handle": None,
                            "size": o_size,
                            "fixed": True,
                        }
                    )
                )
                env_cfg_changed = True
                continue

            obst_state = torch.tensor(
                [*pos, 0, 0, 0, 1, *vel, 0, 0, 0], device="cuda:0"
            )

            # Note: reset simulator if size changed, because this cannot be done at runtime.
            if not all([a == b for a, b in zip(o_size, self.env_cfg[obst_idx].size)]):
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
        line_segments = (
            torch.concat((lines[:-1], lines[1:]), axis=-1)
            .flatten(end_dim=-2)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        num_lines = line_segments.shape[0]
        colors = np.zeros((num_lines, 3), dtype=np.float32)
        colors[:, 1] = 255
        self.gym.add_lines(
            self.viewer, self.envs[env_idx], num_lines, line_segments, colors
        )
