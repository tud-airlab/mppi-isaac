from isaacgym import gymapi
from isaacgym import gymtorch
from dataclasses import dataclass
import torch
import numpy as np

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


class IsaacGymWrapper:
    def __init__(
        self,
        cfg: IsaacGymConfig,
        urdf_file: str,
        fix_base: bool,
        flip_visual: bool,
        num_envs: int = 0,
    ):
        self.gym = gymapi.acquire_gym()

        # Keep track of env idxs. Everytime an actor get added append with a tuple of (idx, type, name)
        self.env_cfg = [
            {"type": "axis", "name": "x", "handle": None, "fixed": True},
            {"type": "axis", "name": "y", "handle": None, "fixed": True},
            {"type": "robot", "name": "main_robot", "handle": None},
        ]
        self.cfg = cfg
        self.num_envs = num_envs
        self._urdf_file = urdf_file
        self._fix_base = fix_base
        self._flip_visual = flip_visual
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

        asset_file = "urdf/" + self._urdf_file
        print(asset_file)
        self._robot_asset = self.load_robot_asset_from_urdf(
            asset_file=asset_file,
            asset_root=f"{file_path}/../../assets",
            fix_base_link=self._fix_base,
            flip_visual_attachments=self._flip_visual,
        )

        self.envs = [self.create_env(i) for i in range(self.num_envs)]

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

        # helpfull slices
        self.robot_positions = self.root_state[:, 2, 0:3]  # [x, y, z]
        self.robot_velocities = self.root_state[:, 2, 7:10]  # [x, y, z]
        self.obstacle_positions = self.root_state[:, 3:, 0:3]  # [x, y, z]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def stop_sim(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def create_env(self, env_idx):
        env = self.gym.create_env(
            self.sim,
            gymapi.Vec3(-self.cfg.spacing, 0.0, -self.cfg.spacing),
            gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing),
            int(self.num_envs**0.5),
        )

        x_axis = self.add_box(
            env,
            env_idx=-2,
            name=self.env_cfg[0]["name"],
            whd=(0.5, 0.05, 0.01),
            pos=gymapi.Vec3(0.25, 0, 0.01),
            color=gymapi.Vec3(1, 0.0, 0.2),
        )
        self.env_cfg[0]["handle"] = x_axis

        y_axis = self.add_box(
            env=env,
            env_idx=-2,
            name=self.env_cfg[1]["name"],
            whd=(0.05, 0.5, 0.01),
            pos=gymapi.Vec3(0, 0.25, 0.01),
            color=gymapi.Vec3(0.0, 1, 0.2),
        )
        self.env_cfg[1]["handle"] = y_axis

        robot_init_pose = gymapi.Transform()
        robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.05)

        robot_handle = self.gym.create_actor(
            env=env,
            asset=self._robot_asset,
            pose=robot_init_pose,
            name=self.env_cfg[2]["name"],
            group=env_idx,
        )
        self.env_cfg[2]["handle"] = robot_handle

        # Update point bot dynamics / control mode
        props = self.gym.get_asset_dof_properties(self._robot_asset)
        props["driveMode"].fill(gymapi.DOF_MODE_VEL)
        props["stiffness"].fill(0.0)
        props["damping"].fill(1e7)
        self.gym.set_actor_dof_properties(env, robot_handle, props)

        for obst_cfg in self.env_cfg[3:]:
            if obst_cfg["type"] == "sphere":
                # add spheres
                handle = self.add_sphere(
                    env=env,
                    env_idx=env_idx,
                    name=obst_cfg["name"],
                    radius=obst_cfg["size"][0],
                    pos=gymapi.Vec3(0, 0, -20),
                    color=gymapi.Vec3(1.0, 1.0, 1.0),
                )
            elif obst_cfg["type"] == "box":
                # add spheres
                handle = self.add_box(
                    env=env,
                    env_idx=env_idx,
                    name=obst_cfg["name"],
                    whd=obst_cfg["size"],
                    pos=gymapi.Vec3(0, 0, -20),
                    color=gymapi.Vec3(1.0, 1.0, 1.0),
                    fixed=obst_cfg["fixed"],
                )
            else:
                raise NotImplementedError(
                    f"obstacles of type {obst_cfg['type']} are not supported!"
                )
            obst_cfg["handle"] = handle
        return env

    def add_box(
        self,
        env,
        env_idx: int,
        name: str,
        whd: list,
        pos: gymapi.Vec3,
        color: gymapi.Vec3,
        fixed: bool = True,
        mass: float = 1.0,
    ) -> int:
        asset_options_objects = gymapi.AssetOptions()
        asset_options_objects.fix_base_link = fixed
        object_asset = self.gym.create_box(
            sim=self.sim,
            width=whd[0],
            height=whd[1],
            depth=whd[2],
            options=asset_options_objects,
        )

        pose = gymapi.Transform()
        pose.p = pos
        handle = self.gym.create_actor(
            env=env,
            asset=object_asset,
            pose=pose,
            name=name,
            group=env_idx,
        )
        self.gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
        )
        props = self.gym.get_actor_rigid_body_properties(env, handle)
        props[0].mass = mass
        self.gym.set_actor_rigid_body_properties(env, handle, props)
        return handle

    def add_sphere(
        self,
        env,
        env_idx: int,
        name: str,
        radius: float,
        pos: gymapi.Vec3,
        color: gymapi.Vec3,
        fixed: bool = True,
    ) -> int:
        asset_options_objects = gymapi.AssetOptions()
        asset_options_objects.fix_base_link = fixed
        object_asset = self.gym.create_sphere(
            sim=self.sim,
            radius=radius,
            options=asset_options_objects,
        )

        pose = gymapi.Transform()
        pose.p = pos
        handle = self.gym.create_actor(
            env=env,
            asset=object_asset,
            pose=pose,
            name=name,
            group=env_idx,
        )
        self.gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
        )
        return handle

    def add_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self.gym.add_ground(self.sim, plane_params)

    def load_robot_asset_from_urdf(
        self,
        asset_file,
        asset_root=f"{file_path}/../../assets",
        fix_base_link=False,
        flip_visual_attachments=False,
    ):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = fix_base_link
        asset_options.armature = 0.01
        asset_options.flip_visual_attachments = flip_visual_attachments
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
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

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
                obst_idx = [actor["name"] for actor in self.env_cfg].index(name)
            except:
                self.env_cfg.append(
                    {"type": o_type, "name": name, "handle": None, "size": o_size}
                )
                env_cfg_changed = True
                continue

            obst_state = torch.tensor(
                [*pos, 0, 0, 0, 1, *vel, 0, 0, 0], device="cuda:0"
            )

            # Note: reset simulator if size changed, because this cannot be done at runtime.
            if not all(
                [a == b for a, b in zip(o_size, self.env_cfg[obst_idx]["size"])]
            ):
                env_cfg_changed = True
                self.env_cfg[obst_idx]["size"] = o_size

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
            self.root_state[:, i + 3] = o_tensor.repeat(self.num_envs, 1)

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_state)
        )
