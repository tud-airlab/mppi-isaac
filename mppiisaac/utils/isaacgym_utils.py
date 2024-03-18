import mppiisaac
from isaacgym import gymapi
from typing import List
import yaml
from yaml import SafeLoader
import numpy as np
import pathlib
import os
from mppiisaac.planner.isaacgym_wrapper import ActorWrapper

FILE_PATH = pathlib.Path(__file__).parent.resolve()


def load_asset(gym, sim, actor_cfg):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = actor_cfg.fixed
    asset_root_path = f"{FILE_PATH}/../../assets"

    if actor_cfg.type == "robot":
        asset_file = "urdf/" + actor_cfg.urdf_file
        asset_options.flip_visual_attachments = actor_cfg.flip_visual
        asset_options.disable_gravity = not actor_cfg.gravity
        actor_asset = gym.load_asset(
            sim=sim,
            rootpath=asset_root_path,
            filename=asset_file,
            options=asset_options,
        )
    elif actor_cfg.type == "box":
        if actor_cfg.noise_sigma_size is not None:
            noise_sigma = np.array(actor_cfg.noise_sigma_size)
        else:
            noise_sigma = np.zeros((3,))
        noise = np.random.normal(loc=0, scale=noise_sigma, size=3)
        actor_asset = gym.create_box(
            sim=sim,
            width=actor_cfg.size[0] + noise[0],
            height=actor_cfg.size[1] + noise[1],
            depth=actor_cfg.size[2] + noise[2],
            options=asset_options,
        )
    elif actor_cfg.type == "sphere":
        if actor_cfg.noise_sigma_size is not None:
            noise_sigma = np.array(actor_cfg.noise_sigma_size)
        else:
            noise_sigma = np.zeros((1,))
        noise = np.random.normal(loc=0, scale=noise_sigma, size=1)
        actor_asset = gym.create_sphere(
            sim=sim,
            radius=actor_cfg.size[0] + noise[0],
            options=asset_options,
        )
    else:
        raise NotImplementedError(
            f"actor asset of type {actor_cfg.type} is not yet implemented!"
        )

    return actor_asset


def add_ground_plane(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1.0
    plane_params.dynamic_friction = 1.0
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

def load_actor_cfgs(actors: List[str]) -> List[ActorWrapper]:
    actor_cfgs = []
    for actor_name in actors:
        with open(
            f"{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml"
        ) as f:
            actor_cfgs.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))
    
    return actor_cfgs