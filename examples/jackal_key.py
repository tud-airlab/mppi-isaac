import gym
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
import yaml
from yaml import SafeLoader
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import hydra
from omegaconf import OmegaConf
import os
import mppiisaac
import torch
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from isaacgym import gymapi
import time

import io


@hydra.main(version_base=None, config_path="../conf", config_name="config_jackal_robot")
def run_jackal_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    actors=[]
    for actor_name in cfg.actors:
        with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml') as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))


    sim = IsaacGymWrapper(
        cfg.isaacgym,
        init_positions=cfg.initial_actor_positions,
        actors=actors,
        num_envs=1,
        viewer=True,
    )

    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0)
    )

    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_A, "left")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_S, "down")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_D, "right")
    sim.gym.subscribe_viewer_keyboard_event(sim.viewer, gymapi.KEY_W, "up")

    action = torch.tensor([0., 0.], device="cuda:0")
    for _ in range(cfg.n_steps):
        for e in sim.gym.query_viewer_action_events(sim.viewer):
            if e.action == "up":
                action[0] = 0.5
            if e.action == "down":
                action[0] = -0.2
                #action[1] = 0.
            if e.action == "left":
                action[1] = 5.8
            if e.action == "right":
                action[1] = -5.8

        # Apply action
        sim.apply_robot_cmd_velocity(torch.unsqueeze(action, axis=0))

        # Step simulator
        sim.step()
        sim.gym.sync_frame_time(sim.sim)
    return {}


if __name__ == "__main__":
    res = run_jackal_robot()
