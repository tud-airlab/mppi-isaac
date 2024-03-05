import gym
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
import yaml
from yaml import SafeLoader
import mppiisaac
import numpy as np
import hydra
from omegaconf import OmegaConf
import torch
import zerorpc
from mppiisaac.utils.config_store import ExampleConfig
from isaacgym import gymapi
import time
from examples.boxer_push_client import Objective
import sys
import os
import io

def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)

def set_viewer(sim):
    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1., 6.5, 4), gymapi.Vec3(1., 0, 0)        # CAMERA LOCATION, CAMERA POINT OF INTEREST
    )

@hydra.main(version_base=None, config_path="../conf", config_name="config_anymal")
def run_anymal(cfg: ExampleConfig):
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
        device=cfg.mppi.device,
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")
    
    set_viewer(sim)
 
    while True:
        # Reset state
        planner.reset_rollout_sim(
            torch_to_bytes(sim.dof_state[0]),
            torch_to_bytes(sim.root_state[0]),
            torch_to_bytes(sim.rigid_body_state[0]),
        )
       
        # Compute action
        action = bytes_to_torch(planner.command())
        if torch.any(torch.isnan(action)):
            print("nan action")
            action = torch.zeros_like(action)

        # Apply action
        #sim.set_dof_velocity_target_tensor(10*action)
        sim.apply_robot_cmd_velocity(torch.unsqueeze(action, axis=0))

        # Step simulator
        sim.step()


    return {}

if __name__ == "__main__":
    res = run_anymal()
