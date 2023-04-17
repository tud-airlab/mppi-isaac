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


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


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

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")

    for _ in range(cfg.n_steps):
        t = time.time()
        # Reset state
        planner.reset_rollout_sim(
            torch_to_bytes(sim.dof_state[0]),
            torch_to_bytes(sim.root_state[0]),
            torch_to_bytes(sim.rigid_body_state[0]),
        )

        # Compute action
        action = bytes_to_torch(planner.command())
        #print(action)
        if torch.any(torch.isnan(action)):
            print("nan action")
            action = torch.zeros_like(action)

        # Apply action
        sim.apply_robot_cmd_velocity(torch.unsqueeze(action, axis=0))

        # Step simulator
        sim.step()
        sim.gym.sync_frame_time(sim.sim)

        # Print error
        pos = sim.robot_positions[:, :2]
        print(torch.linalg.norm(pos - torch.tensor(cfg.goal, device="cuda:0"), axis=1))
    return {}


if __name__ == "__main__":
    res = run_jackal_robot()
