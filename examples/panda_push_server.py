import gym
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
import numpy as np
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import hydra
from omegaconf import OmegaConf
import os
import torch
from mppiisaac.priors.fabrics_panda import FabricsPandaPrior
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


@hydra.main(version_base=None, config_path="../conf", config_name="config_panda_push")
def run_panda_robot(cfg: ExampleConfig):
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)
    cfg.isaacgym.dt=0.05

    sim = IsaacGymWrapper(
        cfg.isaacgym,
        cfg.urdf_file,
        cfg.fix_base,
        cfg.flip_visual,
        num_envs=1,
        ee_link=cfg.ee_link,
        disable_gravity=cfg.disable_gravity,
        viewer=True,
    )

    # Manually add table + block and restart isaacgym
    additions = [
        {
            "type": "box",
            "name": "table",
            "size": [0.5, 1.0, 0.112],
            "init_pos": [0.5, 0, 0.112 / 2],
            "fixed": True,
            "handle": None,
        },
        {
            "type": "box",
            "name": "block",
            "size": [0.105, 0.063, 0.063],
            "init_pos": [0.55, 0.2, 0.3],
            "mass": 0.250,
            "fixed": False,
            "handle": None,
            "color": [0.2, 0.2, 1.0]
        }
    ]
    for a in additions:
        sim.env_cfg.append(a)
    sim.stop_sim()
    sim.start_sim()

    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0)
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")

    planner.add_to_env(additions)

    pi = 3.14
    sim.set_dof_state_tensor(torch.tensor([0, 0, 0, 0, 0, 0, -pi/2, 0, 0, 0, pi/2, 0, pi/4, 0], device="cuda:0"))

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
        if torch.any(torch.isnan(action)):
            action = torch.zeros_like(action)

        # Apply action
        sim.set_dof_velocity_target_tensor(action)

        # Step simulator
        sim.step()

        # Print error of block
        pos = sim.root_state[0, -1][:2].cpu().numpy()
        goal = np.array([0.5, 0])
        print(f"L2: {np.linalg.norm(pos - goal)} FPS: {1/(time.time() - t)}")
    return {}


if __name__ == "__main__":
    res = run_panda_robot()
