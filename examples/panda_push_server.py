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
    obj_index = 5
    
                #  l      w     h     mu      m     x    y
    obj_set =  [[0.100, 0.100, 0.05, 0.150, 0.150, 0.40, 0.],     # Baseline 1
                [0.116, 0.116, 0.06, 0.637, 0.016, 0.37, 0.],     # Baseline 2, A
                [0.168, 0.237, 0.05, 0.232, 0.615, 0.40, 0.],     # Baseline 2, B
                [0.198, 0.198, 0.06, 0.198, 0.565, 0.42, 0.],     # Baseline 2, C
                [0.166, 0.228, 0.08, 0.312, 0.587, 0.42, 0.],     # Baseline 2, D
                [0.153, 0.462, 0.05, 0.181, 0.506, 0.40, 0.],]    # Baseline 2, E
    obj_ = obj_set[obj_index][:]
    table_dim = [0.8, 1.0, 0.108]
    table_pos = [0.5, 0., table_dim[-1]/2]
    
    additions = [
        {
            "type": "box",
            "name": "table",
            "size": table_dim,
            "init_pos": table_pos,
            "fixed": True,
            "handle": None,
        },
        {
            "type": "box",
            "name": "block",
            "size": [obj_[0], obj_[1], obj_[2]],
            "init_pos": [obj_[5], obj_[6], table_dim[-1] + obj_[2] / 2],
            "mass": obj_[4],
            "fixed": False,
            "handle": None,
            "color": [4 / 255, 160 / 255, 218 / 255],
            "friction": obj_[3]
        }
    ]

    sim.add_to_envs(additions)

    sim.gym.viewer_camera_look_at(
        sim.viewer, None, gymapi.Vec3(1.5, 2, 3), gymapi.Vec3(1.5, 0, 0)
    )

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Mppi server found!")

    planner.add_to_env(additions)

    pi = 3.14
    init_pos = [0.0, -0.94, 0., -2.8, 0., 1.8675, 0.]
    init_vel = [0., 0., 0., 0., 0., 0., 0.,]

    sim.set_dof_state_tensor(torch.tensor([init_pos[0], init_vel[0], init_pos[1], init_vel[1], init_pos[2], init_vel[2],
                                           init_pos[3], init_vel[3], init_pos[4], init_vel[4], init_pos[5], init_vel[5],
                                           init_pos[6], init_vel[6]], device="cuda:0"))

    for _ in range(cfg.n_steps):
        t = time.time()
        # Reset state
        planner.reset_rollout_sim(
            torch_to_bytes(sim.dof_state[0]),
            torch_to_bytes(sim.root_state[0]),
            torch_to_bytes(sim.rigid_body_state[0]),
        )
        sim.gym.clear_lines(sim.viewer)

        # Compute action
        action = bytes_to_torch(planner.command())
        if torch.any(torch.isnan(action)):
            print("nan action")
            action = torch.zeros_like(action)

        # Apply action
        sim.set_dof_velocity_target_tensor(action)

        # Visualize samples
        # rollouts = bytes_to_torch(planner.get_rollouts())
        # sim.draw_lines(rollouts)

        # Step simulator
        sim.step()

        # Print error of block
        pos = sim.root_state[0, -1][:2].cpu().numpy()
        goal = np.array([0.5, 0])
        print(f"L2: {np.linalg.norm(pos - goal)} FPS: {1/(time.time() - t)} RT-factor: {cfg.isaacgym.dt/(time.time() - t)}")
    return {}


if __name__ == "__main__":
    res = run_panda_robot()
