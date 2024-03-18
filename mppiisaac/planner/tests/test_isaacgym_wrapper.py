from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
from mppiisaac.utils.config_store import ExampleConfig, MPPIConfig
from hydra import initialize, compose
import torch
import mppiisaac
import os
import yaml
from yaml import SafeLoader


def test_body_force() -> None:
    config_path = "."
    with initialize(version_base=None, config_path=config_path):
        cfg_isaacgym = compose(config_name="test_isaacgym_config", overrides=["viewer=True"])
        cfg_boxer = compose(config_name="test_boxer_config")
        cfg_wall = compose(config_name="test_wall_config")

        num_envs = 600
        sim = IsaacGymWrapper(
            cfg_isaacgym,
            actors=[ActorWrapper(**cfg_boxer), ActorWrapper(**cfg_wall)],
            num_envs=num_envs
        )

        assert sim.dof_state.size() == torch.Size([num_envs, 4])

        cmd = torch.Tensor([0.2, 0.])
        cmd = cmd.repeat(num_envs, 1)
        assert cmd.size() == torch.Size([num_envs, 2])
        sim.apply_robot_cmd_velocity(cmd)

        for i in range(200):
            sim.step()

        assert all(sim.net_cf[:, 0] == sim.net_cf[:, -1])
