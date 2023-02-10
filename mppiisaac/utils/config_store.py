from dataclasses import dataclass
from mppiisaac.planner.mppi import MPPIConfig
from mppiisaac.planner.isaacgym_wrapper import IsaacGymConfig
from hydra.core.config_store import ConfigStore

from typing import List


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacgym: IsaacGymConfig
    goal: List[float]


cs = ConfigStore.instance()
cs.store(name="config", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)
