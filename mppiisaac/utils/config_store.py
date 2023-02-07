from dataclasses import dataclass
from mppiisaac.planner.mppi import MPPIConfig
from mppiisaac.planner.mppi_isaac import IsaacSimConfig
from hydra.core.config_store import ConfigStore
from typing import List


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacsim: IsaacSimConfig
    goal: List[float]


cs = ConfigStore.instance()
cs.store(name="config", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacsim", name="base_isaacsim", node=IsaacSimConfig)
