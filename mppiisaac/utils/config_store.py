from dataclasses import dataclass, field
from mppiisaac.planner.mppi import MPPIConfig
from mppiisaac.planner.isaacgym_wrapper import IsaacGymConfig, ActorWrapper
from hydra.core.config_store import ConfigStore

from typing import List, Optional


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacgym: IsaacGymConfig
    goal: List[float]
    nx: int
    urdf_file: str
    fix_base: bool
    flip_visual: bool
    initial_position: List[float] = field(default_factory=lambda: [0, 0, 0])
    disable_gravity: bool = False
    differential_drive: bool = False
    wheel_base: float = 0
    wheel_radius: float = 0
    wheel_count: int = 0
    ee_link: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config_point_robot", node=ExampleConfig)
cs.store(name="config_heijn_robot", node=ExampleConfig)
cs.store(name="config_boxer_robot", node=ExampleConfig)
cs.store(name="config_jackal_robot", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(name="config_omnipanda", node=ExampleConfig)
cs.store(name="config_panda_push", node=ExampleConfig)
cs.store(name="config_heijn_push", node=ExampleConfig)
cs.store(name="config_boxer_push", node=ExampleConfig)
cs.store(name="config_panda_c_space_goal", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)
