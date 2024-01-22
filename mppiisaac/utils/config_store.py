from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from typing import List, Optional
from enum import Enum

class SupportedActorTypes(Enum):
    Axis = 1
    Robot = 2
    Sphere = 3
    Box = 4


@dataclass
class ActorWrapper:
    type: SupportedActorTypes
    name: str
    init_pos: List[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0  # kg
    color: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fixed: bool = False
    collision: bool = True
    friction: float = 1.
    handle: Optional[int] = None
    flip_visual: bool = False
    urdf_file: str = None
    ee_link: str = None
    gravity: bool = True
    differential_drive: bool = False
    wheel_radius: Optional[float] = None
    wheel_base: Optional[float] = None
    wheel_count: Optional[float] = None
    left_wheel_joints: Optional[List[int]] = None
    right_wheel_joints: Optional[List[int]] = None
    caster_links: Optional[List[str]] = None
    noise_sigma_size: Optional[List[float]] = None
    noise_percentage_mass: float = 0.0
    noise_percentage_friction: float = 0.0


@dataclass
class MPPIConfig(object):
    """
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param mppi_mode: 'halton-spline' or 'simple' corresponds to the type of mppi.
        :param sampling_method: 'halton' or 'random', sampling strategy while using mode 'halton-spline'. In 'simple', random sampling is forced to 'random' 
        :param noise_sigma: variance per action
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean        
        :param device: pytorch device
        :param lambda_: inverse temperature, positive scalar where smaller values will allow more exploration
        :param update_lambda: flag for updating inv temperature
        :param update_cov: flag for updating covariance
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param rollout_var_discount: Discount cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost   
    """

    num_samples: int = 100
    horizon: int = 30
    mppi_mode: str = 'halton-spline'
    sampling_method: str = "halton"
    noise_sigma: Optional[List[List[float]]] = None
    noise_mu: Optional[List[float]] = None
    device: str = "cuda:0"
    lambda_: float = 0.0
    update_lambda: bool = False
    update_cov: bool = False
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None
    u_init: float = 0.0
    U_init: Optional[List[List[float]]] = None
    u_scale: float = 1
    u_per_command: int = 1
    rollout_var_discount: float = 0.95
    sample_null_action: bool = False
    noise_abs_cost: bool = False
    filter_u: bool = False
    use_priors: bool = False

@dataclass
class IsaacGymConfig(object):
    dt: float = 0.05
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_client_threads: int = 0
    viewer: bool = False
    num_obstacles: int = 10
    spacing: float = 6.0


@dataclass
class ExampleConfig:
    render: bool
    n_steps: int
    mppi: MPPIConfig
    isaacgym: IsaacGymConfig
    goal: List[float]
    nx: int
    actors: List[str]
    initial_actor_positions: List[List[float]]


cs = ConfigStore.instance()
cs.store(name="config_anymal", node=ExampleConfig)
cs.store(name="config_point_robot", node=ExampleConfig)
cs.store(name="config_multi_point_robot", node=ExampleConfig)
cs.store(name="config_heijn_robot", node=ExampleConfig)
cs.store(name="config_boxer_robot", node=ExampleConfig)
cs.store(name="config_jackal_robot", node=ExampleConfig)
cs.store(name="config_multi_jackal", node=ExampleConfig)
cs.store(name="config_panda", node=ExampleConfig)
cs.store(name="config_omnipanda", node=ExampleConfig)
cs.store(name="config_panda_push", node=ExampleConfig)
cs.store(name="config_heijn_push", node=ExampleConfig)
cs.store(name="config_boxer_push", node=ExampleConfig)
cs.store(name="config_panda_c_space_goal", node=ExampleConfig)
cs.store(group="mppi", name="base_mppi", node=MPPIConfig)
cs.store(group="isaacgym", name="base_isaacgym", node=IsaacGymConfig)
