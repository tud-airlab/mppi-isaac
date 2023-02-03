import torch
import functools
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import signal
from dataclasses import dataclass
from abc import ABC, abstractmethod


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[
                    :-1
                ]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [
            v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v
            for v in args
        ]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [
                v
                if (not is_tensor_like(v) or len(v.shape) == 0)
                else (
                    v.view(*batch_dims, v.shape[-1])
                    if len(v.shape) == 2
                    else v.view(*batch_dims)
                )
                for v in ret
            ]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


# TODO: integrate with localplannerbench, using class inheritence
@dataclass
class MPPIConfig(object):
    """
    :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
    :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
    :param nx: state dimension
    :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
    :param num_samples: K, number of trajectories to sample
    :param horizon: T, length of each trajectory
    :param device: pytorch device
    :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
    :param lambda_: temperature, positive scalar where larger values will allow more exploration
    :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
    :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
    :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
    :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
    :param U_init: (T x nu) initial control sequence; defaults to noise
    :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
    :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
        (should be 1 for deterministic dynamics and more for models that output a distribution)
    :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
    :param rollout_var_discount: Discount of variance cost over control horizon
    :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
    :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
    """

    nx: int = 4
    num_samples: int = 100
    horizon: int = 30
    noise_sigma = torch.tensor([[1, 0], [0, 1]], device="cuda:0", dtype=torch.float32)
    device: str = "cuda:0"
    lambda_: float = 1.0
    noise_mu = None
    u_min = None
    u_max = None
    u_init = None
    U_init = None
    u_scale = 1
    u_per_command = 1
    step_dependent_dynamics = False
    rollout_samples = 1
    rollout_var_cost = 0
    rollout_var_discount = 0.95
    sample_null_action = False
    use_vacuum = False
    robot: str = "point_robot"
    noise_abs_cost: bool = False
    actors_per_env = None
    env_type = "normal"
    bodies_per_env = None
    filter_u: bool = False


class MPPIPlanner(ABC):
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(
        self,
        cfg: MPPIConfig
    ):
        self.d = cfg.device
        self.dtype = cfg.noise_sigma.dtype
        self.K = cfg.num_samples  # N_SAMPLES
        self.T = cfg.horizon  # TIMESTEPS
        self.filter_u = cfg.filter_u
        # dimensions of state and control
        self.nx = cfg.nx
        self.nu = 1 if len(cfg.noise_sigma.shape) == 0 else cfg.noise_sigma.shape[0]
        self.lambda_ = cfg.lambda_

        if cfg.noise_mu is None:
            cfg.noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if cfg.u_init is None:
            cfg.u_init = torch.zeros_like(cfg.noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            cfg.noise_mu = cfg.noise_mu.view(-1)
            cfg.noise_sigma = cfg.noise_sigma.view(-1, 1)

        # bounds
        self.u_min = cfg.u_min
        self.u_max = cfg.u_max
        self.u_scale = cfg.u_scale
        self.u_per_command = cfg.u_per_command
        # make sure if any of thecfg.m is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = cfg.noise_mu.to(self.d)
        self.noise_sigma = cfg.noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        # T x nu control sequence
        self.U = cfg.U_init
        self.u_init = cfg.u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = cfg.step_dependent_dynamics
        self.F = self.dynamics
        self.terminal_state_cost = None
        self.sample_null_action = cfg.sample_null_action
        self.noise_abs_cost = cfg.noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = cfg.rollout_samples
        self.rollout_var_cost = cfg.rollout_var_cost
        self.rollout_var_discount = cfg.rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # filtering
        if cfg.robot == "point_robot":
            if self.T < 20:
                self.sgf_window = self.T
            else:
                self.sgf_window = 10
            self.sgf_order = 2
        elif cfg.robot == "heijn":
            if self.T < 20:
                self.sgf_window = self.T
            else:
                self.sgf_window = 20
            self.sgf_order = 1
        elif cfg.robot == "boxer":
            if self.T < 20:
                self.sgf_window = self.T
            else:
                self.sgf_window = 10
            self.sgf_order = 3

    @handle_batch_input
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        cost_total = self._compute_total_cost_batch()

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1.0 / eta) * self.cost_total_non_zero

        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[: self.u_per_command]

        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]

        if self.sample_null_action and cost_total[-1] <= 0.01:
            action = torch.zeros_like(action)

        # Smoothing with Savitzky-Golay filter
        if self.filter_u:
            u_ = action.cpu().numpy()
            u_filtered = signal.savgol_filter(
                u_,
                self.sgf_window,
                self.sgf_order,
                deriv=0,
                delta=1.0,
                axis=0,
                mode="interp",
                cval=0.0,
            )
            action = torch.from_numpy(u_filtered).to("cuda")

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        states = []
        actions = []

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)

            # Last rollout is a breaking manover
            if self.sample_null_action:
                u[:, self.K - 1, :] = torch.zeros_like(u[:, self.K - 1, :])
                # Update perturbed action sequence for later use in cost computation
                self.perturbed_action[self.K - 1][t] = u[:, self.K - 1, :]

            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state, u)

            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:, t] = u
            cost_samples += c

            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount**t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise

        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(
            self.perturbed_action
        )
        self.actions /= self.u_scale

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = (
                self.lambda_ * self.noise @ self.noise_sigma_inv
            )  # Like original paper

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def get_rollouts(self, state, num_rollouts=1):
        """
        :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
        :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                             dynamics
        :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros(
            (num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device
        )
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(
                states[:, t].view(num_rollouts, -1),
                self.u_scale * self.U[t].view(num_rollouts, -1),
                t,
            )
        return states[:, 1:]

    @abstractmethod
    def dynamics(self):
        raise NotImplementedError

    @abstractmethod
    def running_cost(self):
        raise NotImplementedError

#    @abstractmethod
#    def terminal_cost(self):
#        raise NotImplementedError

    @abstractmethod
    def compute_action(self, q, qdot):
        raise NotImplementedError
