import torch
import functools
import numpy as np
from typing import Optional, List, Callable
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import signal
import scipy.interpolate as si
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from mppiisaac.utils.mppi_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def bspline(c_arr, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    return samples


# TODO: integrate with localplannerbench, using class inheritence
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
    lambda_: float = 1.0
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

class MPPIPlanner(ABC):
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories thus it scales with the number of samples K. 

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'  
    and 'STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation'

    Code based off and https://github.com/NVlabs/storm

    This mppi can run in two modes: 'simple' and a 'halton-spline':
        - simple:           random sampling at each MPPI iteration from normal distribution with simple mean update. To use this set 
                            mppi_mode = 'simple_mean'
        - halton-spline:    samples only at the start a halton-spline which is then shifted according to the current moments of the control distribution. 
                            Moments are updated using gradient. To use this set
                            mppi_mode = 'halton-spline', sample_mode = 'halton'
                            Alternatively, one can also sample random trajectories at each iteration using gradient mean update by setting
                            mppi_mode = 'halton-spline', sample_mode = 'random'
    """

    def __init__(self, cfg: MPPIConfig, nx: int, dynamics: Callable, running_cost: Callable, prior: Optional[Callable] = None):

        # Parameters for mppi and sampling method
        self.mppi_mode = cfg.mppi_mode
        self.sample_method = cfg.sampling_method

        # Utility vars
        self.K = cfg.num_samples        # N_SAMPLES 
        self.T = cfg.horizon            # TIMESTEPS
        self.filter_u = cfg.filter_u    # Flag for Sav-Gol filter
        self.lambda_ = cfg.lambda_
        self.tensor_args={'device':cfg.device, 'dtype':torch.float32}
        self.delta = None
        self.sample_null_action = cfg.sample_null_action
        self.u_per_command = cfg.u_per_command
        self.terminal_state_cost = None
        self.update_lambda = cfg.update_lambda
        self.update_cov = cfg.update_cov

        # Bound actions
        self.u_min = cfg.u_min
        self.u_max = cfg.u_max
        self.u_scale = cfg.u_scale        

        # Noise and input initialization
        self.noise_abs_cost = cfg.noise_abs_cost
        
        if not cfg.noise_sigma:
            cfg.noise_sigma = np.identity(int(nx/2)).tolist()
        assert all([len(cfg.noise_sigma[0]) == len(row) for row in cfg.noise_sigma])

        if not cfg.noise_mu:
            cfg.noise_mu = [0.0] * len(cfg.noise_sigma)
        if not cfg.U_init:
            cfg.U_init = [[0.0] * len(cfg.noise_mu)] * cfg.horizon

        # Make sure if any of the input limits are specified, both are specified
        if cfg.u_max and not cfg.u_min:
            cfg.u_min = -cfg.u_max
        if cfg.u_min and not cfg.u_max:
            cfg.u_max = -cfg.u_min
        self.cfg = cfg

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.prior = prior

        # Convert lists in cfg to tensors and put them on device
        self.noise_sigma = torch.tensor(cfg.noise_sigma, device=cfg.device)
        self.noise_mu = torch.tensor(cfg.noise_mu, device=cfg.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        self.u_init = torch.tensor(cfg.u_init, device=cfg.device)
        self.U = torch.tensor(cfg.U_init, device=cfg.device)
        # self.U = self.noise_dist.sample((self.T,))
        self.u_max = torch.tensor(cfg.u_max, device=cfg.device)
        self.u_min = torch.tensor(cfg.u_min, device=cfg.device)

        # Dimensions of state nx and control nu
        self.nx = nx
        self.nu = 1 if len(self.noise_sigma.shape) == 0 else self.noise_sigma.shape[0]
        
        # Moments and best trajectory
        self.mean_action = torch.zeros(self.nu, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        self.best_traj = self.mean_action.clone()

        # Sampled results from last command
        self.state = None
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # handle 1D edge case
        if self.nu == 1:
            self.noise_mu = self.noise_mu.view(-1)
            self.noise_sigma = self.noise_sigma.view(-1, 1)
    
        # Halton sampling 
        self.knot_scale = 2             # From mppi config storm is 4
        self.seed_val = 0               # From mppi config storm
        self.n_knots = self.T//self.knot_scale
        self.ndims = self.n_knots * self.nu
        self.degree = 1                 # From sample_lib storm is 2
        self.Z_seq = torch.zeros(1, self.T, self.nu, **self.tensor_args)
        self.cov_action = torch.diagonal(self.noise_sigma, 0)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.squash_fn = 'clamp'
        self.step_size_mean = 0.98      # From storm

        # Discount
        self.gamma = cfg.rollout_var_discount 
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.T - 1)),dim=0).reshape(1, self.T)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self.beta = 1 # param storm

        # Filtering
        self.sgf_window = 9
        self.sgf_order = 2
        if (self.sgf_window % 2) == 0:
            self.sgf_window -=1       # Some versions of the sav-go filter require odd window size

        # Lambda update, for now the update of lambda is not performed
        self.eta_max = 0.1      # 10%
        self.eta_min = 0.01     # 1%
        self.lambda_mult = 0.1  # Update rate

        # covariance update  for now the update of lambda is not performed
        self.step_size_cov = 0.7
        self.kappa = 0.005

    def _dynamics(self, state, u, t=None):
        return self.dynamics(state, u, t=None)

    def _running_cost(self, state):
        return self.running_cost(state)

    def _exp_util(self, costs, actions):
        """
           Calculate weights using exponential utility given cost
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)
        traj_costs = traj_costs[:,0]

        #control_costs = self._control_costs(actions)
        total_costs = traj_costs - torch.min(traj_costs) #+ self.beta * control_costs

        # Normalization of the weights
        exp_ = torch.exp((-1.0/self.beta) * total_costs)
        eta = torch.sum(exp_)       # tells how many significant samples we have, more or less
        w = 1/eta*exp_
        # print(self.beta)
        eta_u_bound = 50
        eta_l_bound = 20
        beta_lm = 0.9
        beta_um = 1.2
        # beta update 
        if eta > eta_u_bound:
            self.beta = self.beta*beta_lm
        elif eta < eta_l_bound:
            self.beta = self.beta*beta_um
        
        #w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w

    def get_samples(self, sample_shape, **kwargs): 
        """
        Gets as input the desired number of samples and returns the actual samples. 

        Depending on the method, the samples can be Halton or Random. Halton samples a 
        number of knots, later interpolated with a spline
        """
        if(self.sample_method=='halton'):
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape,               # Number of samples
                self.ndims,                 # n_knots * nu (knots per number of actions)
                use_ghalton=True,
                seed_val=self.seed_val,     # seed val is 0 
                device=self.tensor_args['device'],
                float_dtype=self.tensor_args['dtype'])
            
            # Sample splines from knot points:
            # iteratre over action dimension:
            knot_samples = self.knot_points.view(sample_shape, self.nu, self.n_knots) # n knots is T/knot_scale (30/4 = 7)
            self.samples = torch.zeros((sample_shape, self.T, self.nu), **self.tensor_args)
            for i in range(sample_shape):
                for j in range(self.nu):
                    self.samples[i,:,j] = bspline(knot_samples[i,j,:], n=self.T, degree=self.degree)

        elif(self.sample_method == 'random'):
            self.samples = self.noise_dist.sample((self.K, self.T))
        
        return self.samples

    def command(self, state):
        """
            Given a state, returns the best action sequence
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.tensor_args['dtype'], device=self.tensor_args['device'])

        if self.mppi_mode == 'simple':
            self.U = torch.roll(self.U, -1, dims=0)

            cost_total = self._compute_total_cost_batch_simple()

            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

            eta = torch.sum(self.cost_total_non_zero)
            self.omega = (1. / eta) * self.cost_total_non_zero
            
            self.U += torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

            action = self.U

        elif self.mppi_mode == 'halton-spline':
            # shift command 1 time step
            self.mean_action = torch.roll(self.mean_action, -1, dims=0)
            # Set first sequence to zero, otherwise it takes the last of the sequence
            self.mean_action[0].zero_()

            cost_total = self._compute_total_cost_batch_halton()
              
            action = torch.clone(self.mean_action)

        # Lambda update
        if self.update_lambda:
            if eta > self.eta_max*self.K:
                self.lambda_ = (1+self.lambda_mult)*self.lambda_
            elif eta < self.eta_min*self.K:
                self.lambda_ = (1-self.lambda_mult)*self.lambda_

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
                mode='interp', 
                cval=0.0
                )
            if self.tensor_args['device'] == "cpu":
                action = torch.from_numpy(u_filtered).to('cpu')
            else:
                action = torch.from_numpy(u_filtered).to('cuda')
        
        # Reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]

        return action

    def _compute_rollout_costs(self, perturbed_actions):
        """
            Given a sequence of perturbed actions, forward simulates their effects and calculates costs for each rollout
        """
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_horizon = torch.zeros([K, T], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_samples = cost_total

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        states = []
        actions = []

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]

            # Last rollout is a braking manover
            if self.sample_null_action:
                u[self.K - 1, :] = torch.zeros_like(u[self.K -1, :])
                self.perturbed_action[self.K - 1][t] = u[self.K -1, :]

            if self.prior:
                u[self.K - 2] = self.prior(state, t)
                self.perturbed_action[self.K - 2][t] = u[self.K - 2]
                
            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state)

            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:,t] = u
            cost_samples += c
            cost_horizon[:, t] = c 

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
        
        if self.mppi_mode == 'halton-spline':
            self.noise = self._update_distribution(cost_horizon, actions)

        return cost_total, states, actions
 
    def _update_distribution(self, costs, actions):
        """
            Update moments using sample trajectories.
            So far only mean is updated, eventually one could also update the covariance
        """
        w = self._exp_util(costs, actions)
        
        # Compute also top n best actions to plot
        # top_values, top_idx = torch.topk(self.total_costs, 10)
        # self.top_values = top_values
        # self.top_idx = top_idx
        # self.top_trajs = torch.index_select(actions, 0, top_idx).squeeze(0)

        # Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)
       
        weighted_seq = w * actions.T

        sum_seq = torch.sum(weighted_seq.T, dim=0)
        new_mean = sum_seq

        # Gradient update for the mean
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean 
       
        delta = actions - self.mean_action.unsqueeze(0)

        #Update Covariance
        if self.update_cov:
            #Diagonal covariance of size AxA
            weighted_delta = w * (delta ** 2).T
            # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
            cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
    
            self.cov_action = (1.0 - self.step_size_cov) * self.cov_action + self.step_size_cov * cov_update
            self.cov_action += self.kappa #* self.init_cov_action
            # self.cov_action[self.cov_action < 0.0005] = 0.0005
            self.scale_tril = torch.sqrt(self.cov_action)
        return delta

    def get_action_cost(self):
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper
        return action_cost

    def _compute_total_cost_batch_simple(self):
        """
            Samples random noise and computes perturbed action sequence at each iteration. Returns total cost
        """
        # Resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # Broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        
        # Naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions /= self.u_scale

        # Bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

        action_cost = self.get_action_cost()

        # Action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _compute_total_cost_batch_halton(self):
        """
            Samples Halton splines once and then shifts mean according to control distribution. If random sampling is selected 
            then samples random noise at each step. Mean of control distribution is updated using gradient
        """
        if self.sample_method == 'random':
            self.delta = self.get_samples(self.K, base_seed=0)
        elif self.delta == None and self.sample_method == 'halton':
            self.delta = self.get_samples(self.K, base_seed=0)
            #add zero-noise seq so mean is always a part of samples

        # Add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        # Keeps the size but scales values
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0], self.T, self.nu)
        
        # First time mean is zero then it is updated in the distribution
        act_seq = self.mean_action + scaled_delta

        # Scales action within bounds. act_seq is the same as perturbed actions
        act_seq = scale_ctrl(act_seq, self.u_min, self.u_max, squash_fn=self.squash_fn)
        act_seq[self.nu, :, :] = self.best_traj
        
        self.perturbed_action = torch.clone(act_seq)

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)

        self.actions /= self.u_scale

        action_cost = self.get_action_cost()

        # Action perturbation cost
        perturbation_cost = torch.sum(self.mean_action * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    
    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action
