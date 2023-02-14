#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ghalton

def scale_ctrl(ctrl, action_lows, action_highs, squash_fn='clamp'):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == 'clamp':
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == 'clamp_rescale':
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == 'tanh':
        ctrl = torch.tanh(ctrl)
    elif squash_fn == 'identity':
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)

###########################
## Quasi-Random Sampling ##
###########################

def generate_prime_numbers(num):
    def is_prime(n):
        for j in range(2, ((n //2) + 1),1):
            if n % j == 0:
                return False
        return True

    primes = [0] * num #torch.zeros(num, device=device)
    primes[0] = 2
    curr_num = 1
    for i in range(1, num):
        while True:
            curr_num += 2
            if is_prime(curr_num):
                primes[i] = curr_num
                break
            
    return primes

def generate_van_der_corput_samples_batch(idx_batch, base):
    inp_device = idx_batch.device
    batch_size = idx_batch.shape[0]
    f = 1.0 #torch.ones(batch_size, device=inp_device)
    r = torch.zeros(batch_size, device=inp_device)
    while torch.any(idx_batch > 0):
        f /= base*1.0
        r += f * (idx_batch % base) #* (idx_batch > 0)
        idx_batch = idx_batch // base
    return r

def generate_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123, device=torch.device('cpu'), float_dtype=torch.float64):
    if not use_ghalton:
        samples = torch.zeros(num_samples, ndims, device=device, dtype=float_dtype)
        if not bases:
            bases = generate_prime_numbers(ndims)
        idx_batch = torch.arange(1,num_samples+1, device=device)
        for dim in range(ndims):
            samples[:, dim] = generate_van_der_corput_samples_batch(idx_batch, bases[dim])
    else:
        
        if ndims <= 100:
            perms = ghalton.EA_PERMS[:ndims]
            sequencer = ghalton.GeneralizedHalton(perms)
        else:
            sequencer = ghalton.GeneralizedHalton(ndims, seed_val)
        samples = torch.tensor(sequencer.get(num_samples), device=device, dtype=float_dtype)
    return samples


def generate_gaussian_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123, device=torch.device('cpu'), float_dtype=torch.float64):
    uniform_halton_samples = generate_halton_samples(num_samples, ndims, bases, use_ghalton, seed_val, device, float_dtype)

    gaussian_halton_samples = torch.sqrt(torch.tensor([2.0],device=device,dtype=float_dtype)) * torch.erfinv(2 * uniform_halton_samples - 1)
    
    return gaussian_halton_samples

def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    cost_seq = torch.fliplr(torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq