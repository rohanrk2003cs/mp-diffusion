import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

class GaussianNormalizer:
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, input):
        self.input = input.astype(np.float32)
        self.means = self.input.mean(axis=0)
        self.stds = self.input.std(axis=0)
    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x
# change to cuda after
def batch_to_device(batch, device='cpu'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def to_device(x, device="cpu"):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')