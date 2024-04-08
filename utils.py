import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import cvxpy as cp
import pdb

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()
    
class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

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

def apply_conditioning(x, conditions, observation_dim):
    for t, val in conditions.items():
        x[:, t, :observation_dim] = val.clone()
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
     


#-----------------------------------------------------------------------------#
#--------------------------- collision constraints ---------------------------#
#-----------------------------------------------------------------------------#

class Collision_Constraints:
    def __init__(self, obstacles, height, width):
        self.obstacles = obstacles
        self.height = height
        self.width = width
        self.obst_center_x, self.obst_center_y, self.obst_radii = self.obstacle_vals()
    def compute_cost(self, y, i):
        A, b = self.get_constraint_tensor(y)
        n = A.shape[1]
        m = A.shape[0]
        t = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(t))
        b_numpy = b.cpu().detach().numpy()
        A_numpy = A.cpu().detach().numpy()
       
        constraints = []
        x = cp.Variable(n)
        C, d = self.get_opt_vars(self.obstacles[i], False)
        print(C)
        print(d)
        print(isinstance(A_numpy, np.ndarray))
        constraints.append(A_numpy @ (x + t) <= b_numpy)
        constraints.append((C @ x) - d <= np.zeros(m))
        # Create problem instance
        problem = cp.Problem(objective, constraints)
   
        # Solve the problem
        problem.solve()
        print(problem.status)
       
        if objective.value == 0:
            obs_x = self.obst_center_x[i]
            obs_y = self.obst_center_y[i]
            obs_rad = self.obst_radii[i]
            vehicle_width = b[0] - b[1]
            vehicle_height = b[2] - b[3]
            vehicle_x = (b[0] + b[1])/2
            vehicle_y = (b[2] + b[3])/2
            vehicle_rad = (vehicle_height**2 + vehicle_width**2)**0.5
            x_dists = obs_x - vehicle_x
            y_dists = obs_y - vehicle_y
            center_dists = (x_dist**2 + y_dist**2)**0.5
            sphere_dists = center_dists-obs_rad-vehicle_rad
            return sphere_dists
        else:
            # plug primal and duel variables into the lagrangian
            lagrangian = torch.tensor(objective.value, dtype=torch.float32)
            print(lagrangian, objective.value)
            x_opt = torch.tensor(x.value, dtype=torch.float32)
            t_opt = torch.tensor(t.value, dtype=torch.float32)
            C = torch.from_numpy(C)
            d = torch.from_numpy(d)
            lagrangian += torch.dot(torch.tensor(constraints[1].dual_value, dtype=torch.float32), (C @ x_opt - d))
            lagrangian += torch.dot(torch.tensor(constraints[0].dual_value, dtype=torch.float32),(A @ (x_opt + t_opt) - b))
        print("this is the lagrangian", x.value, t.value)
        return lagrangian
       
    def compute_cost_grads(self, trajectory):
        # trajectory shape: [batch_size, trajectory_size, dimensionality]
        batch_size, trajectory_size, _ = trajectory.shape
        cost_grads = []
        for batch in range(batch_size):
            batch_costs = []
            for step in range(trajectory_size):
                step_tensor = trajectory[batch, step, :]
                obs_costs = []
                for i in range(len(self.obstacles)):
                    cost = self.compute_cost(step_tensor, i)
                    obs_costs.append(cost)
                obs_costs = torch.stack(obs_costs)
                cost_min = torch.min(obs_costs)
                print(cost_min, step_tensor)
                cost_grad = torch.autograd.grad(cost_min, step_tensor)[0]
                print(cost_grad)
                batch_costs.append(cost_grad)
            # Stack the costs for each step in the trajectory of the current batch
            batch_costs_tensor = torch.stack(batch_costs)
            cost_grads.append(batch_costs_tensor)
        # Stack the costs for all batches to form the final cost tensor
        cost_grads = torch.stack(cost_grads)
        return cost_grads
         
    def get_constraint_tensor(self, input_tensor):
        x_front = input_tensor[0]
        y_front = input_tensor[1]
        theta = input_tensor[2]
   
        # Calculate back middle point
        x_back = x_front - self.height * torch.cos(theta)
        y_back = y_front - self.height * torch.sin(theta)
       
        # Calculate vertices
        A = torch.stack([x_front + self.width/2 * torch.sin(theta), y_front - self.width/2 * torch.cos(theta)])
        B = torch.stack([x_front - self.width/2 * torch.sin(theta), y_front + self.width/2 * torch.cos(theta)])
        C = torch.stack([x_back + self.width/2 * torch.sin(theta), y_back - self.width/2 * torch.cos(theta)])
        D = torch.stack([x_back - self.width/2 * torch.sin(theta), y_back + self.width/2 * torch.cos(theta)])
        vertices = torch.stack([A,B,D, C])
        return self.get_opt_vars(vertices, True)
       
    def get_opt_vars(self, vertices, tensor):
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        def line_from_points(p1, p2):
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]
            if x1 == x2:  
                return torch.tensor(1.), torch.tensor(0.), x1
            else:
                m = (y2 - y1) / (x2 - x1)
                a = -m
                b = torch.tensor(1.)
                c = y1 - (m * x1)
            if a < 0.:  
                print(p1, p2)
                a, b, c = -a, -b, -c
            return a, b, c
        A = []
        B = []
        min_int_negative_slope = None
        min_int_positive_slope = None
        for i in range(4):
            a,b,c = line_from_points(vertices[i], vertices[(i+1)%4])
            if b > 0:
                if min_int_negative_slope != None:
                    if min_int_negative_slope[0] < c:
                        A[min_int_negative_slope[1]] = torch.stack([-1.0 * item for item in A[min_int_negative_slope[1]]])
                        B[min_int_negative_slope[1]] = -1.0 * B[min_int_negative_slope[1]]
                        A.append(torch.stack([a,b]))
                        B.append(c)
                    else:
                        A.append(torch.stack([-a, -b]))
                        B.append(-c)
                else:
                    A.append(torch.stack([a, b]))
                    B.append(c)
                    min_int_negative_slope = (c, i)
            else:
                if min_int_positive_slope != None:
                    if min_int_positive_slope[0] < c:
                        A[min_int_positive_slope[1]] = torch.stack([-1.0 * item for item in A[min_int_positive_slope[1]]])
                        B[min_int_positive_slope[1]] = -1.0 * B[min_int_positive_slope[1]]
                        A.append(torch.stack([a,b]))
                        B.append(c)
                    else:
                        A.append(torch.stack([-a, -b]))
                        B.append(-c)
                else:
                    A.append(torch.stack([a, b]))
                    B.append(c)
                    min_int_positive_slope = (c, i)
        A = torch.stack(A)
        B = torch.stack(B)
        if not tensor:
            A = A.numpy()
            B = B.numpy()
        return A,B
       
    def obstacle_vals(self):
        # [[xr, xl, yr, yl], [a, b, c, d]
        center_x = (self.obstacles[:, 0] + -1.0 * self.obstacles[:, 1])/2
        center_y = (self.obstacles[:, 2] + -1.0 * self.obstacles[:, 3])/2
        x_dist = self.obstacles[:, 0] - (-1.0) * self.obstacles[:, 1]
        y_dist = self.obstacles[:, 2] - (-1.0) * self.obstacles[:, 3]
        obstacle_radii = np.linalg.norm([x_dist, y_dist], axis=0)
        return (center_x, center_y, obstacle_radii)
