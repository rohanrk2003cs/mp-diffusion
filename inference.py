import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import argparse
from PIL import Image
from diffusion import GaussianDiffusion
from infra import TemporalUnet, ResidualTemporalBlock
from utils import Collision_Constraints, batch_to_device


HEIGHT = 5
WIDTH = 2.5
TRAJECTORY_SIZE = 28
NUM_TRAJECTORIES = 10
GRAD_INCLUDED = True

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('start_state_x', type=float)
    parser.add_argument('start_state_y', type=float)
    parser.add_argument('obstacles', type=float, nargs="+")
    args = parser.parse_args()
    env_obstacles = obstacle_parser(args.obstacles)
    collision_env = Collision_Constraints(env_obstacles, HEIGHT, WIDTH)
    conditions = {0: np.repeat(np.array([args.start_state_x, args.start_state_y]), NUM_TRAJECTORIES), 
                TRAJECTORY_SIZE-1: np.repeat(np.array([0,0]), NUM_TRAJECTORIES)}
    diffusion_model = instantiate_trained_model()
    if GRAD_INCLUDED:
        diffusion_model(conditions, guide=collision_env, n_guide_steps=0)
    else:
        diffusion_model(conditions, guide=collision_env)



def obstacle_parser(obstacles_list):
    assert len(obstacles_list) % 8 == 0, "Input passed in incorrectly"
    i = 0
    obstacles =[]
    while i < len(obstacles_list):
        obstacle = []
        for j in range(4):
            obstacle.append([obstacles_list[i + j * 2], obstacles_list[i + j * 2 + 1]])
        i += 8
    return np.array(obstacles)
def instantiate_trained_model():
    old_horizon = 28
    old_transition_dim = 4
    old_action_dim = 2
    old_observation_dim = 2
    temporal_net = TemporalUnet(old_horizon, old_transition_dim, dim_mults=(1, 4, 8))
    diffusion_model = GaussianDiffusion(temporal_net, old_horizon, old_observation_dim, old_action_dim, loss_type="l2", predict_epsilon=False, action_weight=1, n_timesteps=256)
    diffusion_model.action_dim = 2
    diffusion_model.observation_dim = 2
    diffusion_model.model.downs[0][0] = ResidualTemporalBlock(diffusion_model.action_dim + diffusion_model.observation_dim, 32, 32, 4)
    diffusion_model.model.final_conv[-1] = nn.Conv1d(32, diffusion_model.action_dim + diffusion_model.observation_dim, 1)
    state_file_contents = torch.load("diff_state.pt")
    diffusion_model.load_state_dict(state_file_contents)
    diffusion_model.to("cuda:0")
    return diffusion_model


if __name__ == '__main__':
    main()
