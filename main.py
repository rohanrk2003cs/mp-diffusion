import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusion import GaussianDiffusion
from infra import TemporalUnet, ResidualTemporalBlock
from utils import GaussianNormalizer, batch_to_device
import pickle
from dataset import ParkingDataset
from torch.utils.data import DataLoader, TensorDataset
#Will clean this up after trainging
def cycle(dl):
    while True:
        for data in dl:
            yield data
class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



#####Training
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
def main():
    old_action_dim = 6
    old_observation_dim = 17
    horizon = 4
    state_file_contents = torch.load("state_800000.pt", map_location=torch.device('cpu'))
    model_params = state_file_contents["model"]
    ema_params = state_file_contents["ema"]


    temporal_net = TemporalUnet(horizon, old_action_dim + old_observation_dim, old_observation_dim, dim_mults=(1, 4, 8), attention=True)
    new_diffusion_model = GaussianDiffusion(temporal_net, horizon, old_observation_dim, old_action_dim, loss_type="l2", predict_epsilon=False, action_weight=10, n_timesteps=20)
    new_diffusion_model.load_state_dict(model_params)
    new_diffusion_model.action_dim = 1
    new_diffusion_model.observation_dim = 2
    #Add new residual block as first layer
    new_diffusion_model.model.downs[0][0] = ResidualTemporalBlock(new_diffusion_model.action_dim + new_diffusion_model.observation_dim, 32, 32, 4)
    #Change final conv block
    new_diffusion_model.model.final_conv[-1] = nn.Conv1d(32, new_diffusion_model.action_dim + new_diffusion_model.observation_dim, 1)
    parking_dataset = ParkingDataset(32, False, GaussianNormalizer)
    print(parking_dataset.data)
    diffusion_trainer = Trainer(new_diffusion_model, parking_dataset)
    n_epochs = 50
    for i in range(50):
        diffusion_trainer.train(3000) 


main()