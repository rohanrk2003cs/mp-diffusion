from collections import namedtuple
import numpy as np
import torch
import pdb
from torch.utils.data import DataLoader, TensorDataset
from dlp.dataset import Dataset

Batch = namedtuple('Batch', 'trajectories conditions')
# make sure data is coming in properly

class ParkingDataset(torch.utils.data.Dataset):

    def __init__(self, horizon, horizon_enabled, normalizer):
        self.horizon = horizon
        self.horizon_enabled = horizon_enabled
        self.get_parking_trajectories()
        self.normalizer = normalizer
        self.normalize()

    def normalize(self):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        n_traj = len(self.data)
        array = self.data.reshape(n_traj * 28, -1)
        normed = self.normalizer(array).normalize(array)
        self.data = normed.reshape(n_traj, 28, -1)
    # add speed
    def get_parking_trajectories(self):
        # Initialize an empty list for trajectories
        path = '../dlp-dataset/data/DJI_00'
        for i in range(12,13):
            ds = Dataset()
            if i >= 10:
                ds.load(path + str(i))
            else:
                ds.load(path + "0" + str(i))
            
            scene = ds.get('scene', ds.list_scenes()[0])
            agents = scene['agents']
            self.data = []

            for agent_token in agents:
                agent = ds.get('agent', agent_token)
                if agent["type"] == "Car":
                    instance_token = agent["first_instance"]
                    not_parked = 0
                    traj = []
                    while instance_token != '':
                        instance = ds.get('instance', instance_token)
                        traj.append([instance["coords"][0], instance["coords"][1], instance["speed"]])
                        if ds._inside_parking_area(instance_token) and instance['speed'] < 0.02:
                            if not_parked > 700:
                                traj = traj[-700:]  
                                sample_traj = np.array(traj[::25])  
                                
                                adjusted_traj = sample_traj - np.array([instance["coords"][0], instance["coords"][1], 0.0])
                                reflectx1_traj = adjusted_traj * np.array([1.0, -1.0, 1.0])
                                reflectx2_traj = adjusted_traj * np.array([-1.0, -1.0, 1.0])
                                reflectx3_traj = adjusted_traj * np.array([-1.0, 1.0, 1.0])
                                
                                reflecty1_traj = adjusted_traj[:, [1, 0, 2]]
                                reflecty2_traj = reflectx1_traj[:, [1, 0, 2]]
                                reflecty3_traj = reflectx2_traj[:, [1, 0, 2]]
                                reflecty4_traj = reflectx3_traj[:, [1, 0, 2]]
                                
                                self.data.append(adjusted_traj)  
                                self.data.append(reflectx1_traj)
                                self.data.append(reflectx2_traj)
                                self.data.append(reflectx3_traj)
                                
                                self.data.append(reflecty1_traj)
                                self.data.append(reflecty2_traj)
                                self.data.append(reflecty3_traj)
                                self.data.append(reflecty4_traj)
                            traj = []
                            not_parked = 0
                        else:
                            not_parked += 1
                        instance_token = instance["next"]
        self.data = np.array(self.data)

    #WILL UTILIZE IF IMPLEMENT HORIZON BASED APPROACH

    # def make_indices(self, path_lengths, horizon):
    #     '''
    #         makes indices for sampling from dataset;
    #         each index maps to a datapoint
    #     '''
    #     indices = []
    #     for i, path_length in enumerate(path_lengths):
    #         max_start = min(path_length - 1, self.max_path_length - horizon)
    #         if not self.use_padding:
    #             max_start = min(max_start, path_length - horizon)
    #         for start in range(max_start):
    #             end = start + horizon
    #             indices.append((i, start, end))
    #     indices = np.array(indices)
    #     return indices

    def get_conditions(self, observation):
        return {0: observations[0], 
               len(observations) - 1: observation[-1]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, eps=1e-4):
        observations = self.data[idx] 

        conditions = self.get_conditions(observations)
        batch = Batch(observations, conditions)
        return batch

