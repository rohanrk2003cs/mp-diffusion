from collections import namedtuple
import numpy as np
import math
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
        self.normalizer = self.normalizer(array)
        normed = self.normalizer.normalize(array)
        self.data = normed.reshape(n_traj, 28, -1)
    # add speed
    def get_parking_trajectories(self):
        # Initialize an empty list for trajectories
        path = '../dlp-dataset/data/DJI_00'
        self.data = []
        for i in range(1, 31):
            ds = Dataset()
            if i >= 10:
                ds.load(path + str(i))
            else:
                ds.load(path + "0" + str(i))
            
            scene = ds.get('scene', ds.list_scenes()[0])
            agents = scene['agents']
            print("LOADING TRAINING DATA FILE: " + str(i))

            for agent_token in agents:
                agent = ds.get('agent', agent_token)
                if agent["type"] == "Car":
                    instance_token = agent["first_instance"]
                    not_parked = 0
                    traj = []
                    while instance_token != '':
                        instance = ds.get('instance', instance_token)
                        #included heading here
                        traj.append([instance["coords"][0], instance["coords"][1], instance["speed"], instance["heading"]])
                        if ds._inside_parking_area(instance_token) and instance['speed'] < 0.02:
                            if not_parked > 700:
                                traj = traj[-700:]  
                                sample_traj = np.array(traj[::25]) 

                                # adjusted_traj[3] doesnt necesarly correspond to the correct heading angle, <I 
                                
                                adjusted_traj = sample_traj - np.array([instance["coords"][0], instance["coords"][1], 0.0, 0.0])
                                reflectx2_traj = adjusted_traj * np.array([1.0, -1.0, 1.0, 1.0])
                                reflectx2_traj[:, 3] = 2*math.pi - reflectx2_traj[:, 3]
                                reflectx3_traj = adjusted_traj * np.array([-1.0, -1.0, 1.0, 1.0])
                                reflectx3_traj[:, 3] = math.pi + reflectx3_traj[:, 3]
                                reflectx4_traj = adjusted_traj * np.array([-1.0, 1.0, 1.0, 1.0])
                                reflectx4_traj[:, 3] = math.pi - reflectx4_traj[:, 3]
                                
                                reflecty1_traj = adjusted_traj[:, [1, 0, 2, 3]]
                                reflecty2_traj = reflectx2_traj[:, [1, 0, 2, 3]]
                                reflecty3_traj = reflectx3_traj[:, [1, 0, 2, 3]]
                                reflecty4_traj = reflectx4_traj[:, [1, 0, 2, 3]]

                                
                                condition = adjusted_traj[:, 3] > np.pi
                                reflecty1_traj[condition, 3] = np.asin(np.cos(np.pi - adjusted_traj[condition, 3])) + np.pi
                                reflecty1_traj[~condition, 3] = np.asin(np.cos(adjusted_traj[~condition, 3]))

                                condition = reflectx2_traj[:, 3] > np.pi
                                reflecty2_traj[condition, 3] = np.asin(np.cos(np.pi - reflectx2_traj[condition, 3])) + np.pi
                                reflecty2_traj[~condition, 3] = np.asin(np.cos(reflectx2_traj[~condition, 3]))

                                condition = reflectx3_traj[:, 3] > np.pi
                                reflecty3_traj[condition, 3] = np.asin(np.cos(np.pi - reflectx3_traj[condition, 3])) + np.pi
                                reflecty3_traj[~condition, 3] = np.asin(np.cos(reflectx3_traj[~condition, 3]))

                                condition = reflectx4_traj[:, 3] > np.pi
                                reflecty4_traj[condition, 3] = np.asin(np.cos(np.pi - reflectx4_traj[condition, 3])) + np.pi
                                reflecty4_traj[~condition, 3] = np.asin(np.cos(reflectx4_traj[~condition, 3]))
                                
                                self.data.append(adjusted_traj)  
                                self.data.append(reflectx2_traj)
                                self.data.append(reflectx3_traj)
                                self.data.append(reflectx4_traj)
                                
                                self.data.append(reflecty1_traj)
                                self.data.append(reflecty2_traj)
                                self.data.append(reflecty3_traj)
                                self.data.append(reflecty4_traj)
                            traj = []
                            not_parked = 0
                        else:
                            not_parked += 1
                        instance_token = instance["next"]
        self.data = np.array(self.data, dtype=np.float32)

    #WILL NEED TO MODIFY IF DECIDE FOR HORIZON BASED APPROACH
    def get_conditions(self, observation):
        return {0: observation[0], 
               len(observation) - 1: observation[-1]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, eps=1e-4):
        trajectory = self.data[idx] 

        conditions = self.get_conditions(trajectory[:,:2])
        batch = Batch(trajectory, conditions)
        return batch

