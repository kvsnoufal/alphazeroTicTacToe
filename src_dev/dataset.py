import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pickle
from copy import copy
from config import Config as cfg

class TicTacToeDataset:
    def __init__(self,dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        datapoint = self.data[index]
        state = datapoint[0]
        v = datapoint[3]
        p = datapoint[1]
        return torch.tensor(state,dtype=torch.float),torch.tensor(v,dtype=torch.float),torch.tensor(p,dtype=torch.float)

class TrainingDataset:
    def __init__(self):
        self.training_dataset = []
    def calculate_values(self,dataset,winner):
        for ind,step in enumerate(dataset):
            step_ = copy(step)
            step_player = step_[2]
            if winner == 0:
                value = 0
            else:
                if winner==step_player:
                    value = 1
                else:
                    value =-1
            step_.append(value)
            dataset[ind] = step_
        return dataset
    def add_game_to_training_dataset(self,dataset,winner):
        data = self.calculate_values(dataset,winner)
        self.training_dataset.extend(data)
        self.training_dataset = self.training_dataset[-1*cfg.DATASET_QUEUE_SIZE:]    
    
    def save(self,path):
#         pickle.dump(self.training_dataset,path)
        with open(path, 'wb') as handle:
            pickle.dump(self.training_dataset,handle)
    def load(self,path):
#         self.training_dataset = pickle.load(path)
        with open(path, 'rb') as handle:
            self.training_dataset = pickle.load(handle)
    def retreive_test_train_data(self):
        data = np.array(self.training_dataset)
        train_idx = np.random.choice(np.arange(len(data)),int(cfg.TRAIN_TEST_SPLIT*len(data)))
        val_idx = [t for t in np.arange(len(data)) if t not in train_idx]
        return TicTacToeDataset(data[train_idx]),TicTacToeDataset(data[val_idx])
    