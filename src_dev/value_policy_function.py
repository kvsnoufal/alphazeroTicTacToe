from config import Config as cfg

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from model import NeuralNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"

# model = NeuralNetwork().to(device)
class ValuePolicyNetwork:
    def __init__(self,path=None):
        self.model = NeuralNetwork().to(device)
        if path:
            self.model.load_state_dict(torch.load(path))
         
        self.model.eval()
    def get_vp(self,state):
        state = state.reshape(1,cfg.ACTION_SIZE)
        state = torch.tensor(state,dtype=torch.float).to(device)
        
        with torch.no_grad():
            value,policy = self.model(state)
        value = value.cpu().numpy().flatten()[0]
        policy = torch.nn.functional.softmax(policy)
        policy = policy.cpu().numpy().flatten()
        
        return value,policy




