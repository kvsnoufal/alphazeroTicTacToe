import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from config import Config as cfg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(cfg.ACTION_SIZE,64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.value_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.policy_network = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,cfg.ACTION_SIZE)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        value = self.value_network(logits)
        actions = self.policy_network(logits)
        return value,actions