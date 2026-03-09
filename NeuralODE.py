# NeuralODE.py
#
# Defines the PyTorch model architecture for predicting battery state of charge (SoC) from voltage and current.
# Uses a neural ordinary differential equation (ODE) approach, where the model learns a continuous-time representation of the battery dynamics.

# imports
import torch
import torch.nn as nn
import torchdiffeq

# model 
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(), # we use tanh activation to ensure the output is smooth (ODEs are smooth by nature)
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
