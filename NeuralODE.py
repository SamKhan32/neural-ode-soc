# NeuralODE.py
#
# Defines the PyTorch model architecture for predicting battery state of charge (SoC) from voltage and current.
# Uses a neural ordinary differential equation (ODE) approach, where the model learns a continuous-time representation of the battery dynamics.

# imports
import torch
import torch.nn as nn
import torchdiffeq
from scipy.interpolate import interp1d

# We will use the training data to create an interpolation function that maps time to input features (voltage and current).
# This allows the ODE function to access the input features at any time point during integration.
# model 
class ODEFunc(nn.Module):
    def __init__(self, latent_dim=2, input_dim=3):
        super(ODEFunc, self).__init__()
        self.interpolator = None

        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, latent_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    # call this once per cycle before odeint to set the interpolator for that cycle's input features
    def set_interpolator(self, t_np, x_np):
        self.interpolator = interp1d(t_np, x_np, axis=0,bounds_error=False, fill_value="extrapolate")

    def forward(self, t, z):
        # look up input features at current time t (scalar)
        x_t = torch.tensor(self.interpolator(t.item()), dtype=torch.float32)
        inp = torch.cat([z, x_t])
        return self.net(inp)