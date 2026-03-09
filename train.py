# train.py 
#
# Main training loop for the Neural ODE model. Loads preprocessed data, applies normalization, trains the model, and evaluates on validation set.
# Also "trains" baseline models (Coulomb counting and OCV lookup) for comparison.

# imports
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_cycles, remove_relaxation, split_cycles, compute_norm_stats
# neural ode model
from NeuralODE import ODEFunc
from torchdiffeq import odeint
# baselines
from baselines import coulomb_counting, fit_ocv_lookup, ocv_lookup_predict
