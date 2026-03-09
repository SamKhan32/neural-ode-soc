# train.py 
#
# Main training loop for the Neural ODE model. Loads preprocessed data, applies normalization, trains the model, and evaluates on validation set.
# Also "trains" baseline models (Coulomb counting and OCV lookup) for comparison.

# imports
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_cycles, remove_relaxation, split_cycles, compute_norm_stats, normalize_cycles
from NeuralODE import ODEFunc # neural ode model architecture
from torchdiffeq import odeint # for solving the ODE during training
from baselines import coulomb_counting, fit_ocv_lookup, ocv_lookup_predict # baseline models for comparison

# global variables

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20




def main():
    print("Running Main Training Loop")
    # load and preprocess data
    cycle_dfs = load_cycles()
    print(f"loaded {len(cycle_dfs)} cycles")
    cycle_dfs = remove_relaxation(cycle_dfs)
    train_cycles, val_cycles, test_cycles = split_cycles(cycle_dfs)
    print(f"train: {len(train_cycles)} cycles, val: {len(val_cycles)} cycles, test: {len(test_cycles)} cycles")

    # normalize using training stats only
    stats = compute_norm_stats(train_cycles)
    np.save("data/processed/norm_stats.npy", stats)
    train_cycles = normalize_cycles(train_cycles, stats)
    val_cycles = normalize_cycles(val_cycles, stats)
    test_cycles = normalize_cycles(test_cycles, stats)
    # prepare training data for Neural ODE

if __name__ == "__main__":
    main()