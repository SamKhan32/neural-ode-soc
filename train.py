# train.py 
#
# Main training loop for the Neural ODE model. Loads preprocessed data, applies normalization, trains the model, and evaluates on validation set.
# Also "trains" baseline models (Coulomb counting and OCV lookup) for comparison.

# imports
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_cycles, remove_relaxation, split_cycles, compute_norm_stats, normalize_cycles
from NeuralODE import ODEFunc # neural ode model architecture
from torchdiffeq import odeint # for solving the ODE during training
from baselines import coulomb_counting, fit_ocv_lookup, ocv_lookup_predict # baseline models for comparison
import time
import matplotlib.pyplot as plt
import pickle

# global variables
LEARNING_RATE = 1e-3
EPOCHS = 20
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def train_neural_ode(train_cycles, val_cycles):
    # initialize model components
    odefunc = ODEFunc(latent_dim=2, input_dim=3)
    encoder = nn.Linear(3, 2)   # maps first observation [V, I, T] -> z0
    decoder = nn.Linear(2, 1)   # maps latent state z -> SOC
    odefunc = odefunc.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    params = list(odefunc.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    train_losses = []
    epoch_times = []
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0.0
        odefunc.train()
        encoder.train()
        decoder.train()

        for idx, df in enumerate(train_cycles):
            optimizer.zero_grad()
         
            # build tensors for this cycle
            t = torch.tensor(df["time_s"].values, dtype=torch.float32)
            x = torch.tensor(df[["voltage_V", "current_A", "temperature_C"]].values, dtype=torch.float32)
            y = torch.tensor(df["soc"].values, dtype=torch.float32)
            # subsample every 20 timesteps
            t = t[::20]
            x = x[::20]
            y = y[::20]
            # normalize time to start at 0, and to be between 0 and 1
            t = t - t[0]
            t = t / t[-1]  # normalize to [0, 1]

            # send everything to gpu if using
            t = t.to(device)
            x = x.to(device)
            y = y.to(device)
            # set interpolator for this cycle
            odefunc.set_interpolator(t.cpu().numpy(), x.cpu().numpy())

            # initial latent state from first observation
            z0 = encoder(x[0])

            # integrate latent state forward through time
            z_t = odeint(odefunc, z0, t, method="dopri5", rtol=1e-3, atol=1e-3) # [T, latent_dim]
            # decode to SOC
            soc_pred = decoder(z_t).squeeze()  # [T]

            loss = nn.functional.mse_loss(soc_pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_cycles)
        train_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"epoch {epoch+1}/{EPOCHS} — train loss: {avg_loss:.6f} — time: {epoch_time:.1f}s")
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Training Loss - Neural ODE")
    plt.xticks(range(EPOCHS))
    plt.tight_layout()
    plt.savefig("figures/training_loss.png", dpi=150)
    plt.show()

    print(f"avg epoch time: {np.mean(epoch_times):.1f}s")
    print(f"total training time: {sum(epoch_times):.1f}s")
    return odefunc, encoder, decoder

        



def main():
    print("Running Main Training Loop")
    # verify gpu or cpu for training
    print(device)
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
    # train neural ode
    odefunc, encoder, decoder = train_neural_ode(train_cycles, val_cycles)
    # save models
    torch.save({"odefunc": odefunc.state_dict(),"encoder": encoder.state_dict(),"decoder": decoder.state_dict(),"stats":   stats,}, "data/processed/checkpoint.pt")
    print("checkpoint saved")
    # "fit" the ocv-soc lookup table on train
    ocv_lookup = fit_ocv_lookup(train_cycles)
    with open("data/processed/ocv_lookup.pkl", "wb") as f:
        pickle.dump(ocv_lookup, f)

if __name__ == "__main__":
    main()