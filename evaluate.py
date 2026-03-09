# evaluate.py
#
# Loads trained Neural ODE checkpoint and baseline models, runs inference on
# test cycles, computes error metrics, and generates figures for comparison.
# Run after train.py completes.

import pickle
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from preprocess import load_cycles, remove_relaxation, split_cycles, compute_norm_stats, normalize_cycles
from NeuralODE import ODEFunc
from baselines import coulomb_counting, ocv_lookup_predict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_checkpoint():
    checkpoint = torch.load("data/processed/checkpoint.pt", map_location=device)

    odefunc = ODEFunc(latent_dim=2, input_dim=3).to(device)
    encoder = nn.Linear(3, 2).to(device)
    decoder = nn.Linear(2, 1).to(device)

    odefunc.load_state_dict(checkpoint["odefunc"])
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    odefunc.eval()
    encoder.eval()
    decoder.eval()

    stats = checkpoint["stats"]
    return odefunc, encoder, decoder, stats


def load_ocv_lookup():
    with open("data/processed/ocv_lookup.pkl", "rb") as f:
        return pickle.load(f)


def predict_node(df, odefunc, encoder, decoder):
    # build tensors
    t = torch.tensor(df["time_s"].values, dtype=torch.float32)
    x = torch.tensor(df[["voltage_V", "current_A", "temperature_C"]].values, dtype=torch.float32)

    # subsample to match training
    t = t[::20]
    x = x[::20]

    # normalize time
    t = t - t[0]
    t = t / t[-1]

    t = t.to(device)
    x = x.to(device)

    odefunc.set_interpolator(t.cpu().numpy(), x.cpu().numpy())
    z0 = encoder(x[0])

    with torch.no_grad():
        z_t = odeint(odefunc, z0, t, method="dopri5", rtol=1e-3, atol=1e-3)

    soc_pred = decoder(z_t).squeeze().cpu().numpy()
    return soc_pred


def predict_coulomb(df):
    # note: we use raw (unnormalized) capacity here
    # coulomb counting operates on physical units
    return coulomb_counting(
        df["time_s"].values,
        df["current_A"].values,
        df["capacity_Ah"].iloc[0],
    )


def predict_ocv(df, lookup):
    return ocv_lookup_predict(df, lookup)


def compute_mae(pred, true):
    return np.mean(np.abs(pred - true))

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def plot_single_cycle(df, soc_node, soc_coulomb, soc_ocv, cycle_num, save=True):
    # subsampled time and ground truth to match node prediction length
    t = df["time_s"].values[::20]
    soc_true = df["soc"].values[::20]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, soc_true,     color="black",      linewidth=2,   label="Ground Truth (Coulomb)")
    ax.plot(t, soc_node,     color="steelblue",  linewidth=1.5, label="Neural ODE",   linestyle="--")
    ax.plot(t, soc_coulomb[::20], color="firebrick",  linewidth=1.5, label="Coulomb Counting", linestyle=":")
    ax.plot(t, soc_ocv[::20],     color="darkorange", linewidth=1.5, label="OCV Lookup",       linestyle="-.")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SOC Estimation — Test Cycle {cycle_num}")
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"figures/soc_comparison_cycle{cycle_num}.png", dpi=150)
    plt.show()


def plot_mae_over_cycles(node_maes, coulomb_maes, ocv_maes, save=True):
    # shows how each method degrades as the battery ages
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(node_maes,     color="steelblue",  linewidth=1.5, label="Neural ODE")
    ax.plot(coulomb_maes,  color="firebrick",  linewidth=1.5, label="Coulomb Counting")
    ax.plot(ocv_maes,      color="darkorange", linewidth=1.5, label="OCV Lookup")

    ax.set_xlabel("Test Cycle Index")
    ax.set_ylabel("MAE")
    ax.set_title("SOC Estimation Error Over Aging — B0005 Test Set")
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig("figures/mae_over_cycles.png", dpi=150)
    plt.show()


def main():
    print("Loading data...")
    cycle_dfs = load_cycles()
    cycle_dfs = remove_relaxation(cycle_dfs)
    train_cycles, val_cycles, test_cycles = split_cycles(cycle_dfs)

    # normalize using training stats
    stats = compute_norm_stats(train_cycles)
    train_cycles = normalize_cycles(train_cycles, stats)
    val_cycles   = normalize_cycles(val_cycles,   stats)
    test_cycles  = normalize_cycles(test_cycles,  stats)

    print("Loading models...")
    odefunc, encoder, decoder, _ = load_checkpoint()
    ocv_lookup = load_ocv_lookup()

    # run inference on all test cycles and collect metrics
    node_maes     = []
    coulomb_maes  = []
    ocv_maes      = []

    for i, df in enumerate(test_cycles):
        t_sub    = df["time_s"].values[::20]
        soc_true = df["soc"].values[::20]

        soc_node    = predict_node(df, odefunc, encoder, decoder)
        soc_coulomb = predict_coulomb(df)[::20]
        soc_ocv     = predict_ocv(df, ocv_lookup)[::20]

        node_maes.append(compute_mae(soc_node, soc_true))
        coulomb_maes.append(compute_mae(soc_coulomb, soc_true))
        ocv_maes.append(compute_mae(soc_ocv, soc_true))

    # print summary metrics
    print(f"\n{'':20} {'MAE':>8} {'RMSE':>8}")
    print(f"{'Neural ODE':20} {np.mean(node_maes):8.4f}")
    print(f"{'Coulomb Counting':20} {np.mean(coulomb_maes):8.4f}")
    print(f"{'OCV Lookup':20} {np.mean(ocv_maes):8.4f}")

    # plot a few individual cycles — early, mid, late test set
    for cycle_num in [0, len(test_cycles)//2, len(test_cycles)-1]:
        df = test_cycles[cycle_num]
        soc_node    = predict_node(df, odefunc, encoder, decoder)
        soc_coulomb = predict_coulomb(df)
        soc_ocv     = predict_ocv(df, ocv_lookup)
        plot_single_cycle(df, soc_node, soc_coulomb, soc_ocv, cycle_num)

    # plot mae over aging
    plot_mae_over_cycles(node_maes, coulomb_maes, ocv_maes)

    print("\nDone. Figures saved to figures/")


if __name__ == "__main__":
    main()