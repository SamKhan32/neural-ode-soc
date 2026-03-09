# evaluate_scenario.py
#
# Visualizes the core failure mode of Coulomb counting: sensitivity to the
# initial SOC estimate. CC is a pure integrator, so a wrong starting value
# produces a persistent offset that never corrects itself across the discharge.
# The Neural ODE sidesteps this by inferring z0 from the first observed signal
# rather than relying on a user-supplied scalar.
#
# Generates two figures:
#   scenario_fixed_offset.png  -- CC with a fixed bad init vs. Neural ODE
#   scenario_noise_sweep.png   -- CC uncertainty band across N noisy inits
#
# Run after train.py:
#   python evaluate_scenario.py

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from preprocess import load_cycles, remove_relaxation, split_cycles, compute_norm_stats, normalize_cycles
from NeuralODE import ODEFunc
from baselines import coulomb_counting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# which test cycle to visualize: "early", "mid", "late", or an integer index
SCENARIO_CYCLE_IDX = "mid"

# how far off is the bad soc_init in figure 1 (subtracted from true init)
CC_INIT_OFFSET = 0.15

# sigma levels for the noise sweep in figure 2
NOISE_SIGMAS = [0.05, 0.10, 0.20]
N_SAMPLES = 100


def load_checkpoint():
    checkpoint = torch.load("data/processed/checkpoint.pt", map_location=device, weights_only=False)
    odefunc = ODEFunc(latent_dim=2, input_dim=3).to(device)
    encoder = nn.Linear(3, 2).to(device)
    decoder = nn.Linear(2, 1).to(device)
    odefunc.load_state_dict(checkpoint["odefunc"])
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    odefunc.eval()
    encoder.eval()
    decoder.eval()
    return odefunc, encoder, decoder


def predict_node(df, odefunc, encoder, decoder):
    t = torch.tensor(df["time_s"].values, dtype=torch.float32)[::20]
    x = torch.tensor(df[["voltage_V", "current_A", "temperature_C"]].values, dtype=torch.float32)[::20]
    t = (t - t[0]) / (t[-1] - t[0])
    t, x = t.to(device), x.to(device)
    odefunc.set_interpolator(t.cpu().numpy(), x.cpu().numpy())
    z0 = encoder(x[0])
    with torch.no_grad():
        z_t = odeint(odefunc, z0, t, method="dopri5", rtol=1e-3, atol=1e-3)
    return decoder(z_t).squeeze().detach().cpu().numpy()


def predict_coulomb_with_init(raw_df, soc_init):
    return coulomb_counting(
        raw_df["time_s"].values,
        raw_df["current_A"].values,
        raw_df["capacity_Ah"].iloc[0],
        soc_init=soc_init,
    )[::20]


def resolve_cycle_idx(test_cycles, key):
    n = len(test_cycles)
    if isinstance(key, int):
        return key
    return {"early": 0, "mid": n // 2, "late": n - 1}[key]


def plot_fixed_offset(t_sub, soc_true, soc_node, soc_cc_correct, soc_cc_offset, offset, save=True):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t_sub, soc_true,       color="black",     linewidth=2.0, label="Ground Truth")
    ax.plot(t_sub, soc_node,       color="steelblue", linewidth=1.8, label="Neural ODE (init inferred from signal)", linestyle="--")
    ax.plot(t_sub, soc_cc_correct, color="gray",      linewidth=1.0, label="Coulomb Counting (correct init)", linestyle=":", alpha=0.5)
    ax.plot(t_sub, soc_cc_offset,  color="firebrick", linewidth=1.8, label=f"Coulomb Counting (init − {offset:.2f})", linestyle=":")

    # show that the error at the end of the cycle is the same as at the start
    gap = abs(soc_cc_offset[-1] - soc_true[-1])
    ax.annotate(
        f"error = {gap:.3f}\n(never corrects)",
        xy=(t_sub[-1], soc_cc_offset[-1]),
        xytext=(t_sub[-1] * 0.75, soc_cc_offset[-1] + 0.05),
        arrowprops=dict(arrowstyle="->", color="firebrick"),
        color="firebrick",
        fontsize=9,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(
        "SOC Estimation — Sensitivity to Initial Condition\n"
        "A bad soc_init offsets Coulomb counting for the entire discharge; Neural ODE is unaffected"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save:
        plt.savefig("figures/scenario_fixed_offset.png", dpi=150)
        print("saved: figures/scenario_fixed_offset.png")
    plt.show()


def plot_noise_sweep(t_sub, soc_true, soc_node, raw_df, true_init, sigmas, n_samples, save=True):
    # three shades of red, one per sigma level, drawn widest-first so narrow bands aren't hidden
    band_colors = ["#f4a582", "#d6604d", "#b2182b"]

    fig, ax = plt.subplots(figsize=(11, 5))

    rng = np.random.default_rng(seed=42)
    for sigma, color in zip(reversed(sigmas), reversed(band_colors)):
        inits = np.clip(rng.normal(loc=true_init, scale=sigma, size=n_samples), 0.01, 1.0)
        samples = np.stack([predict_coulomb_with_init(raw_df, s) for s in inits])
        ax.fill_between(t_sub, samples.min(axis=0), samples.max(axis=0), color=color, alpha=0.25)
        ax.plot(t_sub, samples.mean(axis=0), color=color, linewidth=1.2, linestyle=":",
                label=f"CC  σ={sigma:.2f}  (n={n_samples})")

    ax.plot(t_sub, soc_true, color="black",     linewidth=2.0, label="Ground Truth")
    ax.plot(t_sub, soc_node, color="steelblue", linewidth=1.8, label="Neural ODE (single prediction, no init required)", linestyle="--")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(
        "SOC Estimation — Initialization Noise Robustness\n"
        "CC uncertainty band widens with σ; Neural ODE produces one consistent curve"
    )
    ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    if save:
        plt.savefig("figures/scenario_noise_sweep.png", dpi=150)
        print("saved: figures/scenario_noise_sweep.png")
    plt.show()


def main():
    print("loading data...")
    cycle_dfs = load_cycles()
    cycle_dfs = remove_relaxation(cycle_dfs)
    train_cycles, val_cycles, test_cycles = split_cycles(cycle_dfs)
    raw_test_cycles = test_cycles.copy()

    stats = compute_norm_stats(train_cycles)
    train_cycles = normalize_cycles(train_cycles, stats)
    val_cycles   = normalize_cycles(val_cycles,   stats)
    test_cycles  = normalize_cycles(test_cycles,  stats)

    print("loading checkpoint...")
    odefunc, encoder, decoder = load_checkpoint()

    idx = resolve_cycle_idx(test_cycles, SCENARIO_CYCLE_IDX)
    df     = test_cycles[idx]
    raw_df = raw_test_cycles[idx]
    print(f"using test cycle {idx} ('{SCENARIO_CYCLE_IDX}')")

    t_sub     = raw_df["time_s"].values[::20]
    soc_true  = df["soc"].values[::20]
    true_init = soc_true[0]
    print(f"true initial SOC: {true_init:.3f}")

    print("running Neural ODE inference...")
    soc_node = predict_node(df, odefunc, encoder, decoder)

    print("\nfigure 1: fixed offset")
    soc_cc_correct = predict_coulomb_with_init(raw_df, soc_init=true_init)
    soc_cc_offset  = predict_coulomb_with_init(raw_df, soc_init=true_init - CC_INIT_OFFSET)
    print(f"  correct init:  {true_init:.3f}")
    print(f"  offset init:   {true_init - CC_INIT_OFFSET:.3f}")
    print(f"  final error — correct CC: {abs(soc_cc_correct[-1] - soc_true[-1]):.4f}")
    print(f"  final error — offset  CC: {abs(soc_cc_offset[-1]  - soc_true[-1]):.4f}")
    print(f"  final error — Neural ODE: {abs(soc_node[-1]       - soc_true[-1]):.4f}")
    plot_fixed_offset(t_sub, soc_true, soc_node, soc_cc_correct, soc_cc_offset, CC_INIT_OFFSET)

    print("\nfigure 2: noise sweep")
    plot_noise_sweep(t_sub, soc_true, soc_node, raw_df, true_init, NOISE_SIGMAS, N_SAMPLES)

    print("\ndone.")


if __name__ == "__main__":
    main()