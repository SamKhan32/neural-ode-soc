# evaluate_scenario.py
#
# Two scenarios that expose real failure modes of Coulomb counting, each with
# a corresponding Neural ODE comparison.
#
# Scenario A — bad initial SOC (perturbed z0):
#   Both CC and the Neural ODE are given the same wrong initial state.
#   CC integrates forward from that error with no way to recover. The Neural
#   ODE has signal feedback through the ODE dynamics, so the latent state gets
#   pulled back toward the true trajectory as the discharge progresses.
#   The z0 perturbation is computed analytically: since the decoder is a single
#   linear layer, we can solve exactly for the delta that shifts the initial
#   decoded SOC by CC_INIT_OFFSET, making the comparison apples-to-apples.
#
# Scenario B — capacity fade (stale nominal capacity):
#   In a deployed BMS, the CC capacity parameter is set at manufacture and
#   rarely updated. As the cell ages, true capacity fades, so CC systematically
#   overestimates remaining SOC late in life. The Neural ODE learned from the
#   voltage/current signal shape, not the nominal capacity, so it degrades more
#   gracefully across the aging test set.
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

# which test cycle to use for scenario A: "early", "mid", "late", or an int
SCENARIO_CYCLE_IDX = "mid"

# how far off the initial SOC estimate is in scenario A
CC_INIT_OFFSET = 0.15


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


def predict_node(df, odefunc, encoder, decoder, z0_override=None):
    t = torch.tensor(df["time_s"].values, dtype=torch.float32)[::20]
    x = torch.tensor(df[["voltage_V", "current_A", "temperature_C"]].values, dtype=torch.float32)[::20]
    t = (t - t[0]) / (t[-1] - t[0])
    t, x = t.to(device), x.to(device)
    odefunc.set_interpolator(t.cpu().numpy(), x.cpu().numpy())
    z0 = z0_override if z0_override is not None else encoder(x[0])
    with torch.no_grad():
        z_t = odeint(odefunc, z0, t, method="dopri5", rtol=1e-3, atol=1e-3)
    return decoder(z_t).squeeze().detach().cpu().numpy()


def perturb_z0(df, encoder, decoder, soc_offset):
    # find the delta in latent space that shifts the decoded initial SOC by soc_offset
    # decoder is Linear(2->1): soc = w @ z + b, so d(soc)/dz = w
    # we want decoder(z0 + delta) = decoder(z0) + soc_offset
    # => w @ delta = soc_offset => delta = soc_offset * w^T / ||w||^2
    x0 = torch.tensor(
        df[["voltage_V", "current_A", "temperature_C"]].values[0],
        dtype=torch.float32
    ).to(device)
    z0 = encoder(x0)
    w = decoder.weight[0]  # shape [latent_dim]
    delta = soc_offset * w / (w @ w)
    return (z0 + delta).detach()


def predict_coulomb(raw_df, capacity_Ah, soc_init=1.0):
    return coulomb_counting(
        raw_df["time_s"].values,
        raw_df["current_A"].values,
        capacity_Ah,
        soc_init=soc_init,
    )[::20]


def resolve_cycle_idx(test_cycles, key):
    n = len(test_cycles)
    if isinstance(key, int):
        return key
    return {"early": 0, "mid": n // 2, "late": n - 1}[key]


def compute_mae(pred, true):
    return np.mean(np.abs(pred - true))


def plot_scenario_a(t_sub, soc_true, soc_node_perturbed, soc_cc_offset, offset, save=True):
    # three curves only: ground truth, CC with bad init, NODE with same bad init
    # keeping the "correct init" references out avoids distracting from the main point
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t_sub, soc_true,           color="black",     linewidth=2.0, label="Ground Truth")
    ax.plot(t_sub, soc_cc_offset,      color="firebrick", linewidth=1.8, label=f"Coulomb Counting (init − {offset:.2f})", linestyle=":")
    ax.plot(t_sub, soc_node_perturbed, color="steelblue", linewidth=1.8, label=f"Neural ODE (init − {offset:.2f})", linestyle="--")

    cc_gap   = abs(soc_cc_offset[-1]      - soc_true[-1])
    node_gap = abs(soc_node_perturbed[-1] - soc_true[-1])

    ax.annotate(
        f"final error = {cc_gap:.3f}",
        xy=(t_sub[-1], soc_cc_offset[-1]),
        xytext=(t_sub[-1] * 0.70, soc_cc_offset[-1] - 0.08),
        arrowprops=dict(arrowstyle="->", color="firebrick"),
        color="firebrick", fontsize=9,
    )
    ax.annotate(
        f"final error = {node_gap:.3f}",
        xy=(t_sub[-1], soc_node_perturbed[-1]),
        xytext=(t_sub[-1] * 0.70, soc_node_perturbed[-1] + 0.06),
        arrowprops=dict(arrowstyle="->", color="steelblue"),
        color="steelblue", fontsize=9,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(
        f"Scenario A — both methods start with soc_init − {offset:.2f}\n"
        "Coulomb counting carries the error forward; Neural ODE pulls back toward ground truth"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig("figures/scenario_a_bad_init.png", dpi=150)
        print("saved: figures/scenario_a_bad_init.png")
    plt.show()


def plot_scenario_b(test_cycle_indices, node_maes, cc_true_maes, cc_stale_maes, capacity_fade_pct, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: MAE over aging
    ax = axes[0]
    ax.plot(test_cycle_indices, cc_stale_maes, color="firebrick", linewidth=1.8, label="Coulomb Counting (stale capacity)", linestyle=":")
    ax.plot(test_cycle_indices, cc_true_maes,  color="gray",      linewidth=1.0, label="Coulomb Counting (true capacity)",  linestyle=":", alpha=0.5)
    ax.plot(test_cycle_indices, node_maes,     color="steelblue", linewidth=1.8, label="Neural ODE", linestyle="--")
    ax.set_xlabel("Test Cycle Index (older →)")
    ax.set_ylabel("MAE")
    ax.set_title("SOC Error Over Aging")
    ax.legend(fontsize=8.5)

    # right: capacity fade across the same test cycles
    ax = axes[1]
    ax.plot(test_cycle_indices, capacity_fade_pct, color="darkorange", linewidth=1.8)
    ax.set_xlabel("Test Cycle Index (older →)")
    ax.set_ylabel("Capacity remaining (%)")
    ax.set_title("True Cell Capacity Fade")
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Scenario B — Capacity Fade: CC uses a stale nominal capacity from training time\n"
        "Neural ODE tracks the signal shape and degrades more gracefully",
        fontsize=10,
    )
    plt.tight_layout()
    if save:
        plt.savefig("figures/scenario_b_capacity_fade.png", dpi=150)
        print("saved: figures/scenario_b_capacity_fade.png")
    plt.show()


def main():
    print("loading data...")
    cycle_dfs = load_cycles()
    cycle_dfs = remove_relaxation(cycle_dfs)
    train_cycles, val_cycles, test_cycles = split_cycles(cycle_dfs)
    raw_train_cycles = train_cycles.copy()
    raw_test_cycles  = test_cycles.copy()

    stats = compute_norm_stats(train_cycles)
    train_cycles = normalize_cycles(train_cycles, stats)
    val_cycles   = normalize_cycles(val_cycles,   stats)
    test_cycles  = normalize_cycles(test_cycles,  stats)

    print("loading checkpoint...")
    odefunc, encoder, decoder = load_checkpoint()

    # nominal capacity: mean over training cycles, representing what a BMS would
    # have been calibrated to at the start of the cell's life
    nominal_capacity = np.mean([df["capacity_Ah"].iloc[0] for df in raw_train_cycles])
    print(f"nominal capacity (training mean): {nominal_capacity:.4f} Ah")

    # scenario A
    print("\nscenario A: bad initial SOC")
    idx    = resolve_cycle_idx(test_cycles, SCENARIO_CYCLE_IDX)
    df     = test_cycles[idx]
    raw_df = raw_test_cycles[idx]
    print(f"  using test cycle {idx} ('{SCENARIO_CYCLE_IDX}')")

    t_sub     = raw_df["time_s"].values[::20]
    soc_true  = df["soc"].values[::20]
    true_init = soc_true[0]

    soc_node_clean     = predict_node(df, odefunc, encoder, decoder)
    z0_bad             = perturb_z0(df, encoder, decoder, soc_offset=-CC_INIT_OFFSET)
    soc_node_perturbed = predict_node(df, odefunc, encoder, decoder, z0_override=z0_bad)
    soc_cc_correct     = predict_coulomb(raw_df, raw_df["capacity_Ah"].iloc[0], soc_init=true_init)
    soc_cc_offset      = predict_coulomb(raw_df, raw_df["capacity_Ah"].iloc[0], soc_init=true_init - CC_INIT_OFFSET)

    print(f"  true init: {true_init:.3f}  |  perturbed init: {true_init - CC_INIT_OFFSET:.3f}")
    print(f"  final error — CC (offset):       {abs(soc_cc_offset[-1]       - soc_true[-1]):.4f}")
    print(f"  final error — NODE (perturbed):  {abs(soc_node_perturbed[-1]  - soc_true[-1]):.4f}")
    plot_scenario_a(t_sub, soc_true, soc_node_perturbed, soc_cc_offset, CC_INIT_OFFSET)

    # scenario B
    print("\nscenario B: capacity fade")
    node_maes      = []
    cc_true_maes   = []
    cc_stale_maes  = []
    capacities     = []
    test_indices   = list(range(len(test_cycles)))

    for df, raw_df in zip(test_cycles, raw_test_cycles):
        soc_true = df["soc"].values[::20]
        true_cap = raw_df["capacity_Ah"].iloc[0]
        capacities.append(true_cap)

        soc_node      = predict_node(df, odefunc, encoder, decoder)
        soc_cc_true   = predict_coulomb(raw_df, true_cap,        soc_init=soc_true[0])
        soc_cc_stale  = predict_coulomb(raw_df, nominal_capacity, soc_init=soc_true[0])

        node_maes.append(compute_mae(soc_node,     soc_true))
        cc_true_maes.append(compute_mae(soc_cc_true,  soc_true))
        cc_stale_maes.append(compute_mae(soc_cc_stale, soc_true))

    capacity_fade_pct = 100.0 * np.array(capacities) / capacities[0]
    print(f"  capacity range: {min(capacities):.4f} – {max(capacities):.4f} Ah  ({capacity_fade_pct[-1]:.1f}% remaining at end)")
    print(f"  mean MAE — NODE:        {np.mean(node_maes):.4f}")
    print(f"  mean MAE — CC (true):   {np.mean(cc_true_maes):.4f}")
    print(f"  mean MAE — CC (stale):  {np.mean(cc_stale_maes):.4f}")
    plot_scenario_b(test_indices, node_maes, cc_true_maes, cc_stale_maes, capacity_fade_pct)

    print("\ndone.")


if __name__ == "__main__":
    main()