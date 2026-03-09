# baselines.py
#
# Simple baseline models for SOC estimation.  and an OCV-SOC lookup table (voltage-only).
# Includes Coulomb counting (physics-based integration)
# and an OCV-SOC lookup table (voltage-only).
# Used as reference points to evaluate Neural ODE performance.

# imports
import numpy as np
from scipy.interpolate import interp1d


# coulomb counting baseline
# integrates current over time to estimate SOC
# requires a known starting SOC and the true capacity of the cell
def coulomb_counting(time_s, current_A, capacity_Ah, soc_init=1.0):
    capacity_As = capacity_Ah * 3600.0
    dt = np.diff(time_s, prepend=time_s[0])
    charge_removed = np.cumsum(np.abs(current_A) * dt)
    soc = soc_init - (charge_removed / capacity_As)
    return np.clip(soc, 0.0, 1.0)


# fits an OCV-SOC lookup table from training cycles
# maps voltage directly to SOC — no current or temperature information used
# returns a callable that takes voltage array and returns SOC array
def fit_ocv_lookup(train_cycles):
    # concatenate all training cycles and compute mean SOC per voltage bin
    all_voltage = np.concatenate([df["voltage_V"].values for df in train_cycles])
    all_soc     = np.concatenate([df["soc"].values for df in train_cycles])

    # build a lookup table by sorting voltages and corresponding SOCs, then using interpolation
    sort_idx = np.argsort(all_voltage)
    voltage_sorted = all_voltage[sort_idx]
    soc_sorted     = all_soc[sort_idx]

    lookup = interp1d(
        voltage_sorted,
        soc_sorted,
        bounds_error=False,
        fill_value=(soc_sorted[0], soc_sorted[-1])
    )
    return lookup


# applies the OCV lookup to a single cycle
def ocv_lookup_predict(df, lookup):
    return lookup(df["voltage_V"].values)
