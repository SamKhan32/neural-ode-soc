# preprocess.py
#
# This file contains functions that are ran during train.py to load and preprocess the raw discharge cycle data for B0005.
# Loads raw discharge cycles for B0005, removes relaxation rows, splits into
# train/val/test by cycle index, and computes normalization statistics from
# training data only. 
#
# Split is chronological - early (healthy) cycles are training, late (degraded)
# cycles are test. This simulates real deployment where you train on a fresh
# battery and evaluate on an aged one.

# imports
import os
import numpy as np
import pandas as pd

# global variables
DATA_DIR_PATH = "data/processed/B0005/"
TRAIN_SPLIT = 0.7  # first 70% of cycles for training
VAL_SPLIT = 0.15   # next 15% for validation, remaining 15% for test

# function that builds a list of per-cycle dataframes from all discharge cycles
def load_cycles():
    cycle_dfs = []
    for filename in sorted(os.listdir(DATA_DIR_PATH)):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR_PATH, filename))
            cycle_dfs.append(df)
    return cycle_dfs

# function that removes relaxation rows from a cycle
# relaxation occurs after the load is removed - current drops to near zero and voltage recovers
# we keep only rows where current is below -1.0A (active discharge)
def remove_relaxation(cycle_dfs):
    cleaned = []
    for df in cycle_dfs:
        df = df[df["current_A"] < -1.0].reset_index(drop=True)
        cleaned.append(df)
    return cleaned

# function that splits cycles into train, val, and test sets
# we split by cycle index to simulate real deployment - train on healthy, test on degraded
def split_cycles(cycle_dfs):
    n = len(cycle_dfs)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    train = cycle_dfs[:train_end]
    val = cycle_dfs[train_end:val_end]
    test = cycle_dfs[val_end:]
    return train, val, test

# function that computes normalization statistics from training cycles only
# we never fit on val or test to avoid data leakage
def compute_norm_stats(train_cycles):
    train_df = pd.concat(train_cycles, ignore_index=True)
    stats = {
        "voltage_mean": train_df["voltage_V"].mean(),
        "voltage_std":  train_df["voltage_V"].std(),
        "current_mean": train_df["current_A"].mean(),
        "current_std":  train_df["current_A"].std(),
        "temp_mean":    train_df["temperature_C"].mean(),
        "temp_std":     train_df["temperature_C"].std(),
    }
    return stats

# function that applies z-score normalization to a list of cycles using precomputed stats
def normalize_cycles(cycle_dfs, stats):
    normalized = []
    for df in cycle_dfs:
        df = df.copy()
        df["voltage_V"]     = (df["voltage_V"]     - stats["voltage_mean"]) / stats["voltage_std"]
        df["current_A"]     = (df["current_A"]     - stats["current_mean"]) / stats["current_std"]
        df["temperature_C"] = (df["temperature_C"] - stats["temp_mean"])    / stats["temp_std"]
        normalized.append(df)
    return normalized


if __name__ == "__main__":
    cycle_dfs = load_cycles()
    print(f"loaded {len(cycle_dfs)} cycles")

    cycle_dfs = remove_relaxation(cycle_dfs)

    train_cycles, val_cycles, test_cycles = split_cycles(cycle_dfs)
    print(f"train: {len(train_cycles)} cycles, val: {len(val_cycles)} cycles, test: {len(test_cycles)} cycles")

    stats = compute_norm_stats(train_cycles)
    np.save("data/processed/norm_stats.npy", stats)
    print("normalization stats saved")