"""
mat_to_csv.py

Converts NASA Battery Dataset .mat files (B0005, B0006, B0007, B0018)
into organized CSVs, one per discharge cycle.

Usage:
    python mat_to_csv.py --data_dir ./data/raw --out_dir ./data/processed

Output structure:
    data/processed/
        B0005/
            discharge_001.csv
            discharge_002.csv
            ...
        B0006/
            ...
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


BATTERIES = ["B0005", "B0006", "B0007", "B0018"]


def extract_discharge_cycles(mat_path, battery_name):
    """
    Parse a .mat file and return a list of DataFrames,
    one per discharge cycle.

    The .mat structure is deeply nested:
        mat[battery][0,0]['cycle'][0, i] -> one cycle
    Each cycle has a 'type' field ('charge', 'discharge', 'impedance').
    We only extract discharge cycles.
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    battery = mat[battery_name]
    cycles = battery.cycle  # array of cycle structs

    discharge_cycles = []

    for i, cycle in enumerate(cycles):
        if cycle.type.strip() != "discharge":
            continue

        data = cycle.data

        # Some cycles have malformed or empty data — skip them
        try:
            time = np.atleast_1d(data.Time).astype(float)
            voltage = np.atleast_1d(data.Voltage_measured).astype(float)
            current = np.atleast_1d(data.Current_measured).astype(float)
            temperature = np.atleast_1d(data.Temperature_measured).astype(float)
            capacity = float(data.Capacity)  # scalar: measured Ah this cycle
        except Exception as e:
            print(f"  Skipping cycle {i} ({battery_name}): {e}")
            continue

        # Skip obviously bad cycles (too short or zero capacity)
        if len(time) < 10 or capacity <= 0:
            print(f"  Skipping cycle {i} ({battery_name}): too short or zero capacity")
            continue

        # Compute SOC via Coulomb counting using measured capacity
        # Current is negative during discharge, so we take abs
        q_actual_as = capacity * 3600.0  # Ah -> As
        dt = np.diff(time, prepend=time[0])  # same length as time
        charge_removed = np.cumsum(np.abs(current) * dt)
        soc = 1.0 - (charge_removed / q_actual_as)
        soc = np.clip(soc, 0.0, 1.0)

        df = pd.DataFrame({
            "time_s":       time,
            "voltage_V":    voltage,
            "current_A":    current,
            "temperature_C": temperature,
            "soc":          soc,
            "capacity_Ah":  capacity,        # constant per cycle, useful for aging plots
            "cycle_index":  i,               # original cycle number in the .mat file
            "battery":      battery_name,
        })

        discharge_cycles.append(df)

    return discharge_cycles


def main(data_dir, out_dir):
    for battery_name in BATTERIES:
        mat_path = os.path.join(data_dir, f"{battery_name}.mat")

        if not os.path.exists(mat_path):
            print(f"[SKIP] {mat_path} not found")
            continue

        print(f"Processing {battery_name}...")
        cycles = extract_discharge_cycles(mat_path, battery_name)
        print(f"  Found {len(cycles)} discharge cycles")

        battery_out_dir = os.path.join(out_dir, battery_name)
        os.makedirs(battery_out_dir, exist_ok=True)

        for j, df in enumerate(cycles):
            out_path = os.path.join(battery_out_dir, f"discharge_{j+1:03d}.csv")
            df.to_csv(out_path, index=False)

        print(f"  Saved to {battery_out_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/raw",
                        help="Directory containing B0005.mat etc.")
    parser.add_argument("--out_dir", default="./data/processed",
                        help="Output directory for CSVs")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)