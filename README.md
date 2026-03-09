# neural-ode-soc

Neural ODE-based State of Charge (SOC) estimation for Li-ion batteries using the NASA Battery Dataset (B0005).

Built as a weekend project to explore continuous-time deep learning for battery degradation modeling.

---

## What This Is

State of Charge is the battery equivalent of a fuel gauge — it tells you what fraction of usable energy remains. Estimating it accurately is hard because the usable capacity shrinks as the battery ages, which causes traditional Coulomb counting to accumulate error over hundreds of cycles.

This project trains a Neural ODE to learn the continuous-time latent dynamics of a discharging cell directly from measurable signals — voltage, current, and temperature — without hand-coding a physics model.

**Design framing:** the Neural ODE is an *offline analysis tool*, not a real-time BMS replacement. The adaptive ODE solver (dopri5) is too slow for embedded real-time use. The intended use case is as a correction or initialization signal for an online estimator (e.g., an Extended Kalman Filter), or for offline capacity fade analysis across a fleet of cells.

---

## Architecture

```
x = [V, I, T]  (3 measurable signals)

encoder:   Linear(3 → 2)         maps first observation to initial latent state z₀
ODEFunc:   MLP(2+3 → 50 → 2)    learned dynamics  dz/dt = f(z, u(t))
decoder:   Linear(2 → 1)         maps latent state z(t) → SOC(t)
```

Input features are interpolated across the time grid so the ODE function can query `u(t)` at any solver step. Time is normalized to `[0, 1]` per cycle.

---

## Data

**NASA Li-ion Battery Aging Dataset — B0005**
- 18650-format Li-ion cell cycled to end-of-life (~30% capacity fade)
- Constant current discharge at 2A, room temperature
- 168 discharge cycles extracted and converted from raw `.mat` files
- Split: 70% train / 15% val / 15% test, **chronologically by cycle index**

The chronological split is intentional — early (healthy) cycles are training data, late (degraded) cycles are the test set. This simulates real deployment where a model trained on a fresh cell must generalize to an aged one.

> Raw `.mat` files are not included. Download from [NASA PCoE](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) and place in `data/raw/`.

---

## Baselines

| Method | Description |
|---|---|
| **Coulomb Counting** | Physics-based current integration. Requires known capacity and initial SOC. Degrades as capacity fades. |
| **OCV Lookup** | Voltage-to-SOC interpolation table fit on training cycles. No current or temperature information. |
| **Neural ODE** | Learns latent continuous-time dynamics from all three signals. |

---

## Structure

```
neural-ode-soc/
├── data/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
├── 01_eda.ipynb        # Discharge curve visualization, capacity fade, data quality checks
├── mat_to_csv.py       # Convert raw .mat files to per-cycle CSVs
├── preprocess.py       # Load cycles, remove relaxation rows, split, normalize
├── NeuralODE.py        # ODEFunc (MLP dynamics) with set_interpolator pattern
├── baselines.py        # Coulomb counting and OCV-SOC lookup table
├── train.py            # Training loop — saves checkpoint.pt and ocv_lookup.pkl
├── evaluate.py         # Inference, MAE/RMSE metrics, comparison figures
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Convert raw .mat files to CSVs
python mat_to_csv.py --data_dir ./data/raw --out_dir ./data/processed

# Train
python train.py

# Evaluate
python evaluate.py
```

---

## Background

This project grew out of research applying Neural ODEs to ocean sensor data reconstruction — both problems involve learning latent continuous-time dynamics from sparse, noisy observations of a physical system.

For battery SOC, the Neural ODE learns:

```
dz/dt = f(z, u(t))    where z is a 2D latent state, u = [V, I, T]
SOC(t) = g(z(t))
```

rather than relying on the integral form of Coulomb counting, which accumulates measurement error over time and degrades as cell capacity fades.

The latent ODE formulation is also a natural fit for future extensions: encoding degradation state (capacity fade, internal resistance growth) as additional latent dimensions, or coupling the dynamics model with a Kalman filter for real-time deployment.