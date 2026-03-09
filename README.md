# neural-ode-soc

Neural ODE-based State of Charge (SOC) estimation for Li-ion batteries using the NASA Battery Dataset.

> **Status:** Work in progress

---

## What This Is

State of Charge is the battery equivalent of a fuel gauge — it tells you what fraction of usable energy remains. Estimating it accurately is hard because the "tank size" shrinks as the battery ages.

This project uses a Neural ODE to learn the continuous-time dynamics of SOC directly from measurable signals (voltage, current, temperature), without hand-coding a physics model. The key advantage over traditional Coulomb counting is that the model remains accurate as the cell degrades across hundreds of cycles.

## Data

NASA Li-ion Battery Aging Dataset (B0005) from the NASA Prognostics Center of Excellence. 18650-format Li-ion cells cycled to end-of-life (30% capacity fade). Room temperature, constant current discharge at 2A.

> Raw `.mat` files not included. Download from [NASA PCoE](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) and place in `data/raw/`.

## Structure

```
neural-ode-soc/
├── data/
│   └── .gitkeep
├── figures/
│   └── .gitkeep
├── 01_eda.py           # Exploratory data analysis and discharge curve visualization
├── preprocess.py       # Coulomb counting, normalization, dataset construction
├── model.py            # ODEFunc and NeuralODE (torchdiffeq)
├── train.py            # Training loop
├── evaluate.py         # Load checkpoint, generate figures
├── mat_to_csv.py       # Convert raw .mat files to processed CSVs
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Convert raw data
python mat_to_csv.py --data_dir ./data/raw --out_dir ./data/processed
```

## Background

This project grew out of research applying Neural ODEs to ocean sensor data reconstruction. The core idea transfers naturally — both problems involve learning latent continuous-time dynamics from sparse, noisy observations of a physical system.

For battery SOC specifically, the Neural ODE learns:

```
dz/dt = f(z, u, t)   where z is latent state, u = [V, I, T]
SOC(t) = g(z(t))
```

rather than relying on the integral form of Coulomb counting, which accumulates error over time and degrades as cell capacity fades.