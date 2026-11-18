"""Generate the NN training dataset by running the physics model on random parameters.

Outputs
  data/regen_dataset.csv        in-range samples (train/val/test come from this)
  data/regen_extrapolation.csv  out-of-range samples (to test generalisation)
"""
import os
import numpy as np
import pandas as pd
from model import simulate

# (low, high) – identical to the Streamlit slider ranges
RANGES = {
    'mass_t': (80, 400),
    'speed_kmh': (20, 120),
    'decel': (0.3, 1.5),
    'efficiency': (0.40, 0.95),
    'max_power_kw': (0, 5000),
}
# Outside the training box – used only for the extrapolation test
EXTRA_RANGES = {
    'mass_t': (400, 500),
    'speed_kmh': (120, 150),
    'decel': (0.3, 1.5),
    'efficiency': (0.40, 0.95),
    'max_power_kw': (5000, 7000),
}


def sample(n, ranges, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = {k: rng.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
        s, _ = simulate(p['mass_t'], p['speed_kmh'], p['decel'], p['efficiency'], p['max_power_kw'])
        p.update(initial_kwh=s['initial_kwh'], recovered_kwh=s['recovered_kwh'], recovery_pct=s['recovery_pct'])
        rows.append(p)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    main = sample(20000, RANGES, seed=42)
    main.to_csv('data/regen_dataset.csv', index=False)
    extra = sample(2000, EXTRA_RANGES, seed=7)
    extra.to_csv('data/regen_extrapolation.csv', index=False)
    print(f'regen_dataset.csv: {len(main)} rows | regen_extrapolation.csv: {len(extra)} rows')
    print(main.describe().loc[['min', 'mean', 'max']].round(3).to_string())
