"""Compare NN predictions against the physics model: metrics + figures.

Outputs: models/metrics.json, figures/*.png
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nn_model import FEATURES, predict_df, initial_kwh

os.makedirs('figures', exist_ok=True)


def predict_any(name, df):
    """Return DataFrame[recovered_kwh, recovery_pct] for any saved model."""
    if name == 'mlp_direct':
        return predict_df(df, 'direct')
    if name == 'mlp_fraction':
        return predict_df(df, 'fraction')
    out = joblib.load(f'models/{name}.joblib').predict(df[FEATURES])
    return pd.DataFrame(out, columns=['recovered_kwh', 'recovery_pct'], index=df.index)


def metrics(true, pred):
    r = {}
    for col in ['recovered_kwh', 'recovery_pct']:
        t, p = true[col].values, pred[col].values
        mask = t > 1e-3 * max(t.max(), 1)  # MAPE only where the target is not ~0
        r[col] = {'R2': float(r2_score(t, p)),
                  'MAE': float(mean_absolute_error(t, p)),
                  'RMSE': float(np.sqrt(mean_squared_error(t, p))),
                  'MAPE_%': float(100 * np.mean(np.abs(t[mask] - p[mask]) / t[mask]))}
    return r


def cap_ratio(df):
    """Power cap / peak ideal electrical power. <1 means the cap binds (the kink region)."""
    peak_kw = df['efficiency'] * df['mass_t'] * 1000 * df['decel'] * (df['speed_kmh'] / 3.6) / 1000
    return df['max_power_kw'] / peak_kw


if __name__ == '__main__':
    names = ['mlp_direct', 'mlp_fraction', 'linear', 'rf']
    sets = {'test': pd.read_csv('data/test_split.csv'),
            'extrapolation': pd.read_csv('data/regen_extrapolation.csv')}
    results = {k: {} for k in sets}
    preds = {k: {} for k in sets}
    for sname, df in sets.items():
        for n in names:
            preds[sname][n] = predict_any(n, df)
            results[sname][n] = metrics(df, preds[sname][n])
    json.dump(results, open('models/metrics.json', 'w'), indent=2)

    # --- console summary
    for sname in sets:
        print(f'\n== {sname} set ({len(sets[sname])} rows) ==')
        rows = []
        for n in names:
            m = results[sname][n]
            rows.append({'model': n,
                         'R2_kWh': m['recovered_kwh']['R2'], 'MAE_kWh': m['recovered_kwh']['MAE'],
                         'RMSE_kWh': m['recovered_kwh']['RMSE'],
                         'R2_pct': m['recovery_pct']['R2'], 'MAE_pct': m['recovery_pct']['MAE'],
                         'RMSE_pct': m['recovery_pct']['RMSE']})
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    # --- Fig 1: parity plots, main MLP, test set
    t = sets['test']; p = preds['test']['mlp_direct']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, unit in zip(axs, ['recovered_kwh', 'recovery_pct'], ['kWh', '%']):
        ax.scatter(t[col], p[col], s=3, alpha=.35)
        lim = [0, max(t[col].max(), p[col].max()) * 1.02]
        ax.plot(lim, lim, 'r--', lw=1, label='ideal')
        ax.set(xlabel=f'Physics model ({unit})', ylabel=f'NN prediction ({unit})',
               title=f'{col}  (R²={results["test"]["mlp_direct"][col]["R2"]:.4f})')
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig('figures/parity_test.png', dpi=150); plt.close(fig)

    # --- Fig 2: residual vs cap ratio (the kink region)
    ratio = cap_ratio(t)
    res = p['recovery_pct'] - t['recovery_pct']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ratio.clip(upper=3), res, s=3, alpha=.35)
    ax.axvline(1, color='r', ls='--', label='cap = peak ideal power (kink)')
    ax.axhline(0, color='k', lw=.6)
    ax.set(xlabel='Power cap / peak ideal electrical power (clipped at 3)', ylabel='NN − physics (recovery %)',
           title='Residuals are larger where the power cap binds (ratio < 1)')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig('figures/residual_vs_cap_ratio.png', dpi=150); plt.close(fig)

    # --- Fig 3: extrapolation parity (recovery %)
    e = sets['extrapolation']
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, c in zip(['mlp_direct', 'linear', 'rf'], ['C0', 'C1', 'C2']):
        ax.scatter(e['recovery_pct'], preds['extrapolation'][n]['recovery_pct'], s=4, alpha=.4, c=c,
                   label=f'{n} (R²={results["extrapolation"][n]["recovery_pct"]["R2"]:.3f})')
    ax.plot([0, 100], [0, 100], 'k--', lw=1)
    ax.set(xlabel='Physics (%)', ylabel='Model (%)', title='Out-of-range inputs (speed 120–150, mass 400–500, cap 5–7 MW)')
    ax.legend(markerscale=3); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig('figures/parity_extrapolation.png', dpi=150); plt.close(fig)

    # --- Fig 4: model comparison (MAE on recovery %, test vs extrapolation)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = .38; x = np.arange(len(names))
    ax.bar(x - w / 2, [results['test'][n]['recovery_pct']['MAE'] for n in names], w, label='in-range test')
    ax.bar(x + w / 2, [results['extrapolation'][n]['recovery_pct']['MAE'] for n in names], w, label='extrapolation')
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set(ylabel='MAE of recovery % (points)', title='Model comparison')
    ax.legend(); ax.grid(alpha=.3, axis='y'); fig.tight_layout(); fig.savefig('figures/model_comparison.png', dpi=150); plt.close(fig)

    # --- Fig 5: 1-D sweep of power cap, NN vs physics (default train)
    from model import simulate
    caps = np.linspace(0, 5000, 101)
    base = dict(mass_t=200, speed_kmh=60, decel=0.9, efficiency=0.8)
    phys = [simulate(max_power_kw=c, **base)[0]['recovery_pct'] for c in caps]
    sweep = pd.DataFrame({**{k: v for k, v in base.items()}, 'max_power_kw': caps})
    nn = predict_df(sweep)['recovery_pct']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(caps, phys, label='Physics'); ax.plot(caps, nn, '--', label='NN')
    ax.set(xlabel='Accepted power limit (kW)', ylabel='Recovery (%)', title='Power-limit sweep: 200 t, 60 km/h, 0.9 m/s², η=0.8')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig('figures/power_cap_sweep.png', dpi=150); plt.close(fig)
    print('\nfigures saved to figures/')
