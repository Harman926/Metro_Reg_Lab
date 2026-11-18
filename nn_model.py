"""Thin wrapper around the trained MLP so the app/tests can call it like simulate()."""
import os
import joblib
import numpy as np
import pandas as pd

FEATURES = ['mass_t', 'speed_kmh', 'decel', 'efficiency', 'max_power_kw']
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_CACHE = {}


def model_available(name='mlp_direct'):
    return os.path.exists(os.path.join(_DIR, f'{name}.joblib'))


def load_model(name='mlp_direct'):
    if name not in _CACHE:
        path = os.path.join(_DIR, f'{name}.joblib')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found - run `python train_nn.py` first')
        _CACHE[name] = joblib.load(path)
    return _CACHE[name]


def initial_kwh(mass_t, speed_kmh):
    """Analytic kinetic energy (kWh) - exact, no learning needed."""
    return 0.5 * (np.asarray(mass_t) * 1000) * (np.asarray(speed_kmh) / 3.6) ** 2 / 3.6e6


def predict_df(df, variant='direct'):
    """Vectorised prediction for a DataFrame with FEATURES columns -> DataFrame[recovered_kwh, recovery_pct].
    variant='direct'  : MLP predicts both outputs.
    variant='fraction': MLP predicts recovery %, kWh = pct/100 * initial KE."""
    X = df[FEATURES]
    ek = initial_kwh(df['mass_t'], df['speed_kmh'])
    if variant == 'fraction':
        pct = np.clip(load_model('mlp_fraction').predict(X), 0, 100)
        kwh = pct / 100 * ek
    else:
        out = load_model('mlp_direct').predict(X)
        kwh, pct = out[:, 0], out[:, 1]
        kwh = np.clip(kwh, 0, ek)
        pct = np.clip(pct, 0, 100)
    return pd.DataFrame({'recovered_kwh': kwh, 'recovery_pct': pct}, index=df.index)


def predict(mass_t, speed_kmh, decel, efficiency, max_power_kw, variant='direct'):
    row = pd.DataFrame([dict(zip(FEATURES, [mass_t, speed_kmh, decel, efficiency, max_power_kw]))])
    r = predict_df(row, variant).iloc[0]
    return {'recovered_kwh': float(r.recovered_kwh), 'recovery_pct': float(r.recovery_pct)}
