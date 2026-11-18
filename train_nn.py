"""Train the MLP surrogate (plus baselines) on the physics-generated dataset.

Models saved to models/:
  mlp_direct.joblib    MLP, 2 outputs: recovered_kwh and recovery_pct   <- main model
  mlp_fraction.joblib  MLP, 1 output: recovery_pct; kWh = pct/100 * initial KE (variant)
  linear.joblib        LinearRegression baseline (2 outputs)
  rf.joblib            RandomForest baseline (2 outputs)
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['mass_t', 'speed_kmh', 'decel', 'efficiency', 'max_power_kw']
TARGETS = ['recovered_kwh', 'recovery_pct']
SEED = 42


def make_mlp(seed=SEED):
    """5 -> 64 -> 64 -> 32 -> outputs, ReLU, Adam, early stopping; inputs and targets standardised."""
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='relu', solver='adam',
                             learning_rate_init=1e-3, batch_size=128, max_iter=600,
                             early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
                             random_state=seed)),
    ])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def make_linear():
    return TransformedTargetRegressor(
        regressor=Pipeline([('scale', StandardScaler()), ('lin', LinearRegression())]),
        transformer=StandardScaler())


def make_rf():
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=SEED)


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/regen_dataset.csv')
    train, test = train_test_split(df, test_size=0.15, random_state=SEED)
    train, val = train_test_split(train, test_size=0.15 / 0.85, random_state=SEED)  # 70 / 15 / 15
    test.to_csv('data/test_split.csv', index=False)
    print(f'train={len(train)} val={len(val)} test={len(test)}')

    X, Y = train[FEATURES], train[TARGETS]
    models = {'mlp_direct': make_mlp(), 'linear': make_linear(), 'rf': make_rf()}
    for name, m in models.items():
        m.fit(X, Y)
        joblib.dump(m, f'models/{name}.joblib', compress=3)
        print(f'trained {name}')

    # Fraction variant: predict recovery % only, then multiply by the analytic initial kinetic energy
    frac = make_mlp(seed=SEED + 1)
    frac.fit(X, train['recovery_pct'])
    joblib.dump(frac, 'models/mlp_fraction.joblib')
    print('trained mlp_fraction')

    mlp = models['mlp_direct'].regressor_.named_steps['mlp']
    info = {'hidden_layers': list(mlp.hidden_layer_sizes), 'iterations': int(mlp.n_iter_),
            'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test)}
    json.dump(info, open('models/training_info.json', 'w'), indent=2)
    print(info)
