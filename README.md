# Metro Regen Lab — UEE Group Project

## Objective
Simulate a level-track metro train braking at a station, calculate stopping time/distance, kinetic energy, instantaneous braking power, electrical energy accepted by the supply/storage, and unrecovered energy.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
5. Tests: `python -m unittest test_model.py`

## Tabs
- Live braking: adjustable inputs and time-position train illustration
- Energy analysis: energy balance and conventional-versus-regenerative comparison
- Scenario comparison: sensitivity curves and CSV
- Model & validation: equations and assumptions

## Physics
With mass m (kg), initial speed v0 (m/s), and constant deceleration a (m/s²):
- stopping time = v0/a
- stopping distance = v0²/(2a)
- kinetic energy = 0.5*m*v0²
- mechanical braking power = m*a*(v0-a*t)
- recovered electrical power = min(efficiency*mechanical braking power, acceptance power cap)
- recovered energy = integral of recovered electrical power over the stop
- conversion loss = integral((1-efficiency)*mechanical power)
- unaccepted energy = integral(ideal converted power - accepted power)

## Interpretation and limitations
This is a teaching model, **not** a validated metro train design. Constant deceleration is maintained by supplementary braking when regeneration is limited. The supply/storage is assumed capable of accepting electrical power up to the selected cap. Unaccepted energy and conversion losses are not equivalent to all being friction brake heat. Resistive forces, gradients, rotating inertia, low-speed cutout, auxiliary loads, actual traction curves, substations, and subsequent energy use are not modelled. The dashboard's time slider is a frame-by-frame visualization, not an automatically moving animation.

## Suggested 5-member responsibilities
1. Train dynamics and mathematical derivation
2. Energy conversion and power-cap model
3. Streamlit dashboard and visualization
4. Scenario analysis and testing
5. Report, slides, limitations and viva coordination

---

## Neural-network surrogate (added)

An MLP learns the physics model's mapping from the five inputs (train mass, initial speed, deceleration, regeneration efficiency, power acceptance limit) to **recovered energy (kWh)** and **recovery (%)**, and is compared against `simulate()`.

### Run order
```
pip install -r requirements.txt
python generate_dataset.py   # 20,000 in-range + 2,000 out-of-range physics runs -> data/
python train_nn.py           # MLP + baselines -> models/
python evaluate.py           # metrics + figures -> models/metrics.json, figures/
python -m unittest test_nn.py test_model.py
streamlit run app.py         # 5th tab: "NN vs Physics"
```

### Files
| File | Purpose |
|---|---|
| `generate_dataset.py` | Samples inputs uniformly over the slider ranges, labels them with `simulate()` |
| `train_nn.py` | 70/15/15 split; trains MLP (64-64-32, ReLU, Adam, early stopping), a fraction-output MLP variant, Linear and Random Forest baselines |
| `nn_model.py` | `predict()` / `predict_df()` wrapper with physical clipping (0 ≤ kWh ≤ initial KE, 0 ≤ % ≤ 100) |
| `evaluate.py` | R², MAE, RMSE, MAPE; parity, residual, extrapolation and sweep figures |
| `test_nn.py` | R² > 0.99, bounds, zero-cap and uncapped sanity checks |

### Results (held-out test set, 3,000 rows)
| Model | R² (kWh) | MAE (kWh) | R² (%) | MAE (% pts) |
|---|---|---|---|---|
| MLP direct | 0.9995 | 0.119 | 0.9992 | 0.393 |
| MLP fraction | 0.9998 | 0.051 | 0.9988 | 0.355 |
| Linear | 0.746 | 2.443 | 0.698 | 9.881 |
| Random Forest | 0.981 | 0.604 | 0.983 | 1.817 |

Out-of-range inputs (speed 120–150 km/h, mass 400–500 t, cap 5–7 MW): MLP direct R² (%) = 0.944, MLP fraction R² (%) = 0.972, Random Forest 0.856, Linear 0.310 (see `models/metrics.json`).

### Interpretation and limitations
- The physics is closed-form and exact, so the NN is a **surrogate**: it reproduces the model approximately, it does not improve on it. Its practical value is as a template for cases where equations are unavailable or too slow.
- Errors are largest where the power cap binds (cap below peak ideal power), because recovery has a kink there (see `figures/residual_vs_cap_ratio.png`).
- Accuracy degrades outside the training ranges; the physics model does not. Predicting recovery % and multiplying by the analytic kinetic energy (the "fraction" variant) extrapolates better, because it hard-codes the part of the physics that is exactly known.
- Results depend on the random seed and sampling; the figures above come from seed 42.
