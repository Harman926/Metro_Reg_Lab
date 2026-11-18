"""Tests for the NN surrogate. Run `python generate_dataset.py && python train_nn.py` first."""
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from model import simulate
from nn_model import predict, predict_df, model_available, FEATURES


@unittest.skipUnless(model_available(), 'train the model first: python train_nn.py')
class TestNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test = pd.read_csv('data/test_split.csv')
        cls.pred = predict_df(cls.test)

    def test_r2_recovered_kwh(self):
        self.assertGreater(r2_score(self.test.recovered_kwh, self.pred.recovered_kwh), 0.99)

    def test_r2_recovery_pct(self):
        self.assertGreater(r2_score(self.test.recovery_pct, self.pred.recovery_pct), 0.99)

    def test_fraction_variant_r2(self):
        p = predict_df(self.test, 'fraction')
        self.assertGreater(r2_score(self.test.recovered_kwh, p.recovered_kwh), 0.99)

    def test_physical_bounds(self):
        self.assertTrue((self.pred.recovery_pct.between(0, 100)).all())
        self.assertTrue((self.pred.recovered_kwh >= 0).all())

    def test_default_case_close_to_physics(self):
        s, _ = simulate()
        p = predict(200, 60, 0.9, 0.8, 1500)
        self.assertAlmostEqual(p['recovery_pct'], s['recovery_pct'], delta=2.0)
        self.assertAlmostEqual(p['recovered_kwh'], s['recovered_kwh'], delta=0.5)

    def test_zero_acceptance_gives_near_zero(self):
        p = predict(200, 60, 0.9, 0.8, 0)
        self.assertLess(p['recovery_pct'], 3.0)

    def test_uncapped_recovery_near_efficiency(self):
        p = predict(200, 60, 0.9, 0.8, 5000)  # cap far above peak power -> ~efficiency
        self.assertAlmostEqual(p['recovery_pct'], 80, delta=3.0)


if __name__ == '__main__':
    unittest.main()
