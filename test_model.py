import unittest
from model import simulate
class TestModel(unittest.TestCase):
    def test_default_energy_balance(self):
        s,_=simulate()
        self.assertAlmostEqual(s['initial_kwh'],s['recovered_kwh']+s['conversion_loss_kwh']+s['unaccepted_kwh'],places=8)
    def test_stopping(self):
        s,d=simulate()
        self.assertAlmostEqual(s['stop_time_s'],(60/3.6)/0.9)
        self.assertAlmostEqual(d.speed_kmh.iloc[-1],0)
        self.assertAlmostEqual(d.distance_m.iloc[-1],s['stop_distance_m'])
    def test_no_acceptance(self):
        s,_=simulate(max_power_kw=0)
        self.assertAlmostEqual(s['recovered_kwh'],0)
    def test_full_ideal(self):
        s,_=simulate(efficiency=1,max_power_kw=100000)
        self.assertAlmostEqual(s['recovered_kwh'],s['initial_kwh'],places=8)
    def test_power_cap(self):
        s,d=simulate(max_power_kw=100)
        self.assertLessEqual(d.recovered_power_kw.max(),100.00001)
    def test_invalid(self):
        with self.assertRaises(ValueError):simulate(decel=0)
if __name__=='__main__':unittest.main()
