"""Energy conversion and failure cases must not fabricate efficiency."""
import unittest
import tempfile
from pathlib import Path
from fpga_ml.power_reporting import CpuEnergyMeter, power_summary


class EnergyTests(unittest.TestCase):
    def test_fpga_live_estimate_uses_elapsed_window(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            reports = Path(directory)
            (reports / 'power_uart_activity.rpt').write_text('| Total On-Chip Power (W) | 0.606 |')
            (reports / 'performance_summary.json').write_text('{"uart_simulation": [{"total_ns": 1000000}]}')
            text = '\n'.join(power_summary('fpga', 20, 600, reports=reports))
            self.assertIn('363.600000 J', text)
            self.assertIn('18.180000 J', text)
            self.assertIn('0.055005501 decisions/J', text)
            empty = '\n'.join(power_summary('fpga', 0, 600, reports=reports))
            self.assertIn('Estimated live energy efficiency: unavailable', empty)

    def setUp(self):
        self.first = dict(energy_pwh=100, meter_time_ms=1000, monotonic_ns=1000000000)
        self.last = dict(energy_pwh=1000000100, meter_time_ms=2000, monotonic_ns=2000000000)

    def test_energy_units_and_efficiency(self):
        m = CpuEnergyMeter.calculate(self.first, self.last, 10, 'unit-test workload')
        self.assertAlmostEqual(m['energy_j'], 3.6)
        self.assertAlmostEqual(m['average_power_w'], 3.6)
        self.assertAlmostEqual(m['energy_per_decision_j'], .36)
        self.assertAlmostEqual(m['efficiency_decisions_per_j'], 10 / 3.6)
        self.assertIn('CPU package average power', '\n'.join(power_summary('cpu', 10, 1, measurement=m)))

    def test_no_predictions_does_not_claim_efficiency(self):
        m = CpuEnergyMeter.calculate(self.first, self.last, 0, 'waiting only')
        self.assertIsNone(m['energy_per_decision_j'])
        self.assertIsNone(m['efficiency_decisions_per_j'])

    def test_stalled_reset_and_short_counters_rejected(self):
        for last in [self.first, {**self.last, 'energy_pwh': 0},
                     {**self.last, 'meter_time_ms': 1001, 'monotonic_ns': 1001000000}]:
            with self.assertRaises(ValueError):
                CpuEnergyMeter.calculate(self.first, last, 10, 'invalid')

    def test_clock_disagreement_rejected(self):
        with self.assertRaises(ValueError):
            CpuEnergyMeter.calculate(self.first, {**self.last, 'monotonic_ns': 9000000000}, 10, 'invalid')

    def test_missing_sensor_stays_unavailable(self):
        m = CpuEnergyMeter()
        m.reason = 'permission denied'
        result = m.stop(10, 'test')
        text = '\n'.join(power_summary('cpu', 10, 1, measurement=result))
        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('permission denied', text)
        self.assertNotIn('CPU power efficiency:', text)


if __name__ == '__main__':
    unittest.main()
