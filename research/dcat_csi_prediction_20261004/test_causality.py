import tempfile
from pathlib import Path
import unittest
import numpy as np
from csi_predictors import (Predictor, channel_info, fit_empirical, fit_acf,
                            predict_at, samples, simple_predictor, weights)


def synthetic():
    t = np.arange(128000)
    h = np.exp(1j * .0021 * t) + .3 * np.exp(-1j * .004 * t)
    pilots = h[::128].copy()
    return {"CSI": h, "NoPred": np.repeat(pilots, 128), "pilots": pilots,
            "AgingTime": (t % 128)*.03125, "Doppler": np.ones(len(t)), "period": 128}


class CausalityTests(unittest.TestCase):
    def test_block_separation(self):
        trace = synthetic()
        for block in ([0, 600], [600, 800], [800, 1000]):
            anchors, x, y = samples(trace, block)
            self.assertGreaterEqual(anchors.min()-31, block[0])
            self.assertLess(anchors.max()*128+190, block[1]*128)
            self.assertEqual(x.shape, (len(anchors), 32))
            self.assertEqual(y.shape, (len(anchors), 191))

    def test_hold_and_receiver_update(self):
        trace = synthetic()
        start = 850*128 + 112
        out = channel_info(trace, start)
        np.testing.assert_array_equal(out["csi_encoder"][0], np.full(64, trace["pilots"][850]))
        np.testing.assert_array_equal(out["csi_txrx"][0][:16], np.full(16, trace["pilots"][850]))
        np.testing.assert_array_equal(out["csi_txrx"][0][16:], np.full(48, trace["pilots"][851]))
        np.testing.assert_array_equal(out["aging_time_encoder"][0], np.arange(112,176)*.03125)

    def test_future_pilot_cannot_change_tx(self):
        trace = synthetic()
        model = simple_predictor("hold")
        target = 850*128 + np.arange(112,176)
        before_tx = predict_at(trace, model, target, target[0])
        before_rx = predict_at(trace, model, target, target)
        trace["pilots"][851:] += 100+30j
        np.testing.assert_array_equal(before_tx, predict_at(trace, model, target, target[0]))
        after_rx = predict_at(trace, model, target, target)
        np.testing.assert_array_equal(before_rx[:16], after_rx[:16])
        self.assertGreater(np.max(abs(before_rx[16:]-after_rx[16:])), 100)

    def test_truth_not_a_predictor_input(self):
        trace = synthetic()
        model = simple_predictor("hold")
        start = 850*128+16
        before = channel_info(trace, start, tx_model=model, rx_model=model)
        trace["CSI"][:] += 200j
        after = channel_info(trace, start, tx_model=model, rx_model=model)
        for key in ("csi_encoder", "csi_txrx"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(np.array_equal(before["csi_gt"], after["csi_gt"]))

    def test_complex_regression_orientation_and_no_test_leak(self):
        trace = synthetic()
        _, x, y = samples(trace, [0,600])
        model = fit_empirical(x, y, 4, 1e-6)
        _, xv, yv = samples(trace, [800,1000])
        self.assertLess(np.mean(abs(model.predict(xv)-yv)**2), 1e-6)
        original = model.coef.copy()
        trace["CSI"][800*128:] *= -100
        _, x2, y2 = samples(trace, [0,600])
        np.testing.assert_array_equal(original, fit_empirical(x2,y2,4,1e-6).coef)

    def test_acf_phase_orientation(self):
        t = np.arange(30000)
        raw = np.exp(1j*.0021*t)
        model = fit_acf(raw, 1, 1e-6)
        expected = np.exp(1j*.0021*np.arange(191))
        self.assertLess(np.max(abs(model.coef[0]-expected)), .02)

    def test_roundtrip(self):
        model = simple_predictor("hold")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"p.npz"
            model.save(path)
            loaded = Predictor.load(path)
            np.testing.assert_array_equal(model.coef, loaded.coef)

    def test_future_access_rejected_and_weights(self):
        trace = synthetic()
        with self.assertRaises(ValueError):
            predict_at(trace, simple_predictor("hold"), [1000], [1001])
        for w in weights():
            self.assertAlmostEqual(w.sum(), 1)


if __name__ == "__main__":
    unittest.main()
