"""Causal pilot-history predictors; no dependency on DCAT or GPU packages."""
from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat
from scipy.linalg import toeplitz
from scipy.signal import correlate


@dataclass
class Predictor:
    name: str
    coef: np.ndarray
    bias: np.ndarray
    ridge: float = 0.0

    @property
    def order(self):
        return self.coef.shape[0]

    def predict(self, history):
        return np.asarray(history)[..., :self.order] @ self.coef + self.bias

    def save(self, path):
        np.savez_compressed(path, name=self.name, coef=self.coef, bias=self.bias, ridge=self.ridge)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(str(d["name"]), d["coef"], d["bias"], float(d["ridge"]))


def load_trace(path, period=128):
    data = loadmat(path)
    result = {k: np.asarray(data[k]).reshape(-1).astype(
        np.complex128 if k in ("CSI", "NoPred") else np.float64)
        for k in ("CSI", "NoPred", "AgingTime", "Doppler")}
    for v in result.values():
        if len(v) != 128000 or not np.isfinite(v).all():
            raise ValueError("Unexpected trace size or nonfinite values")
    pilots = np.flatnonzero(result["AgingTime"] == 0)
    if not np.array_equal(pilots, np.arange(0, 128000, period)):
        raise ValueError("Pilot grid differs from locked protocol")
    # Observations, including when fitting a predictor, come from NoPred.
    result["pilots"] = result["NoPred"][::period].copy()
    result["period"] = period
    return result


def samples(trace, block, max_history=32, horizons=191):
    period = trace["period"]
    anchors = np.arange(block[0] + max_history - 1, block[1])
    anchors = anchors[anchors * period + horizons <= block[1] * period]
    history_idx = anchors[:, None] - np.arange(max_history)[None, :]
    target_idx = anchors[:, None] * period + np.arange(horizons)[None, :]
    assert history_idx.min() >= block[0] and target_idx.max() < block[1] * period
    return anchors, trace["pilots"][history_idx], trace["CSI"][target_idx]


def weights(horizons=191):
    rx = np.zeros(horizons)
    rx[:128] = 1 / 128
    tx = np.convolve(np.ones(128), np.ones(64)) / (128 * 64)
    assert len(tx) == horizons
    return rx, tx


def nmse(prediction, truth, weight):
    numerator = np.sum(np.mean(abs(prediction - truth)**2, axis=0) * weight)
    denominator = np.sum(np.mean(abs(truth)**2, axis=0) * weight)
    return float(numerator / denominator)


def objective(prediction, truth):
    return sum(nmse(prediction, truth, w) for w in weights(truth.shape[1])) / 2


def exact_observed_pilot(model):
    model.coef[:, 0] = 0
    model.coef[0, 0] = 1
    model.bias[0] = 0
    return model


def fit_empirical(x, y, order, ridge, name="wiener_empirical"):
    x = x[:, :order]
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mx, y - my
    if ridge == 0:
        coef = np.linalg.lstsq(xc, yc, rcond=None)[0]
    else:
        covariance = xc.conj().T @ xc / len(x)
        cross = xc.conj().T @ yc / len(x)
        scale = max(float(np.trace(covariance).real / order), 1e-12)
        coef = np.linalg.solve(covariance + ridge * scale * np.eye(order), cross)
    return exact_observed_pilot(Predictor(name, coef, my - mx @ coef, ridge))


def fit_acf(raw_h, order, ridge, horizons=191, period=128, name="wiener_acf"):
    mu = raw_h.mean()
    h = raw_h - mu
    # Biased covariance is positive semidefinite. r[l] = E[h(t+l) conj(h(t))].
    acf = correlate(h, h, mode="full", method="fft")[len(h)-1:] / len(h)
    max_lag = (order - 1) * period + horizons - 1
    if max_lag >= len(acf):
        raise ValueError("Insufficient block length")
    lags = np.arange(order) * period
    covariance = toeplitz(acf[lags], acf[lags].conj())
    cross = acf[lags[:, None] + np.arange(horizons)[None, :]]
    coef = np.linalg.solve(covariance + ridge * max(float(acf[0].real), 1e-12) * np.eye(order), cross)
    bias = mu - np.full(order, mu) @ coef
    return exact_observed_pilot(Predictor(name, coef, bias, ridge))


def fit_ar1(pilots, horizons=191):
    mu = pilots.mean()
    previous, future = pilots[:-1] - mu, pilots[1:] - mu
    rho = np.vdot(previous, future) / max(float(np.vdot(previous, previous).real), 1e-12)
    if abs(rho) > 1:
        rho /= abs(rho)
    if abs(rho) < 1e-15:
        gain = np.zeros(horizons, dtype=np.complex128)
        gain[0] = 1
    else:
        gain = np.exp(np.log(complex(rho)) * np.arange(horizons) / 128)
    return exact_observed_pilot(Predictor("ar1_pilot", gain[None, :], mu * (1-gain)))


def simple_predictor(name, horizons=191):
    if name not in ("hold", "zero_nmse_reference"):
        raise ValueError(name)
    coef = np.ones((1, horizons), dtype=np.complex128) if name == "hold" else np.zeros((1, horizons), dtype=np.complex128)
    return Predictor(name, coef, np.zeros(horizons, dtype=np.complex128))


def predict_at(trace, model, targets, available_at):
    """Predict each target using only the latest pilot available by available_at.

    A scalar available_at freezes the TX information set for the entire image;
    an array equal to targets permits causal RX pilot updates.
    """
    targets = np.asarray(targets, dtype=int)
    available = np.broadcast_to(np.asarray(available_at, dtype=int), targets.shape)
    if np.any(available > targets):
        raise ValueError("Future observation requested")
    pilots = available // trace["period"]
    horizons = targets - pilots * trace["period"]
    history = pilots[:, None] - np.arange(model.order)[None, :]
    if history.min() < 0 or horizons.min() < 0 or horizons.max() >= model.coef.shape[1]:
        raise ValueError("Insufficient history or horizon")
    return (trace["pilots"][history] * model.coef[:, horizons].T).sum(axis=1) + model.bias[horizons]


def channel_info(trace, start, size=64, tx_model=None, rx_model=None, oracle=False, zero_age=False):
    t = np.arange(start, start + size)
    p = trace["period"]
    true = trace["CSI"][t]
    tx = np.full(size, trace["pilots"][start // p], dtype=np.complex128)
    rx = trace["NoPred"][t].copy()
    if tx_model is not None:
        tx = predict_at(trace, tx_model, t, start)
    if rx_model is not None:
        rx = predict_at(trace, rx_model, t, t)
    if oracle:
        tx, rx = true.copy(), true.copy()
    if zero_age and not oracle:
        raise ValueError("Age-zeroing is reserved for the labelled oracle")
    age_rx = trace["AgingTime"][t].copy()
    age_tx = (t - (start // p) * p) * .03125
    if zero_age:
        age_rx[:], age_tx[:] = 0, 0
    return {"csi_gt": [true.tolist()], "csi_txrx": [rx.tolist()], "csi_encoder": [tx.tolist()],
            "speed": [trace["Doppler"][t].tolist()], "aging_time": [age_rx.tolist()],
            "aging_time_encoder": [age_tx.tolist()]}
