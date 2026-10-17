"""Fit and evaluate the locked CPU-only B0 causal-prediction protocol."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import scipy
from csi_predictors import (load_trace, samples, weights, nmse, objective,
                            fit_empirical, fit_acf, fit_ar1, simple_predictor)

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    out = args.output or HERE / "results" / datetime.now(timezone.utc).strftime("b0_%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "run_b0.py", "csi_predictors.py"):
        shutil.copy2(HERE / name, out / name)
    manifest = {"status": "RUNNING", "pid": __import__("os").getpid(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": digest(HERE / "protocol.json"),
                "source_hashes": {n: digest(HERE / n) for n in ("run_b0.py", "csi_predictors.py")},
                "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
                "channels_completed": []}
    write(out / "manifest.json", manifest)
    summaries, horizon_rows, candidates = [], [], []
    started = time.monotonic()
    print("B0_OUTPUT", out, flush=True)
    try:
        for name in cfg["channel_cases"]:
            if time.monotonic() - started > cfg["cpu_timeout_seconds"]:
                raise TimeoutError("B0 CPU hard timeout")
            path = Path(cfg["baseline_root"]) / "SoM-DCAT-main/Channel_inputs" / (name + ".mat")
            trace = load_trace(path)
            hmax = max(cfg["history_candidates"])
            blocks = cfg["pilot_blocks"]
            ds = {split: samples(trace, block, hmax, cfg["horizons_steps"]) for split, block in blocks.items()}
            atr, xtr, ytr = ds["train"]
            aval, xval, yval = ds["validation"]
            ate, xte, yte = ds["diagnostic_test"]
            train_h = trace["CSI"][blocks["train"][0]*128:blocks["train"][1]*128]
            test_h = trace["CSI"][blocks["diagnostic_test"][0]*128:blocks["diagnostic_test"][1]*128]
            models = {n: simple_predictor(n) for n in ("hold", "zero_nmse_reference")}
            models["ar1_pilot"] = fit_ar1(trace["pilots"][slice(*blocks["train"])])
            selected = {}
            for family in ("wiener_empirical", "wiener_acf"):
                best_score, best_model = float("inf"), None
                for order in cfg["history_candidates"]:
                    for ridge in cfg["ridge_candidates"]:
                        model = (fit_empirical(xtr, ytr, order, ridge) if family == "wiener_empirical"
                                 else fit_acf(train_h, order, ridge))
                        score = objective(model.predict(xval), yval)
                        candidates.append({"channel": name, "family": family, "order": order,
                                           "ridge": ridge, "validation_objective": score})
                        if score < best_score:
                            best_score, best_model = score, model
                models[family] = best_model
                selected[family] = {"order": best_model.order, "ridge": best_model.ridge,
                                    "validation_objective": best_score}
            models["genie_acf_test_statistics"] = fit_acf(test_h, hmax, 1e-6, name="genie_acf_test_statistics")
            models["genie_empirical_test_fit"] = fit_empirical(xte, yte, hmax, 0, name="genie_empirical_test_fit")
            folder = out / name
            folder.mkdir()
            for key, model in models.items():
                model.save(folder / (key + ".npz"))
            rx, tx = weights()
            truth_energy = np.mean(abs(yte)**2, axis=0)
            results = {}
            for key in cfg["predictors"]:
                pred = yte.copy() if key == "ideal" else models[key].predict(xte)
                err = np.mean(abs(pred - yte)**2, axis=0)
                results[key] = {"rx_nmse": nmse(pred, yte, rx), "tx_nmse": nmse(pred, yte, tx),
                                "objective": objective(pred, yte), "mean_abs_prediction": float(np.mean(abs(pred))),
                                "abs_prediction_p01": float(np.quantile(abs(pred), .01))}
                for horizon in range(yte.shape[1]):
                    horizon_rows.append({"channel": name, "method": key, "horizon_steps": horizon,
                                         "horizon_ms": horizon*.03125, "mse": float(err[horizon]),
                                         "signal_power": float(truth_energy[horizon]),
                                         "nmse": float(err[horizon] / truth_energy[horizon]), "anchors": len(ate)})
            result = {"channel": name, "source_sha256": digest(path), "selected": selected,
                      "split_anchors": {k: {"count": len(v[0]), "first": int(v[0][0]), "last": int(v[0][-1])} for k, v in ds.items()},
                      "hold_full_trace_nmse": float(np.sum(abs(trace["CSI"]-trace["NoPred"])**2) / np.sum(abs(trace["CSI"])**2)),
                      "diagnostic_test": results}
            write(folder / "summary.json", result)
            summaries.append(result)
            manifest["channels_completed"].append(name)
            manifest["elapsed_seconds"] = time.monotonic() - started
            write(out / "manifest.json", manifest)
            print(json.dumps({"channel": name, "test_rx_nmse": {m: round(v["rx_nmse"], 6) for m, v in results.items()},
                              "selected": selected, "elapsed_s": round(time.monotonic()-started, 1)}), flush=True)
        write(out / "summary.json", {"protocol_id": cfg["id"], "channels": summaries})
        for filename, rows in (("horizon_metrics.csv", horizon_rows), ("validation_candidates.csv", candidates)):
            with (out / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        manifest.update(status="COMPLETED", elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write(out / "manifest.json", manifest)
        write(HERE / "latest_b0.json", {"run": str(out), "protocol_sha256": manifest["protocol_sha256"]})
    except BaseException:
        manifest.update(status="FAILED", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        write(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
