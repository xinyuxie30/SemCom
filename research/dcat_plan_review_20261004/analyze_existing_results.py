"""CPU-only audit of existing P1 outcomes and released DCAT channel traces.

No model inference, training, API calls, or modification of baseline files.
Run with D:/anconda/envs/pytorch/python.exe.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
P1 = (HERE.parent / "jev_image_transmission_20260926" / "results"
      / "counterfactual_20260926T112638517856Z")
CHANNEL_ROOT = Path("D:/PyChram_project/DCAT-SoM/SoM-DCAT-main/Channel_inputs")
ACTIONS = ["upper_left", "upper_right", "lower_left", "lower_right", "global", "stop"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    return {"mean_mse": statistics.mean(values),
            "mean_psnr_db": statistics.mean(-10 * math.log10(x) for x in values)}


def phase0():
    path = P1 / "action_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    groups = collections.defaultdict(dict)
    for row in rows:
        key = (row["case_id"], row["alpha"])
        assert row["action"] not in groups[key], "Duplicate state/alpha/action"
        assert row["mse"] > 0 and math.isfinite(row["mse"])
        assert abs(row["psnr_db"] + 10 * math.log10(row["mse"])) < 1e-4
        groups[key][row["action"]] = row
    assert len(rows) == 4608 and len(groups) == 768
    assert all(set(group) == set(ACTIONS) for group in groups.values())
    out = {"source": str(path), "sha256": sha256(path), "rows": len(rows),
           "states": len({row["case_id"] for row in rows}),
           "primary_alpha": 0.5, "strata": []}
    for c in (64, 128):
        keys = sorted(k for k in groups
                      if k[1] == 0.5 and groups[k]["stop"]["base_c"] == c)
        assert len(keys) == 192
        data = [[groups[k][a]["mse"] for a in ACTIONS] for k in keys]
        fixed = min(range(6), key=lambda j: statistics.mean(x[j] for x in data))
        fixed_q = min(range(4), key=lambda j: statistics.mean(x[j] for x in data))
        vectors = {
            "best_fixed": [x[fixed] for x in data],
            "best_fixed_quadrant": [x[fixed_q] for x in data],
            "oracle_quadrants": [min(x[:4]) for x in data],
            "oracle_enhancement": [min(x[:5]) for x in data],
            "oracle_all": [min(x) for x in data],
            "oracle_global_or_stop": [min(x[4:]) for x in data],
            "stop": [x[5] for x in data],
            "single": [groups[k]["stop"]["single_mse"] for k in keys],
        }
        scalar_oracle = []
        for key, x in zip(keys, data):
            assert abs(x[5] - groups[(key[0], 1.0)]["stop"]["mse"]) < 1e-10
            candidates = [x[5]]
            for j, action in enumerate(ACTIONS[:5]):
                # Inputs were clipped before convex blending. For alpha in [0,1],
                # MSE is a quadratic; three saved MSEs identify its coefficients.
                d0, d05 = x[5], x[j]
                d1 = groups[(key[0], 1.0)][action]["mse"]
                a = 2 * d1 - 4 * d05 + 2 * d0
                b = d1 - d0 - a
                assert a >= -1e-8, "Nonconvex fit: investigate numerical/input mismatch"
                alpha = min(1, max(0, -b / (2 * a))) if a > 1e-12 else float(d1 < d0)
                value = d0 + b * alpha + a * alpha**2
                assert 0 < value <= min(d0, d05, d1) + 1e-8
                candidates.append(value)
            scalar_oracle.append(min(candidates))
        vectors["oracle_action_and_scalar_alpha"] = scalar_oracle
        scores = {name: summarize(v) for name, v in vectors.items()}
        stages = ["best_fixed_quadrant", "oracle_quadrants", "oracle_enhancement", "oracle_all"]
        counts = collections.Counter(ACTIONS[min(range(6), key=lambda j: x[j])] for x in data)
        roi_count = sum(counts[a] for a in ACTIONS[:4])
        global_close = sum(x[4] <= 1.01 * min(x[:4]) for x in data)
        out["strata"].append({
            "base_c": c, "single_comparator_c": c + 64, "states": len(keys),
            "best_fixed": ACTIONS[fixed], "best_fixed_quadrant": ACTIONS[fixed_q],
            "oracle_counts": {a: counts[a] for a in ACTIONS},
            "roi_oracle_fraction": roi_count / len(keys),
            "global_no_worse_best_quadrant_count": sum(x[4] <= min(x[:4]) for x in data),
            "global_within_1pct_best_quadrant_count": global_close,
            "quadrant_beats_global_or_stop_by_1pct_count": sum(min(x[:4]) < .99 * min(x[4:]) for x in data),
            "phase0_kill": roi_count / len(keys) < .05 or global_close / len(keys) >= .9,
            "scores": scores,
            "oracle_mse_reduction_vs_best_fixed": 1 - scores["oracle_all"]["mean_mse"] / scores["best_fixed"]["mean_mse"],
            "decomposition_note": "Fixed quadrant -> best quadrant -> add global -> add stop. Order-dependent, not a causal attribution. C64 starts at best fixed QUADRANT, not its best fixed action (global).",
            "decomposition": [{"from": a, "to": b,
                               "delta_psnr_db": scores[b]["mean_psnr_db"] - scores[a]["mean_psnr_db"],
                               "absolute_mse_drop": scores[a]["mean_mse"] - scores[b]["mean_mse"]}
                              for a, b in zip(stages, stages[1:])],
        })
    out["fusion_scope"] = "Post-hoc ground-truth oracle over one scalar alpha per state/action in [0,1]; NOT an upper bound on CNN fusion, spatial weights, or a new codec."
    return out


def channel_audit():
    out = []
    for name in ("A1", "A3", "A5", "A7", "A9"):
        path = CHANNEL_ROOT / (name + ".mat")
        mat = loadmat(path)
        h, known, age = (mat[key].reshape(-1) for key in ("CSI", "NoPred", "AgingTime"))
        pilots = np.flatnonzero(age == 0)
        last = np.maximum.accumulate(np.where(age == 0, np.arange(len(age)), 0))
        assert pilots[0] == 0 and len(h) == 128000
        out.append({"trace": name, "source": str(path), "sha256": sha256(path),
                    "samples": len(h), "zero_age_pilots": len(pilots),
                    "pilot_spacing_steps": np.unique(np.diff(pilots)).tolist(),
                    "age_step_ms": float(np.min(np.diff(np.unique(age)))),
                    "hold_equals_pilot_csi_max_abs_diff": float(np.max(np.abs(known - h[last]))),
                    "hold_nmse": float(np.sum(np.abs(known - h)**2) / np.sum(np.abs(h)**2))})
    return out


if __name__ == "__main__":
    result = {"status": "Existing-artifact arithmetic verified; no new model experiment",
              "date": "2026-10-04", "phase0": phase0(), "channel_audit": channel_audit()}
    target = HERE / "audit_results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "states": result["phase0"]["states"],
                      "phase0_kill_by_base_c": {s["base_c"]: s["phase0_kill"] for s in result["phase0"]["strata"]}}))
