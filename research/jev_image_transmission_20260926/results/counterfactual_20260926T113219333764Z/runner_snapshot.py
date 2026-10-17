"""Frozen DCAT two-round counterfactual diagnostic, not a deployable policy.

External baseline: D:/PyChram_project/DCAT-SoM, official PKU-PCNI/SoM-DCAT.
All new files are written inside this research directory. No baseline is edited.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BASELINE = Path("D:/PyChram_project/DCAT-SoM")
sys.path.insert(0, str(BASELINE / "reproduction"))
from common import SOURCE, build_model, checkpoint_name, channel_slice, digest, load_channel
from evaluate_batched import adapt

REGIONS = {"upper_left": (0, 64, 0, 64), "upper_right": (0, 64, 64, 128),
           "lower_left": (64, 128, 0, 64), "lower_right": (64, 128, 64, 128)}
ACTIONS = list(REGIONS) + ["global", "stop"]


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def metric_mse(reference, estimates):
    return (reference - estimates).square().mean(dim=(1, 2, 3))


def metrics(reference, estimates, lpips):
    mse = metric_mse(reference, estimates)
    perceptual = lpips(reference.expand_as(estimates), estimates, normalize=True).reshape(-1)
    return [{"mse": float(m), "psnr_db": float(-10 * torch.log10(m)),
             "lpips_normalized": float(p)} for m, p in zip(mse, perceptual)]


def blind_region_features(image):
    """Only received pixels. No reference image, distortion, or future outcome."""
    array = image.detach().cpu().numpy()[0]
    result = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        tile = array[:, y0:y1, x0:x1]
        gray = tile.mean(axis=0)
        dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
        lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        histogram = np.histogram(gray, bins=16, range=(0, 1))[0].astype(float)
        histogram /= histogram.sum()
        result[name] = {"mean": float(gray.mean()), "std": float(gray.std()),
                        "gradient": float((np.abs(dx).mean() + np.abs(dy).mean()) / 2),
                        "laplacian_variance": float(lap.var()),
                        "entropy_bits": float(-(histogram[histogram > 0] * np.log2(histogram[histogram > 0])).sum()),
                        "clipped_fraction": float(((tile <= 0) | (tile >= 1)).mean())}
    return result


def summarize(rows):
    summaries = []
    for base_c in sorted({r["base_c"] for r in rows}):
        for alpha in (0.5, 1.0):
            selected = [r for r in rows if r["base_c"] == base_c and r["alpha"] == alpha]
            keys = sorted({r["case_id"] for r in selected})
            if not keys:
                continue
            grouped = {key: {r["action"]: r for r in selected if r["case_id"] == key} for key in keys}
            array = np.array([[grouped[key][a]["mse"] for a in ACTIONS] for key in keys])
            psnr = -10 * np.log10(array)
            oracle = np.argmin(array, axis=1)
            fixed = np.argmin(array.mean(axis=0))
            # Second-best oracle is never silently substituted for the prespecified objective.
            first = [grouped[key]["stop"] for key in keys]
            single = np.array([r["single_mse"] for r in first])
            geometer = np.array([r["perfect_roi_mse"] for r in first])
            best = array[np.arange(len(keys)), oracle]
            best_psnr = psnr[np.arange(len(keys)), oracle]
            summaries.append({
                "base_c": base_c, "enhancement_c": 64, "alpha": alpha, "states": len(keys),
                "images": len({r["image_id"] for r in first}),
                "best_fixed_action_on_this_development_set": ACTIONS[fixed],
                "mean_psnr_db": {a: float(psnr[:, j].mean()) for j, a in enumerate(ACTIONS)},
                "mean_mse": {a: float(array[:, j].mean()) for j, a in enumerate(ACTIONS)},
                "oracle_mean_psnr_db": float(best_psnr.mean()),
                "oracle_mse_reduction_vs_best_fixed_fraction": float(1 - best.mean() / array[:, fixed].mean()),
                "oracle_gain_vs_best_fixed_mean_db": float((best_psnr - psnr[:, fixed]).mean()),
                "oracle_gain_vs_expected_uniform_random_roi_mean_db": float((best_psnr - psnr[:, :4].mean(axis=1)).mean()),
                "single_round_mean_psnr_db": float((-10 * np.log10(single)).mean()),
                "oracle_minus_single_mean_db": float((best_psnr + 10 * np.log10(single)).mean()),
                "oracle_mse_reduction_vs_single_fraction": float(1 - best.mean() / single.mean()),
                "oracle_beats_single_fraction": float((best < single).mean()),
                "perfect_roi_mean_psnr_db": float((-10 * np.log10(geometer)).mean()),
                "oracle_action_counts": {a: int((oracle == j).sum()) for j, a in enumerate(ACTIONS)},
                "at_least_5pct_oracle_headroom": bool(1 - best.mean() / array[:, fixed].mean() >= 0.05),
                "oracle_matches_single_mean_mse": bool(best.mean() <= single.mean()),
                "resource_scope": "Equal forward payload only; control, pilots, feedback, service waits not debited",
            })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based development/reserved image index")
    parser.add_argument("--snrs", default="0,6")
    parser.add_argument("--ages", default="young,old")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    assert 1 <= args.images <= 16 and 0 <= args.start_index <= 23 and args.start_index + args.images <= 24 and 1 <= args.segments <= 3
    snrs, age_names = list(map(float, args.snrs.split(","))), args.ages.split(",")
    assert set(snrs) <= {0, 6} and set(age_names) <= {"young", "old"}
    protocol = json.loads((ROOT / "PILOT_PROTOCOL.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "kodak_manifest.json").read_text(encoding="utf-8"))["images"][args.start_index:args.start_index+args.images]
    preflight = json.loads((BASELINE / "reproduction/preflight_report.json").read_text())
    assert all(digest(SOURCE / path) == value for path, value in preflight["source_hashes"].items())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / ("counterfactual_" + stamp)
    output.mkdir(parents=True)
    (output / "receiver_arrays").mkdir()
    manifest = {"status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": vars(args), "protocol": protocol, "protocol_sha256": digest(ROOT / "PILOT_PROTOCOL.json"),
                "script_sha256": digest(Path(__file__)), "dataset_sha256": digest(ROOT / "kodak_manifest.json"),
                "baseline_source_hashes": preflight["source_hashes"],
                "external_helper_hashes": {p: digest(BASELINE / "reproduction" / p) for p in ("common.py", "evaluate_batched.py")},
                "checkpoint_hashes": {str(c): digest(SOURCE / "Checkpoints" / checkpoint_name(c)) for c in (64, 128, 192)},
                "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name(0)},
                "scope": "Development-only counterfactual diagnosis; no trained selector, no live API",
                "completed_states": 0}
    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_manifest()
    print("PILOT_OUTPUT " + str(output), flush=True)
    started = time.monotonic()
    rows = []
    try:
        from utils import seed_torch
        seed_torch(protocol["seed"])
        models = {}
        noise_context = {}
        with (output / "model.log").open("w", encoding="utf-8") as model_log:
            with contextlib.redirect_stdout(model_log):
                for c in (64, 128, 192):
                    model = adapt(build_model(c))
                    def noise(self, inputs, std, name=None):
                        additive = noise_context["noise"].to(inputs.device)
                        assert tuple(additive.shape) == tuple(inputs.shape[1:])
                        return inputs + additive.unsqueeze(0) * std
                    model.ours_channel.gaussian_noise_layer = types.MethodType(noise, model.ours_channel)
                    models[c] = model
            channel = load_channel("A9")
            manifest["channel_sha256"] = digest(SOURCE / "Channel_inputs/A9.mat")
            row_path = output / "action_outcomes.jsonl"
            state_path = output / "receiver_states.jsonl"
            with torch.inference_mode(), row_path.open("w", encoding="utf-8") as row_file, state_path.open("w", encoding="utf-8") as state_file:
                for item in dataset:
                    path = ROOT / "datasets/kodak/128" / item["image_id"]
                    assert digest(path) == item["prepared_sha256"]
                    with Image.open(path) as image:
                        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255).permute(2, 0, 1).unsqueeze(0).cuda()
                    crops = [F.interpolate(x[:, :, y0:y1, x0:x1], size=(128, 128), mode="bilinear", align_corners=False)
                             for y0, y1, x0, x1 in REGIONS.values()]
                    enhancement_inputs = torch.cat(crops + [x])
                    for age_name in age_names:
                        low, high = protocol["age_at_enhancement_start_ms"][age_name]
                        age = channel["AgingTime"][0]
                        for segment in range(args.segments):
                            rng = np.random.default_rng(stable_seed(protocol["seed"], item["image_id"], age_name, segment))
                            left, right = 256 + segment * 42000, 256 + (segment + 1) * 42000
                            candidates = np.flatnonzero((age[left:right] >= low) & (age[left:right] <= high)) + left
                            assert len(candidates)
                            anchor = int(rng.choice(candidates))
                            origin = anchor - 128
                            bank = (rng.standard_normal((256, 64), dtype=np.float32) + 1j * rng.standard_normal((256, 64), dtype=np.float32)).astype(np.complex64)
                            for snr in snrs:
                                def infer(c, inputs, start):
                                    noise_context["noise"] = torch.from_numpy(bank[start-origin:start-origin+c//2])
                                    one = channel_slice(channel, start, c // 2, False)
                                    info = {key: value * inputs.shape[0] for key, value in one.items()}
                                    with contextlib.redirect_stdout(model_log):
                                        result, _, _ = models[c](inputs, info, snr)
                                    assert bool(torch.isfinite(result).all())
                                    return result.clamp(0, 1), one
                                enhancement, _ = infer(64, enhancement_inputs, anchor)
                                for base_c in (64, 128):
                                    if time.monotonic() - started > args.timeout_seconds:
                                        raise TimeoutError("Pilot hard timeout reached")
                                    base_start = anchor - base_c // 2 - protocol["feedback_wait_steps"]
                                    base, base_info = infer(base_c, x, base_start)
                                    if manifest["completed_states"] == 0:
                                        repeat, _ = infer(base_c, x, base_start)
                                        difference = float((repeat - base).abs().max())
                                        assert difference <= 1e-6
                                        manifest["repeat_max_abs_difference"] = difference
                                    single, _ = infer(base_c + 64, x, base_start)
                                    single_metrics = metrics(x, single, models[64].lpips_loss)[0]
                                    case_id = f"{Path(item['image_id']).stem}_A9_snr{snr:g}_{age_name}_s{segment}_C{base_c}"
                                    perfects = []
                                    for y0, y1, x0, x1 in REGIONS.values():
                                        perfect = base.clone()
                                        perfect[:, :, y0:y1, x0:x1] = x[:, :, y0:y1, x0:x1]
                                        perfects.append(perfect)
                                    perfect_mse = float(metric_mse(x, torch.cat(perfects)).min())
                                    np.savez_compressed(output / "receiver_arrays" / (case_id + ".npz"), received=base.cpu().numpy())
                                    state = {"case_id": case_id, "image_id": item["image_id"], "base_c": base_c, "snr_db": snr,
                                             "csi_age_at_last_base_symbol_ms": base_info["aging_time"][0][-1],
                                             "last_known_csi_magnitude": abs(base_info["csi_txrx"][0][-1]),
                                             "region_features": blind_region_features(base),
                                             "array": "receiver_arrays/" + case_id + ".npz"}
                                    state_file.write(json.dumps(state) + "\n")
                                    for alpha in (0.5, 1.0):
                                        fused = []
                                        for index, (y0, y1, x0, x1) in enumerate(REGIONS.values()):
                                            patch = F.interpolate(enhancement[index:index+1], size=(64, 64), mode="bilinear", align_corners=False)
                                            final = base.clone()
                                            final[:, :, y0:y1, x0:x1] = (1-alpha)*base[:, :, y0:y1, x0:x1] + alpha*patch
                                            fused.append(final)
                                        fused += [(1-alpha)*base + alpha*enhancement[4:5], base]
                                        all_metrics = metrics(x, torch.cat(fused), models[64].lpips_loss)
                                        for action, metric in zip(ACTIONS, all_metrics):
                                            row = {"case_id": case_id, "image_id": item["image_id"], "base_c": base_c, "snr_db": snr,
                                                   "age_group": age_name, "segment": segment, "base_start": base_start,
                                                   "enhancement_start": anchor, "enhancement_start_age_ms": float(age[anchor]),
                                                   "alpha": alpha, "action": action, **metric,
                                                   "single_mse": single_metrics["mse"], "single_lpips_normalized": single_metrics["lpips_normalized"],
                                                   "perfect_roi_mse": perfect_mse,
                                                   "payload_complex_symbols": 32*(base_c + (0 if action == "stop" else 64))}
                                            row_file.write(json.dumps(row) + "\n")
                                            rows.append(row)
                                    manifest["completed_states"] += 1
                                row_file.flush()
                                state_file.flush()
                    save_manifest()
                    print(json.dumps({"image": item["image_id"], "states": manifest["completed_states"],
                                      "elapsed_seconds": round(time.monotonic()-started, 2)}), flush=True)
        summary = summarize(rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (output / "action_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest.update(status="COMPLETED", elapsed_seconds=time.monotonic()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        ended_utc=datetime.now(timezone.utc).isoformat())
        save_manifest()
        print("PILOT_SUMMARY " + json.dumps(summary), flush=True)
    except BaseException as error:
        manifest.update(status="FAILED", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        save_manifest()
        raise


if __name__ == "__main__":
    main()
