"""Small fixed-checkpoint DCAT evaluation of frozen B0 predictors."""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

import numpy as np
from PIL import Image
import torch
from csi_predictors import Predictor, channel_info, load_trace, samples
from run_b0 import digest, write

HERE = Path(__file__).resolve().parent


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def select_images(cfg):
    records = json.loads(Path(cfg["image_manifest"]).read_text(encoding="utf-8"))["images"]
    # The locked protocol uses "val"; the existing manifest spells it "validation".
    # This alias was corrected at preflight, before selecting or inferring images.
    split = {"val": "validation"}.get(cfg["image_split"], cfg["image_split"])
    scenes = {}
    for item in sorted(records, key=lambda x: x["image_id"]):
        if item["split"] == split:
            scenes.setdefault(item["scene_id"], item)
    selected = sorted(scenes.values(), key=lambda x: seed_for(cfg["seed"], x["scene_id"]))[:cfg["images"]]
    if len(selected) != cfg["images"]:
        raise ValueError("Insufficient independent validation scenes")
    return selected


def policy_info(trace, start, policy, empirical, acf):
    tx = empirical if policy in ("wiener_tx", "wiener_both") else None
    rx = empirical if policy in ("wiener_rx", "wiener_both") else None
    if policy == "acf_both":
        tx = rx = acf
    return channel_info(trace, start, tx_model=tx, rx_model=rx,
                        oracle=policy.startswith("ideal_"), zero_age=policy == "ideal_zero_age")


def report(out, rows, manifest):
    groups = {}
    for row in rows:
        key = (row["channel"], row["policy"])
        groups.setdefault(key, []).append(row)
    paired = {(r["channel"], r["snr_db"], r["phase"], r["image_id"]): r for r in rows if r["policy"] == "hold"}
    summary = []
    for (name, policy), data in groups.items():
        deltas = [r["psnr_db"] - paired[(name,r["snr_db"],r["phase"],r["image_id"])]["psnr_db"] for r in data]
        summary.append({"channel": name, "policy": policy, "states": len(data),
                        **{k: float(np.mean([r[k] for r in data])) for k in
                           ("psnr_db", "mse", "lpips_normalized", "tx_nmse", "rx_nmse", "tx_error_estimate", "rx_error_estimate", "prediction_ms", "codec_ms")},
                        "delta_psnr_vs_hold_db": float(np.mean(deltas)),
                        "psnr_p05_db": float(np.quantile([r["psnr_db"] for r in data], .05)),
                        "below_20db_fraction_exploratory": float(np.mean([r["psnr_db"] < 20 for r in data]))})
    write(out / "summary.json", {"scope": manifest["scope"], "rows": summary})
    for r in summary:
        print(json.dumps({k:r[k] for k in ("channel","policy","psnr_db","delta_psnr_vs_hold_db")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-family", choices=["without", "with"], default="without")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.b0 / "protocol.json").read_text(encoding="utf-8"))
    assert json.loads((args.b0 / "manifest.json").read_text())["status"] == "COMPLETED"
    e = cfg["e2e"]
    selected = select_images(e)
    channels, snrs, phases = e["channels"], e["snr_db"], e["start_phases"]
    if args.smoke:
        selected, channels, snrs, phases = selected[:4], ["A1", "A7"], [6], [112]
    out = args.output or args.b0 / datetime.now(timezone.utc).strftime("e2e_" + ("smoke_" if args.smoke else "pilot_") + args.checkpoint_family + "_%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("run_e2e.py", "csi_predictors.py", "run_b0.py"):
        shutil.copy2(HERE / filename, out / filename)
    write(out / "images.json", selected)
    manifest = {"status": "RUNNING", "scope": f"B0 development pilot, fixed {args.checkpoint_family}_CP_C128; no new neural training",
                "checkpoint_family": args.checkpoint_family,
                "secondary_checkpoint_diagnostic": args.checkpoint_family == "with",
                "preflight_amendment": "Protocol val maps to manifest validation; corrected before any image inference. The first preflight failed with no output run and no GPU inference.",
                "pid": __import__("os").getpid(), "created_utc": datetime.now(timezone.utc).isoformat(),
                "b0": str(args.b0), "protocol_sha256": digest(args.b0 / "protocol.json"),
                "smoke": args.smoke, "images": len(selected), "channels": channels, "snrs": snrs, "phases": phases,
                "policies": e["policies"], "completed_batches": 0,
                "expected_rows": len(selected)*len(channels)*len(snrs)*len(phases)*len(e["policies"]),
                "code_hashes": {n:digest(HERE/n) for n in ("run_e2e.py","csi_predictors.py","run_b0.py")}}
    write(out / "manifest.json", manifest)
    began = time.monotonic()
    print("E2E_OUTPUT", out, flush=True)
    try:
        root = Path(cfg["baseline_root"])
        source = root / "SoM-DCAT-main"
        old_manifest = json.loads((root / "reproduction/preflight_report.json").read_text())
        for file, expected in old_manifest["source_hashes"].items():
            assert digest(source / file) == expected, f"Baseline source changed: {file}"
        manifest["baseline_source_hashes"] = old_manifest["source_hashes"]
        manifest["baseline_helper_hashes"] = {f:digest(root/"reproduction"/f) for f in ("common.py","evaluate_batched.py")}
        checkpoint = f"DCAT_{args.checkpoint_family}_CP_C128.model"
        manifest["checkpoint"] = checkpoint
        manifest["checkpoint_sha256"] = digest(source/"Checkpoints"/checkpoint)
        manifest["predictor_hashes"] = {name:{kind:digest(args.b0/name/(kind+".npz")) for kind in ("wiener_empirical","wiener_acf")} for name in channels}
        manifest["environment"] = {"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}
        sys.path.insert(0, str(root / "reproduction"))
        from common import build_model, channel_slice, load_channel
        from evaluate_batched import adapt
        from utils import seed_torch
        seed_torch(e["seed"])
        torch.set_num_threads(4)
        images = []
        for item in selected:
            assert digest(item["path"]) == item["sha256"]
            with Image.open(item["path"]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)/255
            images.append(torch.from_numpy(pixels.copy()).permute(2,0,1))
        all_x = torch.stack(images)
        noise_state, error_estimates = {}, {}
        with (out / "model.log").open("w", encoding="utf-8") as log, torch.inference_mode():
            with contextlib.redirect_stdout(log):
                model = adapt(build_model(128, args.checkpoint_family == "with"))
            def noise(_self, inputs, std, name=None):
                n = noise_state["value"].to(inputs.device)
                assert n.shape == inputs.shape
                return inputs + n*std
            model.ours_channel.gaussian_noise_layer = types.MethodType(noise, model.ours_channel)
            model.pred_nmse_tx.register_forward_hook(lambda module, args, output: error_estimates.update(tx=output.detach().cpu().numpy()))
            model.pred_nmse_rx.register_forward_hook(lambda module, args, output: error_estimates.update(rx=output.detach().cpu().numpy()))
            torch.cuda.reset_peak_memory_stats()
            rows = []
            fields = ["channel","snr_db","phase","image_id","scene_id","start","policy","mse","psnr_db","lpips_normalized",
                      "tx_nmse","rx_nmse","tx_error_estimate","rx_error_estimate","prediction_ms","codec_ms","reconstruction_finite"]
            with (out / "per_state.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                first_call = True
                for name in channels:
                    trace = load_trace(source / "Channel_inputs" / (name+".mat"))
                    original = load_channel(name)
                    empirical = Predictor.load(args.b0/name/"wiener_empirical.npz")
                    acf = Predictor.load(args.b0/name/"wiener_acf.npz")
                    anchors = samples(trace, cfg["pilot_blocks"]["diagnostic_test"])[0]
                    # Use the same initial 32-image schedule even in the four-image smoke.
                    indices = np.linspace(0, len(anchors)-1, e["images"]).astype(int)[:len(selected)]
                    anchors = anchors[indices]
                    for snr in snrs:
                        for phase in phases:
                            starts = anchors*128+phase
                            for offset in range(0,len(selected),e["batch_size"]):
                                end = min(offset+e["batch_size"],len(selected))
                                subset = selected[offset:end]
                                batch_starts = starts[offset:end]
                                x = all_x[offset:end].cuda()
                                bank = []
                                for item,start in zip(subset,batch_starts):
                                    rng = np.random.default_rng(seed_for(e["seed"],name,item["image_id"],int(start)))
                                    bank.append((rng.standard_normal((64,64),dtype=np.float32)+1j*rng.standard_normal((64,64),dtype=np.float32)).astype(np.complex64))
                                noise_state["value"] = torch.from_numpy(np.stack(bank))
                                for policy in e["policies"]:
                                    if time.monotonic()-began > e["timeout_seconds"]:
                                        raise TimeoutError("B0 E2E hard timeout")
                                    infos, timings = [], []
                                    for start in batch_starts:
                                        tick = time.perf_counter()
                                        info = policy_info(trace,int(start),policy,empirical,acf)
                                        timings.append((time.perf_counter()-tick)*1000)
                                        infos.append(info)
                                        if policy == "hold":
                                            expected = channel_slice(original,int(start),64,False)
                                            for key in expected:
                                                np.testing.assert_array_equal(info[key],expected[key])
                                    info = {key:[one[key][0] for one in infos] for key in infos[0]}
                                    if first_call:
                                        with contextlib.redirect_stdout(log):
                                            warm,_,_ = model(x,info,snr)
                                        torch.cuda.synchronize()
                                    tick = time.perf_counter()
                                    with contextlib.redirect_stdout(log):
                                        y,_,_ = model(x,info,snr)
                                    torch.cuda.synchronize()
                                    codec_ms = (time.perf_counter()-tick)*1000/len(subset)
                                    if first_call:
                                        maximum = float((y-warm).abs().max())
                                        assert maximum <= 1e-6, maximum
                                        manifest["repeated_call_max_abs_difference"] = maximum
                                        first_call = False
                                    assert bool(torch.isfinite(y).all()), f"Nonfinite reconstruction: {name} {policy}"
                                    y = y.clamp(0,1)
                                    mse = (x-y).square().mean(dim=(1,2,3)).cpu().numpy()
                                    psnr = -10*np.log10(mse)
                                    lpips = model.lpips_loss(x,y,normalize=True).reshape(-1).cpu().numpy()
                                    assert np.isfinite([mse,psnr,lpips]).all()
                                    for i,(item,start,one) in enumerate(zip(subset,batch_starts,infos)):
                                        true = np.array(one["csi_gt"][0])
                                        power = np.sum(abs(true)**2)
                                        row = {"channel":name,"snr_db":snr,"phase":phase,"image_id":item["image_id"],"scene_id":item["scene_id"],
                                               "start":int(start),"policy":policy,"mse":float(mse[i]),"psnr_db":float(psnr[i]),"lpips_normalized":float(lpips[i]),
                                               "tx_nmse":float(np.sum(abs(np.array(one["csi_encoder"][0])-true)**2)/power),
                                               "rx_nmse":float(np.sum(abs(np.array(one["csi_txrx"][0])-true)**2)/power),
                                               "tx_error_estimate":float(error_estimates["tx"][i]),"rx_error_estimate":float(error_estimates["rx"][i]),
                                               "prediction_ms":timings[i],"codec_ms":codec_ms,"reconstruction_finite":True}
                                        writer.writerow(row)
                                        rows.append(row)
                                stream.flush()
                                manifest["completed_batches"] += 1
                                manifest["completed_rows"] = len(rows)
                                manifest["elapsed_seconds"] = time.monotonic()-began
                                write(out/"manifest.json",manifest)
                            print(json.dumps({"channel":name,"snr":snr,"phase":phase,"rows":len(rows),"total":manifest["expected_rows"],"elapsed_s":round(time.monotonic()-began,1)}),flush=True)
            assert len(rows) == manifest["expected_rows"]
            report(out,rows,manifest)
            manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            manifest["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        manifest.update(status="COMPLETED",elapsed_seconds=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/"manifest.json",manifest)
        pointer = ("latest_e2e_smoke" if args.smoke else "latest_e2e") + ("_with" if args.checkpoint_family == "with" else "") + ".json"
        write(HERE/pointer,{"run":str(out)})
    except BaseException:
        manifest.update(status="FAILED",error=traceback.format_exc(),elapsed_seconds=time.monotonic()-began)
        write(out/"manifest.json",manifest)
        raise


if __name__ == "__main__":
    main()
