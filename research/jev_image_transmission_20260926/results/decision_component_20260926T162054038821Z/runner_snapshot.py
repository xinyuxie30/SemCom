"""32-state decision-component pilot. Reads only receiver states, never labels."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/Visual-Jev/code"))
from vdm.models.vdm_model import VDM
from probe_typesafe import ENDPOINT, MODEL, NoRedirect

REGIONS = ["upper_left", "upper_right", "lower_left", "lower_right"]
ACTIONS = REGIONS + ["global", "stop"]
ARTIFACTS = {
    "low": "Edges and textures mostly clear; little visible damage.",
    "noticeable": "Noticeable blur, noise, or broken edges.",
    "severe": "Strong blur, noise, or destroyed texture/edges.",
    "uncertain": "Cannot reliably distinguish artifacts from the scene content.",
}
CRITERIA = {r: "Enhance only the " + r.replace("_", " ") + " quadrant." for r in REGIONS}
CRITERIA.update(global_="unused")
del CRITERIA["global_"]
CRITERIA["global"] = "Send an additional low-rate full image and blend it with the first reconstruction."
CRITERIA["stop"] = "Keep the current reconstruction without further transmission."
QUESTION = ("Choose the next transmission action most likely to give the lowest reconstruction error "
            "over the ENTIRE image after the fixed blending procedure. The original image and future "
            "channel are unavailable. Use present evidence; an image that looks plausible may still "
            "contain errors. Consider all six candidates. Do not prioritize an object merely because it is salient.")
PROTOCOL = ("A static 128x128 image was received using a fixed C128 DCAT encoder. Each enhancement uses "
            "the same C64 payload budget. A local enhancement independently codes one original quadrant "
            "after resizing it to 128x128, then shrinks its reconstruction to that quadrant. A global "
            "enhancement codes the whole original image with C64. The affected pixels are an equal "
            "blend of the first and second reconstructions. The second reconstruction can be noisy; "
            "an enhancement may worsen quality. All five transmission actions are currently feasible. "
            "This diagnostic asks about quality alone, with no penalty for unused budget and a fixed execution time.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def question_record(instruction, criteria):
    return {"qtype": "choice", "instruction": instruction,
            "candidates": [key + ": " + value for key, value in criteria.items()]}


def visual_predict(model, image, questions, context):
    torch.cuda.synchronize()
    started = time.perf_counter()
    group = model.prepare_group(image, questions, shared_context=json.dumps(context))
    call = model.run_independent if len(questions) == 1 else model.run_prefix_share_batch
    output = call(group)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    probabilities = [torch.softmax(output["lm_option_logits"][i, :n].float(), -1).cpu().tolist()
                     for i, n in enumerate(group.n_options)]
    return probabilities, elapsed


def prediction(options, probabilities, elapsed_ms):
    values = dict(zip(options, probabilities))
    assert len(values) == len(options) and all(np.isfinite(v) for v in probabilities)
    assert abs(sum(probabilities)-1) < 0.002
    return {"action": max(values, key=values.get), "probabilities": values, "elapsed_ms": elapsed_ms}


def official_request(state, criteria):
    questions = {"action": {"type": "choice", "instructions": QUESTION, "criteria": criteria}}
    for action in ACTIONS[:-1]:
        questions["benefit_" + action] = {
            "type": "noul",
            "instructions": ("Given only the receiver evidence and protocol, is action '" + action +
                             "' likely to reduce whole-image mean squared reconstruction error compared "
                             "with keeping the current reconstruction? Future true channel and original "
                             "pixels are unavailable. This is a prediction, not a calculation."),
        }
    return {"model": MODEL, "state": state, "questions": questions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args()
    source = args.states.resolve()
    states = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    states = [s for s in states if s["base_c"] == 128]
    assert len(states) == 32 and len({s["image_id"] for s in states}) == 8
    assert all(17 <= int(s["image_id"][5:7]) <= 24 for s in states)
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is absent; no live request sent.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "results" / ("decision_component_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"status": "RUNNING", "scope": "32 held-aside DEVELOPMENT states; not a full experiment",
                "states_path": str(source), "states_sha256": digest(source),
                "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(ROOT / "COMPONENT_PILOT_PROTOCOL.md"),
                "assets": json.loads((ROOT / "visual_assets_manifest.json").read_text()),
                "versions": {p: version(p) for p in ("torch", "transformers", "peft", "huggingface_hub", "numpy", "pillow")},
                "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                "question": QUESTION, "protocol_context": PROTOCOL, "artifact_criteria": ARTIFACTS,
                "future_outcome_files_read": [], "completed_states": 0,
                "api_upload_content": "Public Kodak-derived receiver descriptions and known protocol; no original pixels or labels"}
    def save_manifest():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_manifest()
    print("DECISION_OUTPUT " + str(out), flush=True)
    started = time.monotonic()
    try:
        torch.manual_seed(20260926)
        model = VDM(str(ROOT / "models/qwen_base"), dtype=torch.bfloat16, device="cuda", with_heads=False)
        model.processor.image_processor.max_pixels = 200704
        model.backbone = PeftModel.from_pretrained(model.backbone, str(ROOT / "models/visual_adapter")).eval()
        opener = urllib.request.build_opener(NoRedirect())
        gray_image = Image.new("RGB", (448, 448), (127, 127, 127))
        description_questions = [question_record(
            "Consider only the " + region.replace("_", " ") + " quadrant of the RECEIVED image. "
            "Which description best matches visible blur/noise/blocking affecting edge and texture fidelity? "
            "Do not assume that smooth scene content is necessarily damaged. The original is unavailable.", ARTIFACTS)
            for region in REGIONS]
        api_disabled = False
        with torch.inference_mode(), (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for state in states:
                if time.monotonic() - started > args.timeout_seconds:
                    raise TimeoutError("Decision component hard timeout")
                stage = time.perf_counter()
                with np.load(source.parent / state["array"]) as archive:
                    array = archive["received"][0].transpose(1, 2, 0)
                received = Image.fromarray(np.clip(np.rint(array*255), 0, 255).astype(np.uint8)).resize((448, 448), Image.Resampling.BICUBIC)
                preparation_ms = (time.perf_counter()-stage)*1000
                order = ACTIONS.copy()
                random.Random(int(hashlib.sha256(state["case_id"].encode()).hexdigest()[:8], 16)).shuffle(order)
                criteria = {action: CRITERIA[action] for action in order}
                context = {"protocol": PROTOCOL, "previous_link": {
                    "snr_regime": "low (0 dB)" if state["snr_db"] == 0 else "moderate (6 dB)",
                    "csi_age_at_last_received_symbol_ms": round(state["csi_age_at_last_base_symbol_ms"], 4),
                    "known_channel_magnitude": round(state["last_known_csi_magnitude"], 4)}}
                action_question = question_record(QUESTION, criteria)
                with model.backbone.disable_adapter():
                    raw_probs, raw_ms = visual_predict(model, received, [action_question], context)
                    desc_probs, desc_ms = visual_predict(model, received, description_questions, context)
                descriptions = {region: {"grade": list(ARTIFACTS)[int(np.argmax(probs))],
                                          "distribution": dict(zip(ARTIFACTS, [round(v, 6) for v in probs]))}
                                for region, probs in zip(REGIONS, desc_probs)}
                text_context = {**context, "receiver_region_descriptions": descriptions,
                                "description_origin": "Frozen local visual front end on received pixels; uncertain observations, not ground truth"}
                with model.backbone.disable_adapter():
                    text_probs, text_ms = visual_predict(model, gray_image, [action_question], text_context)
                adapted_probs, adapted_ms = visual_predict(model, received, [action_question], context)
                result = {"case_id": state["case_id"], "image_id": state["image_id"], "option_order": order,
                          "input_preparation_ms": preparation_ms, "description_ms": desc_ms, "descriptions": descriptions,
                          "qwen_visual": prediction(order, raw_probs[0], raw_ms),
                          "qwen_description_only": prediction(order, text_probs[0], text_ms),
                          "visual_jev": prediction(order, adapted_probs[0], adapted_ms)}
                severity = {r: sum(p*w for p, w in zip(probs, (0, 0.5, 1, 0.5))) for r, probs in zip(REGIONS, desc_probs)}
                result["severity_rule"] = {"action": max(severity, key=severity.get), "scores": severity}
                request = official_request(text_context, criteria)
                result["official_request"] = request
                began = time.perf_counter()
                try:
                    if api_disabled:
                        raise RuntimeError("API disabled after authorization failure")
                    payload = json.dumps(request).encode("utf-8")
                    http_request = urllib.request.Request(ENDPOINT, data=payload, method="POST", headers={
                        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    with opener.open(http_request, timeout=20) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    elapsed = (time.perf_counter()-began)*1000
                    assert body["model"] == MODEL
                    answer = body["answers"]["action"]
                    assert answer["type"] == "choice" and set(answer["probabilities"]) == set(ACTIONS)
                    official = prediction(order, [answer["probabilities"][a] for a in order], elapsed)
                    assert official["action"] == answer["choice"]
                    official.update(confidence=answer["confidence"], response=body, request_sha256=hashlib.sha256(payload).hexdigest())
                    result["official_jev"] = official
                except Exception as error:
                    code = error.code if isinstance(error, urllib.error.HTTPError) else None
                    if code in (401, 403):
                        api_disabled = True
                    result["official_jev"] = {"action": "upper_right", "fallback": True, "error_type": type(error).__name__,
                                              "http_status": code, "elapsed_ms": (time.perf_counter()-began)*1000}
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                manifest["completed_states"] += 1
                save_manifest()
                print(json.dumps({"completed_states": manifest["completed_states"], "elapsed_seconds": round(time.monotonic()-started, 2),
                                  "api_fallback": result["official_jev"].get("fallback", False)}), flush=True)
        manifest.update(status="COMPLETED", elapsed_seconds=time.monotonic()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save_manifest()
    except BaseException as error:
        manifest.update(status="FAILED", error_type=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        save_manifest()
        raise


if __name__ == "__main__":
    main()
