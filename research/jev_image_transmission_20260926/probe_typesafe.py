"""Bounded TypeSafe interface probe using synthetic, non-sensitive text only.

This is NOT an image transmission accuracy experiment. No image, checkpoint,
dataset, local document, or ground-truth outcome is sent to the service.
Reference: https://docs.typesafe.ai/api (checked 2026-09-26).
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request


MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
REGIONS = ("upper_left", "upper_right", "lower_left", "lower_right")
CLEAR = "Edges and textures are clear; no visible damage is reported."
DAMAGED = "Edges are doubled and fine textures are visibly destroyed."


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def cases():
    items = []
    for region in REGIONS:
        items.append((region, {r: DAMAGED if r == region else CLEAR for r in REGIONS}, region, False))
    items.append(("global_damage", {r: DAMAGED for r in REGIONS}, "global", False))
    items.append(("all_clear", {r: CLEAR for r in REGIONS}, "stop", False))
    items.append(("upper_left_reordered", items[0][1], "upper_left", True))
    items.append(("upper_left_repeat", items[0][1], "upper_left", False))
    return items


def make_request(observations, reverse=False):
    options = [(r, "Repair the " + r.replace("_", " ") + " region.") for r in REGIONS]
    options += [("global", "Repair the whole image when all four regions show damage."),
                ("stop", "No repair when all four regions are reported clear.")]
    if reverse:
        options.reverse()
    return {
        "model": MODEL,
        "state": {"observations": observations},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "Classify the reported visible damage. Choose the single damaged region "
                    "when only one region is damaged, global when all four are damaged, "
                    "and stop when all four are clear. Use only observations. "
                    "This is a textual interface test, not a prediction of image quality gain."
                ),
                "criteria": dict(options),
            },
            "upper_left_damaged": {
                "type": "noul",
                "instructions": "Does observations.upper_left report visible damage?",
            },
            "upper_left_severity": {
                "type": "score",
                "instructions": "Classify visible damage described in observations.upper_left.",
                "criteria": ["Clear; no damage reported", "Mild damage", "Severe damage; destroyed detail"],
            },
        },
    }


def validate_response(response, request):
    errors = []
    if response.get("model") != MODEL:
        errors.append("returned model differs from pinned version")
    answers = response.get("answers", {})
    for qid, question in request["questions"].items():
        answer = answers.get(qid, {})
        if answer.get("type") != question["type"]:
            errors.append(qid + ": missing or incorrect type")
            continue
        if question["type"] == "noul":
            value = answer.get("noul")
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(qid + ": invalid noul")
            continue
        probabilities = answer.get("probabilities", {})
        values = list(probabilities.values())
        if (not values or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                              or not 0 <= v <= 1 for v in values)):
            errors.append(qid + ": invalid probability values")
        elif abs(sum(values) - 1) > 0.002:
            errors.append(qid + ": probabilities do not sum to one")
        confidence = answer.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(qid + ": invalid confidence")
        if question["type"] == "choice":
            if set(probabilities) != set(question["criteria"]):
                errors.append(qid + ": option keys do not match")
            if answer.get("choice") not in question["criteria"]:
                errors.append(qid + ": unrecognized choice")
        else:
            score = answer.get("score")
            if not isinstance(score, (int, float)) or not 0 <= score <= 2:
                errors.append(qid + ": score outside rubric")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send at most eight synthetic requests.")
    args = parser.parse_args()
    records = []
    for name, observations, expected, reverse in cases():
        records.append({"case": name, "request": make_request(observations, reverse),
                        "expected_choice_synthetic_only": expected})
    if not args.live:
        print(json.dumps({"mode": "offline_preview", "request_count": len(records),
                          "example": records[0]["request"]}, indent=2))
        return
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not configured; no request sent.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parent / "results" / ("interface_probe_" + stamp)
    output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect())
    completed = []
    for record in records:
        payload = json.dumps(record["request"], ensure_ascii=True).encode("utf-8")
        record["request_sha256"] = hashlib.sha256(payload).hexdigest()
        record["sent_at_utc"] = datetime.now(timezone.utc).isoformat()
        request = urllib.request.Request(ENDPOINT, data=payload, method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with opener.open(request, timeout=20) as result:
                record["http_status"] = result.status
                response = json.loads(result.read().decode("utf-8"))
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            record["response"] = response
            record["validation_errors"] = validate_response(response, record["request"])
            record["matches_synthetic_expectation"] = (
                response.get("answers", {}).get("action", {}).get("choice")
                == record["expected_choice_synthetic_only"])
        except Exception as exc:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            record["error_type"] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                record["http_status"] = exc.code
            # Do not print arbitrary provider/proxy error bodies or credentials.
        completed.append(record)
        with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record[k] for k in ("case", "http_status", "elapsed_ms",
              "matches_synthetic_expectation", "validation_errors", "error_type") if k in record}), flush=True)
        if "error_type" in record or record.get("validation_errors"):
            break
    successful = [r for r in completed if "response" in r]
    summary = {
        "scope": "Synthetic API conformance probe; NOT transmission accuracy evidence",
        "endpoint": ENDPOINT, "requested_model": MODEL, "attempted_requests": len(completed),
        "successful_requests": len(successful), "retries": 0,
        "synthetic_expectation_matches": sum(r.get("matches_synthetic_expectation", False) for r in completed),
        "response_validation_failures": sum(bool(r.get("validation_errors")) for r in completed),
        "input_tokens": sum(r["response"].get("usage", {}).get("input_tokens", 0) for r in successful),
        "output_tokens": sum(r["response"].get("usage", {}).get("output_tokens", 0) for r in successful),
        "observed_median_ms_including_first_request": statistics.median(r["elapsed_ms"] for r in successful) if successful else None,
        "observed_min_ms": min((r["elapsed_ms"] for r in successful), default=None),
        "observed_max_ms": max((r["elapsed_ms"] for r in successful), default=None),
        "latency_scope": "One local client, serial HTTP; includes network and service; tiny sample, no latency guarantee",
        "output_directory": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
