# B0.1 — validation-only error calibration and stable equalization

## Completed review

All six runs completed (69,120 transmissions, 490.51 seconds). The frozen-model
plug-in variants failed the A7 +0.3 dB gate. The strongest predictor arm for the
primary checkpoint, calibrated ACF, was -0.03754 dB versus original hold and
only +0.00436 dB versus equally calibrated hold. Do not start full training from
this evidence alone. Review the complete Chinese report:

[B0.1 审查报告](/D:/PyChram_project/CognitiveSemCom/research/dcat_csi_prediction_20261008/results/b01_20261008T004209Z/review/REVIEW_zh-CN.md)

`review_b01.py` recomputes the statistics and checks the frozen artifacts on CPU.
Its output is an analysis, not a new independent test-set result.

## Original protocol and execution notes

Date: 2026-10-08. Status: supplementary development gate, not a formal experiment.

The prior B0 showed lower CSI NMSE but degraded image reconstruction. This run
isolates attention reliability calibration and error-aware MMSE equalization.
It uses a complete 3 × 2 × 2 factorial comparison, giving hold the same upgrades
as both Wiener baselines. Both frozen checkpoint families and three additive
noise seeds are evaluated. Noise seeds are not independent propagation traces
or training seeds. The same 32 validation scenes, five released traces, three
SNRs and two transmission phases are reused: 69,120 image transmissions.

Calibration fits only validation-block errors, never diagnostic-test truth.
The validation block was already used for predictor selection; this limitation
and the reuse of B0 development outcomes preclude a formal generalization claim.
The physical channel, decoder CSI, transmit power and age fields are unchanged.
MMSE changes only the equalizer. Its variance and thermal-noise terms are
included for hold as well as the predictors. No API or neural training is used.

`latest_job.json` points to the run. Each job freezes its own source snapshots,
protocols, logs, per-state metrics, combined summaries and status. The local
supervisor runs six jobs sequentially with a 30-minute timeout per job and stops
on the first error. It does not automatically retry or launch full training.

User preference: long jobs run in the background without conversational progress
polling; check readiness and launch once, then review when the user returns.
No reminder or recurring automation is scheduled.

## Interrupted launch and recovery (2026-10-08)

The first job `b01_20261008T003501Z` lost its supervisor before creating an
inference output directory. Its original `RUNNING` status and logs are preserved
in `status_before_recovery.json` and in the recovery job's `prior_status.json`.
The old job is marked `INTERRUPTED`, not completed. The logs contain only an
interpreter path warning; that warning also occurs on successful invocations,
so the root cause is unconfirmed.

Recovery `b01_20261008T004209Z` uses the exact frozen protocol, worker and
prediction/evaluation sources from the interrupted job. Only the supervisor
changed: child PIDs, timestamps, local heartbeats, exit codes and completed
artifact counts are recorded. It runs in a persistent execution session.
No scientific factor, data selection or seed changed. A failed run stops the
sequence; recovery requires an explicit invocation and refuses a live old job.

The latest job's `status.json` is the authoritative execution state. A successful
end writes `REVIEW_READY.md` and `combined_summary.csv`. A stale `RUNNING` value
alone is not proof of a live process or completed experiment.

Windows sandbox note: `Get-Process` in a restricted execution can omit live
host processes. A host-side read-only check confirmed the recovery supervisor
(PID 34588) was alive, and the old supervisor (PID 28200) was absent. Use the
persistent session and local heartbeat, or a host-side process check, for
liveness. Do not declare a job dead from a restricted process query alone.
