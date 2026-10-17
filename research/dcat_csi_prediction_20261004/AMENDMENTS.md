# Protocol amendments

## 2026-10-04 / preflight 001

Before any image inference, `run_e2e.py --smoke` exited because the locked
protocol names the validation split `val`, whereas `suite_data.json` uses
`validation` (128 images from 64 scenes). No image was selected and no model
was loaded. The loader now maps this spelling explicitly; the locked CPU run
and its protocol snapshot remain unchanged. The intended validation-only,
one-image-per-scene sample, size, hash ordering, and metrics are unchanged.

## 2026-10-04 / diagnostic 002

After the primary without_CP pilot completed, execute the secondary fixed
with_CP_C128 comparison already described in the preceding plan review. Add
`--checkpoint-family with` without changing predictor coefficients, images,
channel windows, noises, policies, or age semantics. Record this as an
exploratory checkpoint diagnostic, not an additional success criterion and
not a replacement of the primary result. Each checkpoint is compared to its
own hold; cross-checkpoint differences are not attributed to prediction.
