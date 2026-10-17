# Experiment report

Mode: **smoke**. Smoke runs are not scientific evidence.

Official API failures retained: 1. Input/output tokens: {'input_tokens': 13808, 'output_tokens': 1738}.

Main table: means over states, with each trained-seed family averaged within state. See analysis.json for scene-cluster paired intervals and stratification.

## fixed_time: external_test/A1/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 19.975 | 19.479 | 0.2680 | 0.500 | 326.75 | 6096.0 |
| fixed_global | 19.995 | 19.661 | 0.3065 | 0.500 | 0.00 | 6464.0 |
| fixed_stop | 19.747 | 19.137 | 0.2461 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.057 | 19.398 | 0.2442 | 0.500 | 0.00 | 6464.0 |
| gradient_importance | 20.263 | 19.800 | 0.2411 | 0.500 | 1.95 | 6464.0 |
| hindsight_oracle | 20.269 | 19.801 | 0.2390 | 0.500 | 0.00 | 6464.0 |
| numeric | 19.995 | 19.367 | 0.2376 | 0.500 | 2.08 | 6464.0 |
| numeric_no_age | 19.995 | 19.367 | 0.2376 | 0.500 | 2.06 | 6464.0 |
| official | 20.000 | 19.449 | 0.2550 | 0.500 | 1793.09 | 6096.0 |
| official_jev_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 1791.02 | 6464.0 |
| official_no_age | 20.000 | 19.449 | 0.2550 | 0.500 | 1793.10 | 6096.0 |
| official_no_stop | 20.034 | 19.452 | 0.2517 | 0.500 | 1793.09 | 6464.0 |
| qwen | 20.013 | 19.402 | 0.2402 | 0.500 | 245.20 | 6464.0 |
| qwen_description_only_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 705.36 | 6464.0 |
| qwen_visual_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 243.13 | 6464.0 |
| random_roi | 20.263 | 19.800 | 0.2411 | 0.500 | 0.00 | 6464.0 |
| severity_rule | 19.990 | 19.418 | 0.2279 | 0.500 | 324.69 | 6464.0 |
| single_C192 | 20.780 | 20.441 | 0.1877 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 19.995 | 19.367 | 0.2376 | 0.500 | 2.06 | 6464.0 |
| visual | 20.013 | 19.402 | 0.2402 | 0.500 | 309.52 | 6464.0 |
| visual_jev_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 307.46 | 6464.0 |
| visual_no_age | 20.013 | 19.402 | 0.2402 | 0.500 | 309.52 | 6464.0 |
| visual_no_stop | 20.013 | 19.402 | 0.2402 | 0.500 | 309.52 | 6464.0 |

## fixed_time: external_test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 20.411 | 20.047 | 0.2504 | 0.167 | 341.10 | 6464.0 |
| fixed_global | 20.255 | 20.082 | 0.2887 | 0.000 | 0.00 | 6464.0 |
| fixed_stop | 20.012 | 19.574 | 0.2417 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.361 | 19.918 | 0.2391 | 0.500 | 0.00 | 6464.0 |
| gradient_importance | 20.499 | 20.129 | 0.2359 | 0.000 | 1.90 | 6464.0 |
| hindsight_oracle | 20.499 | 20.129 | 0.2359 | 0.000 | 0.00 | 6464.0 |
| numeric | 20.312 | 19.955 | 0.2468 | 0.333 | 2.03 | 6464.0 |
| numeric_no_age | 20.336 | 19.931 | 0.2426 | 0.333 | 2.01 | 6464.0 |
| official | 20.319 | 19.898 | 0.2306 | 0.500 | 1824.72 | 6464.0 |
| official_jev_raw | 20.255 | 20.082 | 0.2887 | 0.000 | 1822.66 | 6464.0 |
| official_no_age | 20.319 | 19.898 | 0.2306 | 0.500 | 1824.70 | 6464.0 |
| official_no_stop | 20.319 | 19.898 | 0.2306 | 0.500 | 1824.72 | 6464.0 |
| qwen | 20.326 | 19.861 | 0.2338 | 0.500 | 245.01 | 6464.0 |
| qwen_description_only_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 714.83 | 5360.0 |
| qwen_visual_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 242.97 | 5360.0 |
| random_roi | 20.362 | 19.869 | 0.2221 | 0.500 | 0.00 | 6464.0 |
| severity_rule | 20.277 | 19.861 | 0.2231 | 0.500 | 339.04 | 6464.0 |
| single_C192 | 21.470 | 21.183 | 0.1691 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.304 | 19.869 | 0.2306 | 0.500 | 2.01 | 6464.0 |
| visual | 20.347 | 19.901 | 0.2303 | 0.500 | 305.85 | 6464.0 |
| visual_jev_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 303.82 | 5360.0 |
| visual_no_age | 20.347 | 19.901 | 0.2303 | 0.500 | 305.86 | 6464.0 |
| visual_no_stop | 20.347 | 19.901 | 0.2303 | 0.500 | 305.85 | 6464.0 |

## fixed_time: test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 25.530 | 25.407 | 0.1790 | 0.000 | 329.60 | 6464.0 |
| fixed_global | 25.421 | 24.901 | 0.2009 | 0.000 | 0.00 | 6464.0 |
| fixed_stop | 25.514 | 25.384 | 0.1719 | 0.000 | 0.00 | 4224.0 |
| fixed_upper_right | 25.563 | 25.495 | 0.1675 | 0.000 | 0.00 | 6464.0 |
| gradient_importance | 25.851 | 25.792 | 0.1571 | 0.000 | 1.12 | 6464.0 |
| hindsight_oracle | 25.892 | 25.796 | 0.1619 | 0.000 | 0.00 | 6464.0 |
| numeric | 25.529 | 25.396 | 0.1822 | 0.000 | 1.48 | 6464.0 |
| numeric_no_age | 25.529 | 25.396 | 0.1822 | 0.000 | 1.24 | 6464.0 |
| official | 25.574 | 25.510 | 0.1680 | 0.000 | 1872.82 | 6085.3 |
| official_jev_raw | 25.106 | 24.869 | 0.2069 | 0.000 | 1871.57 | 5360.0 |
| official_no_age | 25.574 | 25.510 | 0.1680 | 0.000 | 1872.82 | 6085.3 |
| official_no_stop | 25.581 | 25.511 | 0.1678 | 0.000 | 1872.82 | 6464.0 |
| qwen | 25.523 | 25.395 | 0.1809 | 0.000 | 245.45 | 6464.0 |
| qwen_description_only_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 702.91 | 6464.0 |
| qwen_visual_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 244.21 | 6464.0 |
| random_roi | 25.574 | 25.496 | 0.1743 | 0.000 | 0.00 | 6464.0 |
| severity_rule | 25.851 | 25.792 | 0.1571 | 0.000 | 328.36 | 6464.0 |
| single_C192 | 26.658 | 26.140 | 0.1180 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 25.673 | 25.653 | 0.1676 | 0.000 | 1.23 | 6464.0 |
| visual | 25.523 | 25.395 | 0.1809 | 0.000 | 294.79 | 6464.0 |
| visual_jev_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 293.55 | 6464.0 |
| visual_no_age | 25.516 | 25.388 | 0.1786 | 0.000 | 294.80 | 6464.0 |
| visual_no_stop | 25.523 | 25.395 | 0.1809 | 0.000 | 294.79 | 6464.0 |

## measured_delay_replay: external_test/A1/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 19.992 | 19.500 | 0.2615 | 0.500 | 326.75 | 6064.0 |
| fixed_global | 20.149 | 19.662 | 0.2757 | 0.500 | 0.00 | 6432.0 |
| fixed_stop | 19.747 | 19.137 | 0.2461 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.091 | 19.419 | 0.2433 | 0.500 | 0.00 | 6432.0 |
| gradient_importance | 20.346 | 19.878 | 0.2385 | 0.500 | 1.95 | 6432.0 |
| numeric | 20.026 | 19.385 | 0.2395 | 0.500 | 2.08 | 6432.0 |
| numeric_no_age | 20.026 | 19.385 | 0.2395 | 0.500 | 2.06 | 6432.0 |
| official | 20.015 | 19.444 | 0.2474 | 0.500 | 1793.09 | 6064.0 |
| official_jev_raw | 20.229 | 19.745 | 0.2913 | 0.500 | 1791.02 | 6432.0 |
| official_no_age | 20.015 | 19.444 | 0.2474 | 0.500 | 1793.10 | 6064.0 |
| official_no_stop | 20.055 | 19.448 | 0.2439 | 0.500 | 1793.09 | 6432.0 |
| qwen | 20.084 | 19.486 | 0.2388 | 0.500 | 245.20 | 6432.0 |
| qwen_description_only_raw | 20.345 | 19.815 | 0.2614 | 0.500 | 705.36 | 6432.0 |
| qwen_visual_raw | 20.399 | 20.099 | 0.2632 | 0.000 | 243.13 | 6432.0 |
| random_roi | 20.356 | 19.901 | 0.2382 | 0.500 | 0.00 | 6432.0 |
| severity_rule | 19.976 | 19.399 | 0.2307 | 0.500 | 324.69 | 6432.0 |
| single_C192 | 20.780 | 20.441 | 0.1877 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.026 | 19.385 | 0.2395 | 0.500 | 2.06 | 6432.0 |
| visual | 20.047 | 19.442 | 0.2445 | 0.500 | 309.52 | 6432.0 |
| visual_jev_raw | 20.127 | 19.756 | 0.2815 | 0.500 | 307.46 | 6432.0 |
| visual_no_age | 20.047 | 19.442 | 0.2445 | 0.500 | 309.52 | 6432.0 |
| visual_no_stop | 20.047 | 19.442 | 0.2445 | 0.500 | 309.52 | 6432.0 |

## measured_delay_replay: external_test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 20.665 | 20.425 | 0.2202 | 0.000 | 341.10 | 6432.0 |
| fixed_global | 20.887 | 20.618 | 0.2234 | 0.000 | 0.00 | 6432.0 |
| fixed_stop | 20.012 | 19.574 | 0.2417 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.479 | 20.034 | 0.2314 | 0.500 | 0.00 | 6432.0 |
| gradient_importance | 20.689 | 20.266 | 0.2255 | 0.000 | 1.90 | 6432.0 |
| numeric | 20.648 | 20.206 | 0.2190 | 0.333 | 2.03 | 6432.0 |
| numeric_no_age | 20.508 | 20.157 | 0.2197 | 0.333 | 2.01 | 6432.0 |
| official | 20.218 | 19.876 | 0.2363 | 0.500 | 1824.72 | 6432.0 |
| official_jev_raw | 19.859 | 19.842 | 0.3042 | 1.000 | 1822.66 | 6432.0 |
| official_no_age | 20.218 | 19.876 | 0.2363 | 0.500 | 1824.70 | 6432.0 |
| official_no_stop | 20.218 | 19.876 | 0.2363 | 0.500 | 1824.72 | 6432.0 |
| qwen | 20.130 | 19.351 | 0.2534 | 0.500 | 245.01 | 6432.0 |
| qwen_description_only_raw | 20.086 | 19.715 | 0.2641 | 0.500 | 714.83 | 5328.0 |
| qwen_visual_raw | 19.593 | 18.780 | 0.2996 | 0.500 | 242.97 | 5328.0 |
| random_roi | 20.511 | 19.960 | 0.2138 | 0.500 | 0.00 | 6432.0 |
| severity_rule | 20.280 | 19.928 | 0.2216 | 0.500 | 339.04 | 6432.0 |
| single_C192 | 21.470 | 21.183 | 0.1691 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.407 | 19.966 | 0.2209 | 0.500 | 2.01 | 6432.0 |
| visual | 20.361 | 20.021 | 0.2264 | 0.167 | 305.85 | 6432.0 |
| visual_jev_raw | 20.711 | 20.519 | 0.2168 | 0.000 | 303.82 | 5328.0 |
| visual_no_age | 20.361 | 20.021 | 0.2264 | 0.167 | 305.86 | 6432.0 |
| visual_no_stop | 20.361 | 20.021 | 0.2264 | 0.167 | 305.85 | 6432.0 |

## measured_delay_replay: test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 25.673 | 25.369 | 0.1695 | 0.000 | 329.60 | 6432.0 |
| fixed_global | 26.440 | 26.420 | 0.1411 | 0.000 | 0.00 | 6432.0 |
| fixed_stop | 25.514 | 25.384 | 0.1719 | 0.000 | 0.00 | 4224.0 |
| fixed_upper_right | 25.829 | 25.636 | 0.1584 | 0.000 | 0.00 | 6432.0 |
| gradient_importance | 26.099 | 26.003 | 0.1481 | 0.000 | 1.12 | 6432.0 |
| numeric | 25.990 | 25.899 | 0.1579 | 0.000 | 1.48 | 6432.0 |
| numeric_no_age | 26.016 | 25.902 | 0.1581 | 0.000 | 1.24 | 6432.0 |
| official | 25.693 | 25.671 | 0.1616 | 0.000 | 1872.82 | 6064.0 |
| official_jev_raw | 25.278 | 25.196 | 0.1776 | 0.000 | 1871.57 | 5328.0 |
| official_no_age | 25.693 | 25.671 | 0.1616 | 0.000 | 1872.82 | 6064.0 |
| official_no_stop | 25.695 | 25.671 | 0.1602 | 0.000 | 1872.82 | 6432.0 |
| qwen | 26.048 | 25.949 | 0.1548 | 0.000 | 245.45 | 6432.0 |
| qwen_description_only_raw | 25.015 | 24.322 | 0.1959 | 0.000 | 702.91 | 6432.0 |
| qwen_visual_raw | 26.522 | 26.502 | 0.1356 | 0.000 | 244.21 | 6432.0 |
| random_roi | 25.768 | 25.630 | 0.1685 | 0.000 | 0.00 | 6432.0 |
| severity_rule | 25.820 | 25.549 | 0.1593 | 0.000 | 328.36 | 6432.0 |
| single_C192 | 26.658 | 26.140 | 0.1180 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 25.902 | 25.763 | 0.1604 | 0.000 | 1.23 | 6432.0 |
| visual | 25.915 | 25.833 | 0.1595 | 0.000 | 294.79 | 6432.0 |
| visual_jev_raw | 25.989 | 25.795 | 0.1575 | 0.000 | 293.55 | 6432.0 |
| visual_no_age | 25.907 | 25.830 | 0.1569 | 0.000 | 294.80 | 6432.0 |
| visual_no_stop | 25.915 | 25.833 | 0.1595 | 0.000 | 294.79 | 6432.0 |

## Interpretation limits

- Fixed-time accuracy is an optimistic component diagnostic, not an end-to-end latency claim.
- Latency replay waits for and charges one fresh periodic pilot; no free intermediate pilot observations.
- Complete base codec wall time is conservatively placed after base reception; replay is not a measured hardware link.
- RF energy assumes unit energy per symbol; compute energy and API infrastructure energy are unmeasured.
- Three replicas share released propagation traces with independent noise; they are not independent propagation realizations.
- UDIS aggregate baseline results were previously observed; DIV2K is the new external test.
- 95% intervals are paired scene-cluster bootstrap intervals, descriptive and not multiplicity-adjusted; primary contrast official vs description.
- Exact/near/regret scores compare intended decisions against the fixed-time hindsight oracle, even in replay.
