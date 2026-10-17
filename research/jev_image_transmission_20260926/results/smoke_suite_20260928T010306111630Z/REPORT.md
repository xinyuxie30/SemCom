# Experiment report

Mode: **smoke**. Smoke runs are not scientific evidence.

Official API failures retained: 2. Input/output tokens: {'input_tokens': 12556, 'output_tokens': 1580}.

Main table: means over states, with each trained-seed family averaged within state. See analysis.json for scene-cluster paired intervals and stratification.

## fixed_time: external_test/A1/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 19.975 | 19.479 | 0.2680 | 0.500 | 328.19 | 6096.0 |
| fixed_global | 19.995 | 19.661 | 0.3065 | 0.500 | 0.00 | 6464.0 |
| fixed_stop | 19.747 | 19.137 | 0.2461 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.057 | 19.398 | 0.2442 | 0.500 | 0.00 | 6464.0 |
| gradient_importance | 20.263 | 19.800 | 0.2411 | 0.500 | 2.05 | 6464.0 |
| hindsight_oracle | 20.269 | 19.801 | 0.2390 | 0.500 | 0.00 | 6464.0 |
| numeric | 19.995 | 19.367 | 0.2376 | 0.500 | 2.20 | 6464.0 |
| numeric_no_age | 19.995 | 19.367 | 0.2376 | 0.500 | 2.17 | 6464.0 |
| official | 20.009 | 19.483 | 0.2647 | 0.500 | 2280.24 | 6464.0 |
| official_jev_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 2278.06 | 6464.0 |
| official_no_age | 20.009 | 19.483 | 0.2647 | 0.500 | 2280.26 | 6464.0 |
| official_no_stop | 20.009 | 19.483 | 0.2647 | 0.500 | 2280.24 | 6464.0 |
| qwen | 20.013 | 19.402 | 0.2402 | 0.500 | 261.13 | 6464.0 |
| qwen_description_only_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 708.83 | 6464.0 |
| qwen_visual_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 258.96 | 6464.0 |
| random_roi | 20.263 | 19.800 | 0.2411 | 0.500 | 0.00 | 6464.0 |
| severity_rule | 19.990 | 19.418 | 0.2279 | 0.500 | 326.01 | 6464.0 |
| single_C192 | 20.780 | 20.441 | 0.1877 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 19.995 | 19.367 | 0.2376 | 0.500 | 2.16 | 6464.0 |
| visual | 20.013 | 19.402 | 0.2402 | 0.500 | 299.62 | 6464.0 |
| visual_jev_raw | 19.995 | 19.661 | 0.3065 | 0.500 | 297.45 | 6464.0 |
| visual_no_age | 20.013 | 19.402 | 0.2402 | 0.500 | 299.62 | 6464.0 |
| visual_no_stop | 20.013 | 19.402 | 0.2402 | 0.500 | 299.62 | 6464.0 |

## fixed_time: external_test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 20.411 | 20.047 | 0.2504 | 0.167 | 342.44 | 6464.0 |
| fixed_global | 20.255 | 20.082 | 0.2887 | 0.000 | 0.00 | 6464.0 |
| fixed_stop | 20.012 | 19.574 | 0.2417 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.361 | 19.918 | 0.2391 | 0.500 | 0.00 | 6464.0 |
| gradient_importance | 20.499 | 20.129 | 0.2359 | 0.000 | 1.67 | 6464.0 |
| hindsight_oracle | 20.499 | 20.129 | 0.2359 | 0.000 | 0.00 | 6464.0 |
| numeric | 20.312 | 19.955 | 0.2468 | 0.333 | 1.83 | 6464.0 |
| numeric_no_age | 20.336 | 19.931 | 0.2426 | 0.333 | 1.81 | 6464.0 |
| official | 20.374 | 19.978 | 0.2494 | 0.333 | 1826.30 | 6464.0 |
| official_jev_raw | 20.255 | 20.082 | 0.2887 | 0.000 | 1824.51 | 6464.0 |
| official_no_age | 20.374 | 19.978 | 0.2494 | 0.333 | 1826.30 | 6464.0 |
| official_no_stop | 20.374 | 19.978 | 0.2494 | 0.333 | 1826.30 | 6464.0 |
| qwen | 20.326 | 19.861 | 0.2338 | 0.500 | 260.30 | 6464.0 |
| qwen_description_only_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 721.01 | 5360.0 |
| qwen_visual_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 258.52 | 5360.0 |
| random_roi | 20.362 | 19.869 | 0.2221 | 0.500 | 0.00 | 6464.0 |
| severity_rule | 20.277 | 19.861 | 0.2231 | 0.500 | 340.66 | 6464.0 |
| single_C192 | 21.470 | 21.183 | 0.1691 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.304 | 19.869 | 0.2306 | 0.500 | 1.80 | 6464.0 |
| visual | 20.347 | 19.901 | 0.2303 | 0.500 | 294.69 | 6464.0 |
| visual_jev_raw | 20.280 | 20.085 | 0.2728 | 0.000 | 292.89 | 5360.0 |
| visual_no_age | 20.347 | 19.901 | 0.2303 | 0.500 | 294.68 | 6464.0 |
| visual_no_stop | 20.347 | 19.901 | 0.2303 | 0.500 | 294.69 | 6464.0 |

## fixed_time: test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 25.530 | 25.305 | 0.1790 | 0.000 | 336.18 | 6464.0 |
| fixed_global | 25.421 | 24.901 | 0.2009 | 0.000 | 0.00 | 6464.0 |
| fixed_stop | 25.514 | 25.384 | 0.1719 | 0.000 | 0.00 | 4224.0 |
| fixed_upper_right | 25.563 | 25.495 | 0.1675 | 0.000 | 0.00 | 6464.0 |
| gradient_importance | 25.851 | 25.792 | 0.1571 | 0.000 | 2.16 | 6464.0 |
| hindsight_oracle | 25.892 | 25.796 | 0.1619 | 0.000 | 0.00 | 6464.0 |
| numeric | 25.529 | 25.317 | 0.1822 | 0.000 | 2.46 | 6464.0 |
| numeric_no_age | 25.529 | 25.317 | 0.1822 | 0.000 | 2.27 | 6464.0 |
| official | 25.544 | 25.307 | 0.1778 | 0.000 | 3723.59 | 6464.0 |
| official_jev_raw | 25.106 | 24.869 | 0.2069 | 0.000 | 3721.31 | 5360.0 |
| official_no_age | 25.544 | 25.307 | 0.1778 | 0.000 | 3723.59 | 6464.0 |
| official_no_stop | 25.544 | 25.307 | 0.1778 | 0.000 | 3723.59 | 6464.0 |
| qwen | 25.523 | 25.305 | 0.1809 | 0.000 | 249.76 | 6464.0 |
| qwen_description_only_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 720.04 | 6464.0 |
| qwen_visual_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 247.49 | 6464.0 |
| random_roi | 25.574 | 25.496 | 0.1743 | 0.000 | 0.00 | 6464.0 |
| severity_rule | 25.851 | 25.792 | 0.1571 | 0.000 | 333.90 | 6464.0 |
| single_C192 | 26.658 | 26.140 | 0.1180 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 25.673 | 25.614 | 0.1676 | 0.000 | 2.27 | 6464.0 |
| visual | 25.523 | 25.305 | 0.1809 | 0.000 | 304.53 | 6464.0 |
| visual_jev_raw | 25.421 | 24.901 | 0.2009 | 0.000 | 302.25 | 6464.0 |
| visual_no_age | 25.516 | 25.297 | 0.1786 | 0.000 | 304.53 | 6464.0 |
| visual_no_stop | 25.523 | 25.305 | 0.1809 | 0.000 | 304.53 | 6464.0 |

## measured_delay_replay: external_test/A1/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 19.990 | 19.453 | 0.2592 | 0.500 | 328.19 | 6064.0 |
| fixed_global | 20.147 | 19.588 | 0.2820 | 0.500 | 0.00 | 6432.0 |
| fixed_stop | 19.747 | 19.137 | 0.2461 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.102 | 19.447 | 0.2457 | 0.500 | 0.00 | 6432.0 |
| gradient_importance | 20.361 | 19.880 | 0.2351 | 0.500 | 2.05 | 6432.0 |
| numeric | 20.021 | 19.384 | 0.2378 | 0.500 | 2.20 | 6432.0 |
| numeric_no_age | 20.021 | 19.384 | 0.2378 | 0.500 | 2.17 | 6432.0 |
| official | 19.918 | 19.461 | 0.2564 | 0.500 | 2280.24 | 5360.0 |
| official_jev_raw | 20.058 | 19.728 | 0.2804 | 0.500 | 2278.06 | 5360.0 |
| official_no_age | 19.918 | 19.461 | 0.2564 | 0.500 | 2280.26 | 5360.0 |
| official_no_stop | 19.918 | 19.461 | 0.2564 | 0.500 | 2280.24 | 5360.0 |
| qwen | 20.072 | 19.479 | 0.2406 | 0.500 | 261.13 | 6432.0 |
| qwen_description_only_raw | 20.453 | 19.936 | 0.2726 | 0.500 | 708.83 | 6432.0 |
| qwen_visual_raw | 20.297 | 19.977 | 0.2717 | 0.500 | 258.96 | 6432.0 |
| random_roi | 20.361 | 19.880 | 0.2351 | 0.500 | 0.00 | 6432.0 |
| severity_rule | 19.984 | 19.414 | 0.2325 | 0.500 | 326.01 | 6432.0 |
| single_C192 | 20.780 | 20.441 | 0.1877 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.021 | 19.384 | 0.2378 | 0.500 | 2.16 | 6432.0 |
| visual | 19.999 | 19.372 | 0.2402 | 0.500 | 299.62 | 6432.0 |
| visual_jev_raw | 20.166 | 19.730 | 0.2665 | 0.500 | 297.45 | 6432.0 |
| visual_no_age | 19.999 | 19.372 | 0.2402 | 0.500 | 299.62 | 6432.0 |
| visual_no_stop | 19.999 | 19.372 | 0.2402 | 0.500 | 299.62 | 6432.0 |

## measured_delay_replay: external_test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 20.517 | 20.272 | 0.2295 | 0.167 | 342.44 | 6432.0 |
| fixed_global | 21.034 | 20.633 | 0.2202 | 0.000 | 0.00 | 6432.0 |
| fixed_stop | 20.012 | 19.574 | 0.2417 | 0.500 | 0.00 | 4224.0 |
| fixed_upper_right | 20.510 | 20.037 | 0.2319 | 0.500 | 0.00 | 6432.0 |
| gradient_importance | 20.687 | 20.266 | 0.2254 | 0.000 | 1.67 | 6432.0 |
| numeric | 20.656 | 20.207 | 0.2199 | 0.333 | 1.83 | 6432.0 |
| numeric_no_age | 20.507 | 20.157 | 0.2204 | 0.333 | 1.81 | 6432.0 |
| official | 20.474 | 20.133 | 0.2338 | 0.333 | 1826.30 | 6432.0 |
| official_jev_raw | 20.290 | 20.229 | 0.2702 | 0.000 | 1824.51 | 6432.0 |
| official_no_age | 20.474 | 20.133 | 0.2338 | 0.333 | 1826.30 | 6432.0 |
| official_no_stop | 20.474 | 20.133 | 0.2338 | 0.333 | 1826.30 | 6432.0 |
| qwen | 20.388 | 19.896 | 0.2260 | 0.500 | 260.30 | 6432.0 |
| qwen_description_only_raw | 19.438 | 18.484 | 0.2889 | 0.500 | 721.01 | 5328.0 |
| qwen_visual_raw | 20.596 | 20.507 | 0.2245 | 0.000 | 258.52 | 5328.0 |
| random_roi | 20.526 | 19.961 | 0.2145 | 0.500 | 0.00 | 6432.0 |
| severity_rule | 20.309 | 19.931 | 0.2166 | 0.500 | 340.66 | 6432.0 |
| single_C192 | 21.470 | 21.183 | 0.1691 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 20.406 | 19.966 | 0.2217 | 0.500 | 1.80 | 6432.0 |
| visual | 20.019 | 19.973 | 0.2459 | 0.500 | 294.69 | 6432.0 |
| visual_jev_raw | 20.711 | 20.519 | 0.2168 | 0.000 | 292.89 | 5328.0 |
| visual_no_age | 20.019 | 19.973 | 0.2459 | 0.500 | 294.68 | 6432.0 |
| visual_no_stop | 20.019 | 19.973 | 0.2459 | 0.500 | 294.69 | 6432.0 |

## measured_delay_replay: test/A9/all/all

| Method | PSNR dB | PSNR p05 | LPIPS | Outage | Decision ms | Symbols |
|---|---:|---:|---:|---:|---:|---:|
| description | 25.188 | 24.404 | 0.1879 | 0.000 | 336.18 | 6432.0 |
| fixed_global | 26.440 | 26.420 | 0.1411 | 0.000 | 0.00 | 6432.0 |
| fixed_stop | 25.514 | 25.384 | 0.1719 | 0.000 | 0.00 | 4224.0 |
| fixed_upper_right | 25.829 | 25.636 | 0.1584 | 0.000 | 0.00 | 6432.0 |
| gradient_importance | 26.064 | 26.015 | 0.1503 | 0.000 | 2.16 | 6432.0 |
| numeric | 25.995 | 25.884 | 0.1573 | 0.000 | 2.46 | 6432.0 |
| numeric_no_age | 25.995 | 25.884 | 0.1573 | 0.000 | 2.27 | 6432.0 |
| official | 25.654 | 25.398 | 0.1644 | 0.000 | 3723.59 | 5360.0 |
| official_jev_raw | 25.195 | 25.037 | 0.1902 | 0.000 | 3721.31 | 5328.0 |
| official_no_age | 25.654 | 25.398 | 0.1644 | 0.000 | 3723.59 | 5360.0 |
| official_no_stop | 25.654 | 25.398 | 0.1644 | 0.000 | 3723.59 | 5360.0 |
| qwen | 25.842 | 25.642 | 0.1620 | 0.000 | 249.76 | 6432.0 |
| qwen_description_only_raw | 24.403 | 23.382 | 0.2070 | 0.000 | 720.04 | 6432.0 |
| qwen_visual_raw | 25.879 | 25.280 | 0.1641 | 0.000 | 247.49 | 6432.0 |
| random_roi | 25.768 | 25.630 | 0.1685 | 0.000 | 0.00 | 6432.0 |
| severity_rule | 25.880 | 25.600 | 0.1531 | 0.000 | 333.90 | 6432.0 |
| single_C192 | 26.658 | 26.140 | 0.1180 | 0.000 | 0.00 | 6304.0 |
| uncertainty_only | 25.895 | 25.785 | 0.1608 | 0.000 | 2.27 | 6432.0 |
| visual | 25.815 | 25.434 | 0.1628 | 0.000 | 304.53 | 6432.0 |
| visual_jev_raw | 25.787 | 25.275 | 0.1687 | 0.000 | 302.25 | 6432.0 |
| visual_no_age | 25.814 | 25.408 | 0.1584 | 0.000 | 304.53 | 6432.0 |
| visual_no_stop | 25.815 | 25.434 | 0.1628 | 0.000 | 304.53 | 6432.0 |

## Interpretation limits

- Fixed-time accuracy is an optimistic component diagnostic, not an end-to-end latency claim.
- Latency replay waits for and charges one fresh periodic pilot; no free intermediate pilot observations.
- Complete base codec wall time is conservatively placed after base reception; replay is not a measured hardware link.
- RF energy assumes unit energy per symbol; compute energy and API infrastructure energy are unmeasured.
- Three replicas share released propagation traces with independent noise; they are not independent propagation realizations.
- UDIS aggregate baseline results were previously observed; DIV2K is the new external test.
- 95% intervals are paired scene-cluster bootstrap intervals, descriptive and not multiplicity-adjusted; primary contrast official vs description.
- Exact/near/regret scores compare intended decisions against the fixed-time hindsight oracle, even in replay.
