# Proposal experiment results

Initially recorded on 2026-11-16 (Asia/Ho_Chi_Minh); updated on 2026-11-25.

## ADE20K overlap-calibration paired full validation

On the full ADE20K validation set, overlap-disagreement calibration produced nearly unchanged aggregate scores relative to standard sliding-window merging. Both conditions evaluated the same 2,000 image IDs (seed 42), with guided spatial bias ($\tau=0.33$, temperature $0.10$) and PAMR disabled. Both summaries were valid and reported no PAMR failures.

| Metric | Standard merge | Overlap calibration | Calibrated minus standard |
| --- | ---: | ---: | ---: |
| mIoU (%) | 17.23 | 17.23 | 0.00 |
| Overall pixel accuracy (%) | 44.94 | 44.94 | 0.00 |
| Mean class accuracy (%) | 39.02 | 39.05 | +0.03 |
| Boundary accuracy (%) | 25.4515 | 25.4599 | +0.0085 |
| Interior accuracy (%) | 47.1732 | 47.1698 | -0.0034 |

Calibration changed the aggregate metrics only slightly in this full validation. The sequential run times (3,655.22 seconds standard; 297.30 seconds calibrated) are not a controlled speed comparison and should not be interpreted as an algorithmic speedup; run order, initialization, and cache state can affect wall time. Run directory on Drive: `/content/drive/MyDrive/NACLIP-results/window-calibration/ade20k_full_paired_seed42_20261125T034551682272Z`.

## PC59 overlap-calibration paired full validation

On all 5,105 PC59 validation images, overlap-disagreement calibration produced only small aggregate metric changes relative to standard sliding-window merging. Both conditions used the same image IDs (seed 42), guided spatial bias ($\tau=0.33$, temperature $0.10$), and PAMR disabled. Both summaries were valid and reported no PAMR failures.

| Metric | Standard merge | Overlap calibration | Calibrated minus standard |
| --- | ---: | ---: | ---: |
| mIoU (%) | 34.86 | 34.86 | 0.00 |
| Overall pixel accuracy (%) | 59.06 | 59.07 | +0.01 |
| Mean class accuracy (%) | 57.59 | 57.62 | +0.03 |
| Boundary accuracy (%) | 36.2619 | 36.2765 | +0.0147 |
| Interior accuracy (%) | 61.4405 | 61.4482 | +0.0077 |

These small differences do not establish a reliable improvement. The sequential run times (8,683.64 seconds standard; 615.60 seconds calibrated) are not a controlled speed comparison and should not be interpreted as an algorithmic speedup; run order, initialization, and cache state can affect wall time. Run directory on Drive: `/content/drive/MyDrive/NACLIP-results/window-calibration/pc59_full_paired_seed42_20261125T045238012722Z`.

## PC59 entropy-adaptive paired full validation

Fixed Gaussian and entropy-adaptive Gaussian were evaluated on all 5,105 PC59 validation images with identical image IDs (seed 42) and PAMR disabled. Both summaries were valid. Entropy-adaptive bias improved mIoU by 0.10 points and boundary/interior accuracy by 0.0431/0.0643 points, while mean class accuracy decreased by 0.04 points.

| Metric | Fixed Gaussian | Entropy-adaptive Gaussian | Entropy minus fixed |
| --- | ---: | ---: | ---: |
| mIoU (%) | 35.19 | 35.29 | +0.10 |
| Overall pixel accuracy (%) | 59.34 | 59.41 | +0.07 |
| Mean class accuracy (%) | 57.74 | 57.70 | -0.04 |
| Boundary accuracy (%) | 36.3860 | 36.4291 | +0.0431 |
| Interior accuracy (%) | 61.7395 | 61.8038 | +0.0643 |

Evaluation wall time was 537.92 seconds for fixed Gaussian and 551.90 seconds for entropy-adaptive Gaussian. These sequential timings are descriptive, not a controlled speed comparison. Run directory on Drive: `/content/drive/MyDrive/NACLIP-results/entropy-bias/entropy_full_pc59_seed42_20261125T072807995354Z/pc59`.

## PC59 matched full-validation comparison

**Both fixed and guided completed successfully on all 5,105 images, with PAMR disabled. At tau=0.33 and temperature=0.10, guided did not improve the fixed baseline in this evaluation.**

| Metric | Fixed Gaussian, PAMR off | Guided, PAMR off | Guided minus fixed (percentage points) |
| --- | ---: | ---: | ---: |
| mIoU (%) | 35.19 | 34.86 | -0.33 |
| Overall pixel accuracy (%) | 59.34 | 59.06 | -0.28 |
| Mean class accuracy (%) | 57.74 | 57.59 | -0.15 |
| Boundary accuracy (%) | 36.3862 | 36.2615 | -0.1248 |
| Interior accuracy (%) | 61.7399 | 61.4408 | -0.2990 |

This is the comparison that isolates the configured spatial-bias change: guided versus fixed, with the same source files, package list, image IDs and other evaluation configuration. These aggregate results do not establish statistical significance or performance on other datasets. The current proposal setting lowers both boundary and interior accuracy on PC59.

## Shared protocol and verification

- PC59 validation: 5,105 unique image IDs; seed 42.
- Architecture: reduced; attention: NACLIP key-key; CLIP ViT-B/16.
- Gaussian sigma 5; slide crop 224; slide stride 112; logit scale 40; probability threshold 0.
- PAMR: 0 iterations in both runs.
- Guided-only parameters: tau=0.33; temperature=0.10; epsilon=1e-6.
- Boundary: both sides of valid horizontal/vertical class transitions, Euclidean distance <=3 pixels on the original annotation grid, ignore=255; pooled correct/total pixel accuracy. Interior uses the valid complement.
- Eligible pixels in both runs: boundary 79,132,214; interior 758,315,069.
- Fixed correct pixels: boundary 28,793,241; interior 468,182,694.
- Guided correct pixels: boundary 28,694,507; interior 465,915,111.
- Both summaries: `baseline_valid=true`, `proposal_protocol=true`, no PAMR failures.
- Verified downloaded manifests, source SHA-256 values against local code, identical image ID lists and package files, final progress, pixel ratios, and resolved configurations. The only configuration differences are the output directory, spatial-bias mode, and the inactive/active guidance threshold and temperature.

Tau and temperature were selected before the full validations from unlabeled feature distributions, not segmentation accuracy. Calibration artifacts are under `NACLIP-results/guidance-distribution-20261115T162954Z`.

## Timing

| Measurement | Fixed, PAMR off | Guided, PAMR off |
| --- | ---: | ---: |
| Evaluation wall time (seconds) | 532.14545 | 3396.49886 |
| Initialization (seconds) | 8.79392 | 8.98884 |
| Steady processing mean (seconds/image) | 0.09470 | 0.09770 |
| Steady processing median (seconds/image) | 0.09205 | 0.09524 |
| Peak PyTorch allocation (GiB) | 0.46847 | 0.46847 |

Fixed evaluation took **8 min 52.15 sec**; guided took **56 min 36.50 sec**. Timed processing totaled 483.73 seconds (fixed) versus 498.94 seconds (guided). The remaining 48.42 versus 2897.56 seconds were outside the processing hook.

Guided's mean timed processing was **3.16% higher**. This includes preprocessing after fetch, inference, metric processing and hook ordering overhead; it is not an isolated GPU inference benchmark. The much larger wall-time difference was dominated by loading/other work outside that hook. Final MMSeg `data_time` was 0.5675 seconds/image in guided versus 0.0095 in fixed. Both read the same Drive dataset; a warm cache is a likely explanation, not directly verified. Do not describe guided as 6x slower computationally based on wall time.

For future speed comparisons, cache the same files under `/content` and use matched warm-up and loading conditions.

## Persistent run artifacts

### ADE20K overlap-calibration paired full validation

- Drive directory: `/content/drive/MyDrive/NACLIP-results/window-calibration/ade20k_full_paired_seed42_20261125T034551682272Z`.
- Saved: `paired_comparison.json`; the `standard/` and `calibrated/` subdirectories contain each run's summary and image-ID list.

### PC59 overlap-calibration paired full validation

- Drive directory: `/content/drive/MyDrive/NACLIP-results/window-calibration/pc59_full_paired_seed42_20261125T045238012722Z`.
- Saved: `paired_comparison.json`; the `standard/` and `calibrated/` subdirectories contain each run's summary and image-ID list.

### PC59 entropy-adaptive paired full validation

- Drive directory: `/content/drive/MyDrive/NACLIP-results/entropy-bias/entropy_full_pc59_seed42_20261125T072807995354Z/pc59`.
- Saved: `paired_comparison.json`; the `fixed/` and `entropy/` subdirectories contain each run's summary and image-ID list.

### Fixed

- Drive directory: `/content/drive/MyDrive/NACLIP-results/fixed/pc59_fixed_full_pamr_off_20261115T175629828787Z`.
- Local copy (ignored by Git): `work_logs/pc59_fixed_full_pamr_off_20261115T175629828787Z/`.
- Saved: `summary.json`, `manifest.json`, `progress.json`, `timing.json`, `packages.txt`, `image_ids.json`, `config.py`, and a per-run Markdown report.

### Guided

- Drive directory: `/content/drive/MyDrive/NACLIP-results/guided/pc59_guided_full_tau0p33_temp0p10_20261115T165417632828Z`.
- Local copy (ignored by Git): `work_logs/pc59_guided_full_tau0p33_temp0p10_20261115T165417632828Z/`.
- Saved: `summary.json`, `manifest.json`, `progress.json`, `timing.json`, `packages.txt`, `image_ids.json`, `config.py`, and a per-run Markdown report.

The guided notebook run streamed its output and saved its adjacent `.log` and `.md` report. The fixed run was executed in a new notebook cell and saved its Markdown report as `results.md` inside the run directory. This combined tracker is copied to both `NACLIP-results/guided/ExperimentResults.md` and `NACLIP-results/fixed/ExperimentResults.md` in Drive. No new evaluation was launched when recording these results.

## Historical PAMR-enabled context

| Metric | Earlier fixed Gaussian + PAMR | Fixed, PAMR off | Guided, PAMR off |
| --- | ---: | ---: | ---: |
| mIoU (%) | 38.36 | 35.19 | 34.86 |
| Overall pixel accuracy (%) | 62.27 | 59.34 | 59.06 |
| Mean class accuracy (%) | 59.13 | 57.74 | 57.59 |
| Boundary accuracy (%) | Not measured | 36.3862 | 36.2615 |
| Interior accuracy (%) | Not measured | 61.7399 | 61.4408 |
| Wall time | 36 min 4.10 sec | 8 min 52.15 sec | 56 min 36.50 sec |
| Peak PyTorch allocation (GiB) | 2.1418 | 0.4685 | 0.4685 |

The earlier PAMR-enabled score is higher than both current scores. Guided is 3.50 mIoU points lower than that historical result; fixed/PAMR-off is 3.17 points lower. The historical boundary and interior metrics were not recorded; its timing had different loading conditions and no newly added boundary/interior metric. The fully verified matched spatial-bias comparison is the fixed/PAMR-off versus guided/PAMR-off table above.
