# Proposal experiment results

Initially recorded on 2026-11-16 (Asia/Ho_Chi_Minh); updated on 2026-11-20.

## ADE20K guided full-validation run

On 2026-11-18, the feature-guided condition completed all 2,000 ADE20K validation images on a Tesla T4. The run used seed 42, $\tau=0.33$, temperature $0.10$, PAMR disabled, and standard sliding-window merging. The run log reports `baseline_valid=true` and `proposal_protocol=true`.

| Metric | Guided, PAMR off |
| --- | ---: |
| mIoU (%) | 17.23 |
| Overall pixel accuracy (%) | 44.94 |
| Mean class accuracy (%) | 39.02 |
| Boundary accuracy (%) | 25.4498 |
| Interior accuracy (%) | 47.1734 |
| Evaluation wall time | 61 min 6.47 sec |

The matched full-validation fixed Gaussian and no-spatial-bias conditions are still unverified. The guided score alone gives no estimate of the change in boundary accuracy relative to the fixed Gaussian baseline.

Source: the pasted run log and its `BASELINE_SUMMARY`. The log reports the run directory as `/content/drive/MyDrive/NACLIP-results/guided/ade20k_guided_full_tau0p33_temp0p10_20261118T055542868887Z`, with an adjacent Markdown report and log on Drive.

## ADE20K overlap-calibration paired pilot

On a matched 32-image ADE20K subset (seed 42), overlap-aware window calibration produced nearly unchanged scores relative to standard sliding-window merging. Both conditions completed all images and used the same image IDs. PAMR was disabled; guided spatial bias used $\tau=0.33$ and temperature $0.10$ in both conditions.

| Metric | Standard merge | Overlap calibration | Calibrated minus standard |
| --- | ---: | ---: | ---: |
| mIoU (%) | 13.73 | 13.72 | -0.01 |
| Boundary accuracy (%) | 27.6264 | 27.6353 | +0.0089 |
| Interior accuracy (%) | 51.8792 | 51.8740 | -0.0052 |

This pilot shows no material change in these aggregate metrics; the tiny boundary increase alone is not evidence of a reliable gain. It is a subset result and does not establish full-validation performance. Run directory on Drive: `/content/drive/MyDrive/NACLIP-results/window-calibration/ade20k_paired_seed42_20261118T073454609223Z`.

## PC59 overlap-calibration paired pilot

On a matched 32-image PC59 validation subset (seed 42), overlap-disagreement calibration made almost no change relative to standard sliding-window merging. Both runs used guided spatial bias ($\tau=0.33$, temperature $0.10$), PAMR off, and the same image IDs; both summaries were valid.

| Metric | Standard merge | Overlap calibration | Calibrated minus standard |
| --- | ---: | ---: | ---: |
| mIoU (%) | 27.37 | 27.36 | -0.01 |
| Overall pixel accuracy (%) | 57.53 | 57.53 | 0.00 |
| Mean class accuracy (%) | 57.88 | 57.84 | -0.04 |
| Boundary accuracy (%) | 38.6925 | 38.7147 | +0.0222 |
| Interior accuracy (%) | 59.6326 | 59.6344 | +0.0018 |

The tiny boundary increase is not evidence of a reliable gain; aggregate metrics are effectively unchanged in this pilot. The observed runner times were 45.06 seconds for standard merge and 6.75 seconds for calibration, but the runs were sequential and the second may benefit from warm data/cache state, so this is not a controlled speed comparison. Run directory on Drive: `/content/drive/MyDrive/NACLIP-results/window-calibration/pc59_paired_seed42_20261119T141553841782Z`.

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

Tau and temperature were selected before the guided evaluation from unlabeled feature distributions of 32 ADE20K and 32 PC59 validation images. They were not selected using segmentation accuracy. Calibration artifacts are under `NACLIP-results/guidance-distribution-20261115T162954Z`.

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
