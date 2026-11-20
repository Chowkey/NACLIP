# Local Ubuntu code, Colab GPU

The notebook is currently preset for the requested **PC59 guided full run**
(tau 0.33, temperature 0.10, PAMR off). Select the existing T4 kernel, run
initialization, then the **PC59 guided full validation** cell. Initialization
defaults to `SOURCE_MODE='github'` and clones the current pushed source on a fresh
runtime. Choose `SOURCE_MODE='synced'` only after explicitly syncing source to
that runtime.
This run writes to `NACLIP-results/guided/pc59_guided_full_tau0p33_temp0p10_<timestamp>`
in Drive, with an automatically updated Markdown results report and console log
beside the folder. See [experiment tracker](ImprovementsIdea/ExperimentResults.md).

Open this NACLIP checkout as your VS Code folder on Ubuntu. Choose either workflow:

| Option | Interface | Send local edits | Execute remotely |
| --- | --- | --- | --- |
| A: CLI | Local terminal / coding agent | `python3 tools/colab_cli.py sync` | `setup`, `smoke`, `baseline` commands |
| B: [VS Code Colab extension](#option-b-vs-code-colab-extension) | Notebook cells / optional Colab terminal | Commit and push to `Chowkey/NACLIP`; notebook clones/fetches | `colab_naclip.ipynb` cells |

Both use `/content/NACLIP`, the isolated `/content/naclip-env` model environment,
and the same baseline runner. Follow the user's preferred option. Option A sends
uncommitted local edits directly and needs no push. Option B fetches committed,
pushed source from `https://github.com/Chowkey/NACLIP.git` and needs no local CLI
installation or CLI login. A local commit alone does not update GitHub. See
[user_guide.md](user_guide.md) for connecting, viewing files, and fetching results.

## One-time Ubuntu setup

Run from the repository root **after switching to Ubuntu**:

```bash
sudo apt update
sudo apt install python3 python3-venv git
bash tools/setup_ubuntu.sh
python3 tools/colab_cli.py sessions
```

Setup installs tooling into ignored `.colab-bootstrap` and `.colab-cli` folders.
It uses uv to obtain Python 3.12 for Google's official `google-colab-cli`, so an
older Ubuntu system Python is fine. No local PyTorch or CUDA installation is
needed. Internet access is required. Complete Google login/consent yourself when
prompted. Keep the Ubuntu desktop keyring unlocked to retain authentication.
Do not put tokens in the repository or paste them into agent chats. Windows/VS
Code login does not automatically transfer to the Ubuntu CLI.

## First run on a runtime

Inspect `sessions` first. If no suitable CLI-managed session exists, request T4:

```bash
python3 tools/colab_cli.py new --session naclip
python3 tools/colab_cli.py sync --session naclip
python3 tools/colab_cli.py setup --session naclip
python3 tools/colab_cli.py smoke --session naclip
```

The default alias is `naclip`, so `--session naclip` can be omitted. Each wrapper
call passes the selected alias explicitly. Extension-managed servers may appear
without a usable local CLI alias; do not assume their VS Code display names work
in the CLI. Avoid duplicate GPU runtimes: inspect `sessions` and the official
CLI's current help if reusing an extension-managed runtime. T4 availability varies.

`sync` sends one source archive, preserves subdirectories, and writes only changed
files into `/content/NACLIP`. It includes uncommitted edits and excludes Git
history, environments, credentials, datasets, weights, logs, and notebook outputs.
Preview the file list without connecting:

```bash
python3 tools/colab_cli.py sync --dry-run
```

The source allowlist is in `tools/colab_cli.py`. Add new source directories or
root-level scripts there if needed. Sync does not delete remote files or mirror
local deletions. Deliberately remove obsolete remote modules after renames if
they cause conflicts. Sync before running a job, not while it is importing code.

## Everyday edits

```bash
python3 tools/colab_cli.py sync
python3 tools/colab_cli.py smoke
```

Agents can execute these commands for you. Repeat `setup` after dependency edits.
After a runtime is deleted, create a new session and repeat `sync`, `setup`, and
`smoke`. CLI execution streams remote output back to the terminal and agent.

## Dataset evaluation

### Prepare the proposal datasets

`tools/prepare_datasets.py` follows the sibling notebook
`NaClip_ComputerVision/NaCLIP_dataset_setup_preprocess.ipynb`: download ADE20K,
VOC2010 images and official PASCAL Context annotations; verify the Context archive
MD5; convert 5,105 PC59 validation masks in NACLIP class order; validate 2,000
ADE20K validation image/mask pairs; generate the notebook's boundary/valid-pixel
masks and write `naclip_validation_manifest.json`. Masks retain original resolution.
It uses the synced checkout's class list rather than cloning another checkout.

After `sessions` and `sync` on the selected runtime (model `setup` is not needed):

```bash
.colab-cli/bin/colab drivemount -s naclip
python3 tools/colab_cli.py prepare-data
```

Complete Drive authorization yourself. The default dataset root is
`/content/drive/MyDrive/datasets`; override it with `--data-root` if needed.
Downloads and preprocessing may take a while; this action allows two hours
(`--timeout SECONDS` overrides it). Reruns reuse downloads/extractions; conversion
and boundary generation run again. This is CPU preprocessing and needs no GPU.
The VOC archive's top-level directory is detected to avoid nested
`VOCdevkit/VOCdevkit` paths. Do not run two preparation jobs against the same root.

Prepared evaluation roots and configs:

| Config | Remote data root |
| --- | --- |
| `configs/cfg_ade20k.py` | `/content/drive/MyDrive/datasets/ADEChallengeData2016` |
| `configs/cfg_context59.py` | `/content/drive/MyDrive/datasets/VOCdevkit/VOC2010` |

Preparation writes persistent data to Drive as in the source notebook. For faster
repeated evaluation, copy the prepared dataset to `/content` once per runtime and
point `eval --data-root` at that copy. Boundary masks are additional proposal
artifacts; standard NACLIP evaluation reads the semantic annotation masks.

### Run evaluation

Prepare the dataset following the upstream README. To mount Drive and evaluate:

```bash
.colab-cli/bin/colab drivemount -s naclip
python3 tools/colab_cli.py eval \
  --config configs/cfg_voc21.py \
  --data-root /content/drive/MyDrive/datasets/VOCdevkit/VOC2012 \
  --work-dir /content/drive/MyDrive/NACLIP-results/voc21
```

Complete any Drive authorization yourself. Paths refer to the remote Linux
filesystem; local Ubuntu/Windows paths are unavailable there. Sync does not send
datasets. Save outputs to mounted Drive or download them before the runtime expires.
Stop a CLI-managed runtime when finished (its temporary storage is released):

```bash
.colab-cli/bin/colab stop -s naclip
```

### Evaluate the proposal conditions

After setup, `baseline` evaluates `--condition fixed` by default: original
Gaussian bias with **PAMR disabled**. `guided` and `none` also disable PAMR;
`upstream` is an explicit legacy mode retaining the original PAMR defaults.
The underlying model default outside this runner remains unchanged.

The experimental overlap-calibration mode is off by default. Use `--window-calibration disagreement` to enable it (strength 0.5, confidence power 1.0); compare it on the same seed and subset with PAMR disabled. The ADE20K paired-pilot cell in `colab_naclip.ipynb` runs standard and calibrated modes on identical selected image IDs and records both summaries.

The patch-grid phase ensemble is also off by default. Set `--patch-phase-ensemble two` to average the canonical image grid with one diagonal half-patch shift, or `four` to use all horizontal/vertical half-patch combinations. It increases encoder work by two or four phases. The notebook's **ADE20K patch-phase pilot** cell compares `off` and `two` on the same seeded 32-image subset with fixed Gaussian bias and PAMR disabled.

All conditions report overall mIoU plus boundary and interior accuracy. The
boundary band uses both sides of valid horizontal/vertical label transitions,
Euclidean distance <= 3 pixels on the original annotation grid, and excludes
ignore label 255. Accuracies pool correct/total pixel counts across the split;
they are not averages of per-image percentages. Regions with no eligible pixels
have `null` accuracy in `summary.json`.

Start with a fixed subset to validate the pipeline:

```bash
python3 tools/colab_cli.py baseline --dataset ade20k --stage subset --condition fixed \
  --data-root /content/naclip_data/ADEChallengeData2016 \
  --work-dir /content/drive/MyDrive/NACLIP-results/proposal/ade20k_fixed_subset
```

Use `--dataset pc59` with root `/content/naclip_data/VOCdevkit/VOC2010` for PC59.
The `/content/naclip_data` directories are runtime-local copies of the prepared
Drive data; copy them first, or point directly to Drive. Subsets contain 32 images
selected with seed 42 from sorted image IDs. `--stage full` evaluates all 2,000
ADE20K or 5,105 PC59 validation images; use a separate work directory for each run.
The execution timeout defaults to 12 hours; this does not extend Colab VM lifetime.

For guided runs, supply `--condition guided --guidance-tau VALUE
--guidance-temperature VALUE`. The threshold must be in [-1, 1], temperature
must be positive, and both must be finite. Choose them before evaluation and keep
them identical across ADE20K and PC59; the proposal does not prescribe values.
The implementation uses raw penultimate-block features before the final-block
layer norm, excludes CLS from cosine similarity, computes similarity in float32,
and shares each image's gate across attention heads. Only the fixed Gaussian is
cached; the content-dependent gate is recomputed on each forward pass.

For example, add `--patch-phase-ensemble two` to a fixed-subset command to run
the two-phase variant. The runner records the phase mode in `manifest.json` and
`summary.json` so it can be compared with a matched single-phase run.

Run `--condition none` for key-key attention without spatial bias. Give every
condition a new, empty work directory and use the same seed/subset selection.
Set `--stage full` only after checking the subset. Historical PAMR-enabled scores
are not comparable to these proposal baselines; request `--condition upstream`
when intentionally reproducing the old inference settings.

`summary.json` records `condition`, `resolved_model`, `proposal_protocol`, and
`metrics`, including `localization/boundary_accuracy`,
`localization/interior_accuracy`, and their correct/total pixel counts. Main
comparison: guided boundary accuracy minus fixed boundary accuracy in percentage
points, with overall mIoU and interior accuracy checked alongside it.

Remote correctness and synthetic integration checks (no dataset required):

```bash
cd /content/NACLIP
/content/naclip-env/bin/python -m unittest discover -s tools -p 'test_*.py' -v
/content/naclip-env/bin/python tools/proposal_smoke.py
```

The smoke script uses tau=0.5 and temperature=0.1 solely as test inputs; these are
not selected research hyperparameters. It checks all three conditions through
real CLIP inference and MMSeg's IoU/boundary evaluators on a synthetic image.
It does not establish an accuracy improvement. See
[proposal specification](ImprovementsIdea/Proposal.md).


Outputs include a resolved config, source hashes, package versions, image IDs,
timing, and a metrics summary. Subset runs also save raw class-index PNGs and
comparison JPEGs (input / ground truth overlay / prediction overlay). Class IDs in
predictions are zero-based; use `classes.json` for their meaning. This export does
not use `eval.py --show-dir`, which changes PAMR settings. PAMR failures invalidate
the baseline and are recorded. Timing excludes dataloader fetch and artifact
export; steady-state statistics exclude the first sample, but mIoU uses all images.

Upstream PAMR-enabled README comparison values: ADE20K 19.05 and PC59 38.35 mIoU. The Colab
dependency stack differs from the paper setup; matching the model/config alone
does not establish exact numerical reproduction. Full-run summaries, not subset
scores, are the baseline measurements.

## Option B: VS Code Colab extension

### Connect a notebook

1. Install the official [Google Colab extension](https://github.com/googlecolab/colab-vscode)
   in VS Code and open this checkout as the workspace.
2. Open `colab_naclip.ipynb`, click **Select Kernel → Colab**, and select the
   intended existing GPU server. Create a **New Colab Server** with a GPU only if
   an appropriate server is not already available. Complete Google sign-in in
   your browser. Extension and CLI login/session names are not interchangeable.
3. The notebook's cells execute on Colab. The ordinary VS Code terminal still
   executes locally. Check the extension's Resources and Contents views for the
   chosen server and its `/content` files. Avoid starting a second evaluation on
   a GPU already running another job.

### Get source from GitHub (notebook default)

The first notebook cell clones `https://github.com/Chowkey/NACLIP.git`, branch
`main`, into `/content/NACLIP`. Set `REPO_URL`, `REPO_BRANCH`, or `PROJECT` in
that cell to change these values. No archive upload is needed.

Only committed and pushed files are included. Local edits and untracked scripts
must be committed and pushed deliberately before the runtime can fetch them.
The cell prints the checked-out commit and source preparation time, and checks
that required project scripts exist before dependency installation.

An existing Git checkout is kept unless `UPDATE_SOURCE=True`. Updating requires
a clean checkout on the configured branch and uses a fast-forward merge. Wait
for active jobs to finish before updating. An existing uploaded folder without
Git metadata is preserved; choose a different `PROJECT` path to clone alongside
it. The first cell does not extract old `extension-source.zip` archives.

### Alternative: upload local source

To run uncommitted local edits, use CLI `sync` on a verified CLI session, or
right-click a source-only local folder in Explorer and choose **Upload to Colab**.
The extension uploads the selected folder under `/content/<folder-name>`.
Exclude environments, `.git`, credentials, datasets, weights, and output folders.
The Git-based first cell requires a Git checkout. For uncommitted code, the
recommended alternative is the CLI setup/smoke/baseline workflow above rather
than combining uploaded files with the notebook's Git-managed checkout.

### Run the notebook

1. **Initialize:** check `REPO_URL`, `REPO_BRANCH`, and `PROJECT`, then run the
   first code cell. It clones or checks the checkout and verifies GPU/source.
   Set `UPDATE_SOURCE=True` only when ready to fetch pushed updates.
2. **Install isolated dependencies:** run once per new runtime, and after
   dependency changes. This creates Python 3.10 with PyTorch 2.0.1/CUDA 11.8 and
   MMCV 2.0.1 at `/content/naclip-env`, without changing the notebook kernel's
   Python/PyTorch. The final import check must pass.
3. **GPU inference check:** run the smoke cell. It checks all three proposal
   conditions and boundary/IoU integration on a synthetic image, not accuracy.
   It downloads pretrained CLIP weights when needed.
4. **Mount Drive:** set `MOUNT_DRIVE=True` and run that cell. Alternatively use
   **Colab: Mount Google Drive to Server...** from the command palette. Complete
   authorization yourself and use the Drive account containing your datasets.
5. **Prepare data only if needed:** the optional preparation cell is disabled by
   default. Reuse the already prepared ADE20K/PC59 data in Drive for normal runs.
6. **Baseline:** set `RUN_BASELINE=True`, `DATASET='ade20k'` or `'pc59'`, and
   `STAGE='subset'` or `'full'`, and `CONDITION='fixed'`, `'guided'`, or `'none'`.
   Guided requires explicit `GUIDANCE_TAU` and `GUIDANCE_TEMPERATURE`. Defaults
   select 32 fixed images with seed 42 and PAMR off. Outputs go under
   `extension-<timestamp>/<dataset>_<condition>_<stage>` in Drive.
   The cell invokes `tools/run_baseline.py` with the isolated Python and streams
   its output below the cell. Run one dataset/stage at a time. Full splits contain
   2,000 ADE20K images and 5,105 PC59 images.
7. **Inspect results:** run the last cell after evaluation, or set `RESULT_DIR` to
   an existing run. It reads `progress.json` / `summary.json` and displays a saved
   comparison if available. Read `baseline_valid` and `pamr_failures` along with
   `metrics.mIoU`, `metrics['localization/boundary_accuracy']`, and
   `metrics['localization/interior_accuracy']`. A completed image count alone
   does not prove a clean baseline.

The older optional upstream-evaluation cell remains available for other dataset
configs. Use the baseline cell for saved predictions, timing and `summary.json`.
Subset predictions have zero-based class IDs; `classes.json` names the classes.

A notebook kernel runs one cell at a time: watch the running cell's output while
it is busy. To inspect progress concurrently, use another notebook kernel on the
same server or the Colab Terminal. Merely opening another console does not attach
it to an existing process's stdout; follow that process's saved log instead.

### Optional extension terminal and remote editing

The official guide marks **Colab Terminal** and **Server Mounting** experimental.
Enable the desired experimental feature in the extension settings and reload VS
Code if prompted. **Colab: Open Terminal** opens a shell on the selected server.
From that remote terminal you can run the same model runner directly:

```bash
cd /content/NACLIP
/content/naclip-env/bin/python -u tools/run_baseline.py \
  --dataset pc59 --stage subset \
  --data-root /content/drive/MyDrive/datasets/VOCdevkit/VOC2010 \
  --work-dir /content/drive/MyDrive/NACLIP-results/my-extension-run/pc59_subset
```

Choose a new work directory for every run. **Colab: Mount Server To Workspace...**
allows editing remote files in VS Code; such edits affect the remote copy and do
not update the local checkout automatically. Keep local source as the source of
truth, or deliberately bring remote changes back before the next source upload.

Data under `/content` is temporary. Drive datasets/results persist; source cloning,
local dataset caches and environment setup must be repeated for a fresh runtime.
The Colab dependency stack differs from the paper's original environment; see the
baseline reproduction notes above. The notebook's code cells compile locally. Local Git fixtures checked fresh
cloning, default checkout reuse, fast-forward updates, refusal to update a dirty
checkout, and preservation of existing non-Git folders. The Git-based notebook
workflow has not been run end to end on Colab as part of this update.

Official extension instructions (checked 2026-11-15):
[User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## Verification and references

On 2026-11-15 the Ubuntu CLI reused an existing T4 runtime, synced source, installed
the isolated environment and passed synthetic GPU inference with shape
`(1, 30, 224, 224)`. Python was 3.10.22, PyTorch 2.0.1+cu118, MMCV 2.0.1,
MMEngine 0.8.4 and MMSegmentation 1.1.1. Prepared ADE20K/PC59 validation data was
cached under `/content/naclip_data`. Fixed 32-image subsets (seed 42) completed
without PAMR failures: ADE20K 15.65 subset mIoU, PC59 30.14 subset mIoU. These are
development measurements, not full benchmark results. Their predictions,
comparisons, timing and provenance are under Drive's
`NACLIP-results/baseline-2026-11-15/{ade20k,pc59}_subset`.

Proposal implementation checks on the T4 passed 16 unit tests (including CUDA
half-precision attention), real CLIP/MMSeg inference for fixed/guided/none, and
actual runner execution on two synthetic images per condition. Runner checks
confirmed identical image IDs, PAMR disabled, correct boundary/interior counts
on the original annotation grid, final progress, summaries, and prediction exports.
These synthetic checks establish pipeline execution, not benchmark accuracy.
Local copies of the synthetic summaries are ignored under
`work_logs/proposal-implementation-check/`.

The completed full PC59 summary reported 5,105 images, 38.36 mIoU, zero PAMR
failures, and `baseline_valid=true`, with evaluation taking 2,164.10 seconds.
The full ADE20K run reported 2,000 images, 19.06 mIoU, and seven PAMR failures
(`baseline_valid=false`). These measurements used the earlier uploaded-source
workflow; they do not validate the new Git-based initialization.

On 2026-11-07 the notebook workflow passed dependency imports and synthetic GPU
inference on Tesla T4, producing shape `(1, 30, 224, 224)`. NACLIP is training-free.
This checks execution, not accuracy. The Ubuntu CLI workflow was initially
prepared and checked locally on Windows; see the 2026-11-15 verification above
for Ubuntu execution. No dataset benchmark was run during the original check.
Local validation: four transfer/error-handling tests, Bash syntax checking, and
Python/notebook syntax checking passed. To rerun the local tests:

```bash
python3 -m unittest discover -s tools -p test_colab_cli.py -v
```

Official references (checked 2026-11-07):

- [Colab CLI installation and commands](https://github.com/googlecolab/google-colab-cli)
- [CLI Python requirement](https://github.com/googlecolab/google-colab-cli/blob/main/pyproject.toml)
- [VS Code extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
