# NACLIP user guide: Colab extension and CLI

Use Ubuntu and VS Code to edit source locally, and Google Colab to execute GPU work. You can use either the VS Code Colab extension or the Colab CLI. They have separate connection state: a CLI session listing alone does not establish whether an extension server is still available.

## 1. Understand where your files are

| Location | Meaning | Persists after runtime deletion? |
| --- | --- | --- |
| `/home/chowkey/VSCode Projects/SemanticSegmentation/NACLIP` | Local source checkout | Yes |
| `/content/NACLIP` | Source cloned or uploaded to a Colab runtime | No |
| `/content/naclip_data` | Dataset copies cached on a runtime | No |
| `/content/drive/MyDrive` | Your Google Drive, after mounting | Yes |
| `/tmp/naclip-progress.json` in your Ubuntu terminal | A downloaded local snapshot | Temporary; does not refresh automatically |

Each Colab server has its own `/content`, packages, RAM, and GPU allocation. Mount Drive separately in each server. Files saved to Drive can be accessed again from another runtime after mounting it.

## 2. Using the VS Code Colab extension

### Connect to a runtime

1. Open the local NACLIP folder in VS Code.
2. Install the **Jupyter** and **Google Colab** extensions if needed.
3. Open `colab_naclip.ipynb`, or another `.ipynb` notebook.
4. Click **Select Kernel → Colab**. If Colab is not visible, check **Select Another Kernel**.
5. Select an existing server when possible. Otherwise choose **New Colab Server** and the required hardware. Use **T4** for NACLIP inference/evaluation and **CPU** for inspecting results or preparing data.
6. Complete Google sign-in and consent in your browser if prompted, then select the remote Python kernel.

Run this notebook cell to confirm that execution is remote:

```python
import os
print(os.getcwd())
print(os.path.isdir('/content'))
```

On a GPU server, check the GPU with:

```python
!nvidia-smi
```

The normal VS Code integrated terminal is still local unless you explicitly open a Colab terminal. Choosing a notebook kernel does not turn a local terminal into a remote one.

### Mount Google Drive

With the intended notebook/server selected, open the command palette (`Ctrl+Shift+P`) and choose **Colab: Mount Google Drive to Server...**. Run the inserted cell and complete authorization yourself.

You can also run this cell directly:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Browse, open, and download files

1. Click the **Colab** icon in VS Code's left activity bar.
2. In **Contents**, expand the intended server and browse its `/content` directory.
3. For saved results, expand `drive → MyDrive → NACLIP-results`.
4. Open a file from the tree to view it. Use the tree's **Refresh** action if new files are missing.
5. Right-click a file and choose **Download...** to save a local copy.

For a remote shell, use **Colab: Open Terminal** on the intended server. For example:

```bash
ls /content/drive/MyDrive/NACLIP-results/baseline-2026-11-15
cat /content/drive/MyDrive/NACLIP-results/baseline-2026-11-15/pc59_full/summary.json
```

You can read JSON directly in a notebook cell too:

```python
from pathlib import Path
import json

run_dir = Path('/content/drive/MyDrive/NACLIP-results/baseline-2026-11-15/pc59_full')
print(json.dumps(json.loads((run_dir / 'summary.json').read_text()), indent=2))
```

For a run in progress, replace `summary.json` with `progress.json`. Rerun the cell to refresh the displayed contents.

### Clone source and run NACLIP

The main notebook now clones `https://github.com/Chowkey/NACLIP.git` (`main`) into `/content/NACLIP` in its first cell. Only committed and pushed changes are fetched. It prints the commit and source preparation time. On an existing clean Git checkout, set `UPDATE_SOURCE=True` to fetch updates after active jobs finish. Old source ZIPs are not extracted. If `/content/NACLIP` contains uploaded files without Git metadata, choose a different `PROJECT` path; the cell preserves that folder.

Run setup, smoke, and the desired evaluation cells after source initialization. Required local scripts must be committed and pushed before GitHub cloning can supply them.

The source flow is:

```text
Local edits → local commit → push to Chowkey/NACLIP → notebook clone/fetch
```

Committing alone does not publish changes. On a fresh runtime, run the first cell with `UPDATE_SOURCE=False` (the default); it clones automatically. To fetch new pushed commits on the same runtime, stop active jobs, set `UPDATE_SOURCE=True`, and rerun the first cell. The cell preserves modified checkouts and existing non-Git folders rather than overwriting them.

Baseline results go to `/content/drive/MyDrive/NACLIP-results/extension-<timestamp>/<dataset>_<stage>/` by default. Adjust `BASELINE_WORK_DIR` for a custom destination; the standard evaluation cell instead uses `WORK_DIR`. Use a new folder per run. Mounted Drive outputs survive runtime deletion.

### Alternative: upload uncommitted source

For uncommitted source, use the CLI sync/setup/smoke/baseline workflow below. It does not require a GitHub push. The main notebook's first cell expects a Git checkout, so keep its Git workflow separate from uploaded source. Both workflows use `/content/naclip-env/bin/python` for NACLIP; do not replace the notebook kernel's packages.

For individual file transfers, right-click local source files/folders in VS Code Explorer and choose **Upload to Colab**, then select the intended server. The destination is fixed under `/content/`: selecting a folder named `NACLIP` creates `/content/NACLIP`, while selecting `naclip.py` alone creates `/content/naclip.py`. There is no destination-folder picker in extension 0.9.7. Normal local edits are not automatically uploaded; use CLI upload for an explicit destination.

For a complete source upload with exclusions, use the CLI `sync` workflow below against a verified CLI-managed runtime. Do not assume the extension server's display name is a CLI alias.

### Use a CPU server and GPU server together

Open two notebooks. Connect one to your GPU server and use **New Colab Server** to create a CPU server for the other. Check the selected server in each notebook before running cells.

Multiple servers are supported, subject to Colab account limits and availability. Avoid concurrent jobs that write to the same output directory or modify the same dataset files.

### Monitor resources and stop a server

- The Colab activity bar's **Resources** view shows server RAM, disk, and GPU resources.
- Hover over the Colab status bar indicator to inspect account usage details when available.
- `!nvidia-smi` shows current GPU utilization and VRAM, not remaining GPU allowance.
- When finished, use **Colab: Remove Server** and select the intended server. Save results to Drive or download them before removal.

## 3. Using the Colab CLI

Run the following commands in your **local Ubuntu terminal** from the NACLIP checkout:

```bash
cd "/home/chowkey/VSCode Projects/SemanticSegmentation/NACLIP"
```

### First-time setup and authentication

If local CLI tooling is not installed yet:

```bash
bash tools/setup_ubuntu.sh
```

Inspect sessions and complete any Google login/consent prompts yourself:

```bash
python3 tools/colab_cli.py sessions
```

Keep the Ubuntu keyring unlocked so authentication can be retained. Do not paste credentials into notebooks or source files.

### Connect to or create a runtime

The wrapper defaults to the alias `naclip`. Always inspect sessions first and reuse a suitable server rather than creating duplicates.

Create a T4 server only when needed:

```bash
python3 tools/colab_cli.py new --session naclip
```

For a separate CPU server, use the official CLI directly; omitting `--gpu` and `--tpu` requests CPU:

```bash
.colab-cli/bin/colab new -s naclip-cpu
```

The project wrapper's `new` action always requests T4. Use `-s naclip-cpu` for subsequent official CLI commands targeting the CPU runtime.

### Prepare a GPU runtime for NACLIP

```bash
python3 tools/colab_cli.py sync --session naclip
python3 tools/colab_cli.py setup --session naclip
python3 tools/colab_cli.py smoke --session naclip
```

`sync` sends source, including uncommitted edits, while excluding datasets, environments, credentials, and outputs. It does not delete obsolete remote files. Sync before launching a job, not while that job is importing project source.

`setup` installs the isolated remote environment. `smoke` checks synthetic GPU inference; it does not measure dataset accuracy. Repeat setup after dependency changes and after creating a fresh runtime.

### Mount Drive and list files

```bash
.colab-cli/bin/colab drivemount -s naclip
.colab-cli/bin/colab ls -s naclip content
.colab-cli/bin/colab ls -s naclip content/drive/MyDrive/NACLIP-results
.colab-cli/bin/colab ls -s naclip content/drive/MyDrive/NACLIP-results/baseline-2026-11-15
```

Complete Drive authorization if prompted. For these CLI file commands, use remote paths beginning with `content/`, without a leading slash. Python and remote shell examples use `/content/` instead.

### Fetch and view progress

```bash
.colab-cli/bin/colab download -s naclip \
  content/drive/MyDrive/NACLIP-results/baseline-2026-11-15/ade20k_full/progress.json \
  /tmp/naclip-progress.json

python3 -m json.tool /tmp/naclip-progress.json
```

Repeat both commands to see updated progress. `json.tool` only displays the local file; it does not fetch updates. The baseline runner writes progress every 25 images during a full run, so the count may stay unchanged briefly.

For PC59 progress, replace `ade20k_full` with `pc59_full` in the remote path.

### Fetch final results

```bash
.colab-cli/bin/colab download -s naclip \
  content/drive/MyDrive/NACLIP-results/baseline-2026-11-15/pc59_full/summary.json \
  /tmp/naclip-pc59-summary.json

python3 -m json.tool /tmp/naclip-pc59-summary.json
```

Replace `pc59_full` with `ade20k_full` and choose a different local filename to retrieve ADE20K results. To keep local copies permanently, download them to a folder outside `/tmp` and outside Git-tracked source.

### Run a full evaluation

After verifying the remote dataset exists, run:

```bash
python3 tools/colab_cli.py baseline --session naclip \
  --dataset pc59 --stage full \
  --data-root /content/naclip_data/VOCdevkit/VOC2010 \
  --work-dir /content/drive/MyDrive/NACLIP-results/my-new-run/pc59_full
```

For ADE20K, use `--dataset ade20k`, data root `/content/naclip_data/ADEChallengeData2016`, and a separate output directory. Use a new work directory for each evaluation. Runtime-local dataset caches must be prepared or copied again after runtime deletion; see `COLAB.md` for dataset preparation.

**PAMR is enabled by default in the current baseline runner.** The smoke check disables PAMR, but that setting does not carry over to evaluation. A run intended to disable PAMR needs an explicit model configuration change before launch; the baseline wrapper currently has no `--pamr` option.

### Check usage and stop a server

```bash
.colab-cli/bin/colab usage
.colab-cli/bin/colab stop -s naclip
```

`usage` reports account compute-unit information when available; it is not a guaranteed countdown to runtime termination. Free GPU limits are dynamic and Colab does not publish an exact remaining-hours quota. Mounting a different Drive does not change the runtime account's GPU allowance.

## 4. Read the output and troubleshoot

| File / field | Meaning |
| --- | --- |
| `progress.json` | Latest saved image count, elapsed time, and PAMR failures |
| `summary.json` | Final metrics, timing, and baseline validity |
| `manifest.json` | Source hashes, environment, and resolved model settings |
| `config.py` | Saved evaluation configuration |
| `timing.json` | Per-image inference timing |
| `predictions/`, `comparisons/` | Prediction images and visual comparisons, exported for subset runs |
| `mIoU` | Main segmentation score; larger is better |
| `pamr_failures` | Images where refinement failed and unrefined predictions were used |
| `baseline_valid` | Runner check that no PAMR failures occurred; does not by itself establish exact paper reproduction |

**Progress appears frozen:** Fetch the remote file again. Check the correct server and output directory. If remote progress remains unchanged, inspect the evaluation output and GPU resources. An active server alone does not prove the evaluation process is advancing.

**CLI reports no sessions:** Check the extension separately. Extension-managed connections may not have a usable local CLI alias. Do not conclude that the extension runtime ended from the CLI listing alone.

**The runtime really ended:** Its `/content` files are lost, but Drive outputs remain. Connect a CPU runtime, mount Drive, and read/download the saved summary. No model setup or T4 is needed to inspect JSON files.

**PAMR reports CUDA out of memory:** The runner continues with unrefined predictions for affected images. A final summary can exist even if the runner then raises an error because `baseline_valid` is false. Inspect the failure list before treating the score as a consistent PAMR baseline.

**A downloaded file is missing:** Verify Drive is mounted, list the parent folder, and check the run name. A new runtime does not restore runtime-local files automatically.

## References

- Project-specific setup and evaluation details: `COLAB.md`
- [Official Colab extension user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
- [Official Colab CLI repository](https://github.com/googlecolab/google-colab-cli)
- [Colab FAQ: resources, persistence, and account limits](https://research.google.com/colaboratory/faq.html)

Commands were checked against this checkout's installed Colab CLI 0.7.5 and extension 0.9.7. Menus and options may vary with newer versions; use `.colab-cli/bin/colab help COMMAND` to inspect a command.
