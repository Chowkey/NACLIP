# Local Ubuntu code, Colab GPU

Open this NACLIP checkout as your VS Code folder on Ubuntu. Agents edit local
source and run the commands below to send it to Colab. No Git push or manual
upload is needed. `AGENTS.md` records this workflow for future agents.

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

## VS Code notebook alternative and remote environment

Keep this folder as the source of truth for local editing and agents. Open
`colab_naclip.ipynb`, then Select Kernel > Colab > New Colab Server and choose
a GPU. Google sign-in is completed by you in the browser.

Use Explorer's Upload to Colab action for selected source folders/files.
The notebook expects the uploaded checkout at `/content/NACLIP`; change its
PROJECT setting if the extension uploads it elsewhere. Upload updated source
files again after local edits; this is not automatic synchronization. Do not
upload `.git`, credentials, or large datasets. Alternatively upload the `clip`,
`configs`, `prompts`, `tools` folders and the root Python files individually.

Run notebook cells in order. The setup creates a separate Python 3.10 environment
under `/content/naclip-env`, with PyTorch 2.0.1/CUDA 11.8 and a matching MMCV
2.0.1 binary wheel. This adapts the upstream Python 3.9/PyTorch 1.10 instructions
to hosted Colab; benchmark equivalence has not been established. It leaves the
notebook kernel's Python and PyTorch unchanged. Downloads may take several minutes.

The smoke check loads CLIP weights and runs NACLIP on a synthetic image. It checks
GPU execution, not segmentation quality. For benchmark results, prepare the dataset
as described in the upstream README, mount Drive, and set DATA_ROOT and WORK_DIR
in the evaluation cell. For VOC21, DATA_ROOT should contain JPEGImages,
SegmentationClass, and ImageSets/Segmentation/val.txt.

The notebook's evaluation cell is disabled until RUN_EVALUATION is set to True.
Use a Drive path for WORK_DIR to retain results after the runtime is recycled.
For faster data loading, copy datasets to `/content` before evaluation. Local
Windows paths cannot be read by Colab. The VS Code local terminal remains local;
the notebook launches the isolated remote Python for NACLIP.

After a runtime is deleted, upload the source and rerun setup. Save/download useful
outputs before disconnecting. GPU availability and session duration depend on Colab.

## Verification and references

On 2026-11-07 the notebook workflow passed dependency imports and synthetic GPU
inference on Tesla T4, producing shape `(1, 30, 224, 224)`. NACLIP is training-free.
This checks execution, not accuracy. The Ubuntu CLI workflow was prepared and
checked locally on Windows; Google login, allocation, upload, and execution still
need an end-to-end check after switching to Ubuntu. No dataset benchmark was run.
Local validation: four transfer/error-handling tests, Bash syntax checking, and
Python/notebook syntax checking passed. To rerun the local tests:

```bash
python3 -m unittest discover -s tools -p test_colab_cli.py -v
```

Official references (checked 2026-11-07):

- [Colab CLI installation and commands](https://github.com/googlecolab/google-colab-cli)
- [CLI Python requirement](https://github.com/googlecolab/google-colab-cli/blob/main/pyproject.toml)
- [VS Code extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
