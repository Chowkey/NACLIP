# Working on NACLIP

Read `COLAB.md` before running GPU work. The user's intended workflow is local
editing on Ubuntu (including agent edits), with GPU execution on Google Colab.
Use the command workflow in that document; do not require manual uploads or a
GitHub fork for normal iteration.

## Locations and commands

- Discover the checkout from this file; do not reuse the old Windows path.
- Local setup: `bash tools/setup_ubuntu.sh` (after switching to Ubuntu).
- Commands: `python3 tools/colab_cli.py {sessions,new,sync,setup,smoke,eval}`.
- Default CLI session alias: `naclip`; select another with `--session NAME`.
- Remote source: `/content/NACLIP`; remote Python: `/content/naclip-env/bin/python`.
- First inspect `sessions`. The VS Code extension's session and authentication
  may not be registered in the CLI. Do not assume they carry across OS installs.
- Use `new` only if a suitable CLI-managed session is absent. It requests T4.
- After edits run `sync`, then `smoke` or `eval`. `sync --dry-run` lists uploads.
- On a fresh runtime: `sync`, `setup`, `smoke`. Repeat `setup` after dependency edits.
- The wrapper always names the session; do not run against an ambiguous runtime.
- `sync` uploads a source-only archive including uncommitted edits, preserves paths,
  skips identical remote files, and does not delete remote files. It excludes Git
  history, environments, credentials, datasets, checkpoints, and execution outputs.
- Do not fix errors by installing dependencies in the local host or changing the
  Colab notebook kernel. NACLIP uses its isolated remote Python 3.10 environment.

## Running and reporting

NACLIP is training-free. The smoke check verifies synthetic GPU inference, not
accuracy. Do not describe it as model training or benchmark validation.

The initial notebook workflow passed on a Tesla T4 on 2026-11-07 with Python 3.10,
PyTorch 2.0.1+cu118, MMCV 2.0.1, MMEngine 0.8.4, MMSegmentation 1.1.1, and NumPy
1.26.4. The Ubuntu CLI workflow was prepared on Windows; authenticate and verify
it on Ubuntu before reporting end-to-end success.

Let the user complete Google login/consent themselves. Never print or commit
tokens or copy credentials from VS Code. The Ubuntu desktop keyring should be
unlocked so the official CLI can retain authentication.

Before evaluation, establish the dataset and its remote path. Persist outputs to
mounted Drive (or download them before the runtime expires). Do not put data,
weights, generated notebooks, or logs in Git. Sync before launching a job, not
while another job is importing project files. Sync does not mirror deletions;
remove obsolete remote source deliberately if renamed modules cause conflicts.

The Git remote initially points to the upstream author's repository. Commit
locally when requested; do not push or change remotes without user direction.
