# Working on NACLIP

Current notebook preset: the user requested PC59 full guided evaluation through
`colab_naclip.ipynb`, with tau=0.33, temperature=0.10, and PAMR off. The current
baseline source is committed and pushed to `main`; the notebook defaults to
`SOURCE_MODE='github'` so a fresh runtime clones it. Use `synced` only after
explicitly syncing local source into the selected runtime. Keep live evaluation
in the notebook for this request; do not launch a duplicate CLI evaluation. The
separate tracker is `ImprovementsIdea/ExperimentResults.md`; each notebook run
also writes its own Markdown report beside its named Drive folder.

Read `COLAB.md` before running GPU work. The user's intended workflow is local
editing on Ubuntu (including agent edits), with GPU execution on Google Colab.
Two supported options are documented there: the CLI workflow for terminal/agent
execution, and the official VS Code Colab extension with `colab_naclip.ipynb` for
interactive notebook execution. Follow the user's choice. CLI is the default for
agent-run jobs. The extension notebook now clones `https://github.com/Chowkey/NACLIP.git`
(branch `main`); it fetches pushed commits, not automatic local-file updates.

## Default for user-requested code changes

The user explicitly requests keeping both the local checkout and the Colab T4
runtime source updated whenever an agent makes a code change for them. Treat the
local checkout as the source of truth: edit and save locally, then use
`python3 tools/colab_cli.py sync --session naclip` to update `/content/NACLIP`.
For the CLI workflow, do not require manual uploads if the CLI is connected;
confirm new source files are covered by its sync allowlist. For the Git-based
extension workflow, preserve the runtime checkout: commits must be pushed to
the configured repository before the notebook can fetch them. Commit or push
only when authorized; a request to commit alone does not authorize pushing.
The notebook no longer loads `extension-source.zip`. Do not mix CLI-synced
uncommitted files into a Git checkout the notebook is expected to update.

Inspect the selected runtime and active jobs before syncing or testing. Preserve
an active evaluation's source until it finishes; queue the update rather than
mixing versions within a run. An already running process does not reload changed
files automatically. After syncing, run a relevant smoke or fixed-subset check
with the isolated remote Python on the T4; run full validation when requested or
needed for the task, not after every edit. Keep results in a new persistent run
folder and report which changes were synced and which checks actually ran.

If authentication, runtime availability, or an active job prevents remote work,
complete the local change and state clearly that sync/testing is pending. The
runtime source is `/content/NACLIP`, not a second working checkout in Drive;
Drive holds persistent datasets and results unless the user requests otherwise.

## Locations and commands

- Discover the checkout from this file; do not reuse the old Windows path.
- Local setup: `bash tools/setup_ubuntu.sh` (after switching to Ubuntu).
- Commands: `python3 tools/colab_cli.py {sessions,new,sync,setup,smoke,eval}`.
- Baselines: `baseline --dataset {ade20k,pc59} --stage {subset,full}` with explicit
  remote `--data-root` and persistent `--work-dir`; preparation: `prepare-data`.
  Runner defaults to `--condition fixed` (PAMR off). Proposal modes: fixed,
  guided, none; guided requires explicit tau/temperature. Use upstream only
  for legacy PAMR-on settings. Summaries include pooled 3-pixel boundary and
  interior accuracy; keep seeds, datasets, and settings matched across modes.
- Extension: select a Colab GPU kernel in `colab_naclip.ipynb`; its first cell
  clones the configured GitHub repo into `/content/NACLIP` on a fresh runtime.
  Existing Git checkouts are reused unless `UPDATE_SOURCE=True`; updates require
  a clean matching branch and fast-forward merge. Non-Git folders are preserved.
  Then use the setup/smoke/baseline cells, which share the CLI environment/runner.
- Default CLI session alias: `naclip`; select another with `--session NAME`.
- Remote source: `/content/NACLIP`; remote Python: `/content/naclip-env/bin/python`.
- First inspect `sessions`. The VS Code extension's session and authentication
  may not be registered in the CLI. Do not assume they carry across OS installs.
- Use `new` only if a suitable CLI-managed session is absent. It requests T4.
- CLI after edits: `sync`, then `smoke` or `eval`; `sync --dry-run` lists uploads.
- Fresh CLI runtime: `sync`, `setup`, `smoke`. Fresh extension runtime: clone cell,
  setup cell, smoke cell. Repeat setup after dependency edits.
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
1.26.4. On 2026-11-15 Ubuntu CLI authentication, source sync, isolated setup, GPU
smoke and 32-image ADE20K/PC59 runs passed. Full ADE20K obtained 19.06 mIoU, but
PAMR ran out of memory on seven images, so its summary marks the baseline invalid.
Read persisted summaries before reporting full-run status or reproduction success.

Let the user complete Google login/consent themselves. Never print or commit
tokens or copy credentials from VS Code. The Ubuntu desktop keyring should be
unlocked so the official CLI can retain authentication.

Before evaluation, establish the dataset and its remote path. Persist outputs to
mounted Drive (or download them before the runtime expires). Do not put data,
weights, generated notebooks, or logs in Git. Sync before launching a job, not
while another job is importing project files. Sync does not mirror deletions;
remove obsolete remote source deliberately if renamed modules cause conflicts.

The configured `origin` is `https://github.com/Chowkey/NACLIP.git`. Commit locally
when requested; do not push or change remotes without user direction.
