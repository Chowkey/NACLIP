#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this setup after switching to Ubuntu.' >&2
  exit 1
fi
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
if ! command -v python3 >/dev/null || ! command -v git >/dev/null; then
  echo 'Install prerequisites: sudo apt update && sudo apt install python3 python3-venv git' >&2
  exit 1
fi
if ! python3 -m venv .colab-bootstrap; then
  echo 'Install venv support: sudo apt install python3-venv' >&2
  exit 1
fi
.colab-bootstrap/bin/python -m pip install --upgrade uv
# The CLI needs Python >=3.12. This also works on Ubuntu with an older system Python.
.colab-bootstrap/bin/uv venv --python 3.12 --allow-existing .colab-cli
.colab-bootstrap/bin/uv pip install --python .colab-cli/bin/python --upgrade google-colab-cli
.colab-cli/bin/colab --help
echo 'Ready. Run: python3 tools/colab_cli.py sessions'
echo 'Complete Google login yourself when prompted. Keep your desktop keyring unlocked.'
