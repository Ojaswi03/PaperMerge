#!/usr/bin/env bash
# Create an isolated Linux environment without changing shared system packages.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
venv_dir="$repo_dir/.venv-hpc"
python_cmd=python3
requirements="$repo_dir/environment/requirements-hpc.txt"
while (($#)); do
  case "$1" in
    --venv) venv_dir=${2:?Missing --venv path}; shift 2 ;;
    --python) python_cmd=${2:?Missing Python command}; shift 2 ;;
    --requirements) requirements=${2:?Missing requirements file}; shift 2 ;;
    --help|-h) echo 'Usage: bash scripts/setup_hpc.sh [--venv PATH] [--python python3] [--requirements FILE]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Linux x86_64 is required.' >&2; exit 1; }
command -v "$python_cmd" >/dev/null || { echo 'Python is missing.' >&2; exit 1; }
"$python_cmd" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
[[ -f "$requirements" ]] || { echo 'Requirements file is missing.' >&2; exit 1; }
if [[ -e "$venv_dir" && ! -f "$venv_dir/pyvenv.cfg" ]]; then
  echo 'Existing target is not a virtual environment; refusing to replace it.' >&2; exit 1
fi
if [[ -f "$venv_dir/pyvenv.cfg" && ! -f "$venv_dir/.papermerge-hpc-environment" ]]; then
  echo 'Existing environment was not created by this setup script; choose a fresh path to protect legacy dependencies.' >&2; exit 1
fi
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
  "$python_cmd" -m venv "$venv_dir"
  touch "$venv_dir/.papermerge-hpc-environment"
fi
venv_python="$venv_dir/bin/python"
"$venv_python" -c 'import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3,12)'
"$venv_python" -m pip --require-virtualenv install -r "$requirements"
"$venv_python" -m pip check
CUDA_VISIBLE_DEVICES=-1 "$venv_python" -c 'import tensorflow as tf, keras, numpy; print("Installed:", tf.__version__, keras.__version__, numpy.__version__)'
"$venv_python" -m pip freeze > "$venv_dir/requirements-resolved.txt"
for tool in nvidia-smi tmux gcc g++; do
  if ! command -v "$tool" >/dev/null; then echo "Tool unavailable: $tool (request administrator assistance if needed)."; fi
done
printf 'Environment: %s\n' "$venv_dir"
echo 'Installation completed. Blackwell execution is NOT yet verified.'
echo 'Activate this environment and run scripts/hpc.py doctor --gpus ID before any experiment.'
