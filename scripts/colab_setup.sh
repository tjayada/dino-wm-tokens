#!/usr/bin/env bash
# Install the project on a Google Colab runtime. Usage: bash scripts/colab_setup.sh
set -euo pipefail
apt-get install -y -qq swig > /dev/null
python -m pip install -q uv
if python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)'; then
  # labmaze (a dm_control dependency used only for maze arenas) has no wheels for Python 3.13+
  stub=$(mktemp -d)/labmaze
  mkdir -p "$stub/labmaze" && echo "# stand-in" > "$stub/labmaze/__init__.py"
  printf '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n[project]\nname = "labmaze"\nversion = "1.0.6"\n' > "$stub/pyproject.toml"
  python -m uv pip install -q --system "$stub"
fi
python -m uv pip install -q --system -e .
echo "installed on python $(python -c 'import platform; print(platform.python_version())')"
