#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

VERSION="${VERSION:-v0.16}"
SIMLINGO_ENV_NAME="${SIMLINGO_ENV_NAME:-simlingo}"
DEFAULT_CARLA_ROOT="/home/nvidia/software/carla0915"

if [[ -z "${CARLA_ROOT:-}" && -d "${DEFAULT_CARLA_ROOT}" ]]; then
    export CARLA_ROOT="${DEFAULT_CARLA_ROOT}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${CARLA_ROOT:-}" ]]; then
    CARLA_PYTHON_API_DIR="${CARLA_ROOT}/PythonAPI/carla"
    CARLA_EGG_PATH="${CARLA_PYTHON_API_DIR}/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    if [[ -d "${CARLA_PYTHON_API_DIR}" ]]; then
        export PYTHONPATH="${CARLA_PYTHON_API_DIR}:${PYTHONPATH}"
    fi
    if [[ -f "${CARLA_EGG_PATH}" ]]; then
        export PYTHONPATH="${CARLA_EGG_PATH}:${PYTHONPATH}"
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
USE_CONDA_RUN=0
CURRENT_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || printf 'unknown')"

if [[ "${CURRENT_PYTHON_VERSION}" == "3.7" || "${CURRENT_PYTHON_VERSION}" == "3.8" ]] && \
   "${PYTHON_BIN}" -c "import carla" >/dev/null 2>&1; then
    :
else
    if command -v conda >/dev/null 2>&1 && \
       conda run -n "${SIMLINGO_ENV_NAME}" python -c "import carla" >/dev/null 2>&1; then
        USE_CONDA_RUN=1
    else
        echo "Unable to import the CARLA Python API." >&2
        echo "Set CARLA_ROOT/PYTHONPATH correctly or activate the '${SIMLINGO_ENV_NAME}' env." >&2
        exit 1
    fi
fi

run_python() {
    if [[ "${USE_CONDA_RUN}" -eq 1 ]]; then
        conda run -n "${SIMLINGO_ENV_NAME}" python "$@"
    else
        "${PYTHON_BIN}" "$@"
    fi
}

BENCH2DRIVE_SPLIT_DIR="${REPO_ROOT}/leaderboard/data/bench2drive_split"
OUTPUT_DIR="${REPO_ROOT}/leaderboard/data/language_benchmark/instruction_following_${VERSION}"
SELECTED_DIR="${OUTPUT_DIR}_selected"
SUBSET_DIR="${OUTPUT_DIR}_subset"

run_python "${REPO_ROOT}/language_navigation/generate_language_xml_route.py" \
    "${BENCH2DRIVE_SPLIT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --force-all-green-traffic-lights \
    --seed 42

run_python "${REPO_ROOT}/language_navigation/copy_selected_routes.py" \
    --select-file "${REPO_ROOT}/language_navigation/route.txt" \
    --source-dir "${OUTPUT_DIR}" \
    --output-dir "${SELECTED_DIR}"

run_python "${REPO_ROOT}/language_navigation/copy_selected_routes.py" \
    --select-file "${REPO_ROOT}/language_navigation/route_subset.txt" \
    --source-dir "${OUTPUT_DIR}" \
    --output-dir "${SUBSET_DIR}"

# Debug visualization stays intentionally opt-in.
# mkdir -p "${REPO_ROOT}/debug/${VERSION}"
# run_python "${REPO_ROOT}/language_navigation/route_xml_bev.py" \
#     --input-dir "${SELECTED_DIR}" \
#     --output "${REPO_ROOT}/debug/${VERSION}"
