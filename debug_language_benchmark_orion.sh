#!/usr/bin/env bash
set -euo pipefail

# Debug script for Language-Following Benchmark — Orion agent
# Maps instruction command_id to Orion's ego_fut_cmd for trajectory mode selection.

REPO_ROOT="/home/nvidia/vla-project/simlingo"
CARLA_ROOT="/home/nvidia/software/carla0915"
SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"

# Select which benchmark to test
BENCHMARK_TYPE="instruction_following_v0.15_selected"
BENCHMARK_FILE="bench2drive_10_language_rebuilt_turn_left.xml"

ROUTE_FILE="${REPO_ROOT}/leaderboard/data/language_benchmark/${BENCHMARK_TYPE}/${BENCHMARK_FILE}"

# Extract route name for output directories
ROUTE_NAME=$(basename "${BENCHMARK_FILE}" .xml)
OUTPUT_BASE="${REPO_ROOT}/eval_results/LanguageBenchmark_Orion/${BENCHMARK_TYPE}/${ROUTE_NAME}"

RESULT_FILE="${OUTPUT_BASE}/res/${ROUTE_NAME}_res.json"
LOG_FILE="${OUTPUT_BASE}/out/${ROUTE_NAME}_out.log"
ERR_FILE="${OUTPUT_BASE}/err/${ROUTE_NAME}_err.log"
VIZ_PATH="${OUTPUT_BASE}/viz"

# Orion agent and config
AGENT_FILE="${REPO_ROOT}/Orion/team_code/orion_language_benchmark_agent.py"
AGENT_CONFIG="${REPO_ROOT}/Orion/adzoo/orion/configs/orion_stage3_agent.py+${REPO_ROOT}/Orion/ckpts/orion/Orion.pth+orion_benchmark"

# Check if route file exists
if [[ ! -f "${ROUTE_FILE}" ]]; then
    echo "Error: Route file not found: ${ROUTE_FILE}"
    echo ""
    echo "Available benchmark files:"
    find "${REPO_ROOT}/leaderboard/data/language_benchmark" -name "*.xml" -type f | sort | head -20
    exit 1
fi

echo "=============================================="
echo "Language-Following Benchmark (Orion) Debug"
echo "=============================================="
echo "Benchmark Type: ${BENCHMARK_TYPE}"
echo "Route File:     ${ROUTE_FILE}"
echo "Output Base:    ${OUTPUT_BASE}"
echo "Agent:          ${AGENT_FILE}"
echo "=============================================="

# Create output directories
mkdir -p \
  "$(dirname "${RESULT_FILE}")" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${ERR_FILE}")" \
  "${VIZ_PATH}"

# Source Orion environment (CUDA, LD_LIBRARY_PATH)
source "${REPO_ROOT}/Orion/orion_eval_env.sh"

# Set environment variables
export CARLA_ROOT
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export SAVE_PATH="${VIZ_PATH}"
export ROUTES="${ROUTE_FILE}"
export IS_BENCH2DRIVE=1  # Required for Orion's BEV camera sensor
export WORK_DIR="${REPO_ROOT}"

# Build PYTHONPATH
# Note: we include ${CARLA_ROOT}/PythonAPI/carla for the 'agents' module but
# NOT the .egg file — Orion's conda env has carla pip-installed.
PYTHONPATH_ENTRIES=(
  "${CARLA_ROOT}/PythonAPI/carla"
  "${REPO_ROOT}"
  "${REPO_ROOT}/Orion"
  "${REPO_ROOT}/Orion/Bench2DriveZoo"
  "${LEADERBOARD_ROOT}"
  "${SCENARIO_RUNNER_ROOT}"
)

if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi

export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

# Orion config uses relative paths (e.g. ckpts/pretrain_qformer/) that expect cwd=Orion/
cd "${REPO_ROOT}/Orion"

# Build command
CMD=(
  python -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py"
  "--routes=${ROUTE_FILE}"
  "--repetitions=1"
  "--track=SENSORS"
  "--checkpoint=${RESULT_FILE}"
  "--timeout=99999999"
  "--agent=${AGENT_FILE}"
  "--agent-config=${AGENT_CONFIG}"
  "--traffic-manager-seed=1"
  "--port=2000"
  "--traffic-manager-port=2500"
  "--gpu-rank=0"
)

echo ""
echo "Running command:"
echo "${CMD[*]}"
echo ""
echo "Logs will be saved to:"
echo "  stdout: ${LOG_FILE}"
echo "  stderr: ${ERR_FILE}"
echo ""

# Run the evaluation
"${CMD[@]}" \
  1> >(tee "${LOG_FILE}") \
  2> >(tee "${ERR_FILE}" >&2)
