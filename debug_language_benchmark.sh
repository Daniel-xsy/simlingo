#!/usr/bin/env bash
set -euo pipefail

# Debug script for Language-Following Benchmark
# Tests instruction-following capability of the agent

REPO_ROOT="/home/nvidia/vla-project/simlingo"
CARLA_ROOT="/home/nvidia/software/carla0915"
SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"

# Select which benchmark to test (can be changed)
# Options:
#   lateral_control/lane_change_left_001.xml
#   lateral_control/lane_change_right_001.xml
#   lateral_control/multi_lane_change_001.xml
#   speed_control/speed_5ms_001.xml
#   speed_control/speed_10ms_001.xml
#   speed_control/speed_15ms_001.xml
#   speed_control/speed_transition_001.xml
#   unsafe_commands/accelerate_during_cutin_001.xml
#   unsafe_commands/accelerate_during_cutin_002.xml
#   unsafe_commands/ignore_stop_sign_001.xml

BENCHMARK_TYPE="lateral_control"
BENCHMARK_FILE="lane_change_left_002.xml"

ROUTE_FILE="${REPO_ROOT}/leaderboard/data/language_benchmark/${BENCHMARK_TYPE}/${BENCHMARK_FILE}"

# Extract route name for output directories
ROUTE_NAME=$(basename "${BENCHMARK_FILE}" .xml)
OUTPUT_BASE="${REPO_ROOT}/eval_results/LanguageBenchmark/${BENCHMARK_TYPE}/${ROUTE_NAME}"

RESULT_FILE="${OUTPUT_BASE}/res/${ROUTE_NAME}_res.json"
LOG_FILE="${OUTPUT_BASE}/out/${ROUTE_NAME}_out.log"
ERR_FILE="${OUTPUT_BASE}/err/${ROUTE_NAME}_err.log"
VIZ_PATH="${OUTPUT_BASE}/viz"

AGENT_FILE="${REPO_ROOT}/team_code/agent_simlingo_language_benchmark.py"
AGENT_CONFIG="${REPO_ROOT}/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

# Check if route file exists
if [[ ! -f "${ROUTE_FILE}" ]]; then
    echo "Error: Route file not found: ${ROUTE_FILE}"
    echo ""
    echo "Available benchmark files:"
    find "${REPO_ROOT}/leaderboard/data/language_benchmark" -name "*.xml" -type f | sort
    exit 1
fi

# Check if agent config exists
if [[ ! -f "${AGENT_CONFIG}" ]]; then
    echo "Warning: Agent config not found: ${AGENT_CONFIG}"
    echo "You may need to adjust the AGENT_CONFIG path."
fi

echo "=============================================="
echo "Language-Following Benchmark Debug Script"
echo "=============================================="
echo "Benchmark Type: ${BENCHMARK_TYPE}"
echo "Route File:     ${ROUTE_FILE}"
echo "Output Base:    ${OUTPUT_BASE}"
echo "=============================================="

# Create output directories
mkdir -p \
  "$(dirname "${RESULT_FILE}")" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${ERR_FILE}")" \
  "${VIZ_PATH}"

# Set environment variables
export CARLA_ROOT
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export SAVE_PATH="${VIZ_PATH}"
export ROUTES="${ROUTE_FILE}"  # Used by language benchmark agent to parse instructions

# Build PYTHONPATH
PYTHONPATH_ENTRIES=(
  "${CARLA_ROOT}/PythonAPI/carla"
  "${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
  "${REPO_ROOT}"
  "${LEADERBOARD_ROOT}"
  "${SCENARIO_RUNNER_ROOT}"
)

if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi

export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

cd "${REPO_ROOT}"

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
  # "--debugpy-port=5678"
  # "--debugpy-wait"  # Wait for debugger to attach
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
