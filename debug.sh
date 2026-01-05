#!/usr/bin/env bash
set -euo pipefail

# Recreates the single-route job that run_eval_simlingo_local.py launches.

REPO_ROOT="/home/shaoyux/models/simlingo"
CARLA_ROOT="/home/shaoyux/software/carla0915"
SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"

ROUTE_FILE="${REPO_ROOT}/leaderboard/data/bench2drive_split/bench2drive_01.xml"
RESULT_FILE="${REPO_ROOT}/eval_results/Bench2Drive/simlingo/bench2drive/1/res/001_res.json"
LOG_FILE="${REPO_ROOT}/eval_results/Bench2Drive/simlingo/bench2drive/1/out/001_out.log"
ERR_FILE="${REPO_ROOT}/eval_results/Bench2Drive/simlingo/bench2drive/1/err/001_err.log"
VIZ_PATH="${REPO_ROOT}/eval_results/Bench2Drive/simlingo/bench2drive/1/viz/001"
AGENT_FILE="${REPO_ROOT}/team_code/agent_simlingo.py"
AGENT_CONFIG="${REPO_ROOT}/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

mkdir -p \
  "$(dirname "${RESULT_FILE}")" \
  "$(dirname "${LOG_FILE}")" \
  "$(dirname "${ERR_FILE}")" \
  "${VIZ_PATH}"

export CARLA_ROOT
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export SAVE_PATH="${VIZ_PATH}"

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
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" \
  1> >(tee "${LOG_FILE}") \
  2> >(tee "${ERR_FILE}" >&2)
