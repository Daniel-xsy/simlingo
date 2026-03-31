#!/usr/bin/env bash
set -euo pipefail

# Distributed Bench2Drive 220-Route Evaluation (Local, non-SLURM)
# Evaluates all 220 pre-split route XMLs using multiple GPUs in parallel.
#
# Usage:
#   ./dist_eval_bench2drive.sh [options]
#
# Options:
#   --route-dir <dir>     Route XMLs directory (default: leaderboard/data/bench2drive_split)
#   --output-dir <dir>    Output root (default: eval_results/Bench2Drive/simlingo)
#   --checkpoint <path>   Model checkpoint path
#   --gpus <n>            Number of GPUs to use (default: auto-detect)
#   --max-parallel <n>    Max concurrent jobs (default: --gpus value)
#   --seed <n>            Traffic manager seed (default: 1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="/home/nvidia/vla-project/simlingo"
CARLA_ROOT="/home/nvidia/software/carla0915"
SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"

AGENT_FILE="${REPO_ROOT}/team_code/agent_simlingo.py"
AGENT_CONFIG="${REPO_ROOT}/ckpts/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

# Port base values per slot: slot i gets BASE + i*STEP
BASE_CARLA_PORT=2000
BASE_TM_PORT=2500
PORT_STEP=100

# Retry configuration
MAX_RETRIES=2

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [--route-dir <dir>] [--output-dir <dir>] [--checkpoint <path>] [--gpus <n>] [--max-parallel <n>] [--seed <n>]"
    exit 1
}

ROUTE_DIR="${REPO_ROOT}/leaderboard/data/bench2drive_split"
OUTPUT_DIR=""
NUM_GPUS=""
MAX_PARALLEL=""
SEED=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --route-dir)    ROUTE_DIR="$2";     shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --checkpoint)   AGENT_CONFIG="$2";  shift 2 ;;
        --gpus)         NUM_GPUS="$2";      shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2";  shift 2 ;;
        --seed)         SEED="$2";          shift 2 ;;
        -h|--help)      usage ;;
        -*)             echo "Unknown option: $1"; usage ;;
        *)              echo "Unexpected argument: $1"; usage ;;
    esac
done

[[ -d "$ROUTE_DIR" ]] || { echo "Error: route directory not found: $ROUTE_DIR"; exit 1; }
ROUTE_DIR="$(realpath "$ROUTE_DIR")"

# Collect XML files
mapfile -t XML_FILES < <(find "${ROUTE_DIR}" -maxdepth 1 -name "*.xml" -type f | sort)
[[ ${#XML_FILES[@]} -gt 0 ]] || { echo "Error: no XML files found in ${ROUTE_DIR}"; exit 1; }

# GPU count
if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
fi
MAX_PARALLEL="${MAX_PARALLEL:-${NUM_GPUS}}"

# Output root
OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/Bench2Drive/simlingo/seed_${SEED}}"
RES_DIR="${OUTPUT_ROOT}/res"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export CARLA_ROOT
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:\
${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:\
${REPO_ROOT}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}" "${RES_DIR}"
mkdir -p "${OUTPUT_ROOT}/out" "${OUTPUT_ROOT}/err" "${OUTPUT_ROOT}/viz"
cd "${REPO_ROOT}"

echo "=============================================="
echo "Distributed Bench2Drive Evaluation"
echo "=============================================="
echo "  Route dir     : ${ROUTE_DIR}"
echo "  Output root   : ${OUTPUT_ROOT}"
echo "  Checkpoint    : ${AGENT_CONFIG}"
echo "  GPUs          : ${NUM_GPUS}"
echo "  Max parallel  : ${MAX_PARALLEL}"
echo "  Seed          : ${SEED}"
echo "  Routes        : ${#XML_FILES[@]}"
echo "=============================================="
[[ -f "${AGENT_CONFIG}" ]] || { echo "Error: checkpoint not found: ${AGENT_CONFIG}"; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# Slot management
# ---------------------------------------------------------------------------
declare -a SLOT_PID
declare -a SLOT_NAME
declare -a SLOT_ROUTE
for ((i = 0; i < MAX_PARALLEL; i++)); do
    SLOT_PID[$i]=""
    SLOT_NAME[$i]=""
    SLOT_ROUTE[$i]=""
done

declare -A ROUTE_RETRIES
declare -a RETRY_QUEUE

# ---------------------------------------------------------------------------
# Cleanup on Ctrl+C / SIGTERM: kill all child processes and CARLA instances
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Caught signal — cleaning up all processes..."

    # Kill all tracked slot PIDs and their process trees
    for ((i = 0; i < MAX_PARALLEL; i++)); do
        local pid="${SLOT_PID[$i]}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            # Kill the entire process group rooted at this PID
            kill -TERM -"$pid" 2>/dev/null || true
            kill -9 -"$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Kill any remaining CARLA and evaluator processes spawned by us
    pkill -9 -f "CarlaUE4.*carla-rpc-port" 2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true

    # Wait briefly for processes to die
    sleep 1

    # Final check — force kill any survivors
    killall -9 -r CarlaUE4-Linux 2>/dev/null || true

    echo "[$(date '+%H:%M:%S')] Cleanup complete."
    exit 130
}

trap cleanup SIGINT SIGTERM

launch_job() {
    local slot="$1"
    local route_file="$2"

    local gpu_rank=$(( slot % NUM_GPUS ))
    local carla_port=$(( BASE_CARLA_PORT + slot * PORT_STEP ))
    local tm_port=$(( BASE_TM_PORT    + slot * PORT_STEP ))

    local route_name
    route_name=$(basename "${route_file}" .xml)

    # Route ID for result file naming (matches merge_route_json.py expectations)
    local route_id="${route_name}"

    local viz_path="${OUTPUT_ROOT}/viz/${route_id}"
    mkdir -p "${viz_path}"

    echo "[$(date '+%H:%M:%S')] START  slot=${slot} gpu=${gpu_rank}  port=${carla_port}  ${route_name}"

    setsid \
    env CUDA_VISIBLE_DEVICES="${gpu_rank}" SAVE_PATH="${viz_path}" \
    python -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
        "--routes=${route_file}" \
        "--repetitions=1" \
        "--track=SENSORS" \
        "--checkpoint=${RES_DIR}/${route_id}_res.json" \
        "--timeout=600" \
        "--agent=${AGENT_FILE}" \
        "--agent-config=${AGENT_CONFIG}" \
        "--traffic-manager-seed=${SEED}" \
        "--port=${carla_port}" \
        "--traffic-manager-port=${tm_port}" \
        "--gpu-rank=${gpu_rank}" \
        1>"${OUTPUT_ROOT}/out/${route_id}_out.log" \
        2>"${OUTPUT_ROOT}/err/${route_id}_err.log" &

    SLOT_PID[$slot]=$!
    SLOT_NAME[$slot]="${route_name}"
    SLOT_ROUTE[$slot]="${route_file}"
}

check_slot_result() {
    local slot="$1"
    local route_name="${SLOT_NAME[$slot]}"
    local route_file="${SLOT_ROUTE[$slot]}"
    local res_path="${RES_DIR}/${route_name}_res.json"

    if [[ -z "$route_file" ]]; then
        return
    fi

    local need_retry=0
    if [[ ! -s "$res_path" ]]; then
        need_retry=1
    else
        # Check for failed status in result JSON
        if python3 -c "
import json, sys
try:
    data = json.load(open('${res_path}'))
    progress = data['_checkpoint']['progress']
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    for rec in data['_checkpoint']['records']:
        if 'Failed' in rec.get('status', ''):
            sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            need_retry=0
        else
            need_retry=1
        fi
    fi

    if (( need_retry )); then
        local attempt="${ROUTE_RETRIES[$route_file]:-0}"
        if (( attempt < MAX_RETRIES )); then
            ROUTE_RETRIES["$route_file"]=$(( attempt + 1 ))
            RETRY_QUEUE+=("$route_file")
            echo "[$(date '+%H:%M:%S')] RETRY  ${route_name}  (attempt $((attempt+1))/${MAX_RETRIES})" >&2
        else
            echo "[$(date '+%H:%M:%S')] FAILED ${route_name}  — giving up after ${MAX_RETRIES} retries" >&2
        fi
    fi
}

wait_for_slot() {
    while true; do
        for ((i = 0; i < MAX_PARALLEL; i++)); do
            local pid="${SLOT_PID[$i]}"
            if [[ -z "$pid" ]]; then
                echo "$i"; return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[$(date '+%H:%M:%S')] DONE   slot=${i}  ${SLOT_NAME[$i]}" >&2
                check_slot_result "$i"
                SLOT_PID[$i]=""
                SLOT_NAME[$i]=""
                SLOT_ROUTE[$i]=""
                echo "$i"; return
            fi
        done
        sleep 2
    done
}

drain_slots() {
    for ((i = 0; i < MAX_PARALLEL; i++)); do
        local pid="${SLOT_PID[$i]}"
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
            if [[ -n "${SLOT_NAME[$i]}" ]]; then
                echo "[$(date '+%H:%M:%S')] DONE   slot=${i}  ${SLOT_NAME[$i]}" >&2
                check_slot_result "$i"
                SLOT_PID[$i]=""
                SLOT_NAME[$i]=""
                SLOT_ROUTE[$i]=""
            fi
        fi
    done
}

# ---------------------------------------------------------------------------
# Resume: filter out routes that already have valid results
# ---------------------------------------------------------------------------
is_route_complete() {
    local res_path="$1"
    [[ -s "$res_path" ]] || return 1
    python3 -c "
import json, sys
try:
    data = json.load(open('${res_path}'))
    records = data['_checkpoint']['records']
    progress = data['_checkpoint']['progress']
    # Must have completed progress
    if len(progress) < 2 or progress[0] < progress[1]:
        sys.exit(1)
    # Only retry 'Agent crashed' — other failures (blocked, timed out,
    # deviated) still produce valid scores that merge_route_json.py uses.
    for rec in records:
        if rec.get('status', '') == 'Failed - Agent crashed':
            sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

PENDING_FILES=()
SKIPPED=0
for route_file in "${XML_FILES[@]}"; do
    route_name=$(basename "${route_file}" .xml)
    res_path="${RES_DIR}/${route_name}_res.json"
    if is_route_complete "$res_path"; then
        SKIPPED=$((SKIPPED + 1))
    else
        PENDING_FILES+=("$route_file")
    fi
done

echo "  Skipped (done) : ${SKIPPED}"
echo "  Pending        : ${#PENDING_FILES[@]}"
echo "=============================================="

if (( ${#PENDING_FILES[@]} == 0 )); then
    echo "All routes already completed — nothing to do."
else

# ---------------------------------------------------------------------------
# Submission loop
# ---------------------------------------------------------------------------
for route_file in "${PENDING_FILES[@]}"; do
    slot=$(wait_for_slot)
    launch_job "$slot" "$route_file"
done

echo ""
echo "All ${#PENDING_FILES[@]} jobs submitted — waiting for completion..."
drain_slots

# Process retry queue
while (( ${#RETRY_QUEUE[@]} > 0 )); do
    retry_batch=("${RETRY_QUEUE[@]}")
    RETRY_QUEUE=()

    echo ""
    echo "[$(date '+%H:%M:%S')] Retrying ${#retry_batch[@]} failed route(s)..."

    for route_file in "${retry_batch[@]}"; do
        route_name=$(basename "${route_file}" .xml)
        rm -f "${RES_DIR}/${route_name}_res.json"

        slot=$(wait_for_slot)
        launch_job "$slot" "$route_file"
    done

    echo "Retry batch submitted — waiting for completion..."
    drain_slots
done

echo "[$(date '+%H:%M:%S')] All jobs finished."

fi  # end of pending routes check

# ---------------------------------------------------------------------------
# Merge results using official Bench2Drive tool
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Merging results with Bench2Drive merge tool..."
echo "=============================================="

python "${REPO_ROOT}/Bench2Drive/tools/merge_route_json.py" -f "${RES_DIR}"

MERGED="${RES_DIR}/merged.json"
if [[ -f "$MERGED" ]]; then
    echo ""
    python3 -c "
import json
with open('${MERGED}') as f:
    data = json.load(f)
print(f\"  Driving Score : {data.get('driving score', 'N/A'):.4f}\")
print(f\"  Success Rate  : {data.get('success rate', 'N/A'):.4f}\")
print(f\"  Eval Routes   : {data.get('eval num', 'N/A')}\")
"
else
    echo "Warning: merged.json not found — check individual results in ${RES_DIR}/"
fi
