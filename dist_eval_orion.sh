#!/usr/bin/env bash
set -euo pipefail

# Distributed Language-Following Benchmark Evaluation — Orion agent
# Evaluates all XML files in a directory using multiple GPUs in parallel.
#
# Usage:
#   ./dist_eval_orion.sh <benchmark_dir> [options]
#
# Options:
#   --output-dir <dir>    Output root (default: eval_results/LanguageBenchmark_Orion/<dir_name>)
#   --gpus <n>            Number of GPUs to use (default: auto-detect via nvidia-smi)
#   --max-parallel <n>    Max concurrent jobs (default: --gpus value)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="/home/nvidia/vla-project/simlingo"
CARLA_ROOT="/home/nvidia/software/carla0915"
SCENARIO_RUNNER_ROOT="${REPO_ROOT}/Bench2Drive/scenario_runner"
LEADERBOARD_ROOT="${REPO_ROOT}/Bench2Drive/leaderboard"

AGENT_FILE="${REPO_ROOT}/Orion/team_code/orion_language_benchmark_agent.py"
AGENT_CONFIG="${REPO_ROOT}/Orion/adzoo/orion/configs/orion_stage3_agent.py+${REPO_ROOT}/Orion/ckpts/orion/Orion.pth+orion_benchmark"

# Port base values per slot
BASE_CARLA_PORT=2000
BASE_TM_PORT=2500
PORT_STEP=100

MAX_RETRIES=2

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <benchmark_dir> [--output-dir <dir>] [--gpus <n>] [--gpu-ids <0,2,3,6>] [--max-parallel <n>]"
    exit 1
}

BENCHMARK_DIR=""
OUTPUT_DIR=""
NUM_GPUS=""
GPU_IDS=""
MAX_PARALLEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --gpus)         NUM_GPUS="$2";     shift 2 ;;
        --gpu-ids)      GPU_IDS="$2";      shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        -h|--help)      usage ;;
        -*)             echo "Unknown option: $1"; usage ;;
        *)
            if [[ -z "$BENCHMARK_DIR" ]]; then
                BENCHMARK_DIR="$1"
            else
                echo "Unexpected argument: $1"; usage
            fi
            shift ;;
    esac
done

[[ -z "$BENCHMARK_DIR" ]] && usage
[[ -d "$BENCHMARK_DIR" ]] || { echo "Error: directory not found: $BENCHMARK_DIR"; exit 1; }
BENCHMARK_DIR="$(realpath "$BENCHMARK_DIR")"

# Collect XML files
mapfile -t XML_FILES < <(find "${BENCHMARK_DIR}" -maxdepth 1 -name "*.xml" -type f | sort)
[[ ${#XML_FILES[@]} -gt 0 ]] || { echo "Error: no XML files found in ${BENCHMARK_DIR}"; exit 1; }

# GPU list: --gpu-ids takes precedence over --gpus
declare -a GPU_LIST
if [[ -n "$GPU_IDS" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_LIST[@]}
elif [[ -n "$NUM_GPUS" ]]; then
    for ((i = 0; i < NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    for ((i = 0; i < NUM_GPUS; i++)); do GPU_LIST+=("$i"); done
fi
MAX_PARALLEL="${MAX_PARALLEL:-${NUM_GPUS}}"

# Output root
BENCHMARK_NAME=$(basename "${BENCHMARK_DIR}")
OUTPUT_ROOT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/LanguageBenchmark_Orion/${BENCHMARK_NAME}}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source "${REPO_ROOT}/Orion/orion_eval_env.sh"

export CARLA_ROOT
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export IS_BENCH2DRIVE=1
export WORK_DIR="${REPO_ROOT}"
# Note: we include ${CARLA_ROOT}/PythonAPI/carla for the 'agents' module but
# NOT the .egg file — Orion's conda env has carla pip-installed.
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:\
${REPO_ROOT}:${REPO_ROOT}/Orion:${REPO_ROOT}/Orion/Bench2DriveZoo:\
${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}"
# Orion config uses relative paths (e.g. ckpts/pretrain_qformer/) that expect cwd=Orion/
cd "${REPO_ROOT}/Orion"

echo "=============================================="
echo "Distributed Language Benchmark (Orion)"
echo "=============================================="
echo "  Benchmark dir : ${BENCHMARK_DIR}"
echo "  Output root   : ${OUTPUT_ROOT}"
echo "  GPUs          : ${NUM_GPUS} (${GPU_LIST[*]})"
echo "  Max parallel  : ${MAX_PARALLEL}"
echo "  Jobs          : ${#XML_FILES[@]}"
echo "  Agent         : ${AGENT_FILE}"
echo "=============================================="
[[ -f "${AGENT_CONFIG%%+*}" ]] || echo "Warning: agent config not found: ${AGENT_CONFIG%%+*}"
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
            kill -TERM -"$pid" 2>/dev/null || true
            kill -9 -"$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Kill any remaining CARLA and evaluator processes spawned by us
    pkill -9 -f "CarlaUE4.*carla-rpc-port" 2>/dev/null || true
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true

    sleep 1

    killall -9 -r CarlaUE4-Linux 2>/dev/null || true

    echo "[$(date '+%H:%M:%S')] Cleanup complete."
    exit 130
}

trap cleanup SIGINT SIGTERM

launch_job() {
    local slot="$1"
    local route_file="$2"

    local gpu_rank=${GPU_LIST[$(( slot % NUM_GPUS ))]}
    local carla_port=$(( BASE_CARLA_PORT + slot * PORT_STEP ))
    local tm_port=$(( BASE_TM_PORT    + slot * PORT_STEP ))

    local route_name
    route_name=$(basename "${route_file}" .xml)

    local base="${OUTPUT_ROOT}/${route_name}"
    mkdir -p "${base}/res" "${base}/out" "${base}/err" "${base}/viz"

    echo "[$(date '+%H:%M:%S')] START  slot=${slot} gpu=${gpu_rank}  port=${carla_port}  ${route_name}"

    setsid \
    env CUDA_VISIBLE_DEVICES="${gpu_rank}" SAVE_PATH="${base}/viz" ROUTES="${route_file}" \
    python -u "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
        "--routes=${route_file}" \
        "--repetitions=1" \
        "--track=SENSORS" \
        "--checkpoint=${base}/res/${route_name}_res.json" \
        "--timeout=600" \
        "--agent=${AGENT_FILE}" \
        "--agent-config=${AGENT_CONFIG}" \
        "--traffic-manager-seed=1" \
        "--port=${carla_port}" \
        "--traffic-manager-port=${tm_port}" \
        "--gpu-rank=${gpu_rank}" \
        1>"${base}/out/${route_name}_out.log" \
        2>"${base}/err/${route_name}_err.log" &

    SLOT_PID[$slot]=$!
    SLOT_NAME[$slot]="${route_name}"
    SLOT_ROUTE[$slot]="${route_file}"
}

check_slot_result() {
    local slot="$1"
    local route_name="${SLOT_NAME[$slot]}"
    local route_file="${SLOT_ROUTE[$slot]}"
    local res_path="${OUTPUT_ROOT}/${route_name}/res/${route_name}_res.json"

    if [[ -z "$route_file" ]]; then
        return
    fi

    if [[ ! -s "$res_path" ]]; then
        local attempt="${ROUTE_RETRIES[$route_file]:-0}"
        if (( attempt < MAX_RETRIES )); then
            ROUTE_RETRIES["$route_file"]=$(( attempt + 1 ))
            RETRY_QUEUE+=("$route_file")
            echo "[$(date '+%H:%M:%S')] RETRY  ${route_name}  (attempt $((attempt+1))/${MAX_RETRIES}) — no result JSON produced" >&2
        else
            echo "[$(date '+%H:%M:%S')] FAILED ${route_name}  — no result after ${MAX_RETRIES} retries" >&2
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
# Submission loop
# ---------------------------------------------------------------------------
for route_file in "${XML_FILES[@]}"; do
    slot=$(wait_for_slot)
    launch_job "$slot" "$route_file"
done

echo ""
echo "All ${#XML_FILES[@]} jobs submitted — waiting for completion..."
drain_slots

# Retry queue
while (( ${#RETRY_QUEUE[@]} > 0 )); do
    retry_batch=("${RETRY_QUEUE[@]}")
    RETRY_QUEUE=()

    echo ""
    echo "[$(date '+%H:%M:%S')] Retrying ${#retry_batch[@]} failed route(s)..."

    for route_file in "${retry_batch[@]}"; do
        route_name=$(basename "${route_file}" .xml)
        rm -f "${OUTPUT_ROOT}/${route_name}/res/${route_name}_res.json"
        rm -f "${OUTPUT_ROOT}/${route_name}/out/${route_name}_out.log"
        rm -f "${OUTPUT_ROOT}/${route_name}/err/${route_name}_err.log"

        slot=$(wait_for_slot)
        launch_job "$slot" "$route_file"
    done

    echo "Retry batch submitted — waiting for completion..."
    drain_slots
done

echo "[$(date '+%H:%M:%S')] All jobs finished."

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Results"
echo "=============================================="

OUTPUT_ROOT="${OUTPUT_ROOT}" BENCHMARK_DIR="${BENCHMARK_DIR}" python3 - <<'PYEOF'
import os, json, sys

output_root  = os.environ["OUTPUT_ROOT"]
benchmark_dir = os.environ["BENCHMARK_DIR"]

xml_files = sorted(f for f in os.listdir(benchmark_dir) if f.endswith(".xml"))
ok, failed = [], []

for xml_file in xml_files:
    name = xml_file[:-4]
    res_path = os.path.join(output_root, name, "res", f"{name}_res.json")
    if not os.path.exists(res_path):
        failed.append(name)
        continue
    try:
        with open(res_path) as f:
            data = json.load(f)
        ok.append({"name": name, "data": data})
    except Exception as e:
        failed.append(name)
        print(f"  [PARSE ERROR] {name}: {e}", file=sys.stderr)

print(f"Parsed {len(ok)}/{len(xml_files)} result files\n")

scores = []
for r in ok:
    try:
        records = r["data"]["_checkpoint"]["records"]
        sc = None
        for rec in records:
            sc = rec.get("scores", {}).get("score_composed")
            if sc is not None:
                break
        label = f"{sc:.4f}" if sc is not None else "n/a"
        print(f"  {r['name']:<60}  score_composed={label}")
        if sc is not None:
            scores.append(sc)
    except Exception as e:
        print(f"  [SCORE ERROR] {r['name']}: {e}")

if scores:
    mean = sum(scores) / len(scores)
    print(f"\n  {'--- Summary ---':<60}")
    print(f"  {'Mean':<60}  {mean:.4f}")
    print(f"  {'Min':<60}  {min(scores):.4f}")
    print(f"  {'Max':<60}  {max(scores):.4f}")
    print(f"  N = {len(scores)}")
else:
    print("\n  No score_composed values found in results.")

if failed:
    print(f"\n  Missing / failed ({len(failed)}): {', '.join(failed)}")
PYEOF
