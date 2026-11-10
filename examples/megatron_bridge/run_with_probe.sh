#!/bin/bash
# Generic hcu_probe entry point for Megatron-Bridge examples.
#
# Every model family keeps its own profile under profiles/; this script only
# handles the parts that are identical for all of them: argument parsing,
# pre-flight checks, probe activation, single/multi-node launch.
#
# Usage:
#   ./run_with_probe.sh <profile> [recipe] [mode] [hostfile] [nodes]
#
#   profile   name of a file in profiles/ without the .sh suffix,
#             e.g. gemma3_pretrain, glm52_pretrain, qwen35_vl_pretrain
#   recipe    recipe name; defaults to the profile's PROFILE_DEFAULT_RECIPE
#   mode      dryrun | train | sft   (default: train)
#   hostfile  node list for multi-node runs
#   nodes     number of nodes to use (requires hostfile)
#
# Environment:
#   MEGATRON_PATH                 Bridge checkout root (required)
#   HCU_PROBE_CONFIG              dump config, enables dump
#   HCU_PROBE_MONITOR_CONFIG      monitor config, enables monitor
#   HCU_PROBE_SEED                fix the random seed
#   HCU_PROBE_REPRODUCE           deterministic mode
#   HCU_PROBE_STRICT              abort training if probe fails to initialize
#   PROBE_ALLOW_UNVERIFIED        set to 1 to run a profile marked unverified
#
# See README.md for the full list of profiles and their status.
set -euo pipefail

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
HCU_PROBE_ROOT=$( cd "${CURRENT_DIR}/../.." && pwd )

usage() {
    cat <<'USAGE'
Usage: ./run_with_probe.sh <profile> [recipe] [mode] [hostfile] [nodes]

Profiles (profiles/*.sh):
  gemma3_pretrain            gemma4_pretrain
  glm52_pretrain             glm52_sft            glm53_pretrain
  minimax_m3_pretrain
  qwen35_text_pretrain       qwen35_vl_pretrain   qwen38_2_4t_pretrain
  deepseek_v4_flash_pretrain deepseek_v4_flash_sft deepseek_v4_pro_pretrain

Modes: dryrun (resolve config only) | train | sft
USAGE
}

case "${1:-}" in
    "")
        usage
        exit 1
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
esac

PROFILE=${1}
RECIPE=${2:-}
MODE=${3:-train}
HOSTFILE=${4:-}
NODES=${5:-}

PROFILE_FILE="${CURRENT_DIR}/profiles/${PROFILE}.sh"
if [[ ! -f "${PROFILE_FILE}" ]]; then
    echo "ERROR: unknown profile '${PROFILE}'." >&2
    echo "Available profiles:" >&2
    for f in "${CURRENT_DIR}"/profiles/*.sh; do
        name=$(basename "${f}" .sh)
        [[ ${name} == _* ]] && continue
        echo "  - ${name}" >&2
    done
    exit 1
fi

MEGATRON_PATH=${MEGATRON_PATH:-${MEGATRON_BRIDGE_ROOT:-}}
if [[ -z "${MEGATRON_PATH}" ]]; then
    echo "ERROR: set MEGATRON_PATH to the Megatron-bridge-das root." >&2
    exit 1
fi
MEGATRON_PATH=$( cd "${MEGATRON_PATH}" && pwd )

source "${CURRENT_DIR}/source_bridge_env.sh"
source "${CURRENT_DIR}/profiles/_common.sh"
source "${PROFILE_FILE}"

RECIPE=${RECIPE:-${PROFILE_DEFAULT_RECIPE}}
if [[ -z "${RECIPE}" ]]; then
    echo "ERROR: no recipe given and ${PROFILE} has no PROFILE_DEFAULT_RECIPE." >&2
    exit 1
fi

if [[ ! " ${PROFILE_RECIPES[*]} " =~ " ${RECIPE} " ]]; then
    echo "ERROR: recipe '${RECIPE}' is not listed for profile '${PROFILE}'." >&2
    echo "Supported:" >&2
    printf "  - %s\n" "${PROFILE_RECIPES[@]}" >&2
    exit 1
fi

# Bridge mode: pretrain recipes run in "pretrain" mode, SFT recipes in "sft".
BRIDGE_MODE=${PROFILE_BRIDGE_MODE:-pretrain}
case "${MODE}" in
    dryrun|train|sft) ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use dryrun, train or sft." >&2
        exit 1
        ;;
esac
[[ ${MODE} == sft ]] && BRIDGE_MODE=sft

if [[ ${PROFILE_VERIFIED:-true} != true && ${PROBE_ALLOW_UNVERIFIED:-0} != 1 ]]; then
    echo "======================================================"
    echo "WARNING: profile '${PROFILE}' is not verified on this fork."
    echo "It was derived from the Bridge example script but has not been"
    echo "run end-to-end with hcu_probe. Set PROBE_ALLOW_UNVERIFIED=1 to"
    echo "proceed anyway."
    echo "======================================================"
    exit 1
fi

# ---------------------------------------------------------------- topology --
GPUS="${GPUS:-8}"
PORT="${PORT:-11453}"
MASTER_ADDR=127.0.0.1
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ -n "${HOSTFILE}" ]]; then
    [[ -f "${HOSTFILE}" ]] || { echo "ERROR: ${HOSTFILE} not found." >&2; exit 1; }
    HOSTFILE=$( cd "$(dirname "${HOSTFILE}")" && pwd )/$(basename "${HOSTFILE}")
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' "${HOSTFILE}")
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES > AVAILABLE_NODES )); then
        echo "ERROR: node count ${NNODES} exceeds ${AVAILABLE_NODES} host(s)." >&2
        exit 1
    fi
    MASTER_ADDR=$(grep -ve '^[[:space:]]*$' "${HOSTFILE}" | sed -n 1p | awk '{print $1}')
else
    NNODES=${NODES:-1}
    if (( NNODES > 1 )); then
        echo "ERROR: multi-node run requires a hostfile." >&2
        exit 1
    fi
fi

# Let the profile adapt parallelism to the chosen recipe (test vs full).
if declare -F profile_configure > /dev/null; then
    profile_configure
fi

TP=${TENSOR_PARALLEL:-${PROFILE_TP:-1}}
PP=${PIPELINE_PARALLEL:-${PROFILE_PP:-1}}
EP=${EXPERT_PARALLEL:-${PROFILE_EP:-1}}
ETP=${EXPERT_TENSOR_PARALLEL:-${PROFILE_ETP:-1}}
CP=${CONTEXT_PARALLEL:-${PROFILE_CP:-1}}
SEQ_LENGTH=${SEQ_LENGTH:-${PROFILE_SEQ_LENGTH:-4096}}

TRAIN_ITERS=${TRAIN_ITERS:-${PROFILE_TRAIN_ITERS:-50}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${PROFILE_GLOBAL_BATCH_SIZE:-32}}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${PROFILE_MICRO_BATCH_SIZE:-1}}

export LOCAL_HF_PATH=${LOCAL_HF_PATH:-${PROFILE_LOCAL_HF_PATH:-}}
if [[ -z "${LOCAL_HF_PATH}" ]]; then
    echo "ERROR: set LOCAL_HF_PATH to the local HF checkpoint for ${RECIPE}." >&2
    exit 1
fi

# Profile-specific environment (tokenizer options, queue counts, ...).
for export_line in ${PROFILE_EXPORTS:-}; do
    [[ -n "${export_line}" ]] && export "${export_line}"
done

SAVE_DIR=${SAVE_DIR:-${CURRENT_DIR}/${PROFILE}_${RECIPE}_tp${TP}_pp${PP}_ep${EP}_cp${CP}}
mkdir -p "${SAVE_DIR}/train_logs"

CONFIG=(
    dataset.seq_length="${SEQ_LENGTH}"
    ${PROFILE_DATA_CONFIG[@]+"${PROFILE_DATA_CONFIG[@]}"}
    train.train_iters="${TRAIN_ITERS}"
    train.global_batch_size="${GLOBAL_BATCH_SIZE}"
    train.micro_batch_size="${MICRO_BATCH_SIZE}"
    ${PROFILE_TRAIN_CONFIG[@]+"${PROFILE_TRAIN_CONFIG[@]}"}
    model.tensor_model_parallel_size="${TP}"
    model.pipeline_model_parallel_size="${PP}"
    model.context_parallel_size="${CP}"
    model.seq_length="${SEQ_LENGTH}"
    ${PROFILE_PARALLEL_CONFIG[@]+"${PROFILE_PARALLEL_CONFIG[@]}"}
    ${PROFILE_MODEL_CONFIG[@]+"${PROFILE_MODEL_CONFIG[@]}"}
    checkpoint.load=null
    checkpoint.save=null
    logger.log_interval=1
    logger.tensorboard_dir="${SAVE_DIR}/tb_logs"
    logger.save_config_filepath="${SAVE_DIR}/${RECIPE}.yaml"
)

PROBE_ARGS=()
WANT_PROBE=0
if [[ -n "${HCU_PROBE_CONFIG:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-config "${HCU_PROBE_CONFIG}")
    WANT_PROBE=1
fi
if [[ -n "${HCU_PROBE_MONITOR_CONFIG:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-monitor-config "${HCU_PROBE_MONITOR_CONFIG}")
    WANT_PROBE=1
fi
if [[ -n "${HCU_PROBE_SEED:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-seed "${HCU_PROBE_SEED}")
fi
if [[ -n "${HCU_PROBE_REPRODUCE:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-reproduce)
fi
if [[ -n "${HCU_PROBE_STRICT:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-strict)
fi

LAUNCH_ARGS=(
    --bridge-root "${MEGATRON_PATH}"
    "${PROBE_ARGS[@]}"
    --recipe "${RECIPE}"
    --mode "${BRIDGE_MODE}"
    "${CONFIG[@]}"
    "${PROFILE_EXTRA_ARGS[@]}"
)

# ------------------------------------------------------ pre-flight checks --
check_probe_config() {
    local path=$1 label=$2
    [[ -f "${path}" ]] || { echo "ERROR: ${label} not found: ${path}" >&2; exit 1; }
    python -m json.tool "${path}" > /dev/null \
        || { echo "ERROR: ${label} is not valid JSON: ${path}" >&2; exit 1; }
}

if (( WANT_PROBE )); then
    [[ -n "${HCU_PROBE_CONFIG:-}" ]] && check_probe_config "${HCU_PROBE_CONFIG}" "HCU_PROBE_CONFIG"
    [[ -n "${HCU_PROBE_MONITOR_CONFIG:-}" ]] && check_probe_config "${HCU_PROBE_MONITOR_CONFIG}" "HCU_PROBE_MONITOR_CONFIG"
    python -c "import matplotlib, tensorboard" 2>/dev/null || {
        echo "ERROR: missing probe dependencies. Install with:" >&2
        echo "  pip install matplotlib tensorboard" >&2
        exit 1
    }
    python -c "from hcuprobe.pytorch import PrecisionDebugger" 2>/dev/null || {
        echo "ERROR: hcuprobe.pytorch is not importable; check PYTHONPATH." >&2
        exit 1
    }
fi

echo "======================================"
echo "${PROFILE_NAME:-${PROFILE}}"
echo "Mode:          ${MODE} (bridge mode: ${BRIDGE_MODE})"
echo "Recipe:        ${RECIPE}"
echo "Verified:      ${PROFILE_VERIFIED:-true}"
echo "Nodes:         ${NNODES}"
echo "GPUs:          ${GPUS}/node, $((NNODES * GPUS)) total (TP=${TP}, PP=${PP}, EP=${EP}, CP=${CP})"
echo "Iters:         ${TRAIN_ITERS}"
echo "GBS/MBS:       ${GLOBAL_BATCH_SIZE} / ${MICRO_BATCH_SIZE}"
echo "Seq length:    ${SEQ_LENGTH}"
echo "Local HF path: ${LOCAL_HF_PATH}"
echo "Save dir:      ${SAVE_DIR}"
echo "Dump cfg:      ${HCU_PROBE_CONFIG:-<disabled>}"
echo "Monitor cfg:   ${HCU_PROBE_MONITOR_CONFIG:-<disabled>}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

if ! (( WANT_PROBE )); then
    echo "NOTE: no probe config set; training runs without hcu_probe."
fi

# --------------------------------------------------------------- execution --
case "${MODE}" in
    dryrun)
        source /opt/dtk/env.sh
        source_bridge_env "${MEGATRON_PATH}"
        export PYTHONPATH="${HCU_PROBE_ROOT}/python:${PROFILE_EXTRA_PYTHONPATH:-}:${PYTHONPATH:-}"
        python -m hcuprobe.integrations.megatron_bridge.launcher \
            "${LAUNCH_ARGS[@]}" \
            --dry-run
        echo "Dry run complete. Resolved config: ${SAVE_DIR}/${RECIPE}.yaml"
        ;;

    train|sft)
        TORCHRUN_ARGS=(
            --nnodes="${NNODES}"
            --nproc_per_node="${GPUS}"
            --master_addr="${MASTER_ADDR}"
            --master_port="${MASTER_PORT}"
        )
        LOG_FILE="${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"

        if (( NNODES == 1 )); then
            source /opt/dtk/env.sh
            source_bridge_env "${MEGATRON_PATH}"
            export PYTHONPATH="${HCU_PROBE_ROOT}/python:${PROFILE_EXTRA_PYTHONPATH:-}:${PYTHONPATH:-}"
            torchrun "${TORCHRUN_ARGS[@]}" \
                -m hcuprobe.integrations.megatron_bridge.launcher \
                "${LAUNCH_ARGS[@]}" \
                2>&1 | tee "${LOG_FILE}"
        else
            HOSTFILE_SLOTS="${CURRENT_DIR}/hostfile_slots"
            grep -ve '^[[:space:]]*$' "${HOSTFILE}" | sed -n "1,${NNODES}p" \
                | awk '{print $1" slots=1"}' > "${HOSTFILE_SLOTS}"

            TORCHRUN_QUOTED="$(printf '%q ' "${TORCHRUN_ARGS[@]}")"
            LAUNCH_QUOTED="$(printf '%q ' "${LAUNCH_ARGS[@]}")"
            REMOTE_CMD="torchrun --node_rank=\${OMPI_COMM_WORLD_RANK} ${TORCHRUN_QUOTED} -m hcuprobe.integrations.megatron_bridge.launcher ${LAUNCH_QUOTED}"

            mpirun -np "${NNODES}" --hostfile "${HOSTFILE_SLOTS}" \
                --allow-run-as-root --bind-to none \
                --mca plm_rsh_no_tree_spawn 1 --mca plm_rsh_args "-p ${PORT}" \
                -x PATH -x LD_LIBRARY_PATH -x LOCAL_HF_PATH \
                -x HCU_PROBE_CONFIG -x HCU_PROBE_MONITOR_CONFIG \
                -x HCU_PROBE_SEED -x HCU_PROBE_REPRODUCE -x HCU_PROBE_STRICT \
                -x MASTER_ADDR="${MASTER_ADDR}" -x MASTER_PORT="${MASTER_PORT}" \
                bash -c "
                source /opt/dtk/env.sh && \
                source ${CURRENT_DIR}/source_bridge_env.sh && \
                source_bridge_env ${MEGATRON_PATH} && \
                export PYTHONPATH=${HCU_PROBE_ROOT}/python:${PROFILE_EXTRA_PYTHONPATH:-}:\${PYTHONPATH:-} && \
                ${REMOTE_CMD}" 2>&1 | tee "${LOG_FILE}"
        fi
        echo "Training finished. Logs/checkpoints: ${SAVE_DIR}"
        ;;
esac
