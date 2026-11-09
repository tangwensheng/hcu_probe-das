#!/bin/bash
# gemma3 pretrain on Megatron-Bridge, with hcu_probe dump/monitor attached.
#
# This mirrors Megatron-bridge-das/examples/gemma/train_gemma3_pretrain.sh but
# swaps Bridge's run_recipe.py for the hcu_probe launcher, so the overlay can
# install itself before training starts. The Bridge checkout is never modified.
#
# Usage:
#   HCU_PROBE_CONFIG=/abs/L0.json \
#   HCU_PROBE_MONITOR_CONFIG=/abs/monitor.json \
#   ./train_gemma3_pretrain_with_probe.sh <recipe> <dryrun|train> [hostfile] [nodes]
#
# Environment:
#   MEGATRON_PATH   Bridge checkout root (or pass MEGATRON_BRIDGE_ROOT)
#   HCU_PROBE_CONFIG / HCU_PROBE_MONITOR_CONFIG   optional activation configs
#   GPUS, PORT, TENSOR_PARALLEL, PIPELINE_PARALLEL, CONTEXT_PARALLEL
set -euo pipefail

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
HCU_PROBE_ROOT=$( cd "${CURRENT_DIR}/../.." && pwd )
MEGATRON_PATH=${MEGATRON_PATH:-${MEGATRON_BRIDGE_ROOT:-}}

if [[ -z "${MEGATRON_PATH}" ]]; then
    echo "ERROR: set MEGATRON_PATH (or MEGATRON_BRIDGE_ROOT) to the Megatron-bridge-das root." >&2
    exit 1
fi
MEGATRON_PATH=$( cd "${MEGATRON_PATH}" && pwd )

export LOCAL_HF_PATH=${LOCAL_HF_PATH:-/path/to/gemma-3-1b-it}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-4}
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=${NVTE_USE_HIPBLASLT_GROUPEDGEMM:-1}

RECIPE=${1:-}
MODE=${2:-}
HOSTFILE=${3:-}
NODES=${4:-}

SUPPORTED_RECIPES=(
    gemma3_1b_pretrain_8gpu_bw1000_bf16_config
)

if [[ -z ${RECIPE} ]]; then
    echo "ERROR: missing recipe argument. Supported recipes:"
    printf "  - %s\n" "${SUPPORTED_RECIPES[@]}"
    echo "Usage: $0 <RECIPE> <MODE> [HOSTFILE] [NODES]   (MODE: dryrun | train)"
    exit 1
fi

GPUS="${GPUS:-8}"
PORT="${PORT:-11453}"
MASTER_ADDR=127.0.0.1
MASTER_PORT=29508

if [[ -n ${HOSTFILE} ]]; then
    [[ -f ${HOSTFILE} ]] || { echo "ERROR: ${HOSTFILE} not found."; exit 1; }
    HOSTFILE=$(cd "$(dirname "${HOSTFILE}")" && pwd)/$(basename "${HOSTFILE}")
    AVAILABLE_NODES=$(grep -cve '^[[:space:]]*$' "${HOSTFILE}")
    NNODES=${NODES:-${AVAILABLE_NODES}}
    if (( NNODES > AVAILABLE_NODES )); then
        echo "ERROR: node count ${NNODES} exceeds ${AVAILABLE_NODES} host(s) in ${HOSTFILE}." >&2
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

SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-50}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TP="${TENSOR_PARALLEL:-1}"
PP="${PIPELINE_PARALLEL:-1}"
CP="${CONTEXT_PARALLEL:-1}"

SAVE_DIR=${SAVE_DIR:-${CURRENT_DIR}/${RECIPE}_tp${TP}_pp${PP}_cp${CP}}
mkdir -p "${SAVE_DIR}/train_logs"

CONFIG=(
    dataset.seq_length="${SEQ_LENGTH}"
    dataset.blend=null
    train.train_iters="${TRAIN_ITERS}"
    train.global_batch_size="${GLOBAL_BATCH_SIZE}"
    train.micro_batch_size="${MICRO_BATCH_SIZE}"
    model.tensor_model_parallel_size="${TP}"
    model.pipeline_model_parallel_size="${PP}"
    model.context_parallel_size="${CP}"
    model.seq_length="${SEQ_LENGTH}"
    model.sequence_parallel=true
    model.attention_backend=flash
    mixed_precision=bf16_mixed
    checkpoint.load=null
    checkpoint.save=null
    logger.log_interval=1
    logger.tensorboard_dir="${SAVE_DIR}/tb_logs"
    logger.save_config_filepath="${SAVE_DIR}/${RECIPE}.yaml"
)

PROBE_ARGS=()
if [[ -n "${HCU_PROBE_CONFIG:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-config "${HCU_PROBE_CONFIG}")
fi
if [[ -n "${HCU_PROBE_MONITOR_CONFIG:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-monitor-config "${HCU_PROBE_MONITOR_CONFIG}")
fi
if [[ -n "${HCU_PROBE_SEED:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-seed "${HCU_PROBE_SEED}")
fi
if [[ -n "${HCU_PROBE_REPRODUCE:-}" ]]; then
    PROBE_ARGS+=(--hcu-probe-reproduce)
fi

LAUNCH_ARGS=(
    --bridge-root "${MEGATRON_PATH}"
    "${PROBE_ARGS[@]}"
    --recipe "${RECIPE}"
    --mode pretrain
    "${CONFIG[@]}"
)

echo "======================================"
echo "Gemma 3 1B pretrain with hcu_probe"
echo "Mode:      ${MODE:-}"
echo "Recipe:    ${RECIPE}"
echo "Nodes:     ${NNODES}"
echo "GPUs:      ${GPUS}/node, $((NNODES * GPUS)) total (TP=${TP}, PP=${PP}, CP=${CP})"
echo "Save dir:  ${SAVE_DIR}"
echo "Dump cfg:      ${HCU_PROBE_CONFIG:-<disabled>}"
echo "Monitor cfg:   ${HCU_PROBE_MONITOR_CONFIG:-<disabled>}"
echo "Master:    ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================"

case "${MODE}" in
    dryrun)
        source /opt/dtk/env.sh
        source "${MEGATRON_PATH}/env/env.sh"
        export PYTHONPATH="${HCU_PROBE_ROOT}/python:${PYTHONPATH:-}"
        python -m hcuprobe.integrations.megatron_bridge.launcher \
            "${LAUNCH_ARGS[@]}" \
            --dry-run
        echo "Dry run complete (no callbacks fire; resolved config: ${SAVE_DIR}/${RECIPE}.yaml)"
        ;;
    train)
        TORCHRUN_ARGS=(
            --nnodes="${NNODES}"
            --nproc_per_node="${GPUS}"
            --master_addr="${MASTER_ADDR}"
            --master_port="${MASTER_PORT}"
        )
        if (( NNODES == 1 )); then
            source /opt/dtk/env.sh
            source "${MEGATRON_PATH}/env/env.sh"
            export PYTHONPATH="${HCU_PROBE_ROOT}/python:${PYTHONPATH:-}"
            torchrun "${TORCHRUN_ARGS[@]}" \
                -m hcuprobe.integrations.megatron_bridge.launcher \
                "${LAUNCH_ARGS[@]}" \
                2>&1 | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
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
                -x MASTER_ADDR="${MASTER_ADDR}" -x MASTER_PORT="${MASTER_PORT}" \
                bash -c "
                source /opt/dtk/env.sh && \
                source ${MEGATRON_PATH}/env/env.sh && \
                export PYTHONPATH=${HCU_PROBE_ROOT}/python:\${PYTHONPATH:-} && \
                ${REMOTE_CMD}" 2>&1 | tee "${SAVE_DIR}/train_logs/mpi_$(date +%F-%H%M).log"
        fi
        echo "Training finished. Checkpoints/logs: ${SAVE_DIR}"
        ;;
    *)
        echo "ERROR: unknown mode '${MODE:-}'. Use 'dryrun' or 'train'." >&2
        exit 1
        ;;
esac
