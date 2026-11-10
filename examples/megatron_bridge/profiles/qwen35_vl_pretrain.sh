# Qwen3.5-VL pretrain (multimodal, single node / 8 GPUs).
#
# Source: Megatron-bridge-das/examples/qwen/train_qwen35_vl_pretrain.sh
# Status: UNVERIFIED. Defaults to the mock-data recipe so no Energon dataset is
#         needed; switch to the matching *_config recipe and export DATA_PATH
#         for real data. LM head / vision tower module names may differ from
#         text models -- inspect one step before trusting an L0 dump.

PROFILE_NAME="Qwen3.5-VL pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config
PROFILE_RECIPES=(
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_2b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_4b_pretrain_8gpu_bw1000_bf16_mock_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_config
    qwen35_vl_9b_pretrain_8gpu_bw1000_bf16_mock_config
)

PROFILE_LOCAL_HF_PATH=/path/to/Qwen3.5-VL-4B
PROFILE_EXTRA_PYTHONPATH="${MEGATRON_PATH}/3rdparty/Megatron-Energon/src"

PROFILE_TP=2
PROFILE_PP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=50
PROFILE_GLOBAL_BATCH_SIZE=32
PROFILE_MICRO_BATCH_SIZE=1

# Topology mirrors the recipe: SP requires TP, so 4B is TP2/PP1.
profile_configure() {
    local size
    size=$(echo "${RECIPE}" | sed -n 's/^qwen35_vl_\([0-9]\+b\)_.*/\1/p')
    case "${size}" in
        2b) PROFILE_TP=1; PROFILE_PP=1; PROFILE_CP=1; PROFILE_SP=false ;;
        4b) PROFILE_TP=2; PROFILE_PP=1; PROFILE_CP=1; PROFILE_SP=true ;;
        9b) PROFILE_TP=2; PROFILE_PP=2; PROFILE_CP=1; PROFILE_SP=true
            PROFILE_SEQ_LENGTH=2048 ;;
        *)  PROFILE_SP=true ;;
    esac
    PROFILE_MODEL_CONFIG=(model.sequence_parallel="${PROFILE_SP:-true}")
}

PROFILE_DATA_CONFIG=()

PROFILE_TRAIN_CONFIG=(
    validation.eval_iters=0
)

# Energon backend only: set DATA_PATH to the prepared dataset root.
if [[ ${RECIPE} != *mock* && -n "${DATA_PATH:-}" ]]; then
    PROFILE_DATA_CONFIG+=(dataset.path="${DATA_PATH}")
fi

# Mock data knobs read from the environment by the recipe.
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_PROMPT=${MOCK_PROMPT:-Describe this image.}"
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_NUM_IMAGES=${MOCK_NUM_IMAGES:-1}"
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_NUM_BASE_EXAMPLES=${MOCK_NUM_BASE_EXAMPLES:-1000}"
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_RANDOM_SEED=${MOCK_RANDOM_SEED:-0}"
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_IMAGE_WIDTH=${MOCK_IMAGE_WIDTH:-256}"
PROFILE_EXPORTS="${PROFILE_EXPORTS} MOCK_IMAGE_HEIGHT=${MOCK_IMAGE_HEIGHT:-256}"
