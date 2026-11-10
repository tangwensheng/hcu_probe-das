# Gemma 3 1B pretrain (single node, 8 GPUs).
#
# Source: Megatron-bridge-das/examples/gemma/train_gemma3_pretrain.sh
# Status: verified with hcu_probe dump + monitor on 8x BW1000.

PROFILE_NAME="Gemma 3 1B pretrain"
PROFILE_VERIFIED=true
PROFILE_DEFAULT_RECIPE=gemma3_1b_pretrain_8gpu_bw1000_bf16_config
PROFILE_RECIPES=(
    gemma3_1b_pretrain_8gpu_bw1000_bf16_config
)

PROFILE_LOCAL_HF_PATH=/path/to/gemma-3-1b-it
PROFILE_EXPORTS="GPU_MAX_HW_QUEUES=4 NVTE_USE_HIPBLASLT_GROUPEDGEMM=1"

PROFILE_TP=1
PROFILE_PP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=50
PROFILE_GLOBAL_BATCH_SIZE=32
PROFILE_MICRO_BATCH_SIZE=1

PROFILE_DATA_CONFIG=(
    dataset.blend=null
)

PROFILE_MODEL_CONFIG=(
    model.sequence_parallel=true
    model.attention_backend=flash
    mixed_precision=bf16_mixed
    optimizer.lr=3.0e-5
    optimizer.min_lr=3.0e-6
    scheduler.lr_warmup_iters=1
    model.activation_func=silu
    model.qk_layernorm=true
    model.share_embeddings_and_output_weights=false
)
