# Gemma 4 E4B pretrain (2 nodes, 8 GPUs each).
#
# Source: Megatron-bridge-das/examples/gemma/train_gemma4_pretrain.sh
# Status: UNVERIFIED. Derived from the Bridge script; not yet run with probe.

PROFILE_NAME="Gemma 4 E4B pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=gemma4_e4b_pretrain_16gpu_bw1000_bf16_config
PROFILE_RECIPES=(
    gemma4_e4b_pretrain_16gpu_bw1000_bf16_config
)

PROFILE_LOCAL_HF_PATH=/path/to/gemma-4-E4B-it
PROFILE_EXPORTS=(GEMMA4_CONVERSION_MODE=text GPU_MAX_HW_QUEUES=4 NVTE_USE_HIPBLASLT_GROUPEDGEMM=1)

PROFILE_TP=4
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
    ddp.overlap_grad_reduce=false
    mixed_precision=bf16_mixed
    optimizer.lr=2.0e-5
    optimizer.min_lr=2.0e-6
)
