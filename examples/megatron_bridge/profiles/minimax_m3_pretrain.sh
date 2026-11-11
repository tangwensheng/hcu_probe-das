# MiniMax-M3 pretrain (MoE, 32 nodes / 256 GPUs, or 2 nodes with the test recipe).
#
# Source: Megatron-bridge-das/examples/minimax/train_minimax_m3_pretrain.sh
# Status: UNVERIFIED. Full recipe needs 256 GPUs; start with the test recipe.

PROFILE_NAME="MiniMax-M3 pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=minimax_m3_pretrain_16gpu_bw1000_bf16_test_config
PROFILE_RECIPES=(
    minimax_m3_pretrain_256gpu_bw1000_bf16_config
    minimax_m3_pretrain_16gpu_bw1000_bf16_test_config
)

PROFILE_LOCAL_HF_PATH=/path/to/MiniMax-M3
PROFILE_EXPORTS=(NVTE_USE_HIPBLASLT_GROUPEDGEMM=1)

PROFILE_TP=1
PROFILE_PP=1
PROFILE_EP=4
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=1000000
PROFILE_GLOBAL_BATCH_SIZE=256
PROFILE_MICRO_BATCH_SIZE=1

# The bridge script also flips GLOBAL_BATCH_SIZE to 256 for the test recipe.
profile_configure() {
    if [[ ${RECIPE} == *test* ]]; then
        PROFILE_TP=1
        PROFILE_PP=1
        PROFILE_EP=4
        PROFILE_GLOBAL_BATCH_SIZE=256
        PROFILE_PARALLEL_CONFIG=(
            train.global_batch_size=${PROFILE_GLOBAL_BATCH_SIZE}
            train.exit_interval=30
        )
    else
        PROFILE_TP=2
        PROFILE_PP=4
        PROFILE_EP=32
        PROFILE_GLOBAL_BATCH_SIZE=2048
    fi
}

PROFILE_DATA_CONFIG=(
    dataset.blend=null
)

PROFILE_TRAIN_CONFIG=(
    validation.eval_iters=-1
    ddp.check_for_nan_in_grad=false
)

PROFILE_MODEL_CONFIG=(
    model.sequence_parallel=true
    model.overlap_p2p_comm=true
    model.cross_entropy_loss_fusion=false
    model.cross_entropy_fusion_impl=native
    model.moe_router_fusion=true
    model.moe_permute_fusion=true
    model.moe_router_load_balancing_type=seq_aux_loss
    model.moe_router_pre_softmax=true
    model.moe_token_dispatcher_type=flex
    model.moe_flex_dispatcher_backend=deepep
    model.moe_flex_dispatcher_num_sms=48
    optimizer.optimizer=adam
)
