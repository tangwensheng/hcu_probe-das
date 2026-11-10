# DeepSeek-V4-Pro pretrain (MoE, 32+ nodes / 256 GPUs, or 8 nodes with test).
#
# Source: Megatron-bridge-das/examples/deepseek/train_deepseekv4_pro_pretrain.sh
# Status: UNVERIFIED and resource-heavy. Start with the test recipe, a
#         rank-filtered probe config, and monitor only.

PROFILE_NAME="DeepSeek-V4-Pro pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=deepseek_v4_pro_pretrain_256gpu_bw1000_bf16_test_config
PROFILE_RECIPES=(
    deepseek_v4_pro_pretrain_bw1000_bf16_config
    deepseek_v4_pro_pretrain_256gpu_bw1000_bf16_test_config
)

PROFILE_LOCAL_HF_PATH=/path/to/DeepSeek-V4-Pro
PROFILE_EXPORTS="NVTE_USE_HIPBLASLT_GROUPEDGEMM=1"

PROFILE_TP=1
PROFILE_PP=2
PROFILE_EP=64
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=1000000
PROFILE_GLOBAL_BATCH_SIZE=256
PROFILE_MICRO_BATCH_SIZE=1

profile_configure() {
    if [[ ${RECIPE} == *test* ]]; then
        PROFILE_PP=2
        PROFILE_EP=64
        PROFILE_PARALLEL_CONFIG=(train.exit_interval=30)
    else
        PROFILE_PP=8
        PROFILE_EP=64
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
    model.moe_flex_dispatcher_num_sms=60
    model.apply_dsa_kernel_fusion=null
    model.dsa_kernel_backend=hip
    model.use_fused_mhc=true
    optimizer.optimizer=muon
)
