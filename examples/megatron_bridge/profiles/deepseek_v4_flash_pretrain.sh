# DeepSeek-V4-Flash pretrain (MoE, 16 nodes / 128 GPUs, or 4 nodes with test).
#
# Source: Megatron-bridge-das/examples/deepseek/train_deepseekv4_flash_pretrain.sh
# Status: UNVERIFIED. The Bridge script also switches the Megatron-LM submodule
#         to the `dev` branch before running (scripts/switch_mcore.sh dev); do
#         that manually first or startup will fail.

PROFILE_NAME="DeepSeek-V4-Flash pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=deepseek_v4_flash_pretrain_32gpu_BW1000_bf16_test_config
PROFILE_RECIPES=(
    deepseek_v4_flash_pretrain_128gpu_BW1000_bf16_config
    deepseek_v4_flash_pretrain_32gpu_BW1000_bf16_test_config
)

PROFILE_LOCAL_HF_PATH=/path/to/DeepSeek-V4-Flash-0731
PROFILE_EXPORTS="NVTE_USE_HIPBLASLT_GROUPEDGEMM=1"

PROFILE_TP=1
PROFILE_PP=2
PROFILE_EP=16
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=1000000
PROFILE_GLOBAL_BATCH_SIZE=256
PROFILE_MICRO_BATCH_SIZE=1

profile_configure() {
    if [[ ${RECIPE} == *test* ]]; then
        PROFILE_PP=2
        PROFILE_EP=16
        PROFILE_PARALLEL_CONFIG=(train.exit_interval=30)
    else
        PROFILE_PP=4
        PROFILE_EP=32
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
    model.cross_entropy_loss_fusion=true
    model.cross_entropy_fusion_impl=te
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
