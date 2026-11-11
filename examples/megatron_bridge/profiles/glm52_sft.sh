# GLM-5.2 SFT (MoE, 4 nodes / 32 GPUs).
#
# Source: Megatron-bridge-das/examples/glm/train_glm52_sft.sh
# Status: UNVERIFIED. Goes through Bridge's finetune path; the overlay wrapper
#         has not been checked against it yet.

PROFILE_NAME="GLM-5.2 SFT"
PROFILE_VERIFIED=false
PROFILE_BRIDGE_MODE=sft
PROFILE_DEFAULT_RECIPE=glm52_sft_32gpu_bw1000_bf16_test_config
PROFILE_RECIPES=(
    glm52_sft_32gpu_bw1000_bf16_test_config
)

PROFILE_LOCAL_HF_PATH=/path/to/GLM-5.2
PROFILE_EXPORTS=(NVTE_USE_HIPBLASLT_GROUPEDGEMM=1)

# Bridge script: DATA_PATH=/path/to/tulu-3-sft-mixture/data
DATA_PATH=${DATA_PATH:-/path/to/tulu-3-sft-mixture/data}

PROFILE_TP=1
PROFILE_PP=2
PROFILE_EP=16
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_SEQ_LENGTH=8192
PROFILE_TRAIN_ITERS=100
PROFILE_GLOBAL_BATCH_SIZE=256
PROFILE_MICRO_BATCH_SIZE=1

profile_configure() {
    if [[ ${RECIPE} == *test* ]]; then
        PROFILE_PP=2
        PROFILE_PARALLEL_CONFIG=(train.exit_interval=30)
    else
        PROFILE_PP=8
    fi
}

PROFILE_DATA_CONFIG=(
    dataset.dataset_root=${DATA_PATH}
)

PROFILE_TRAIN_CONFIG=(
    validation.eval_iters=-1
    ddp.check_for_nan_in_grad=false
    tokenizer.tokenizer_model=${LOCAL_HF_PATH:-${PROFILE_LOCAL_HF_PATH}}
)

PROFILE_MODEL_CONFIG=(
    model.sequence_parallel=true
    model.cross_entropy_loss_fusion=True
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
)
