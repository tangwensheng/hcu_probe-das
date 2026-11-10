# Qwen3.5 text (35B-A3B, MoE) pretrain, 4 nodes / 32 GPUs.
#
# Source: Megatron-bridge-das/examples/qwen/train_qwen35_text_pretrain.sh
# Status: UNVERIFIED. MoE: use a rank-filtered probe config.

PROFILE_NAME="Qwen3.5 35B-A3B pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config
PROFILE_RECIPES=(
    qwen35_text_35b_a3b_pretrain_32gpu_bw1000_bf16_config
)

PROFILE_LOCAL_HF_PATH=/path/to/Qwen3.5-35B-A3B
PROFILE_EXPORTS="NVTE_USE_HIPBLASLT_GROUPEDGEMM=1 GPU_MAX_HW_QUEUES=4"

PROFILE_TP=1
PROFILE_PP=4
PROFILE_EP=8
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_TRAIN_ITERS=50
PROFILE_GLOBAL_BATCH_SIZE=64
PROFILE_MICRO_BATCH_SIZE=1

PROFILE_DATA_CONFIG=(
    dataset.blend=null
)

PROFILE_MODEL_CONFIG=(
    model.sequence_parallel=true
    model.init_method_std=0.006
    model.cuda_graph_impl=none
    comm_overlap.tp_comm_overlap=false
    model.bias_activation_fusion=true
    ddp.overlap_grad_reduce=true
    ddp.use_distributed_optimizer=true
    ddp.overlap_param_gather=true
    model.moe_router_fusion=true
    model.moe_permute_fusion=true
    model.moe_router_force_load_balancing=true
    model.moe_grouped_gemm=true
)
