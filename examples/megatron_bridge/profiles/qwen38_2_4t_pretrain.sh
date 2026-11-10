# Qwen3.8 2.4T-A95B pretrain (MoE, 16 nodes / 128 GPUs for the test recipe).
#
# Source: Megatron-bridge-das/examples/qwen/train_qwen38_2.4t_pretrain.sh
# Status: UNVERIFIED and resource-heavy. Start with the test recipe, a
#         rank-filtered probe config, and monitor only.

PROFILE_NAME="Qwen3.8 2.4T-A95B pretrain"
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=qwen38_text_2_4t_pretrain_128gpu_bw1000_bf16_test_config
PROFILE_RECIPES=(
    qwen38_text_2_4t_pretrain_bw1000_bf16_config
    qwen38_text_2_4t_pretrain_128gpu_bw1000_bf16_test_config
)

PROFILE_LOCAL_HF_PATH=/path/to/Qwen3.8-2.4T-A95B
PROFILE_EXPORTS="NVTE_USE_HIPBLASLT_GROUPEDGEMM=1 GPU_MAX_HW_QUEUES=4"

PROFILE_TP=8
PROFILE_PP=4
PROFILE_EP=16
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
    model.moe_token_dispatcher_type=alltoall
    model.cuda_graph_impl=none
    model.use_te_rng_tracker=false
    rng.te_rng_tracker=false
    comm_overlap.tp_comm_overlap=false
    model.bias_activation_fusion=false
    model.moe_router_fusion=true
    model.moe_permute_fusion=true
    model.moe_router_force_load_balancing=true
    model.moe_grouped_gemm=true
)
