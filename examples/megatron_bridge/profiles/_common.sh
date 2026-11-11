# Shared defaults for every profile under profiles/<name>.sh.
#
# Sourced by run_with_probe.sh before the selected profile, so a profile only
# needs to override the values that differ from these. Nothing here may depend
# on a specific model family.

PROFILE_NAME=""
PROFILE_VERIFIED=false
PROFILE_DEFAULT_RECIPE=""
PROFILE_BRIDGE_MODE=pretrain
PROFILE_RECIPES=()

PROFILE_LOCAL_HF_PATH=""
# Array of KEY=VALUE entries exported on every rank; quote values with spaces, e.g.
#   PROFILE_EXPORTS=(GPU_MAX_HW_QUEUES=4 "MOCK_PROMPT=Describe this image.")
PROFILE_EXPORTS=()
# Extra entries prepended to PYTHONPATH on every rank.
PROFILE_EXTRA_PYTHONPATH=""

PROFILE_TP=1
PROFILE_PP=1
PROFILE_EP=1
PROFILE_ETP=1
PROFILE_CP=1
PROFILE_SEQ_LENGTH=4096
PROFILE_TRAIN_ITERS=50
PROFILE_GLOBAL_BATCH_SIZE=32
PROFILE_MICRO_BATCH_SIZE=1

PROFILE_DATA_CONFIG=()
PROFILE_TRAIN_CONFIG=()
PROFILE_PARALLEL_CONFIG=()
PROFILE_MODEL_CONFIG=()
PROFILE_EXTRA_ARGS=()

# Profiles may define profile_configure() to adjust the PROFILE_* defaults
# after the recipe is known (e.g. test recipes use a smaller topology).
