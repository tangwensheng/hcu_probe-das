#!/bin/bash
# Load Megatron-bridge-das/env/env.sh without modifying the Bridge checkout.
#
# Bridge's env.sh infers MEGATRON_PATH from $0.  When it is sourced directly
# from this helper, $0 points at the hcu_probe launcher and resolves to the
# wrong repository.  Run it in a subshell with a synthetic caller located
# under the Bridge root, then import the environment that env.sh exported.

source_bridge_env() {
    local bridge_root=$1
    local env_script="${bridge_root}/env/env.sh"
    local synthetic_caller="${bridge_root}/examples/gemma/train_gemma3_pretrain.sh"
    local exported
    local export_line

    if [[ ! -f "${env_script}" ]]; then
        echo "ERROR: ${env_script} not found." >&2
        return 1
    fi

    exported=$(bash -c 'set -e; source "$1"; export -p' "${synthetic_caller}" "${env_script}") || return 1
    while IFS= read -r export_line; do
        if [[ "${export_line}" == "declare -x "* ]]; then
            eval "export ${export_line#declare -x }"
        fi
    done <<< "${exported}"
}
