"""Megatron-Bridge integration for hcu_probe.

Megatron-Bridge (``Megatron-bridge-das``) ships a callback system that is
attached to the training loop *without* touching framework code:

    train(callbacks=[MyCallback()])   # forwarded down to the training loop

Bridge's recipe runner (``run_recipe.py`` -> ``recipe_runner.run_config``)
however calls the training entry points without a ``callbacks=`` argument.
Because we are not allowed to edit the Bridge repository, this namespace
provides an *in-process overlay*:

    hcu_probe.integrations.megatron_bridge.install    -- patch pretrain/finetune
    hcu_probe.integrations.megatron_bridge.launcher   -- drop-in run_recipe.py
    hcu_probe.integrations.megatron_bridge.adapter    -- Callback <-> lifecycle

The overlay translates Bridge callback events into the existing Megatron
lifecycle hooks (``hcu_probe.integrations.megatron.dump`` /
``...monitor``), so dump, monitor and loss tracking behave exactly as they do
for native Megatron-LM training.

Quick start (single node, gemma3 pretrain):

    python -m hcuprobe.integrations.megatron_bridge.launcher \
        --bridge-root /path/to/Megatron-bridge-das \
        --hcu-probe-config /abs/L0.json \
        --recipe gemma3_1b_pretrain_8gpu_bw1000_bf16_config \
        --mode pretrain \
        dataset.seq_length=4096 train.train_iters=50 \
        train.global_batch_size=32 train.micro_batch_size=1

See ``docs_hcu/zh/user_guide/megatron_bridge_instruct.md`` for the full
runbook (torchrun, multi-node and config layout).

Symbols are imported lazily so that importing this package does not require
``megatron.bridge`` to be installed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRS = {
    "HCUProbeCallback": "adapter",
    "HCUProbeOptions": "adapter",
    "MegatronProbeArgs": "adapter",
    "install": "install",
    "uninstall": "install",
    "is_installed": "install",
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{target}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted({*globals(), *_LAZY_ATTRS})
