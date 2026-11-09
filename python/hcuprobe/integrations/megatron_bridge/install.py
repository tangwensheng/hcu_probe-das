"""In-process overlay that attaches the hcu_probe callback to Megatron-Bridge.

Why an overlay is needed
------------------------
``Megatron-bridge-das`` calls its training entry points without callbacks:

    # scripts/training/recipe_runner.py
    train_func = _load_train_function(mode)      # -> pretrain / finetune
    train_func(config=config, forward_step_func=step_func)

We must not edit that repository, so this module rewrites the two public
training entry points *in the running process* to append the hcu_probe
callback to whatever callback set the caller passed (usually nothing):

    megatron.bridge.training.pretrain.pretrain    # patched
    megatron.bridge.training.finetune.pretrain    # re-bound to the patched one

``finetune()`` delegates to ``pretrain()`` through a module-global reference
(``from ...pretrain import pretrain``). Re-binding that reference is enough to
cover the finetune path, so ``pretrain`` is wrapped exactly once per process.

The patch is applied late (``_load_train_function`` resolves the attribute
with ``getattr(importlib.import_module(...), ...)`` at call time), which is why
installing before the recipe runner calls training is sufficient.

Usage:

    from hcuprobe.integrations.megatron_bridge import install
    install()                      # options read from HCU_PROBE_* env vars
    # ... run training ...
    uninstall()
"""

from __future__ import annotations

import functools
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .adapter import HCUProbeCallback, HCUProbeOptions

logger = logging.getLogger(__name__)

_WRAPPED_SENTINEL = "_hcu_probe_wrapped"
_ATTACHED_SENTINEL = "_hcu_probe_attached"

__all__ = ["install", "uninstall", "is_installed"]


@dataclass
class _InstallHandle:
    """Records patched attributes so ``uninstall()`` can restore them."""

    patch_points: List[Tuple[Any, str, Any]] = field(default_factory=list)

    def restore(self) -> None:
        for owner, attribute, original in reversed(self.patch_points):
            setattr(owner, attribute, original)
        self.patch_points.clear()


_handle: Optional[_InstallHandle] = None


def _import_module(dotted: str) -> Any:
    import importlib

    return importlib.import_module(dotted)


def _attach_callback(manager: Any, probe_options: Optional[HCUProbeOptions]) -> Any:
    """Idempotently add the hcu_probe callback to a CallbackManager."""
    if getattr(manager, _ATTACHED_SENTINEL, False):
        return manager
    manager.add(HCUProbeCallback(probe_options))
    try:
        setattr(manager, _ATTACHED_SENTINEL, True)
    except Exception:  # pragma: no cover - exotic manager implementations
        pass
    return manager


def _wrap_pretrain(original: Any, probe_options: Optional[HCUProbeOptions]) -> Any:
    from megatron.bridge.training.callbacks import CallbackManager, normalize_callbacks

    @functools.wraps(original)
    def wrapped_pretrain(config, forward_step_func, callbacks=None, **kwargs):
        manager = normalize_callbacks(callbacks)
        if manager is None:
            manager = CallbackManager()
        _attach_callback(manager, probe_options)
        return original(config, forward_step_func, callbacks=manager, **kwargs)

    setattr(wrapped_pretrain, _WRAPPED_SENTINEL, True)
    setattr(wrapped_pretrain, "_hcu_probe_original", original)
    return wrapped_pretrain


def is_installed() -> bool:
    """True when the overlay is currently attached to Megatron-Bridge."""
    return _handle is not None and bool(_handle.patch_points)


def install(probe_options: Optional[HCUProbeOptions] = None) -> _InstallHandle:
    """Patch Megatron-Bridge entry points so every run carries hcu_probe.

    Args:
        probe_options: Explicit activation options. When ``None`` (the default)
            options are resolved from ``HCU_PROBE_*`` environment variables at
            training time.

    Returns:
        An install handle whose ``restore()`` reverts the patches.

    Raises:
        RuntimeError: When ``megatron.bridge`` is not importable, which means
            the overlay is being installed outside a Megatron-Bridge
            environment.
    """
    global _handle

    if is_installed():
        return _handle

    try:
        pretrain_module = _import_module("megatron.bridge.training.pretrain")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "install() needs megatron.bridge on sys.path. Run it from a "
            "Megatron-Bridge environment, or use "
            "hcuprobe.integrations.megatron_bridge.launcher which sets sys.path."
        ) from exc

    handle = _InstallHandle()
    original_pretrain = getattr(pretrain_module, "pretrain")

    if getattr(original_pretrain, _WRAPPED_SENTINEL, False):
        wrapped_pretrain = original_pretrain
    else:
        wrapped_pretrain = _wrap_pretrain(original_pretrain, probe_options)
        pretrain_module.pretrain = wrapped_pretrain
        handle.patch_points.append((pretrain_module, "pretrain", original_pretrain))
        logger.info("hcu_probe: patched megatron.bridge.training.pretrain.pretrain")

    # finetune() delegates through its own module-global reference to pretrain.
    finetune_module = sys.modules.get("megatron.bridge.training.finetune")
    if finetune_module is None:
        try:
            finetune_module = _import_module("megatron.bridge.training.finetune")
        except Exception:  # pragma: no cover - finetune module is optional
            finetune_module = None
    if finetune_module is not None and getattr(finetune_module, "pretrain", None) is original_pretrain:
        finetune_module.pretrain = wrapped_pretrain
        handle.patch_points.append((finetune_module, "pretrain", original_pretrain))
        logger.info("hcu_probe: rebound megatron.bridge.training.finetune.pretrain")

    _handle = handle
    return handle


def uninstall() -> None:
    """Revert patches installed by :func:`install` (no-op when not installed)."""
    global _handle
    if _handle is None:
        return
    _handle.restore()
    _handle = None
