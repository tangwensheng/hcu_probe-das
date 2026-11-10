"""Megatron-Bridge -> Megatron integration adapter for hcu_probe.

Megatron-Bridge exposes training-loop callbacks (``Callback`` /
``CallbackManager`` in ``megatron.bridge.training.callbacks``). The hcu_probe
lifecycles were written for the native Megatron-LM hook contract:

    dump.on_train_start(args)
    dump.on_step_start(model, args)
    dump.on_step_end(result, args)
    dump.on_train_end()

    monitor.on_setup_done(models, optimizer, args)
    monitor.on_step_end(result, args)
    monitor.on_train_end()

where ``args`` is a Megatron argparse namespace and ``result`` is the tuple
returned by ``train_step``:

    (loss_dict, skipped_iter, should_checkpoint, should_exit,
     exit_code, grad_norm, num_zeros_in_grad, ...)

This module makes Bridge look like Megatron-LM to those translators:

    Bridge event          ->  Megatron hook
    on_train_start        ->  dump.on_train_start + monitor.on_setup_done
    on_train_step_start   ->  dump.on_step_start
    on_train_step_end     ->  dump.on_step_end + monitor.on_step_end
    on_train_end          ->  dump.on_train_end + monitor.on_train_end

Only argument translation lives here; every lifecycle invariant (DP-rank-0
writing, loss dedup, monitor ``patch_optimizer_step: false``) is inherited
from ``hcu_probe.integrations.megatron``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from megatron.bridge.training.callbacks import Callback, CallbackContext

from ..megatron import dump, monitor

__all__ = ["HCUProbeOptions", "MegatronProbeArgs", "HCUProbeCallback"]

ENV_DUMP_CONFIG = "HCU_PROBE_CONFIG"
ENV_MONITOR_CONFIG = "HCU_PROBE_MONITOR_CONFIG"
ENV_SEED = "HCU_PROBE_SEED"
ENV_REPRODUCE = "HCU_PROBE_REPRODUCE"
ENV_STRICT = "HCU_PROBE_STRICT"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _clean_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return os.path.abspath(os.path.expanduser(text)) if text else None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class HCUProbeOptions:
    """Resolved hcu_probe activation options.

    The same fields are consumed by the native Megatron-LM integration through
    its CLI flags (``--hcu-probe-config`` / ``--hcu-probe-monitor-config`` /
    ``--hcu-probe-seed``). Bridge has no argparse stage we may hook, so the
    overlay reads the equivalent environment variables instead.
    """

    dump_config: Optional[str] = None
    monitor_config: Optional[str] = None
    seed: Optional[int] = None
    reproduce: bool = False
    strict: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "HCUProbeOptions":
        env = os.environ if env is None else env
        return cls(
            dump_config=_clean_path(env.get(ENV_DUMP_CONFIG)),
            monitor_config=_clean_path(env.get(ENV_MONITOR_CONFIG)),
            seed=_to_int(env.get(ENV_SEED)),
            reproduce=_to_bool(env.get(ENV_REPRODUCE)),
            strict=_to_bool(env.get(ENV_STRICT)),
        )

    @property
    def enabled(self) -> bool:
        """True when at least one hcu_probe subsystem is configured."""
        return bool(self.dump_config or self.monitor_config)


@dataclass
class MegatronProbeArgs:
    """Least common denominator of the Megatron ``args`` fields hcu_probe reads.

    ``hcu_probe.integrations.megatron`` accesses these via ``getattr``, so a
    dataclass with matching names is a drop-in replacement for the real
    argparse namespace.
    """

    hcu_probe_config: Optional[str] = None
    hcu_probe_monitor_config: Optional[str] = None
    hcu_probe_seed: Optional[int] = None
    reproduce: bool = False
    global_batch_size: Optional[int] = None
    micro_batch_size: Optional[int] = None
    data_parallel_size: Optional[int] = None


def _cfg_get(config: Any, section: str, field: str, default: Any = None) -> Any:
    holder = getattr(config, section, None)
    if holder is None:
        return default
    return getattr(holder, field, default)


def _resolve_data_parallel_size(config: Any) -> Optional[int]:
    """Return the training data-parallel size for grad-accumulation math.

    ``ConfigContainer.set_data_parallel_size()`` runs during
    ``runtime_config_update`` (before ``on_train_start``), so the attribute is
    normally present. The manual fallback keeps the monitor usable when a
    caller reuses a partially-finalized config.
    """
    if config is None:
        return None
    dp_size = getattr(config, "data_parallel_size", None)
    if dp_size:
        return dp_size
    compute_dp = getattr(config, "get_data_parallel_size", None)
    if compute_dp is None:
        return None
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return compute_dp(dist.get_world_size())
    except Exception:
        return None
    return None


def _train_step_result(context: CallbackContext) -> tuple:
    """Re-shape Bridge's step-end context into Megatron's ``train_step`` tuple."""
    skipped = getattr(context, "skipped_iter", None)
    return (
        getattr(context, "loss_dict", None),
        1 if skipped else 0,
        False,
        0,
        0,
        getattr(context, "grad_norm", None),
        None,
    )


class HCUProbeCallback(Callback):
    """Bridge callback that drives the hcu_probe dump / monitor lifecycles.

    Args:
        probe_options: Pre-resolved options. When ``None`` the options are read
            from the environment on first use, which is what the overlay
            launcher relies on.
    """

    def __init__(self, probe_options: Optional[HCUProbeOptions] = None) -> None:
        self._probe_options = probe_options

    # -- helpers -----------------------------------------------------------

    def _options(self) -> HCUProbeOptions:
        if self._probe_options is None:
            self._probe_options = HCUProbeOptions.from_env()
        return self._probe_options

    def _build_args(self, context: CallbackContext) -> MegatronProbeArgs:
        config = getattr(getattr(context, "state", None), "cfg", None)
        options = self._options()
        return MegatronProbeArgs(
            hcu_probe_config=options.dump_config,
            hcu_probe_monitor_config=options.monitor_config,
            hcu_probe_seed=options.seed,
            reproduce=options.reproduce,
            global_batch_size=_cfg_get(config, "train", "global_batch_size"),
            micro_batch_size=_cfg_get(config, "train", "micro_batch_size"),
            data_parallel_size=_resolve_data_parallel_size(config),
        )

    # -- Bridge events -----------------------------------------------------

    def on_train_start(self, context: CallbackContext) -> None:
        """Fires after model/optimizer setup, before the training loop."""
        from .._common.config import get_dump_output_dir_from_cfg, get_monitor_output_dir_from_cfg
        from .._common.lifecycle import get_dump_lifecycle, get_monitor_lifecycle

        args = self._build_args(context)
        options = self._options()

        dump_out = get_dump_output_dir_from_cfg(args.hcu_probe_config) if args.hcu_probe_config else None
        monitor_out = (
            get_monitor_output_dir_from_cfg(args.hcu_probe_monitor_config)
            if args.hcu_probe_monitor_config
            else None
        )

        banner = [
            "=" * 70,
            "hcu_probe Megatron-Bridge Integration",
            f"  Dump config:    {args.hcu_probe_config or '<disabled>'}",
            f"  Dump output:    {dump_out or '<not set>'}",
            f"  Monitor config: {args.hcu_probe_monitor_config or '<disabled>'}",
            f"  Monitor output: {monitor_out or '<not set>'}",
            f"  Strict mode:    {options.strict}",
            "=" * 70,
        ]
        print("\n".join(banner))

        dump.on_train_start(args)
        monitor.on_setup_done(
            getattr(context, "model", None),
            getattr(context, "optimizer", None),
            args,
        )

        # The lifecycles swallow their own errors and leave the handle as None,
        # so strict mode has to check the resulting state, not rely on raises.
        failures = []
        if args.hcu_probe_config and get_dump_lifecycle().debugger is None:
            failures.append("dump (PrecisionDebugger not initialized)")
        if args.hcu_probe_monitor_config and get_monitor_lifecycle().monitor is None:
            failures.append("monitor (TrainerMonitorV2 not started)")

        if failures:
            message = "hcu_probe failed to start: " + ", ".join(failures)
            print(f"ERROR: {message}. See the [hcu_probe.*] lines above for the cause.")
            if options.strict:
                raise SystemExit(message)

    def on_train_step_start(self, context: CallbackContext) -> None:
        args = self._build_args(context)
        dump.on_step_start(getattr(context, "model", None), args)

    def on_train_step_end(self, context: CallbackContext) -> None:
        args = self._build_args(context)
        result = _train_step_result(context)
        dump.on_step_end(result, args)
        monitor.on_step_end(result, args)

    def on_train_end(self, context: CallbackContext) -> None:
        dump.on_train_end()
        monitor.on_train_end()
