"""Drop-in replacement for ``Megatron-bridge-das/run_recipe.py`` with hcu_probe.

The Bridge example scripts invoke the recipe runner directly, for example
``examples/gemma/train_gemma3_pretrain.sh``:

    python ${MEGATRON_PATH}/run_recipe.py --recipe ... --mode pretrain ...

Because that script may not be edited, this module wraps the very same
``main()`` in the same process and installs the hcu_probe overlay first:

    python -m hcuprobe.integrations.megatron_bridge.launcher \
        --bridge-root ${MEGATRON_PATH} [--hcu-probe-* options] \
        --recipe ... --mode pretrain ...

Everything after the ``--hcu-probe-*`` / ``--bridge-root`` options is forwarded
verbatim (in order) to ``run_recipe.main()``, so all Bridge flags and
``KEY=VALUE`` ConfigContainer overrides keep working.

Under torchrun, replace the script path with ``-m <module>``:

    torchrun --nnodes=1 --nproc_per_node=8 \
        -m hcuprobe.integrations.megatron_bridge.launcher \
        --bridge-root ${MEGATRON_PATH} --hcu-probe-config /abs/L0.json \
        --recipe gemma3_1b_pretrain_8gpu_bw1000_bf16_config --mode pretrain ...

Bridge root resolution order: ``--bridge-root`` > ``MEGATRON_BRIDGE_ROOT`` >
auto-detect from an importable ``hcu_megatron_bridge`` / ``run_recipe`` module.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .adapter import (
    ENV_DUMP_CONFIG,
    ENV_MONITOR_CONFIG,
    ENV_REPRODUCE,
    ENV_SEED,
)
from .install import install

__all__ = ["main"]


def _flag_names(*names: str) -> Tuple[str, ...]:
    """Accept both ``--hcu-probe-config`` and ``--hcu_probe_config`` spellings."""
    expanded: List[str] = []
    for name in names:
        expanded.append(name)
        # Only swap the inner dashes; keep the leading ``--`` intact.
        prefix, _, body = name.partition("--")
        expanded.append(f"{prefix}--{body.replace('-', '_')}")
    return tuple(expanded)


BRIDGE_ROOT_FLAGS = _flag_names("--bridge-root")
DUMP_CONFIG_FLAGS = _flag_names("--hcu-probe-config")
MONITOR_CONFIG_FLAGS = _flag_names("--hcu-probe-monitor-config")
SEED_FLAGS = _flag_names("--hcu-probe-seed")
REPRODUCE_FLAGS = _flag_names("--hcu-probe-reproduce")

_VALUE_FLAGS = {
    **{flag: "bridge_root" for flag in BRIDGE_ROOT_FLAGS},
    **{flag: "dump_config" for flag in DUMP_CONFIG_FLAGS},
    **{flag: "monitor_config" for flag in MONITOR_CONFIG_FLAGS},
    **{flag: "seed" for flag in SEED_FLAGS},
}
_BOOL_FLAGS = {flag: "reproduce" for flag in REPRODUCE_FLAGS}


def _parse_argv(argv: List[str]) -> Tuple[Dict[str, object], List[str]]:
    """Split launcher-owned options from args passed through to Bridge.

    A hand-written scan is used (instead of ``parse_known_args``) to guarantee
    the relative order of the forwarded tokens is preserved exactly.
    """
    options: Dict[str, object] = {}
    passthrough: List[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag, _, inline_value = token.partition("=")
        target = _VALUE_FLAGS.get(flag)
        if target is not None:
            if inline_value:
                options[target] = inline_value
            elif index + 1 < len(argv):
                options[target] = argv[index + 1]
                index += 1
            else:
                raise SystemExit(f"hcu_probe launcher: {flag} requires a value")
            index += 1
            continue
        bool_target = _BOOL_FLAGS.get(flag)
        if bool_target is not None:
            options[bool_target] = True
            index += 1
            continue
        passthrough.append(token)
        index += 1
    return options, passthrough


def _autodetect_bridge_root() -> Optional[Path]:
    env_root = os.environ.get("MEGATRON_BRIDGE_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root).resolve()

    for module_name in ("hcu_megatron_bridge", "hcu_megatron"):
        module = sys.modules.get(module_name)
        if module is None or getattr(module, "__file__", None) is None:
            continue
        for parent in Path(module.__file__).resolve().parents:
            if (parent / "run_recipe.py").is_file():
                return parent

    for entry in sys.path:
        if entry and (Path(entry) / "run_recipe.py").is_file():
            return Path(entry).resolve()
    return None


def _prepare_sys_path(bridge_root: Optional[Path]) -> None:
    """Make Bridge's src, scripts and helper packages importable."""
    if bridge_root is None:
        return
    candidates = [
        bridge_root,
        bridge_root / "hcu_megatron_bridge" / "src",
        bridge_root / "3rdparty" / "Megatron-Bridge" / "src",
        bridge_root / "3rdparty" / "Megatron-Bridge" / "3rdparty" / "Megatron-LM",
        bridge_root / "3rdparty" / "Megatron-LM",
        bridge_root / "3rdparty" / "Megatron-Bridge" / "scripts" / "training",
        bridge_root / "3rdparty" / "Megatron-Bridge" / "scripts" / "common",
    ]
    for path in reversed(candidates):
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def _import_run_recipe(bridge_root: Optional[Path]):
    try:
        return importlib.import_module("run_recipe")
    except ModuleNotFoundError as exc:
        # Only fall back to a path import when *run_recipe itself* is missing;
        # a missing dependency inside run_recipe must keep its own error.
        if getattr(exc, "name", None) != "run_recipe" or bridge_root is None:
            raise

    script = bridge_root / "run_recipe.py"
    spec = importlib.util.spec_from_file_location("run_recipe", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"hcu_probe launcher: cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_recipe"] = module
    spec.loader.exec_module(module)
    return module


def _export_options(options: Dict[str, object]) -> None:
    """Publish launcher flags as environment variables for the overlay.

    Absolute paths are required because ``run_recipe.main()`` changes the
    working directory to the Bridge checkout before training starts.
    """
    if "dump_config" in options:
        os.environ[ENV_DUMP_CONFIG] = os.path.abspath(os.path.expanduser(str(options["dump_config"])))
    if "monitor_config" in options:
        os.environ[ENV_MONITOR_CONFIG] = os.path.abspath(
            os.path.expanduser(str(options["monitor_config"]))
        )
    if "seed" in options:
        os.environ[ENV_SEED] = str(options["seed"])
    if options.get("reproduce"):
        os.environ[ENV_REPRODUCE] = "1"


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    options, passthrough = _parse_argv(argv)

    # Configure logging before install() so the overlay's "patched ..." line is
    # visible; run_recipe.main()'s own basicConfig becomes a no-op.
    logging.basicConfig(level=logging.INFO)

    bridge_root = options.get("bridge_root")
    resolved_root = Path(str(bridge_root)).expanduser().resolve() if bridge_root else _autodetect_bridge_root()
    _prepare_sys_path(resolved_root)

    _export_options(options)
    install()

    run_recipe = _import_run_recipe(resolved_root)
    run_recipe.main(passthrough)


if __name__ == "__main__":
    main()
