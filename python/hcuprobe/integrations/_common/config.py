"""Framework-agnostic JSON config parsing for hcu_probe.

Provides:
- read_json(path)              -- safe JSON loader, returns None on failure
- get_dump_output_dir_from_cfg(cfg_path)
- get_monitor_output_dir_from_cfg(cfg_path)
- resolve_loss_output_dir(*, dump_cfg_path=None, monitor_cfg_path=None)
- rewrite_config_output_paths_to_absolute(cfg_path)
"""

import atexit
import json
import os
import tempfile


def read_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_dump_output_dir_from_cfg(cfg_path):
    cfg = read_json(cfg_path)
    if not cfg:
        return None
    return cfg.get("dump_path") or cfg.get("output_path") or cfg.get("output_dir")


def get_monitor_output_dir_from_cfg(cfg_path):
    cfg = read_json(cfg_path)
    if not cfg:
        return None
    return cfg.get("output_dir")


def resolve_loss_output_dir(*, dump_cfg_path=None, monitor_cfg_path=None):
    """Single source of truth for LossTracker output directory.

    Priority: dump cfg dump_path > monitor cfg output_dir.
    Returns None when neither is set.
    """
    if dump_cfg_path:
        d = get_dump_output_dir_from_cfg(dump_cfg_path)
        if d:
            return d
    if monitor_cfg_path:
        return get_monitor_output_dir_from_cfg(monitor_cfg_path)
    return None


_REWRITTEN_CONFIGS = []


def _cleanup_rewritten_configs():
    for path in _REWRITTEN_CONFIGS:
        try:
            os.unlink(path)
        except OSError:
            pass
    _REWRITTEN_CONFIGS.clear()


atexit.register(_cleanup_rewritten_configs)

_OUTPUT_PATH_KEYS = ("dump_path", "output_path", "output_dir")


def rewrite_config_output_paths_to_absolute(cfg_path, base_dir=None):
    """Return a config path whose output paths are absolute.

    Some training frameworks ``os.chdir()`` into their own checkout after the
    config is resolved but before the probe service reads it, which silently
    redirects relative ``dump_path`` / ``output_dir`` values. This helper makes
    those paths absolute so the output location no longer depends on the
    process working directory.

    Relative values are resolved against ``base_dir`` (default: the current
    working directory, i.e. where the user launched training). The rewritten
    config is written to a temporary file that is removed at interpreter exit.
    If nothing needed rewriting, the original absolute path is returned.
    """
    if not cfg_path:
        return cfg_path

    abs_cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
    cfg = read_json(abs_cfg_path)
    if not cfg:
        return abs_cfg_path

    resolve_root = os.path.abspath(base_dir) if base_dir else os.getcwd()

    modified = False
    for key in _OUTPUT_PATH_KEYS:
        value = cfg.get(key)
        if not value or not isinstance(value, str):
            continue
        expanded = os.path.expanduser(value)
        if os.path.isabs(expanded):
            continue
        cfg[key] = os.path.abspath(os.path.join(resolve_root, expanded))
        modified = True

    if not modified:
        return abs_cfg_path

    # Keep the original file untouched: several ranks may read it concurrently
    # and the source checkout may be read-only.
    fd, temp_path = tempfile.mkstemp(
        suffix=".json", prefix="hcu_probe_cfg_", dir=tempfile.gettempdir(), text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(cfg, handle, indent=2)
    except Exception:
        _discard(temp_path)
        return abs_cfg_path

    _REWRITTEN_CONFIGS.append(temp_path)
    return temp_path


def _discard(path):
    try:
        os.unlink(path)
    except OSError:
        pass
