# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Megatron-Bridge hcu_probe overlay.

These tests do not need Megatron-LM / Megatron-Bridge installed: a faithful
minimal stand-in for ``megatron.bridge.training.callbacks`` is injected into
``sys.modules`` and the overlay is exercised against it. That keeps the test
self-contained while still pinning the exact callback contract the overlay
relies on (``CallbackManager.add`` scanning overridden ``Callback`` methods).
"""

import importlib
import sys
import types

import pytest


VALID_EVENTS = (
    "on_data_init_start",
    "on_train_start",
    "on_train_step_start",
    "on_train_step_end",
    "on_train_end",
    "on_eval_start",
    "on_eval_step_start",
    "on_eval_step_end",
    "on_eval_end",
    "on_test_start",
    "on_test_step_start",
    "on_test_step_end",
    "on_test_end",
    "on_checkpoint_save",
)

_FAKE_MODULE_NAMES = (
    "megatron",
    "megatron.bridge",
    "megatron.bridge.training",
    "megatron.bridge.training.callbacks",
    "megatron.bridge.training.pretrain",
    "megatron.bridge.training.finetune",
)


def _build_callbacks_module():
    module = types.ModuleType("megatron.bridge.training.callbacks")
    module.VALID_EVENTS = frozenset(VALID_EVENTS)

    class Callback:
        def __getattr__(self, name):
            raise AttributeError(name)

    for event in VALID_EVENTS:
        setattr(Callback, event, lambda self, context: None)

    class CallbackContext:
        def __init__(self, state=None, model=None, user_state=None, optimizer=None, **extra):
            self.state = state
            self.model = model
            self.user_state = {} if user_state is None else user_state
            self.optimizer = optimizer
            self.loss_dict = extra.get("loss_dict")
            self.grad_norm = extra.get("grad_norm")
            self.skipped_iter = extra.get("skipped_iter")

    class CallbackManager:
        def __init__(self):
            self._callbacks = {event: [] for event in VALID_EVENTS}
            self._active = set()
            self.user_state = {}

        def add(self, callback):
            callbacks = callback if isinstance(callback, list) else [callback]
            for cb in callbacks:
                for event in VALID_EVENTS:
                    method = getattr(cb, event, None)
                    base = getattr(Callback, event, None)
                    if method is not None and method.__func__ is not base:
                        self._callbacks[event].append(method)
                        self._active.add(event)

        def register(self, event, fn):
            assert event in VALID_EVENTS
            self._callbacks[event].append(fn)
            self._active.add(event)

        def has_callbacks(self, event):
            return event in self._active

        def fire(self, event, context):
            for fn in self._callbacks[event]:
                fn(context)

    def normalize_callbacks(callbacks):
        if callbacks is None:
            return None
        if isinstance(callbacks, CallbackManager):
            return callbacks
        manager = CallbackManager()
        manager.add(callbacks)
        return manager

    module.Callback = Callback
    module.CallbackContext = CallbackContext
    module.CallbackManager = CallbackManager
    module.normalize_callbacks = normalize_callbacks
    return module


@pytest.fixture(autouse=True)
def bridge_env(monkeypatch):
    """Install fake Megatron-Bridge modules for every test in this module.

    The overlay imports ``megatron.bridge.training.callbacks`` at module load,
    so even the pure-helper tests need the stand-in on ``sys.modules``. Making
    the fixture autouse keeps the module self-contained without a real Bridge
    checkout.
    """
    saved = {name: sys.modules.get(name) for name in _FAKE_MODULE_NAMES}
    for name in _FAKE_MODULE_NAMES:
        sys.modules.pop(name, None)

    for name in ("megatron", "megatron.bridge", "megatron.bridge.training"):
        module = types.ModuleType(name)
        module.__path__ = []  # mark as a package
        sys.modules[name] = module

    callbacks_module = _build_callbacks_module()
    sys.modules["megatron.bridge.training.callbacks"] = callbacks_module

    calls = []
    pretrain_module = types.ModuleType("megatron.bridge.training.pretrain")

    def pretrain(config=None, forward_step_func=None, callbacks=None):
        calls.append(callbacks)
        return "pretrain"

    pretrain_module.pretrain = pretrain
    sys.modules["megatron.bridge.training.pretrain"] = pretrain_module

    finetune_module = types.ModuleType("megatron.bridge.training.finetune")
    finetune_module.pretrain = pretrain  # mirrors `from ...pretrain import pretrain`

    def finetune(config=None, forward_step_func=None, callbacks=None):
        return finetune_module.pretrain(config, forward_step_func, callbacks=callbacks)

    finetune_module.finetune = finetune
    sys.modules["megatron.bridge.training.finetune"] = finetune_module

    yield types.SimpleNamespace(
        callbacks=callbacks_module,
        pretrain=pretrain_module,
        finetune=finetune_module,
        calls=calls,
    )

    for name in _FAKE_MODULE_NAMES:
        sys.modules.pop(name, None)
        if saved[name] is not None:
            sys.modules[name] = saved[name]


@pytest.fixture
def overlay(bridge_env):
    """Fresh overlay modules bound to the fake bridge environment."""
    for name in list(sys.modules):
        if name.startswith("hcuprobe.integrations.megatron_bridge"):
            sys.modules.pop(name, None)
    adapter = importlib.import_module("hcuprobe.integrations.megatron_bridge.adapter")
    install_mod = importlib.import_module("hcuprobe.integrations.megatron_bridge.install")
    yield types.SimpleNamespace(adapter=adapter, install=install_mod)
    install_mod.uninstall()


def bridge_env_context(config):
    """Minimal Bridge CallbackContext stand-in for the overlay event tests."""
    state = types.SimpleNamespace(cfg=config)
    return types.SimpleNamespace(
        state=state,
        model=[object()],
        optimizer=object(),
        loss_dict={"lm loss": 0.0},
        grad_norm=None,
        skipped_iter=False,
    )


def test_options_from_env_parses_and_absolutizes(monkeypatch, tmp_path):
    from hcuprobe.integrations.megatron_bridge import adapter

    dump_cfg = tmp_path / "L0.json"
    dump_cfg.write_text("{}", encoding="utf-8")
    env = {
        "HCU_PROBE_CONFIG": str(dump_cfg),
        "HCU_PROBE_MONITOR_CONFIG": "  ",
        "HCU_PROBE_SEED": "1234",
        "HCU_PROBE_REPRODUCE": "true",
    }
    options = adapter.HCUProbeOptions.from_env(env)
    assert options.dump_config == str(dump_cfg)
    assert options.monitor_config is None
    assert options.seed == 1234
    assert options.reproduce is True
    assert options.enabled is True


def test_options_disabled_without_configs():
    from hcuprobe.integrations.megatron_bridge import adapter

    assert adapter.HCUProbeOptions.from_env({}).enabled is False


def test_launcher_parser_separates_flags_and_preserves_order(tmp_path):
    from hcuprobe.integrations.megatron_bridge import launcher

    argv = [
        "--bridge-root",
        str(tmp_path),
        "--hcu-probe-config=/abs/L0.json",
        "--hcu_probe_monitor_config",
        "/abs/monitor.json",
        "--hcu-probe-seed",
        "7",
        "--hcu-probe-reproduce",
        "--recipe",
        "gemma3_1b_pretrain_8gpu_bw1000_bf16_config",
        "--mode",
        "pretrain",
        "-gb",
        "32",
        "train.train_iters=50",
    ]
    options, passthrough = launcher._parse_argv(argv)

    assert options["bridge_root"] == str(tmp_path)
    assert options["dump_config"] == "/abs/L0.json"
    assert options["monitor_config"] == "/abs/monitor.json"
    assert options["seed"] == "7"
    assert options["reproduce"] is True
    assert passthrough == [
        "--recipe",
        "gemma3_1b_pretrain_8gpu_bw1000_bf16_config",
        "--mode",
        "pretrain",
        "-gb",
        "32",
        "train.train_iters=50",
    ]


def test_launcher_missing_value_raises():
    from hcuprobe.integrations.megatron_bridge import launcher

    with pytest.raises(SystemExit):
        launcher._parse_argv(["--hcu-probe-config"])


def test_flag_names_keep_leading_double_dash():
    from hcuprobe.integrations.megatron_bridge import launcher

    names = launcher._flag_names("--hcu-probe-monitor-config")
    assert "--hcu-probe-monitor-config" in names
    assert "--hcu_probe_monitor_config" in names
    # The underscore spelling must not be mangled into "__hcu_...".
    assert all(not name.startswith("__") for name in names)


def test_prepare_sys_path_adds_nested_megatron_lm(tmp_path):
    from hcuprobe.integrations.megatron_bridge import launcher

    bridge = tmp_path / "Megatron-bridge-das"
    nested_lm = bridge / "3rdparty" / "Megatron-Bridge" / "3rdparty" / "Megatron-LM"
    bridge_src = bridge / "3rdparty" / "Megatron-Bridge" / "src"
    hcu_src = bridge / "hcu_megatron_bridge" / "src"
    for path in (nested_lm, bridge_src, hcu_src):
        path.mkdir(parents=True)

    saved = list(sys.path)
    try:
        launcher._prepare_sys_path(bridge)
        for path in (nested_lm, bridge_src, hcu_src):
            assert str(path) in sys.path
    finally:
        sys.path[:] = saved


def test_patch_manager_compat_hides_uninitialized_nvdlfw(monkeypatch):
    from hcuprobe.integrations.megatron_bridge import launcher

    bridge_package = types.ModuleType("hcu_megatron_bridge")
    bridge_package.__path__ = []
    patch_utils = types.ModuleType("hcu_megatron_bridge.patch_utils")
    seen = []

    def scan_modules_like_bridge():
        for module in sys.modules.copy().values():
            hasattr(module, "_hcu_probe_missing")

    class MegatronPatchesManager:
        @staticmethod
        def apply_patches():
            scan_modules_like_bridge()
            seen.append(("apply", "nvdlfw_inspect.api" in sys.modules))

        @staticmethod
        def remove_patches():
            scan_modules_like_bridge()
            seen.append(("remove", "nvdlfw_inspect.api" in sys.modules))

    patch_utils.MegatronPatchesManager = MegatronPatchesManager
    monkeypatch.setitem(sys.modules, "hcu_megatron_bridge", bridge_package)
    monkeypatch.setitem(sys.modules, "hcu_megatron_bridge.patch_utils", patch_utils)

    nvdlfw_api = types.ModuleType("nvdlfw_inspect.api")

    def uninitialized_getattr(name):
        raise RuntimeError("Debug manager not initialized")

    nvdlfw_api.__getattr__ = uninitialized_getattr
    monkeypatch.setitem(sys.modules, "nvdlfw_inspect.api", nvdlfw_api)

    launcher._install_patch_manager_compat()
    MegatronPatchesManager.apply_patches()
    MegatronPatchesManager.remove_patches()

    assert seen == [("apply", False), ("remove", False)]
    assert sys.modules["nvdlfw_inspect.api"] is nvdlfw_api


def test_grad_acc_steps_from_probe_args():
    from hcuprobe.integrations.megatron.config import compute_grad_acc_steps
    from hcuprobe.integrations.megatron_bridge import adapter

    args = adapter.MegatronProbeArgs(
        global_batch_size=32, micro_batch_size=1, data_parallel_size=8
    )
    assert compute_grad_acc_steps(args) == 4


def test_train_step_result_matches_megatron_layout():
    from hcuprobe.integrations.megatron_bridge import adapter

    context = types.SimpleNamespace(
        loss_dict={"lm loss": 1.0}, skipped_iter=False, grad_norm=0.5
    )
    result = adapter._train_step_result(context)
    assert result[0] == {"lm loss": 1.0}
    assert result[1] == 0
    assert result[5] == 0.5

    skipped = types.SimpleNamespace(loss_dict=None, skipped_iter=True, grad_norm=None)
    assert adapter._train_step_result(skipped)[1] == 1


def test_install_wraps_entry_points_and_attaches_callback(overlay, bridge_env):
    handle = overlay.install.install(overlay.adapter.HCUProbeOptions())
    assert overlay.install.is_installed() is True

    wrapped = bridge_env.pretrain.pretrain
    assert getattr(wrapped, "_hcu_probe_wrapped", False) is True
    assert bridge_env.finetune.pretrain is wrapped

    # No caller callbacks -> overlay creates a manager and attaches itself.
    assert wrapped(config=object(), forward_step_func=object()) == "pretrain"
    manager = bridge_env.calls[-1]
    assert isinstance(manager, bridge_env.callbacks.CallbackManager)
    assert manager.has_callbacks("on_train_start")
    assert manager.has_callbacks("on_train_step_end")

    # A caller-provided callback set is preserved and extended in place.
    existing = bridge_env.callbacks.CallbackManager()
    existing.register("on_train_start", lambda ctx: None)
    wrapped(config=object(), forward_step_func=object(), callbacks=existing)
    assert bridge_env.calls[-1] is existing
    assert len(existing._callbacks["on_train_start"]) == 2

    handle.restore()
    assert overlay.install.is_installed() is False
    assert bridge_env.pretrain.pretrain.__name__ == "pretrain"


def test_install_is_idempotent(overlay, bridge_env):
    overlay.install.install()
    first = bridge_env.pretrain.pretrain
    overlay.install.install()
    assert bridge_env.pretrain.pretrain is first

    first(config=object(), forward_step_func=object())
    manager = bridge_env.calls[-1]
    # Attached exactly once even though install() ran twice.
    assert len(manager._callbacks["on_train_step_start"]) == 1


def test_callback_events_are_noops_without_config(overlay):
    callback = overlay.adapter.HCUProbeCallback(overlay.adapter.HCUProbeOptions())
    train_holder = types.SimpleNamespace(global_batch_size=32, micro_batch_size=1)
    config = types.SimpleNamespace(train=train_holder)
    context = bridge_env_context(config)

    callback.on_train_start(context)
    callback.on_train_step_start(context)
    callback.on_train_step_end(context)
    callback.on_train_end(context)
