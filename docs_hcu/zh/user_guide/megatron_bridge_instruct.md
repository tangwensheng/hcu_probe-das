# Megatron-Bridge 接入 hcu_probe（gemma3 示例）

本文说明如何在**不修改 Megatron-Bridge 仓库**的前提下，把 hcu_probe 的
dump（L0/L1 张量采集）、monitor（训练状态监测）和 loss 记录能力挂到
Megatron-Bridge 的训练流程上。示例使用 `Megatron-bridge-das` 中的
`gemma3_1b_pretrain_8gpu_bw1000_bf16_config`。

所有改动都落在 hcu_probe 侧（`hcu_probe-das`），Bridge 仓库保持原样。

## 1. 接入思路

Megatron-Bridge 本身提供了训练回调系统
（`megatron.bridge.training.callbacks`），`pretrain()` / `finetune()` 接受
`callbacks=` 参数，训练循环会在固定事件点触发回调。但 Bridge 的 recipe
入口（`run_recipe.py` → `recipe_runner.run_config`）调用训练函数时**不带
`callbacks=`**：

```python
# 3rdparty/Megatron-Bridge/scripts/training/recipe_runner.py
train_func(config=config, forward_step_func=step_func)   # 没有 callbacks=
```

因此 hcu_probe 采用**进程内 overlay**：在运行训练前，把
`megatron.bridge.training.pretrain.pretrain` 替换为一个薄包装，让它在原
调用基础上自动追加 `HCUProbeCallback`。`finetune()` 内部通过模块级名字
`pretrain` 转发，所以我们同时把那个引用重绑一次即可覆盖 finetune 路径。

`HCUProbeCallback` 负责把 Bridge 的回调事件翻译成 hcu_probe 既有的
Megatron 生命周期调用（dump/monitor 不感知 Bridge 的存在）：

| Bridge 回调事件 | 触发的 hcu_probe 动作 |
|---|---|
| `on_train_start` | `dump.on_train_start(args)` + `monitor.on_setup_done(model, optimizer, args)` |
| `on_train_step_start` | `dump.on_step_start(model, args)` |
| `on_train_step_end` | `dump.on_step_end(result, args)` + `monitor.on_step_end(result, args)` |
| `on_train_end` | `dump.on_train_end()` + `monitor.on_train_end()` |

Bridge 没有可挂钩的 argparse 阶段，所以激活参数改用环境变量传入；
`launcher` 会把命令行参数转换成对应的环境变量。

## 2. 前置条件

- 已有一份可用的 `Megatron-bridge-das`，并且 `source env/env.sh` 后
  `python run_recipe.py ...`（gemma3）能正常跑通。
- hcu_probe 的 Python 包加入 `PYTHONPATH`：

  ```bash
  export PYTHONPATH=<hcu_probe-das>/python:${PYTHONPATH}
  ```

- dump 依赖 `hcu_probe.pytorch.PrecisionDebugger`，monitor 依赖
  `hcu_probe.core.monitor_v2`。两条链路互相独立，可以只开其中一个。

## 3. 准备配置

### 3.1 dump 配置（可选）

按 L0/L1 采集规范准备一个 JSON，例如 `L0.json`，其中至少包含输出目录：

```json
{
  "dump_path": "/abs/out/probe_dump",
  "level": "L0"
}
```

`dump_path`（或 `output_path` / `output_dir`）会同时作为 loss 记录的根目录。

### 3.2 monitor 配置（可选）

按 Monitor V2 规范准备 JSON，例如 `monitor.json`：

```json
{
  "framework": "pytorch",
  "output_dir": "/abs/out/probe_monitor",
  "rank": [0],
  "format": "csv",
  "patch_optimizer_step": false,
  "monitors": {
    "weight_grad": { "enabled": true, "ops": ["min", "max", "mean", "norm", "nans"], "eps": 1e-8 }
  }
}
```

> **重要**：Megatron 下 `patch_optimizer_step` 必须为 `false`。Megatron 的
> 嵌套 optimizer（DistributedOptimizer → ChainedOptimizer）会让上游的自动
> patching 失效，hcu_probe 改为在 `on_step_end` 里手动推进 `mon.step()`。
> 若配成 `true`，会出现重复采集或采集不到的问题。

## 4. 运行方式

### 4.1 直接改环境变量（无需改任何脚本）

如果你已经有一套自己的启动脚本，只要在 `source env/env.sh` 之后追加：

```bash
export PYTHONPATH=<hcu_probe-das>/python:${PYTHONPATH}
export HCU_PROBE_CONFIG=/abs/L0.json                 # 可选，开启 dump
export HCU_PROBE_MONITOR_CONFIG=/abs/monitor.json    # 可选，开启 monitor
export HCU_PROBE_SEED=1234                            # 可选，固定随机种子
export HCU_PROBE_REPRODUCE=1                          # 可选，reproduce 模式
python -m hcuprobe.integrations.megatron_bridge.install ...  # 见 4.3
```

但如果你的入口是 Bridge 的 `run_recipe.py`（它内部调用训练函数），直接跑
`python run_recipe.py ...` 时 overlay 还没有被安装，回调不会生效。这时用
下面两种方式之一。

### 4.2 用 launcher 替换 `run_recipe.py`（推荐）

`launcher` 是 `run_recipe.py` 的直接替代品：它先安装 overlay，再把其余参数
原样、按顺序转发给 `run_recipe.main()`。单机 8 卡 gemma3：

```bash
source /opt/dtk/env.sh
source <Megatron-bridge-das>/env/env.sh
export PYTHONPATH=<hcu_probe-das>/python:${PYTHONPATH}
export LOCAL_HF_PATH=/path/to/gemma-3-1b-it

torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29508 \
  -m hcuprobe.integrations.megatron_bridge.launcher \
  --bridge-root <Megatron-bridge-das> \
  --hcu-probe-config /abs/L0.json \
  --hcu-probe-monitor-config /abs/monitor.json \
  --recipe gemma3_1b_pretrain_8gpu_bw1000_bf16_config \
  --mode pretrain \
  dataset.seq_length=4096 \
  dataset.blend=null \
  train.train_iters=50 \
  train.global_batch_size=32 \
  train.micro_batch_size=1 \
  model.tensor_model_parallel_size=1 \
  model.pipeline_model_parallel_size=1 \
  model.sequence_parallel=true \
  model.attention_backend=flash \
  mixed_precision=bf16_mixed \
  checkpoint.load=null checkpoint.save=null \
  logger.tensorboard_dir=/abs/out/tb_logs \
  logger.save_config_filepath=/abs/out/gemma3.yaml
```

launcher 自有参数（均为可选）：

| 参数 | 作用 |
|---|---|
| `--bridge-root PATH` | Bridge checkout 根目录；缺省时读 `MEGATRON_BRIDGE_ROOT` 或自动探测 |
| `--hcu-probe-config PATH` | 开启 dump，等价于 `HCU_PROBE_CONFIG` |
| `--hcu-probe-monitor-config PATH` | 开启 monitor，等价于 `HCU_PROBE_MONITOR_CONFIG` |
| `--hcu-probe-seed N` | 固定随机种子 |
| `--hcu-probe-reproduce` | 打开 reproduce 模式（布尔开关，无需取值） |

带 `--` 和带 `_` 两种写法都支持（例如 `--hcu_probe_config`）。路径会被
自动转成绝对路径，避免 Bridge 内部 `os.chdir` 之后解析失败。

### 4.3 在自己的启动脚本里手动安装

如果你已经有自己的 Python 入口（不是 `run_recipe.py`），可以直接 import
overlay 并安装：

```python
from hcuprobe.integrations.megatron_bridge import install, uninstall

install()                       # 选项从 HCU_PROBE_* 环境变量读取
# ... 调用 megatron.bridge.training.pretrain.pretrain / finetune ...
uninstall()                     # 可选，恢复原始入口
```

`install()` 是幂等的；重复调用不会重复挂载回调。也可以通过
`HCUProbeOptions(...)` 显式传入 `dump_config` / `monitor_config` / `seed` /
`reproduce`。

### 4.4 gemma3 dryrun

Bridge 的 `--dry-run` 只在配置阶段生成 `ConfigContainer.yaml`，**不会进入
训练循环，因此不会触发任何回调**，dump/monitor 都不会有产物。dryrun 适合
先验证 recipe 和并行配置，再上真实训练：

```bash
source /opt/dtk/env.sh
source <Megatron-bridge-das>/env/env.sh
export PYTHONPATH=<hcu_probe-das>/python:${PYTHONPATH}

python -m hcuprobe.integrations.megatron_bridge.launcher \
  --bridge-root <Megatron-bridge-das> \
  --recipe gemma3_1b_pretrain_8gpu_bw1000_bf16_config \
  --mode pretrain \
  --dry-run \
  dataset.seq_length=4096 train.train_iters=50 \
  train.global_batch_size=32 train.micro_batch_size=1
```

也可以直接使用 `hcu_probe-das/examples/megatron_bridge/train_gemma3_pretrain_with_probe.sh`
包装脚本，用法与 Bridge 的 `train_gemma3_pretrain.sh` 一致
（`<recipe> <dryrun|train> [hostfile] [nodes]`），通过环境变量
`HCU_PROBE_CONFIG` / `HCU_PROBE_MONITOR_CONFIG` 打开采集。

## 5. 产物位置

- 采集数据（dump）：`<dump cfg 的 dump_path>/`
- 监测 CSV（monitor）：`<monitor cfg 的 output_dir>/`
- loss / grad_norm 记录：`<loss 根目录>/loss_tracking/loss_log.csv`

  其中 loss 根目录的取值优先级为：dump 配置的 `dump_path` >
  monitor 配置的 `output_dir`。两个都未配置时不产生该文件。

  表头为 `step, loss_key, loss_value, grad_norm, skipped, timestamp`；只有当
  process 处于 data-parallel rank 0 时才写盘，避免多 rank 竞争。

启动后日志里会看到类似 `hcu_probe: patched
megatron.bridge.training.pretrain.pretrain` 和
`TrainerMonitorV2 initialized and started` 的记录，可作为回调已生效的
信号。

## 6. 已知限制

- 目前只覆盖 `pretrain` / `finetune` 两条训练入口（Bridge 的 `train.py`
  也只在训练事件上触发这些回调）；`--dry-run` 不触发回调。
- `context.model` 是 PP 切分后的 model chunk 列表，dump 只对第一个 chunk
  调用 `PrecisionDebugger.start(model=...)`，与既有 Megatron 集成一致。
- 环境变量方式在进程启动前设置；如果训练中途修改 `os.environ`，需要重新
  构造 `HCUProbeOptions` 并传入 `HCUProbeCallback`。
- overlay 通过替换 Python 模块属性实现，不会落盘到 Bridge 仓库，但同一
  进程内只应安装一次；`install()` 已做幂等处理。
