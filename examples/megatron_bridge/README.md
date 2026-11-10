# Megatron-Bridge hcu_probe 集成示例

本目录演示如何在**不修改 Megatron-Bridge 仓库**的前提下，把 hcu_probe 的 dump（张量采集）、monitor（训练监测）和 loss 记录挂到 Megatron-Bridge 的训练流程上。

所有改动只在 hcu_probe 侧，Bridge 仓库保持原样。

---

## 快速开始

### 1. 准备配置文件

复制模板并修改输出路径为**绝对路径**：

```bash
cd /public/home/tangwsh/myproject/hcu_probe-das
cp .claude/skills/hcu_probe-megatron-integration/config/hcu_probe_config_statistics_L0.json /tmp/my_L0.json
cp .claude/skills/hcu_probe-megatron-integration/config/monitor_v2_config_full_probe.json /tmp/my_monitor.json

# 编辑 /tmp/my_L0.json，把 dump_path 改成：
#   "dump_path": "/public/home/tangwsh/probe_output/statistics_L0"
# 编辑 /tmp/my_monitor.json，把 output_dir 改成：
#   "output_dir": "/public/home/tangwsh/probe_output/monitor_v2"
```

**重要**：输出路径必须是绝对路径，因为 Bridge 会 `chdir` 到自己的目录。

### 2. 运行训练（单机 8 卡示例）

```bash
export MEGATRON_PATH=/public/home/tangwsh/myproject/Megatron-bridge-das
export HCU_PROBE_CONFIG=/tmp/my_L0.json
export HCU_PROBE_MONITOR_CONFIG=/tmp/my_monitor.json

cd /public/home/tangwsh/myproject/hcu_probe-das/examples/megatron_bridge

# 先 dryrun 检查配置
./train_gemma3_pretrain_with_probe.sh gemma3_1b_pretrain_8gpu_bw1000_bf16_config dryrun

# 确认无误后正式训练（默认 50 步，可用 TRAIN_ITERS=5 改小）
TRAIN_ITERS=5 ./train_gemma3_pretrain_with_probe.sh gemma3_1b_pretrain_8gpu_bw1000_bf16_config train
```

### 3. 查看输出

```bash
ls /public/home/tangwsh/probe_output/statistics_L0/step0/rank0/dump.json
ls /public/home/tangwsh/probe_output/monitor_v2/weight_grad_rank0.csv
ls /public/home/tangwsh/probe_output/statistics_L0/loss_tracking/
```

---

## 配置说明

### 环境变量

| 变量 | 必需 | 说明 |
|---|---|---|
| `MEGATRON_PATH` | 是 | Megatron-bridge-das 根目录 |
| `HCU_PROBE_CONFIG` | 否 | dump 配置文件绝对路径；不设置则不开启 dump |
| `HCU_PROBE_MONITOR_CONFIG` | 否 | monitor 配置文件绝对路径；不设置则不开启 monitor |
| `HCU_PROBE_SEED` | 否 | 固定随机种子（整数） |
| `HCU_PROBE_REPRODUCE` | 否 | 设为 1 开启确定性模式（关闭 dropout） |

### 启动脚本参数

```bash
./train_gemma3_pretrain_with_probe.sh <recipe> <mode> [hostfile] [nodes]
```

- `<recipe>`：`gemma3_1b_pretrain_8gpu_bw1000_bf16_config`
- `<mode>`：`dryrun` 或 `train`
- `[hostfile]`：多机时的主机列表文件
- `[nodes]`：使用多少个节点（多机时必需）

可覆盖的环境变量：
- `TRAIN_ITERS=5`：训练步数（默认 50）
- `GPUS=8`：每节点 GPU 数
- `TENSOR_PARALLEL=1`、`PIPELINE_PARALLEL=1`、`CONTEXT_PARALLEL=1`

### 配置文件内容

**dump 配置** (`L0.json` / `L1.json`)：
- `task`：`statistics`（统计量）或 `tensor`（完整张量）
- `dump_path`：**必须是绝对路径**
- `level`：`L0`（模块级）、`L1`（API 级）、`mix`（两者）
- `step`：`[0, 1, 2]` 或 `[]`（所有步）
- `rank`：`[0]` 或 `[]`（所有卡）

**monitor 配置** (`monitor.json`)：
- `output_dir`：**必须是绝对路径**
- `patch_optimizer_step`：**必须为 `false`**（Megatron 嵌套优化器导致自动 patch 失效）
- `monitors`：开启 `weight_grad`、`optimizer`、`param` 等

---

## 启动前检查（自动）

脚本会在启动前检查：

1. 配置文件是否存在
2. 配置文件是否是合法 JSON
3. `matplotlib`、`tensorboard` 是否已安装
4. `hcuprobe.pytorch` 是否可导入

检查失败时会立即退出并提示错误。

---

## Strict 模式

默认情况下，probe 初始化失败时会打印错误但继续训练。要让初始化失败时立即退出：

```bash
export HCU_PROBE_STRICT=1
# 或在启动脚本中加 --hcu-probe-strict（需要先手动传递到 launcher）
```

---

## 多机运行

创建 hostfile（每行一个主机名或 IP）：

```text
node1
node2
```

然后运行：

```bash
./train_gemma3_pretrain_with_probe.sh gemma3_1b_pretrain_8gpu_bw1000_bf16_config train hostfile 2
```

环境变量 `HCU_PROBE_CONFIG` 和 `HCU_PROBE_MONITOR_CONFIG` 会自动通过 mpirun `-x` 下发到所有节点。

**输出目录**：
- 如果输出路径在共享存储（如 `/public/home/...`），所有节点写到同一目录，按 `rank*` 区分。
- 如果是本地路径，每个节点各自落盘，需要分别登录查看。

---

## 人为偏差验证（测试 hcu_probe 能否检测到已知扰动）

### 步骤

1. **baseline 训练**：跑 1 step，生成 baseline dump

```bash
export HCU_PROBE_CONFIG=/tmp/my_L0.json
export HCU_PROBE_MONITOR_CONFIG=  # 关闭 monitor 加快速度
TRAIN_ITERS=1 ./train_gemma3_pretrain_with_probe.sh gemma3_1b_pretrain_8gpu_bw1000_bf16_config train
mv /public/home/tangwsh/probe_output/statistics_L0 /tmp/baseline_dump
```

2. **注入扰动**：在 Megatron-Bridge 代码中找到 `model.module.decoder.layers[0].self_attention.linear_qkv.weight`，乘以 1.01（或直接在 `forward_step` 里临时修改）。

   如果不想改 Bridge 代码，可以用 monkey patch：在 `train_gemma3_pretrain_with_probe.sh` 的 `source_bridge_env` 后加：

   ```python
   python -c "
   import sys
   sys.path.insert(0, '${MEGATRON_PATH}/3rdparty/Megatron-Bridge/src')
   # 在这里写 monkey patch 逻辑注入 1.01 扰动
   "
   ```

3. **perturbed 训练**：用相同配置再跑 1 step

```bash
TRAIN_ITERS=1 ./train_gemma3_pretrain_with_probe.sh gemma3_1b_pretrain_8gpu_bw1000_bf16_config train
mv /public/home/tangwsh/probe_output/statistics_L0 /tmp/perturbed_dump
```

4. **验证检测**：

```bash
python inject_weight_perturbation.py /tmp/baseline_dump /tmp/perturbed_dump --layer layers.0
```

预期输出：`✓ PASS: hcu_probe successfully detected the injected perturbation`

---

## 已知限制

- **输出路径必须是绝对路径**：相对路径会在 Bridge `chdir` 后落到意外位置。
- **monitor 的 `patch_optimizer_step` 必须为 `false`**：Megatron 的 `DistributedOptimizer` → `ChainedOptimizer` 嵌套导致自动 patch 失效。
- **PP（流水线并行）场景**：`model` 可能是列表，目前 callback 只传 `context.model`，若 Bridge 把列表放在别处需要适配。
- **仅支持 pretrain**：finetune 应该也能用（内部调用 pretrain），但未充分测试。

---

## 故障排查

### probe 初始化失败但训练继续

**现象**：日志里有 `[hcu_probe.dump] PrecisionDebugger init failed: No module named 'matplotlib'`，但训练照常跑。

**原因**：lifecycle.py 捕获异常后只打日志。

**解决**：
1. 安装依赖：`pip install matplotlib tensorboard`
2. 开启 strict 模式：`export HCU_PROBE_STRICT=1`

### 找不到输出目录

**现象**：启动目录下没有 `hcu_probe_dump/`，也不在 `/tmp` 下。

**位置**：在 `${MEGATRON_PATH}/3rdparty/Megatron-Bridge/` 下（Bridge 会 `chdir` 到那里）。

**根本解决**：配置文件里的 `dump_path` 和 `output_dir` 改成绝对路径。launcher 会自动把相对路径转成绝对路径（相对于配置文件所在目录）。

### 启动时报 "config file not found"

检查：
```bash
echo $HCU_PROBE_CONFIG
ls -l $HCU_PROBE_CONFIG
```

如果路径包含 `~`，要么展开成绝对路径，要么确保 shell 会展开（不要用单引号）。

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `train_gemma3_pretrain_with_probe.sh` | 主启动脚本，支持 dryrun/train、单机/多机 |
| `source_bridge_env.sh` | 加载 Bridge 环境变量的辅助函数（避免污染 hcu_probe launcher） |
| `inject_weight_perturbation.py` | 人为偏差验证脚本，比对两次 dump 检测扰动 |
| `README.md` | 本文件 |

配置模板在 `hcu_probe-das/.claude/skills/hcu_probe-megatron-integration/config/`。

---

## 实现原理

1. **launcher 替换 run_recipe.py**：`train_gemma3_pretrain_with_probe.sh` 调用 `hcuprobe.integrations.megatron_bridge.launcher`，它最后调用 Bridge 的 `run_recipe.main()`。
2. **overlay 包装 pretrain**：launcher 在 import `run_recipe` 后执行 `install()`，把 `megatron.bridge.training.pretrain.pretrain` 替换成薄包装，自动追加 `HCUProbeCallback`。
3. **callback 翻译事件**：`HCUProbeCallback` 把 Bridge 的 `on_train_start` 等事件翻译成 `hcuprobe.integrations.megatron` 的生命周期调用。
4. **可还原**：`uninstall()` 恢复原始 pretrain 函数。

Bridge 仓库始终不被修改。

---

## 参考

- hcu_probe Megatron 集成文档：`hcu_probe-das/docs_hcu/zh/user_guide/megatron_bridge_instruct.md`
- 项目整体指南：`hcu_probe/CLAUDE.md`、`hcu_probe/docs/integration_guide.md`
