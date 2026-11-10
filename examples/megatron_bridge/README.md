# Megatron-Bridge hcu_probe 集成示例

本目录演示如何在**不修改 Megatron-Bridge 仓库**的前提下，把 hcu_probe 的 dump（张量采集）、monitor（训练监测）和 loss 记录挂到 Megatron-Bridge 的训练流程上。

所有改动只在 hcu_probe 侧，Bridge 仓库保持原样。

---

## 目录结构

```
examples/megatron_bridge/
├── run_with_probe.sh                 # 通用入口：参数解析、启动前检查、单机/多机拉起
├── train_gemma3_pretrain_with_probe.sh  # 兼容入口，转调 run_with_probe.sh gemma3_pretrain
├── source_bridge_env.sh              # 加载 Bridge env.sh 的辅助函数
├── profiles/                         # 每个 Bridge 例子一个 profile，只声明差异
│   ├── _common.sh                    # 所有 profile 共享的默认值
│   ├── gemma3_pretrain.sh
│   ├── gemma4_pretrain.sh
│   ├── glm52_pretrain.sh  glm52_sft.sh  glm53_pretrain.sh
│   ├── minimax_m3_pretrain.sh
│   ├── qwen35_text_pretrain.sh  qwen35_vl_pretrain.sh  qwen38_2_4t_pretrain.sh
│   └── deepseek_v4_flash_pretrain.sh  deepseek_v4_flash_sft.sh  deepseek_v4_pro_pretrain.sh
├── config/                           # 按模型规模分档的 probe 配置模板
│   ├── L0_small.json
│   ├── L0_moe_rank0.json
│   ├── L1_small.json
│   └── monitor_only.json
├── inject_weight_perturbation.py     # 人为偏差验证脚本
└── README.md
```

---

## 快速开始

### 1. 准备 probe 配置

按模型规模选一档，复制后把路径改成**绝对路径**：

| 规模 | 用哪档 | 说明 |
|---|---|---|
| 小模型（gemma3 1B、gemma4 E4B） | `config/L0_small.json` | 全部 rank、step 0-2 |
| MoE 中大模型（glm5x、minimax、qwen35、deepseek flash） | `config/L0_moe_rank0.json` | 只 dump rank 0 的 step 0，控制存储 |
| 只想看单步、要 API 级 | `config/L1_small.json` | L1，step 0 |
| 超大模型（qwen38 2.4T、deepseek pro） | `config/monitor_only.json` | 只开 monitor，先看曲线 |

```bash
cd /public/home/tangwsh/myproject/hcu_probe-das/examples/megatron_bridge
mkdir -p /public/home/tangwsh/probe_output

cp config/L0_small.json /public/home/tangwsh/probe_output/L0.json
python - <<'EOF'
import json
p = "/public/home/tangwsh/probe_output/L0.json"
cfg = json.load(open(p))
cfg["dump_path"] = "/public/home/tangwsh/probe_output/L0_small"
json.dump(cfg, open(p, "w"), indent=2)
EOF

cp config/monitor_only.json /public/home/tangwsh/probe_output/monitor.json
python - <<'EOF'
import json
p = "/public/home/tangwsh/probe_output/monitor.json"
cfg = json.load(open(p))
cfg["output_dir"] = "/public/home/tangwsh/probe_output/monitor"
json.dump(cfg, open(p, "w"), indent=2)
EOF
```

### 2. 运行

```bash
export MEGATRON_PATH=/public/home/tangwsh/myproject/Megatron-bridge-das
export HCU_PROBE_CONFIG=/public/home/tangwsh/probe_output/L0.json
export HCU_PROBE_MONITOR_CONFIG=/public/home/tangwsh/probe_output/monitor.json

# gemma3 是唯一已验证的 profile
LOCAL_HF_PATH=/path/to/gemma-3-1b-it ./run_with_probe.sh gemma3_pretrain "" dryrun
LOCAL_HF_PATH=/path/to/gemma-3-1b-it TRAIN_ITERS=5 ./run_with_probe.sh gemma3_pretrain "" train
```

---

## 调用方式

```bash
./run_with_probe.sh <profile> [recipe] [mode] [hostfile] [nodes]
```

| 参数 | 说明 |
|---|---|
| `profile` | `profiles/` 下的文件名（去掉 `.sh`），见下表 |
| `recipe` | recipe 名；留空用 profile 的默认值 |
| `mode` | `dryrun`（只解析配置）/ `train` / `sft`；默认 `train` |
| `hostfile` | 多机时的主机列表 |
| `nodes` | 节点数，需要 hostfile |

兼容入口 `train_gemma3_pretrain_with_probe.sh` 等价于 `run_with_probe.sh gemma3_pretrain`，老命令不用改。

### Profile 一览

| Profile | 模型 | 规模 | 状态 |
|---|---|---|---|
| `gemma3_pretrain` | Gemma 3 1B | 1 节点 × 8 | **已验证** |
| `gemma4_pretrain` | Gemma 4 E4B | 2 节点 × 8 | 未验证 |
| `glm52_pretrain` | GLM-5.2（MoE） | 4 节点 × 8 | 未验证 |
| `glm52_sft` | GLM-5.2 SFT（MoE） | 4 节点 × 8 | 未验证 |
| `glm53_pretrain` | GLM-5.3（MoE） | 4 节点 × 8 | 未验证 |
| `minimax_m3_pretrain` | MiniMax-M3（MoE） | 2 节点（test）/ 32 节点 | 未验证 |
| `qwen35_text_pretrain` | Qwen3.5-35B-A3B（MoE） | 4 节点 × 8 | 未验证 |
| `qwen35_vl_pretrain` | Qwen3.5-VL 2B/4B/9B | 1 节点 × 8 | 未验证 |
| `qwen38_2_4t_pretrain` | Qwen3.8-2.4T-A95B（MoE） | 16 节点 × 8（test） | 未验证、资源重 |
| `deepseek_v4_flash_pretrain` | DeepSeek-V4-Flash（MoE） | 4 节点（test）/ 16 节点 | 未验证 |
| `deepseek_v4_flash_sft` | DeepSeek-V4-Flash SFT（MoE） | 4 节点（test）/ 16 节点 | 未验证 |
| `deepseek_v4_pro_pretrain` | DeepSeek-V4-Pro（MoE） | 8 节点（test）/ 32 节点 | 未验证、资源重 |

**未验证的 profile 默认拒绝启动**，需要显式确认：

```bash
PROBE_ALLOW_UNVERIFIED=1 ./run_with_probe.sh glm52_pretrain glm52_pretrain_32gpu_bw1000_bf16_test_config dryrun
```

这些 profile 是照着 `Megatron-bridge-das/examples/<family>/train_*.sh` 逐项抄过来的 recipe、模型路径、并行度和 CONFIG 覆盖，但没有在本 fork 上跑通过，第一次使用请先 `dryrun`，再小步数 `train`。

---

## 环境变量

| 变量 | 必需 | 说明 |
|---|---|---|
| `MEGATRON_PATH` | 是 | Megatron-bridge-das 根目录 |
| `LOCAL_HF_PATH` | 是 | 本地 HF 权重目录；profile 里给的是占位路径 |
| `HCU_PROBE_CONFIG` | 否 | dump 配置；不设置则不开启 dump |
| `HCU_PROBE_MONITOR_CONFIG` | 否 | monitor 配置；不设置则不开启 monitor |
| `HCU_PROBE_SEED` | 否 | 固定随机种子 |
| `HCU_PROBE_REPRODUCE` | 否 | 设为 1 开启确定性模式 |
| `HCU_PROBE_STRICT` | 否 | 设为 1 时 probe 初始化失败直接退出 |
| `PROBE_ALLOW_UNVERIFIED` | 否 | 设为 1 才允许跑未验证的 profile |
| `TRAIN_ITERS` / `GLOBAL_BATCH_SIZE` / `MICRO_BATCH_SIZE` | 否 | 覆盖 profile 默认值 |
| `TENSOR_PARALLEL` / `PIPELINE_PARALLEL` / `EXPERT_PARALLEL` / `EXPERT_TENSOR_PARALLEL` / `CONTEXT_PARALLEL` | 否 | 覆盖 profile 并行度 |
| `SEQ_LENGTH` | 否 | 覆盖序列长度 |
| `GPUS` | 否 | 每节点卡数，默认 8 |
| `MASTER_PORT` | 否 | 覆盖 rendezvous 端口 |

profile 里的环境变量（如 `GEMMA4_CONVERSION_MODE=text`、`NVTE_USE_HIPBLASLT_GROUPEDGEMM=1`）由 `PROFILE_EXPORTS` 声明，启动时统一 export。

---

## 启动前检查（自动）

配置了 probe 时，脚本会在启动前检查：

1. 配置文件存在且是合法 JSON
2. `matplotlib`、`tensorboard` 已安装
3. `hcuprobe.pytorch` 可导入

任意一项失败立即退出，不会白跑 8 卡训练。

另外 launcher 会把配置**内部**的 `dump_path` / `output_dir` 按启动目录转成绝对路径再写入临时文件，避免 Bridge `chdir` 后输出落到意外位置，原配置文件不被修改。

---

## 输出位置

- **dump**：`<dump_path>/step<N>/rank<ID>/dump.json`（模板里的 `dump_path`）
- **monitor**：`<output_dir>/` 下的 CSV（模板里的 `output_dir`）
- **loss 记录**：`<dump_path>/loss_tracking/`，没有 dump 时落在 monitor 的 `output_dir` 下
- **训练日志 / tensorboard / 解析后的 yaml**：`examples/megatron_bridge/<profile>_<recipe>_tp.._pp.._ep.._cp../`

启动横幅会打印解析后的绝对输出目录。

---

## 多机运行

```bash
cat > hostfile <<'EOF'
node1
node2
node3
node4
EOF

LOCAL_HF_PATH=/path/to/GLM-5.2 \
HCU_PROBE_CONFIG=/public/.../L0_moe.json \
PROBE_ALLOW_UNVERIFIED=1 \
./run_with_probe.sh glm52_pretrain glm52_pretrain_32gpu_bw1000_bf16_test_config train hostfile 4
```

`HCU_PROBE_CONFIG`、`HCU_PROBE_MONITOR_CONFIG`、`HCU_PROBE_SEED/REPRODUCE/STRICT` 都会通过 `mpirun -x` 下发到所有节点。多机时输出目录要放在共享存储上，否则每个节点各写各的。

---

## 人为偏差验证

验证 hcu_probe 真的能发现已知偏差，而不是只验证正常路径。

```bash
# 1. baseline：跑 1 step
HCU_PROBE_CONFIG=/public/.../L0.json TRAIN_ITERS=1 \
  ./run_with_probe.sh gemma3_pretrain "" train
mv /public/.../L0_small /tmp/baseline_dump

# 2. 注入扰动：单层权重 ×1.01（在 Bridge 的 forward_step 或 monkey patch 里改）

# 3. perturbed：相同配置再跑 1 step
HCU_PROBE_CONFIG=/public/.../L0.json TRAIN_ITERS=1 \
  ./run_with_probe.sh gemma3_pretrain "" train
mv /public/.../L0_small /tmp/perturbed_dump

# 4. 比对
python inject_weight_perturbation.py /tmp/baseline_dump /tmp/perturbed_dump --layer layers.0
```

预期输出 `✓ PASS: hcu_probe successfully detected the injected perturbation`。

---

## 已知限制

1. **输出路径必须是绝对路径**。相对路径会按启动目录解析（launcher 已处理），但写绝对路径最不容易出错。
2. **monitor 的 `patch_optimizer_step` 必须为 `false`**。Megatron 的 `DistributedOptimizer` → `ChainedOptimizer` 嵌套会让上游自动 patch 失效，hcu_probe 在 `on_step_end` 里手动推进 `mon.step()`。
3. **SFT 例子（`glm52_sft`、`deepseek_v4_flash_sft`）走 Bridge 的 `finetune()` 路径**，overlay 只重绑了文档提到的转发点，没有实机验证过。
4. **DeepSeek-V4-Flash 需要在 Bridge 里先切 Megatron-LM 到 `dev` 分支**（原脚本会执行 `scripts/switch_mcore.sh dev`），脚本未自动做这一步。
5. **VL 例子（qwen35_vl）的模块命名**与纯文本不同，第一次 L0 dump 后先确认 layer 名字再写 compare 规则。
6. **超大模型（qwen38 2.4T、deepseek pro）不要用 `L0_small.json`**，全 rank 全步数的统计量会把共享存储打满。

---

## 故障排查

### probe 初始化失败但训练继续

日志里出现 `[hcu_probe.dump] PrecisionDebugger init failed: ...` 时，训练会继续但没有任何数据。安装依赖并开启 strict：

```bash
pip install matplotlib tensorboard
export HCU_PROBE_STRICT=1
```

### 找不到输出目录

先看启动横幅打印的 `Dump output` / `Monitor output`。如果没打印，说明 `on_train_start` 没触发，检查日志里有没有 `hcu_probe Megatron-Bridge Integration` 这段横幅。

### 未验证 profile 直接退出

这是预期行为。先 `dryrun` 确认配置能解析，再加 `PROBE_ALLOW_UNVERIFIED=1` 跑训练。

---

## 新增一个 profile

1. 读 `Megatron-bridge-das/examples/<family>/train_*.sh`，抄下 `SUPPORTED_RECIPES`、`LOCAL_HF_PATH`、`MASTER_PORT`、并行度、`DATA_CONFIG`/`TRAIN_CONFIG`/`MODEL_CONFIG`。
2. 在 `profiles/` 下新建文件，复用 `_common.sh` 的默认值，只写差异；recipe 相关的分支用 `profile_configure()`。
3. `PROFILE_VERIFIED=false`，在文件头注明来源脚本和状态。
4. 在 README 的 Profile 一览表里加一行。
5. 按"快速开始"跑一次 dryrun，确认没有报错。

---

## 实现原理

1. **launcher 替换 `run_recipe.py`**：`run_with_probe.sh` 调用 `hcuprobe.integrations.megatron_bridge.launcher`，后者最后调用 Bridge 的 `run_recipe.main()`。
2. **overlay 包装 `pretrain`**：launcher 在 import `run_recipe` 后执行 `install()`，把 `megatron.bridge.training.pretrain.pretrain` 替换为薄包装，自动追加 `HCUProbeCallback`。
3. **callback 翻译事件**：`HCUProbeCallback` 把 Bridge 的 `on_train_start` / `on_train_step_start` / `on_train_step_end` / `on_train_end` 翻译成 `hcuprobe.integrations.megatron` 的生命周期调用。
4. **可还原**：`uninstall()` 恢复原始 `pretrain`。

Bridge 仓库始终不被修改。

---

## 参考

- hcu_probe Megatron 集成文档：`hcu_probe-das/docs_hcu/zh/user_guide/megatron_bridge_instruct.md`
- 项目整体指南：`hcu_probe/CLAUDE.md`、`hcu_probe/docs/integration_guide.md`
- Bridge 原始例子：`Megatron-bridge-das/examples/`
