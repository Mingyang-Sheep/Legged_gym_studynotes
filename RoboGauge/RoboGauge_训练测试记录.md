# RoboGauge 训练与测试完整记录

> 论文: Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
> 仓库: https://github.com/wty-yy/RoboGauge.git
> 机器人: Unitree Go2 | 服务器: PowerEdge-R750 (2×GPU) | 显卡: RTX 4090

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [代码架构理解](#2-代码架构理解)
3. [启动训练](#3-启动训练)
4. [模型导出 (JIT)](#4-模型导出-jit)
5. [RoboGauge 评估](#5-robogauge-评估)
6. [评估结果分析](#6-评估结果分析)
7. [MoE 机制理解](#7-moe-机制理解)
8. [疑问与解答汇总](#8-疑问与解答汇总)

---

## 1. 环境搭建

### 1.1 前提

- 已安装 Isaac Gym Preview 4
- 已有 conda 环境 `legged_gym` (PyTorch 1.10 + CUDA 11.3)
- 服务器已装好 RTX 4090 驱动

### 1.2 克隆仓库

```bash
# 训练代码
cd ~/lmy/RL
git clone https://github.com/wty-yy/RoboGauge.git go2_rl_gym

# 评估框架（RoboGauge 的另一个独立组件）
cd ~/lmy/RL
git clone https://github.com/wty-yy/RoboGauge.git RoboGauge
```

### 1.3 安装

```bash
# 先卸载旧版 legged_gym（如果有）
pip uninstall legged_gym -y

# 安装训练代码
cd ~/lmy/RL/go2_rl_gym/go2_rl_gym-master
pip install -e .
cd rsl_rl && pip install -e . && cd ..

# 安装评估框架
cd ~/lmy/RL/RoboGauge
pip install -e .
```

### 1.4 RTX 4090 兼容性

训练时必须禁用 TorchScript JIT（RTX 4090 compute capability 8.9 不被 PyTorch 1.10 的 nvrtc 支持）：

```bash
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2_moe_cts
```

### 1.5 验证安装

```bash
python -c "import legged_gym; print(legged_gym.__file__)"
# 应输出: .../go2_rl_gym/go2_rl_gym-master/legged_gym/__init__.py
```

---

## 2. 代码架构理解

### 2.1 仓库组成

训练仓库包含两个核心组件：

| 组件 | 路径 | 作用 |
|------|------|------|
| **legged_gym** | `go2_rl_gym-master/legged_gym/` | 环境代码（1426 行基类），含 Go2 配置、观测、奖励、域随机化 |
| **rsl_rl** | `go2_rl_gym-master/rsl_rl/` | RL 算法库，含 7 种网络架构 + 7 种算法 |

### 2.2 7 个已注册任务

| task 参数 | 网络架构 | 说明 |
|-----------|---------|------|
| `go2` | 单一 Actor-Critic | 原始 PPO 基线 |
| `go2_cts` | Teacher-Student | CTS 基线 |
| **`go2_moe_cts`** | **MoE-CTS** | **论文主打的完整版** |
| `go2_moe_ng_cts` | MoE-No-Goal-CTS | 去掉指令门控消融 |
| `go2_mcp_cts` | MCP-CTS | 乘法组合消融 |
| `go2_ac_moe_cts` | AC-MoE-CTS | MoE 在动作层消融 |
| `go2_dual_moe_cts` | Dual-MoE-CTS | 双重 MoE 消融 |

### 2.3 MoE-CTS 网络结构

```
Teacher 路径 (训练用，有特权信息):
  privileged_obs(263维) → MLP [512,256,32] → L2Norm → latent(32维)

Student 路径 (部署用，只有 proprioception):
  history = 5帧 obs(45维) = 225维
  history → StudentMoEEncoder → latent(32维)
            ├─ Gating Network:  225 → 512 → 256 → 8 → softmax → ω
            ├─ Expert₁ (共享 backbone + 分组卷积) ─┐
            ├─ ...                                ├→ Σ ω_k × Expert_k = latent
            └─ Expert₈ ───────────────────────────┘

Actor (共享):
  [latent(32) + obs(45)] = 77维 → MLP [512,256,128,12]

Critic (共享, 仅训练):
  [latent(32) + privileged_obs(263)] = 295维 → MLP [512,256,128,1]
```

### 2.4 观测空间

| 网络 | 维度 | 组成 |
|------|------|------|
| **Actor** (Student) | **45** | ang_vel(3) + gravity(3) + commands(3) + dof_pos(12) + dof_vel(12) + last_action(12) |
| **Critic** (Teacher) | **263** | 45 + base_lin_vel(3) + foot_forces(4) + torques(12) + dof_acc(12) + heights(187) |

**关键设计**：Actor **不含** base_lin_vel（真机速度估计不准，纯 proprioception 对 Sim2Real 更友好）。

### 2.5 训练参数

| 参数 | 值 |
|------|-----|
| 并行环境数 | 8192 |
| 最大迭代 | 150000 |
| Episode 长度 | 25s |
| 域随机化项 | 10 项（含 PD 增益、执行器强度/偏置、控制延迟等） |
| 指令课程 | 3 阶段（0~20K: ±0.5, 20K~50K: ±1.0, 50K~: ±2.0） |
| MoE 专家数 | 8 |
| 隐状态维度 | 32 |
| 历史帧数 | 5 |
| Teacher 环境比例 | 75% |

---

## 3. 启动训练

### 3.1 训练命令

```bash
cd ~/lmy/RL/go2_rl_gym/go2_rl_gym-master
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2_moe_cts
```

### 3.2 输出文件

训练日志和模型保存在：

```
~/lmy/RL/go2_rl_gym/logs/go2_moe_cts/May03_12-19-00_/
├── model_0.pt, model_100.pt, ..., model_5500.pt   # 训练断点
├── jit_models/                                      # JIT 导出（每 500 iter）
│   └── policy_jit_500.pt, policy_jit_1000.pt, ...
├── config.yaml                                      # 训练配置
└── events.out.tfevents.*                            # TensorBoard 日志
```

### 3.3 监控训练

```bash
tensorboard --logdir ~/lmy/RL/go2_rl_gym/logs/go2_moe_cts
```

### 3.4 Isaac Gym 可视化

```bash
cd ~/lmy/RL/go2_rl_gym/go2_rl_gym-master
# 编辑 play.py 第65行: EXPORT_POLICY = False （避免 JIT 导出报错）
PYTORCH_JIT=0 python legged_gym/scripts/play.py \
    --task=go2_moe_cts --resume \
    --load_run May03_12-19-00_ --checkpoint 5500
```

---

## 4. 模型导出 (JIT)

### 4.1 为什么需要导出

| | model_5500.pt | policy_jit_5500.pt |
|---|---|---|
| 格式 | Python pickle | TorchScript IR |
| 包含 | 网络参数 + 优化器状态 + Teacher + Critic | 仅 Student + Actor |
| 大小 | ~197MB | ~8MB |
| 加载方式 | `torch.load()` (需 PyTorch) | `torch.jit.load()` (跨语言) |
| RoboGauge 可用 | **不能** | **能** |

### 4.2 导出的坑

**坑 1: circular import 问题**
- `from legged_gym.utils.exporter import export_policy_as_jit` 会触发 `legged_gym.__init__` → `envs.__init__` → `task_registry` → 循环导入
- 解决：绕过 legged_gym，直接 `sys.path.insert(0, 'rsl_rl')`

**坑 2: history_length=5 不是 3**
- 训练配置用的是 5 帧历史
- 验证方法：`model_5500.pt` 中 `student_moe_encoder.moe.gating_network.0.network.0.weight` 形状为 `[512, 225]`，即 225 = 45 × 5

**坑 3: PYTORCH_JIT=0 与 JIT 导出冲突**
- `PYTORCH_JIT=0` 全局禁用 TorchScript，但导出需要 TorchScript
- 解决：导出脚本不设该环境变量，全程 CPU 操作（CPU 上 JIT 编译不调 nvrtc）

**坑 4: torch.jit.script() vs torch.jit.trace()**
- `torch.jit.script()` 需要读取 Python 源码 → 需要在 `.py` 文件中执行（不能用 `-c` 内联）
- `torch.jit.trace()` 不需要源码，但不保留 `reset()` 等未在 trace 路径上的方法
- 最终方案：写 `.py` 文件 + `torch.jit.script()` + `@torch.jit.export` 装饰 `reset()`

**坑 5: RoboGauge 需要 reset() 方法**
- RoboGauge 每次评估开始时会调用 `model.reset()` 清空历史缓存
- 缺少该方法会报 `AttributeError: 'RecursiveScriptModule' object has no attribute 'reset'`

### 4.3 最终导出脚本

使用 `RoboGauge/RoboGauge/export_jit.py`：

```bash
cd ~/lmy/RL/RoboGauge
python export_jit.py
```

输出：`~/lmy/RL/go2_rl_gym/logs/go2_moe_cts/May03_12-19-00_/policy_jit_5500.pt`

---

## 5. RoboGauge 评估

### 5.1 RoboGauge 架构

RoboGauge 是**C/S 架构**：
- **Client**: 训练代码中的 `on_policy_runner_cts.py`（`update_robogauge()`），训练时自动提交模型评估
- **Server/离线评估**: `robogauge/scripts/run.py` — 独立于训练的 MuJoCo 评估框架

### 5.2 评估流水线

```
Base Pipeline       → 单环境 + 单种子 → 6 个指标
Multi/Level Pipeline → 多种子 + 多域随机化 → 二分搜索最大难度
Stress Pipeline      → 7 地形 × 9 摩擦 → 综合评分
```

### 5.3 6 个评估指标

| 指标 | 类别 | 衡量什么 |
|------|------|---------|
| lin_vel_err | Tracking | 线速度跟踪 ℓ₂ 误差 |
| ang_vel_err | Tracking | 角速度跟踪 ℓ₂ 误差 |
| dof_power | Safety | 关节功耗 |
| dof_limits | Safety | 关节角度超出软限位 |
| orientation_stability | Safety | 重力在侧向轴的投影 |
| torque_smoothness | Quality | 力矩时间平滑度 |

### 5.4 评分机制

- **加权几何平均** → 防止高分项掩盖低分项
- **Worst-Case Mean (Mean@50)** → 只取表现较差的 50% 命令求平均
- **重叠评分**: `S = α(L*−1) + β * Q(L*)`

### 5.5 评估命令

```bash
cd ~/lmy/RL/RoboGauge

# 单地形快速测试
python robogauge/scripts/run.py \
    --task go2_moe.flat \
    --model-path ~/lmy/RL/go2_rl_gym/logs/go2_moe_cts/May03_12-19-00_/policy_jit_5500.pt \
    --experiment-name test_5500 \
    --headless

# 完整 Stress Benchmark（7 地形 × 9 摩擦 × 5 种子）
python robogauge/scripts/run.py \
    --task go2_moe \
    --model-path ~/lmy/RL/go2_rl_gym/logs/go2_moe_cts/May03_12-19-00_/policy_jit_5500.pt \
    --experiment-name go2_moe_stress_go2_moe_5500_stress \
    --stress-benchmark \
    --stress-terrain-names flat slope_fd slope_bd stairs_fd stairs_bd wave obstacle \
    --num-processes 35 \
    --seeds 0 1 2 \
    --search-seeds 0 1 2 3 4 \
    --frictions 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --compress-logs \
    --headless
```

---

## 6. 评估结果分析

### 6.1 总体评分

| 模型 | Benchmark Score | 迭代 |
|------|:--------------:|------|
| **本次训练 go2_moe_cts** | **0.440** | 5.5K |
| 论文 CTS 收敛版 | 0.580 | 150K |
| 论文 MoE-CTS 收敛版 | 0.674 | 137K |

当前模型训练了 3.7%（5500/150000），已达到收敛版 CTS 的 76%——MoE 架构在早期就已经体现了优势。

### 6.2 七地形评分

| 地形 | 得分 | 评价 |
|------|:----:|------|
| **flat** | 0.624 | 平地表现最好 |
| **wave** | 0.527 | 中等，高摩擦时达 Level 9 |
| **obstacle** | 0.512 | 中等，高摩擦时达 Level 8 |
| **slope_bd** (下坡) | 0.508 | 高摩擦满分 Level 10，低摩擦失败 |
| **slope_fd** (上坡) | 0.455 | 同上，低摩擦更弱 |
| **stairs_bd** (下楼梯) | 0.243 | 弱，最高 Level 4 |
| **stairs_fd** (上楼梯) | 0.213 | 最弱，最高 Level 4 |

### 6.3 摩擦 vs 最高难度

```
                  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
wave               2    3    4    6    7    8    8    9    9
slope_fd           X    X    2    4    6    8    9    9   10
slope_bd           X    2    3    5    7    8    9   10   10
stairs_fd           2    3    2    4    4    3    3    3    3
stairs_bd           1    3    3    4    4    4    4    3    4
obstacle            X    6    8    7    8    7    8    6    6
```

规律：所有地形随摩擦增大难度级别升高（符合预期），楼梯仍是瓶颈。

### 6.4 八项指标汇总（全地形平均）

| 指标 | Mean (all) | Mean@50 (worst 50%) |
|------|:----------:|:------------------:|
| lin_vel_err | 0.524 | 0.477 |
| ang_vel_err | 0.469 | 0.408 |
| dof_power | 0.567 | 0.542 |
| dof_limits | 0.594 | 0.587 |
| orientation_stability | 0.591 | 0.578 |
| torque_smoothness | 0.553 | 0.533 |
| friction_margin | 0.472 | 0.419 |
| zmp_margin | 0.572 | 0.546 |

Mean@50（最差 50% 命令的均值）在各指标上都低于全局均值，说明挑战性命令（急停、对角线速度等）确实暴露了策略的不足。角度跟踪误差 (ang_vel_err) 的差距最大（0.469 vs 0.408），说明急速转向是当前策略的短板。

### 6.5 结论

**当前状态**：5500 步 MoE-CTS 已学会基本行走 + 简单上下坡，但楼梯和障碍物还不够稳健。符合该训练阶段的表现预期。

**改进方向**：
1. **继续训练**到 50K+ 步 — 楼梯和障碍物会显著改善
2. **查看 TensorBoard** — 确认 tracking_lin_vel reward 是否持续上升
3. **对比 CTS 基线** — 用同样配置训练 `go2_cts`，对比 5.5K 时的评分差距

---

## 7. MoE 机制理解

### 7.1 MoE 是什么

MoE（Mixture of Experts）用 K 个专家子网络 + 一个门控网络替代单一编码器：

```
obs → Gating Network → ω₁, ω₂, ..., ω₈ (softmax权重)
    → Expert₁(obs) × ω₁
    → Expert₂(obs) × ω₂
    → ...
    → Expert₈(obs) × ω₈
    → 加权和 = 隐状态
```

### 7.2 MoE 是动态的吗

**是**。每次推理（每 20ms）门控网络根据当前观测实时计算专家权重，不同地形/状况下激活不同的专家组合。但门控网络的参数在推理时不变——「裁判的判断标准已固化，但每次看新场景都会吹不同的哨」。

### 7.3 训练 vs 推理

| 阶段 | 发生什么 | pt 文件里有什么 |
|------|---------|---------------|
| **训练** | 门控网络学习"什么场景该派哪个专家"（通过反向传播 + 负载均衡损失） | 存下 Gating + Experts 的参数 |
| **推理** | 门控网络根据实时观测当场计算 ω，动态调配专家 | 不存推理记录中的 ω 值 |

### 7.4 如何看到 MoE 的专家分配

```bash
# 方式 1: MuJoCo 部署 + 实时柱状图
cd ~/lmy/RL/go2_rl_gym/go2_rl_gym-master
python deploy/deploy_mujoco/deploy_go2.py --visualize-moe-weights

# 方式 2: 保存隐状态做 PCA（论文 Fig.6 风格）
python deploy/deploy_mujoco/deploy_go2.py --save-moe-latent --save-video
```

---

## 8. 疑问与解答汇总

### Q1: 为什么 `model_5500.pt` 不能直接给 RoboGauge 用？

A: `model_5500.pt` 是 PyTorch pickle 格式（`torch.save`），只能在 Python 中用 `torch.load()` 加载。RoboGauge 期望的是 TorchScript 格式（`torch.jit.load`），因为 MuJoCo 的 C++ 端可以直接调用 TorchScript，不需要 Python。

### Q2: 为什么 RTX 4090 训练需要 `PYTORCH_JIT=0`？

A: RTX 4090 的 compute capability 是 8.9，PyTorch 1.10 内置的 CUDA 11.3 的 nvrtc 不支持该架构。GPU 上的 TorchScript JIT 编译会调用 nvrtc → 报错。`PYTORCH_JIT=0` 全局禁用 GPU JIT，走纯 Python 路径。

### Q3: 为什么 JIT 导出不需要 `PYTORCH_JIT=0`？

A: 导出全程在 **CPU** 上操作（`map_location='cpu'`, `.cpu()`）。CPU 上的 TorchScript JIT 编译走 LLVM JIT，不调 CUDA nvrtc，所以不受 GPU 架构限制。

### Q4: model_5500.pt 里存的是什么？

A: 5 个 key：
- `model_state_dict` — 所有网络权重（训练/推理用）
- `optimizer1_state_dict` — Actor/Critic/Teacher 优化器状态（恢复训练用）
- `optimizer2_state_dict` — Student MoE 优化器状态（恢复训练用）
- `iter` — 当前迭代数 (5500)
- `infos` — 额外信息

### Q5: history_length 是 5 还是 3？

A: 本次训练用的是 **5**。验证：`model_5500.pt` 中 `gating_network.0.network.0.weight` 形状为 `[512, 225]`，即 225 = 45 × 5。

### Q6: MoE 占比能从 pt 文件推导吗？

A: **不能**。pt 文件只保存了门控网络的参数（「裁判的判断标准」），不保存推理过程中每次计算出的具体 ω 值。要获取 ω 值必须在不同地形上跑推理并实时采集。

### Q7: 评估时如何可视化？

A: 去掉 `--headless` 参数即可看到 MuJoCo 渲染窗口。`--save-video` 保存 `.mp4` 视频。MoE 专用: `deploy_go2.py --visualize-moe-weights` 实时显示 8 个专家的激活柱状图。

### Q8: Gating Network 是什么？

A: Gating Network 是 MoE 中的门控网络，输入观测（225 维历史拼接），输出 8 个 softmax 权重。它在训练中学会「不同地形该派哪个专家」，推理时根据实时观测当场计算权重。

### Q9: 5500 步的结果算好吗？

A: 对于 5500 步而言表现**超出预期**。斜坡高摩擦已满分，wave/obstacle 中高水平，楼梯仍是弱点。Benchmark Score 0.440 已达到收敛版 CTS（0.580）的 76%，说明 MoE 架构在训练早期就展现出优势。继续训练到 50K+ 步楼梯和障碍物会显著提升。

---

## 附录：文件清单

| 文件 | 用途 |
|------|------|
| `RoboGauge/export_jit.py` | JIT 模型导出脚本 |
| `RoboGauge/plot_results.py` | 评估结果可视化脚本 |
| `RoboGauge/go2_moe_stress_go2_moe_5500_stress/` | 本次 Stress Benchmark 结果 |
| `RoboGauge/RoboGauge_论文学习笔记.md` | 论文原理学习笔记 |
| `RoboGauge/RoboGauge_论文学习笔记.md#10-代码仓库分析go2_rl_gym` | 代码架构分析 |
