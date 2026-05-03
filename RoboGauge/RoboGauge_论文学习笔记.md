# RoboGauge 论文学习笔记

> **论文**: Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
> **作者**: 西安交通大学 (Tianyang Wu et al.)
> **发表**: arXiv:2602.00678, Jan 2026
> **项目页**: https://robogauge.github.io/complete/
> **机器人**: Unitree Go2

---

## 目录

1. [论文概述](#1-论文概述)
2. [与原始 legged_gym 的总体对比](#2-与原始-legged_gym-的总体对比)
3. [MoE 网络架构详解](#3-moe-网络架构详解)
4. [RoboGauge 评估套件](#4-robogauge-评估套件)
5. [训练侧改进：指令与奖励优化](#5-训练侧改进指令与奖励优化)
6. [域随机化：训练 + 评估双重用途](#6-域随机化训练--评估双重用途)
7. [训练闭环流程](#7-训练闭环流程)
8. [真机部署结果](#8-真机部署结果)
9. [关键代码与配置](#9-关键代码与配置)

---

## 1. 论文概述

### 1.1 核心贡献

| 贡献 | 说明 |
|------|------|
| **RoboGauge 评估套件** | 基于 MuJoCo 的 Sim2Sim 评估框架，用 proprioception 指标预判 Sim2Real 迁移能力 |
| **MoE 运动策略** | Mixture-of-Experts 架构替换学生编码器，实现多地形/多指令的专家化表征 |
| **4 m/s 高速运动** | 纯 proprioception 达到 4 m/s，出现自发窄步态 (narrow-width gait) |
| **极强鲁棒性** | 30 cm 障碍物、100 N 侧向冲击、60 cm 跌落恢复、雪地/沙地/冰面 100% 成功率 |

### 1.2 解决的问题

原始 legged_gym 训练存在三个问题：

1. **Training reward 不可靠**：高仿真 reward 不代表真机能跑好（Sim2Real gap）
2. **真机测试风险高**：没有量化指标来选模型，只能频繁物理验证，效率低且危险
3. **单一 MLP 表达能力不足**：对所有地形/指令共用一套表征，复杂场景泛化差

---

## 2. 与原始 legged_gym 的总体对比

| 维度 | 原始 legged_gym | RoboGauge |
|------|----------------|-----------|
| **网络架构** | 单一 Actor MLP | **MoE (Mixture-of-Experts)** — 多专家子网络 + 门控网络 |
| **训练框架** | Teacher-Student (RMA) 或 CTS | **CTS + MoE**：CTS 的学生编码器替换为 MoE |
| **评估方法** | 只看 training reward | **RoboGauge 评估套件**：Sim2Sim 预测 Sim2Real |
| **评估环境** | Isaac Gym（训练环境） | **MuJoCo**（独立于训练的评估环境） |
| **并行环境数** | 4096 | **8192** |
| **最高速度** | < 2 m/s | **4.01 m/s** |
| **奖励函数** | ~15 个标准项 | 新增 hip position、hip symmetry、foot regulation |
| **指令采样** | 均匀随机采样 | 指令课程 + 极端值 20% + 动态采样 |
| **域随机化** | 摩擦 + 质量 + 推搡 | 新增 PD 增益、执行器强度/偏置、延迟、COM、link mass |

---

## 3. MoE 网络架构详解

### 3.1 MoE 替换了什么位置

**MoE 只替换了 Actor 内部的编码器部分，不是替换整个 Actor-Critic 架构。**

RoboGauge 基于 **CTS (Concurrent Teacher-Student)** 框架：

```
原始 legged_gym (单 Actor-Critic):
  obs → [单一 MLP] → Actor 头 → action(12维)
                    → Critic 头 → V(s)

CTS 框架:
  Teacher（有特权信息: 摩擦系数、地形高度等）:
    obs_privileged → MLP 编码器 → Actor/Critic

  Student（只有 proprioception，部署到真机用）:
    obs → MLP 编码器 → Actor/Critic
    └── 蒸馏损失：Student 隐状态逼近 Teacher ──┘

RoboGauge (本文):
  Teacher 不变

  Student:
    obs → MoE 编码器 → Actor/Critic
          ↑ 只有这里换了！其余不变
          ├─ Expert₁(obs) ──┐
          ├─ Expert₂(obs) ──┤
          ├─ ...            ├→ Σ ω_k × Expert_k(obs) → 隐状态 z → Actor/Critic
          ├─ Expert_K(obs) ─┘
          └─ Gating(obs) → [ω₁, ω₂, ..., ω_K]
```

### 3.2 MoE 三个组件

**① 专家子网络 (Experts)**
- K 个并行的 MLP，每个专家独立处理观测
- 不同专家自然分化：有的擅长平地上高速跑，有的擅长爬楼梯，有的擅长斜坡
- 这是"隐式分工"——不需要人工指定哪个专家负责什么，训练过程中自动分化

**② 门控网络 (Gating Network)**
- 输入同一个 obs，输出 K 维 softmax 权重
- `ω = softmax(g(obs))` → 决定"当前场景听哪个专家的话"

**③ 负载均衡损失 (Load Balancing Loss)**
```
L_load_balance = Σ(ω̄_k - 1/K)²
```
防止门控网络"偷懒"只用一两个专家（那就退化回单一 MLP），强制均匀利用所有专家。

### 3.3 消融实验：MoE 放在哪最好？

| 放置方式 | 效果 | 原因 |
|---------|------|------|
| **MoE 在编码器（本文选择）** | 最好，训练稳定 | 编码器输出隐状态，组合后平滑 |
| AC-MoE（MoE 在动作输出层） | 容易 loss 发散 | 门控+专家的同时学习导致控制信号剧烈波动 |
| MCP（乘法组合策略） | 同样不稳定 | 同上 |
| MoE-NG（去掉门控） | 稍差 | 去掉门控后专家间缺乏协调 |

### 3.4 PCA 可视化证据

论文通过 PCA 降维可视化 Student 编码器的隐空间：
- **有 MoE**：不同地形 (flat/wave/stairs/obstacle) 的隐状态在 PCA 空间中清晰分离 → 证明不同专家学到了针对不同场景的专用表征
- **无 MoE (CTS only)**：各种地形的隐状态混在一起 → 单一 MLP 无法区分场景

---

## 4. RoboGauge 评估套件

### 4.1 为什么需要独立评估

Isaac Gym 的 **PhysX 物理引擎**在接触力计算、碰撞响应、摩擦模型等方面与真实世界有系统性偏差。因此高 training reward ≠ 真机能跑好。

RoboGauge 用 **MuJoCo** 做评估，因为 MuJoCo 的物理参数更接近真实世界。**RoboGauge 不参与训练，是训练后的离线评估工具。**

### 4.2 三层流水线

```
┌─────────────────────────────────────────────────────────┐
│  Base Pipeline（基础流水线）                              │
│  - 单种子、单地形、单域随机化                               │
│  - 计算 6 个 proprioception 指标                          │
│  - 运行 3 种运动目标：最大速度、对角线速度、目标位置到达      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Multi/Level Pipeline（多难度流水线）                      │
│  - 5 种随机种子并行评估                                    │
│  - 9 种域随机化配置                                       │
│  - 二分搜索找最大可行难度：成功率 ≥ 80% 升级，否则降级       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Stress Pipeline（压力测试流水线）                          │
│  - 汇总 7 种地形 × 10 级难度 × 9 种域随机化                 │
│  - 综合评分 → 选出最优模型                                  │
└─────────────────────────────────────────────────────────┘
```

### 4.3 6 个评估指标（全部只用 proprioception）

| 指标 | 类别 | 衡量什么 |
|------|------|---------|
| Linear Velocity Error | Tracking（任务完成） | 线速度跟踪 ℓ₂ 误差 |
| Angular Velocity Error | Tracking（任务完成） | 角速度跟踪 ℓ₂ 误差 |
| DOF Power | Safety（安全性） | 关节功耗 |
| DOF Limits | Safety（安全性） | 关节角度超出软限位的程度 |
| Orientation Stability | Safety（安全性） | 重力在机器人侧向 (y) 轴的投影 |
| Torque Smoothness | Quality（运动质量） | 力矩的时间平滑度 |

### 4.4 评分机制

**加权几何平均**（防止单项高分掩盖另一项低分）：
```
Q(L) = ∏ m_k^(w_k) ^ (1/Σ w_k)
```
- 任务完成类指标权重 w=2，其余 w=1
- 如果任何一项接近 0，几何平均会直接拉低总分

**Worst-Case Mean（最差 50% 均值）**：对所有运动目标计算得分后，只取表现较差的 50% 求平均。简单指令的高分被忽略，专注评估困难场景。

**重叠评分函数**（平衡难度和质量）：
```
S = α × (L* − 1) + β × Q(L*)
```
- α = 0.09, β = 0.19
- L* = 二分搜索找到的最高可行难度级别
- Q(L*) = 该难度下的执行质量评分

### 4.5 MuJoCo vs Isaac Gym：为什么 MuJoCo 更准？

论文表 III 的量化对比（与真机测量值对比的误差）：

| 指标类别 | Isaac Gym 误差 | RoboGauge(MuJoCo) 误差 | 改善 |
|---------|---------------|----------------------|------|
| Tracking | 0.0883 | **0.0558** | 误差小 37% |
| Safety | 0.0333 | **0.0117** | 误差小 65% |
| Quality | 0.0380 | **0.0120** | 误差小 68% |

---

## 5. 训练侧改进：指令与奖励优化

### 5.1 动态速度跟踪精度 (Dynamic σ)

原始 legged_gym 的 `tracking_sigma = 0.25` 固定不变。RoboGauge 根据**地形难度 L 和指令大小 v** 动态调整：

```
σ_now = σ + min(e^(L/10) − 1, 1) × (σ_vel − σ)
```

- 简单地形 + 低速 → σ 宽松（不要求精确跟踪）
- 困难地形 + 高速 → σ 收紧（严格要求精确性）

不同地形的 σ_max：

| 地形 | σ_max |
|------|-------|
| Flat | 1/4 |
| Wave | 5/12 |
| Slope | 1/4 |
| Stairs Up | 1/2 |
| Stairs Down | 1/2 |
| Obstacle | 3/4 |

### 5.2 指令课程 (Command Curriculum)

避免"一上来全速"导致的步态不稳（跳跃、高频抖腿）：

| 阶段 | 训练步数 | v_x | v_y | ω_z |
|------|---------|-----|------|------|
| 初期 | 0 ~ 20K | ±0.5 m/s | ±0.5 m/s | ±1.0 rad/s |
| 中期 | 20K ~ 50K | ±1.0 | ±1.0 | ±1.5 |
| 后期 | 50K ~ | ±2.0 | ±1.0 | ±2.0 |

### 5.3 极端指令采样

原始框架均匀随机采样 → 边界值（最大速度）概率极低。RoboGauge 额外分配：

| 指令类型 | 概率 | 目的 |
|---------|------|------|
| 静止 (v=0) | 10% | 学习站立不动 |
| 最大速度组合 (三维都取最大) | 20% | 覆盖极限工况 |
| 最大角速度 (线速度=0时) | 20% | 强化原地转向 |
| 其余常规指令 | 50% | 保持多样性 |

### 5.4 动态指令采样

保证采样的命令序列能让机器人走够 >4m（地形长度的一半），否则课程学习无法升到更高级别。这确保 Agent 在探索更高难度地形前已有足够的前进能力。

### 5.5 新增奖励项

| 奖励项 | 公式 | 目的 |
|--------|------|------|
| **Hip Joint Position** | `|q_hip − q_hip_default|` | 防止高速时大腿外展 |
| **Hip Symmetry** | `|v_cmd_x| × symmetry_penalty` | 高速直线时保持对称姿态 |
| **Foot Regulation (rfr)** | 来自 CTS | 摆动相时保证足部有足够离地高度 |

---

## 6. 域随机化：训练 + 评估双重用途

### 6.1 训练阶段的域随机化 —— 让策略"见多识广"

| 随机化项 | 范围 | 为什么加 |
|---------|------|---------|
| 摩擦系数 | [0.5, 1.5] | 适应冰面→橡胶不同地面 |
| 负载质量 | [-1, +1] kg | 适应不同载荷 |
| **连杆质量** | × [0.9, 1.1] | 制造公差导致各连杆质量偏差 |
| **质心 COM** | ±3 cm (xyz) | 实际质心与 URDF 模型的差异 |
| 恢复系数 | [0.0, 0.5] | 地面软硬程度 |
| **PD kp** | × [0.9, 1.1] | 真机 PD 控制器参数与仿真的差异 |
| **PD kd** | × [0.9, 1.1] | 同上 |
| **执行器强度** | × [0.8, 1.2] | 电机力矩输出的实际误差 |
| **执行器偏置** | ±0.035 rad | 关节零点偏移、齿轮间隙等 |
| **控制延迟** | [0, 20] ms | 通信延迟（策略输出→电机执行） |

对比原始 legged_gym 新增了 **6 项**（加粗标注）。这些随机化在**每次训练 step 都重新采样**，策略在"千奇百怪的机器人"上训练后，到真机上不会因为某个参数偏差就崩溃。

### 6.2 评估阶段的域随机化 —— 给策略做压力测试

RoboGauge 评估时用 **9 种不同的域随机化配置（M=9）**对同一个策略做压力测试：

```
对于每种地形 × 每个难度:
  DR₁: 低摩擦 + 有负载 + 有延迟
  DR₂: 高摩擦 + 无负载 + 无延迟
  DR₃: ...
  ...
  DR₉: ...

每个配置 × 3 个随机种子 → 取平均
计算 6 个指标的加权几何平均 → Quality Score
```

**评估阶段的域随机化不参与训练**，只用于测试策略在各种环境扰动下的鲁棒性。

---

## 7. 训练闭环流程

与原始 legged_gym 的"训练一次→部署"不同，RoboGauge 形成了一个**闭环**：

```
┌─────────────────────────────────────────┐
│  阶段 1: Isaac Gym 训练                   │
│  - CTS + MoE 架构                        │
│  - 8192 并行环境                          │
│  - 指令课程 + 极端采样 + 动态 σ            │
│  - 扩展域随机化                            │
│  - 保存模型 checkpoint                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  阶段 2: RoboGauge 离线评估 (MuJoCo)       │
│  - 7 地形 × 10 难度 × 9 域随机化           │
│  - 6 个指标 → 综合评分                     │
│  - 评分准确反映真机表现（误差远小于           │
│    Isaac Gym 直接评估）                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  阶段 3: 根据评分调整                       │
│  - 评分不够好？→ 调整奖励/域随机化/课程      │
│  - 回到阶段 1 重新训练                      │
│  - 循环直到 RoboGauge 评分满意              │
│  - 满足阈值 → 部署真机                      │
└─────────────────────────────────────────┘
```

**核心价值**：开发者不需要频繁在真机上冒险测试，靠 RoboGauge 评分就能判断模型好坏，直到评分达标才上真机。

---

## 8. 真机部署结果

### 8.1 生存率对比

| 模型 | 侧向冲击 (80-100N) | 瓷砖楼梯 15.5cm | 30cm 障碍物 |
|------|-------------------|----------------|------------|
| **RoboGauge (Ours)** | **18/20** | **85/85** | **17/20** |
| Built-in RL (宇树官方) | 5/20 | 85/85 | 0/20 |
| CTS | 11/20 | 18/85 | 0/20 |
| HIM | 8/20 | 24/85 | 0/20 |
| DreamWaQ | 7/20 | 12/85 | 0/20 |

**只有 RoboGauge 成功跨越了 30 cm 障碍物**，且侧向冲击抗扰能力远超所有 baseline。

### 8.2 速度跟踪能力

| 场景 | 平均速度 | 跟踪误差 |
|------|---------|---------|
| 平地 (高速) | **4.01 m/s** (峰值) | 0.20 m/s |
| 楼梯 | 1.31 m/s | 0.15 m/s |
| 30° 斜坡 | 1.53 m/s | 未知 |

### 8.3 野外测试

雪地、沙地、冰面、不平坦地形 → **100% 成功率，零意外终止**。

### 8.4 自发性窄步态

高速运动时 (>3.5 m/s)，机器人自发出现 **narrow-width gait**（窄步态），四肢落点更靠近身体中线。这不是人为设计的奖励导致的，而是策略为保持高速稳定**自发学会**的行为。

---

## 9. 关键代码与配置

### 9.1 训练规模

- 8192 并行环境（Isaac Gym）
- 3 个随机种子训练，选 RoboGauge 评分最高的
- 训练步数：120K（对比原始 legged_gym 1500 iter × 24 step × 4096 = ~147M transitions，这里约 120K × 8192 ≈ 983M → 实际是 120K steps × 8192 = ~983M transitions）

### 9.2 奖励函数配置

详细奖励表见论文 Table VIII。核心与原始 legged_gym 一致，新增：

- `Hip regulation`: `|q_hip − q_hip_default|`, weight = -0.05
- `Hip symmetry`: 高速直线运动时对称约束, weight = -1.0
- `Foot regulation`: 来自 CTS 的足部离地高度奖励

### 9.3 代码链接

项目页：https://robogauge.github.io/complete/
代码包含完整 Train / Evaluate / Deploy 流程

---

## 10. 代码仓库分析（go2_rl_gym）

> 基于实际代码 `go2_rl_gym-master/` 的逐文件分析

### 10.1 仓库结构

```
go2_rl_gym-master/
├── legged_gym/                       # 环境代码（1426行基类，重度修改自原始框架）
│   ├── envs/base/
│   │   ├── legged_robot.py           #   1426行，融合 turn_over / 动态σ / 指令课程
│   │   │                             #   / PD随机化 / 执行器随机化 / 延迟等新功能
│   │   └── legged_robot_config.py    #   407行，新增 CTS/MoE 等配置基类
│   ├── envs/go2/
│   │   ├── go2_env.py                #   Go2 专用：重写观测(45维) + 新 reward
│   │   ├── go2_config.py             #   完整版配置 (论文主打)
│   │   ├── go2_config_vanilla.py     #   简化版 (接近原始 legged_gym)
│   │   ├── go2_config_vanilla_with_dynamic_cmd.py
│   │   └── go2_config_fast_flat_move.py  # 高速平地专用
│   ├── envs/__init__.py              #   注册 7 个 task
│   ├── scripts/train.py              #   训练入口 (仅23行)
│   └── utils/
│       ├── exporter.py               #   新增：导出 JIT/ONNX 模型
│       └── isaacgym_utils.py         #   新增：Isaac Gym 工具函数
│
├── rsl_rl/                           #   自带的 RL 算法库
│   ├── modules/
│   │   ├── actor_critic.py           #   原始单 Actor-Critic
│   │   ├── actor_critic_cts.py       #   CTS (Teacher-Student)
│   │   ├── actor_critic_moe_cts.py   #   MoE-CTS (论文主打)
│   │   ├── actor_critic_moe_ng_cts.py
│   │   ├── actor_critic_ac_moe_cts.py
│   │   ├── actor_critic_mcp_cts.py
│   │   ├── actor_critic_dual_moe_cts.py
│   │   └── utils.py                  #   MoE/Experts/Gating/StudentMoEEncoder 实现
│   ├── algorithms/
│   │   ├── ppo.py                    #   原始 PPO
│   │   ├── cts.py                    #   CTS 算法
│   │   ├── moe_cts.py                #   MoE-CTS 算法 (论文主打)
│   │   ├── moe_ng_cts.py
│   │   ├── ac_moe_cts.py
│   │   ├── mcp_cts.py
│   │   └── dual_moe_cts.py
│   ├── storage/                      #   含 RolloutStorageCTS
│   └── runners/
│       ├── on_policy_runner.py       #   原始 runner + 导出功能
│       └── on_policy_runner_cts.py   #   CTS runner
│
├── resources/robots/go2/             #   Go2 URDF + 7个 mesh 文件
├── deploy/                           #   部署代码
│   ├── deploy_mujoco/                #   MuJoCo 评估 (= RoboGauge 评估套件)
│   ├── deploy_real/                  #   真机部署 (Unitree Go2)
│   └── pre_train/go2/                #   预训练模型
├── tools/                            #   辅助脚本
└── setup.py
```

### 10.2 MoE 核心代码实现

MoE 在 `rsl_rl/modules/utils.py` 中由三个类组成：

**① MLP** (通用多层感知机)
```python
class MLP(nn.Module):
    def __init__(self, dims, activation='elu', last_activation=False):
        # dims = [input_dim, h1, h2, ..., output_dim]
        # 中间层自动插入激活函数，最后一层默认无激活
```

**② Experts** (专家组)
```python
class Experts(nn.Module):
    # backbone: 共享的 MLP → (B, expert_num × expert_hidden_dim)
    # experts: 分组 Conv1d (groups=expert_num) → 每个专家独立输出
    # 输出形状: (B, expert_num, output_dim)
```

**③ MoE** (门控 + 专家)
```python
class MoE(nn.Module):
    def forward(self, x):
        weights = self.gating_network(x)      # (B, expert_num), softmax
        expert_outs = self.experts(x)          # (B, expert_num, output_dim)
        output = sum(weights * expert_outs)    # (B, output_dim) 加权和
        return output, weights
```

**④ StudentMoEEncoder** (学生编码器)
```python
class StudentMoEEncoder(nn.Module):
    # MoE → L2Norm/SimNorm
    # 输出归一化的 32 维隐状态
```

### 10.3 Actor-Critic MoE-CTS 网络结构

`actor_critic_moe_cts.py` — 论文主打的完整网络：

```
Teacher 路径:
  privileged_obs(263维) → MLP([263, 512, 256, 32]) → L2Norm → latent(32维)

Student 路径:
  history = 3帧 obs(45维) = 135维  # 3帧历史拼接
  history → StudentMoEEncoder(8 experts) → latent(32维)
            ├─ Gating(135 → ... → 8) → softmax → ω
            ├─ Expert₁  ─┐
            ├─ ...       ├→ Σ ω_k × Expert_k(135) → 32维 → L2Norm
            └─ Expert₈  ─┘

Actor (共享):
  [latent(32) + obs(45)] = 77维 → MLP([77, 512, 256, 128, 12])

Critic (共享):
  [latent(32) + privileged_obs(263)] = 295维 → MLP([295, 512, 256, 128, 1])
```

**关键设计决策**：
- Teacher 和 Student 共享 Actor 和 Critic（只有编码器不同）
- Student 使用 **3 帧观测历史** (`history_length=3`)，即 `45×3=135` 维输入
- Gating 网络的输入是 **history 拼接后的 135 维**，即门控基于观测历史做决策
- 蒸馏损失：`L_latent = MSE(teacher_latent, student_latent)`
- 负载均衡损失：`L_balance = Σ(ω̄_k − 1/K)²`

### 10.4 MoE-CTS 算法的两个优化器

`moe_cts.py` 使用**双优化器**设计：

| 优化器 | 管理参数 | 学习率 |
|--------|---------|--------|
| optimizer1 | Teacher编码器 + Actor + Critic + std | 1e-3 (自适应) |
| optimizer2 | Student MoE 编码器 | 1e-3 (独立) |

**每轮更新分两步**：
1. **RL 更新** (optimizer1)：PPO surrogate loss + value loss + entropy — 让 Actor 学会好的动作
2. **蒸馏更新** (optimizer2)：latent loss + load_balance loss — 让 Student 编码器逼近 Teacher

**CTS 的 Teacher/Student 混合机制**：
- `teacher_env_ratio = 0.75`：75% 环境用 Teacher 做决策，25% 环境用 Student 做决策
- Teacher 有特权信息 → 更容易做出好决策 → 产生的高质量数据同时训练两个网络
- Student 虽然只有 proprioception，但通过蒸馏逼近 Teacher 的隐状态，间接学到特权信息

### 10.5 Go2 观测空间

#### Actor 观测 (45维，纯 proprioception，Sim2Real 友好)

```python
# go2_env.py compute_observations()
base_ang_vel (3)       # 角速度（不含线速度！）
projected_gravity (3)   # 重力方向
commands (3)            # 速度指令
dof_pos - default (12)  # 关节位置偏差
dof_vel (12)            # 关节速度
last_actions (12)       # 历史动作
```

**与原始 legged_gym 的关键区别**：去掉了 `base_lin_vel`。因为真机上速度估计不准（需要状态估计器），不让策略依赖它直接提升 Sim2Real 效果。

#### Critic 特权观测 (263维，仅训练用)

Actor 的 45 维 + **base_lin_vel(3)** + **足端接触力(4)** + **关节力矩(12)** + **关节加速度(12)** + **高度采样(187)**

### 10.6 已注册的 7 个任务

| task 参数 | 网络架构 | 论文对应 |
|-----------|---------|---------|
| `--task=go2` | 单一 Actor-Critic (PPO) | 原始 PPO 基线 |
| `--task=go2_cts` | CTS (Teacher-Student) | CTS 基线 |
| `--task=go2_moe_cts` | **MoE-CTS** | **论文主打的完整版** |
| `--task=go2_moe_ng_cts` | MoE-No-Goal-CTS | 去掉门控消融 |
| `--task=go2_mcp_cts` | MCP-CTS | 乘法组合消融 |
| `--task=go2_ac_moe_cts` | AC-MoE-CTS | MoE在动作层消融 |
| `--task=go2_dual_moe_cts` | Dual-MoE-CTS | 双重MoE消融 |

### 10.7 Go2 配置关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | **8192** | 并行环境数（原始 4096 的 2 倍） |
| `max_iterations` | **150000** | 总迭代数（不乘 num_steps_per_env） |
| `episode_length_s` | 25 | Episode 长度 |
| `num_observations` | 45 | Actor 观测维度 |
| `num_privileged_obs` | 263 | Critic 特权观测维度 |
| `latent_dim` | 32 | Teacher/Student 隐状态维度 |
| `student_expert_num` | 8 | MoE 专家数量 |
| `teacher_env_ratio` | 0.75 | Teacher 决策的环境比例 |
| `load_balance_coef` | 0.01 | 负载均衡损失权重 |
| `norm_type` | `l2norm` | 隐状态归一化方式 |
| `history_length` | 3 | Student 观测历史帧数 |

### 10.8 训练命令

```bash
# 进入代码目录
cd go2_rl_gym-master

# 安装（需先 pip uninstall legged_gym 移除旧版）
pip install -e .
cd rsl_rl && pip install -e . && cd ..

# RTX 40 系显卡需要禁用 TorchScript JIT
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2_moe_cts

# 其他可选任务
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2           # 原始 PPO
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2_cts       # CTS 基线
```

### 10.9 与原始 legged_gym 的代码差异总结

| 改动点 | 原始 legged_gym | go2_rl_gym |
|--------|----------------|-----------|
| `legged_robot.py` 行数 | ~900 | **1426** |
| `legged_robot_config.py` 行数 | ~240 | **407** |
| 观测组成 | 含 base_lin_vel | **不含** base_lin_vel |
| 域随机化项数 | 3 | **10** |
| 网络架构 | 1种 (Actor-Critic) | **7种** (PPO/CTS/MoE/...) |
| 算法文件 | 1个 (ppo.py) | **7个** |
| 指令采样 | 均匀随机 | **课程 + 极端 + 动态** |
| tracking_sigma | 固定 0.25 | **动态**（地形+指令相关） |
| reward curriculum | 无 | **有**（部分 reward 系数随时间变化） |
| turn_over 恢复 | 无 | **有**（摔倒翻转后自动恢复） |

---

## 总结

RoboGauge 论文在 legged_gym 基础上做了三个层次的创新：

| 层次 | 创新 | 核心价值 |
|------|------|---------|
| **网络层** | MoE 编码器替换单一 MLP | 多地形/多指令的专家化表征，泛化能力大幅提升 |
| **评估层** | RoboGauge (MuJoCo) | 不依赖真机测试预判 Sim2Real 表现，安全迭代 |
| **训练层** | 指令课程 + 极端采样 + 动态 σ + 扩展域随机化 | 训练更稳定，高速性能显著提升 |

最终在 Go2 上实现了 4 m/s 高速运动、30 cm 障碍跨越、100 N 冲击抗扰，且全部只用 proprioception。
