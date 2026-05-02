# legged_gym 强化学习训练框架 —— 学习笔记

> 基于 legged_gym + rsl_rl + Isaac Gym 的四足机器人 PPO 强化学习完整解析

---

## 目录

1. [整体架构：三件套的分工](#1-整体架构三件套的分工)
2. [强化学习基础与 PPO 算法原理](#2-强化学习基础与-ppo-算法原理)
3. [Actor-Critic 架构详解](#3-actor-critic-架构详解)
4. [神经网络结构 —— MLP](#4-神经网络结构--mlp)
5. [GAE 广义优势估计](#5-gae-广义优势估计)
6. [PPO 完整训练流程](#6-ppo-完整训练流程)
7. [legged_gym 环境侧：从观测到力矩](#7-legged_gym-环境侧从观测到力矩)
8. [奖励函数设计](#8-奖励函数设计)
9. [训练监控与曲线解读](#9-训练监控与曲线解读)
10. [Go2 配置实战](#10-go2-配置实战)
11. [关键代码文件索引](#11-关键代码文件索引)

---

## 1. 整体架构：三件套的分工

legged_gym 的训练依赖于三个组件的协作：

```
┌──────────────────────────────────────────────────────────────┐
│                    Isaac Gym (NVIDIA)                        │
│  GPU 并行物理仿真引擎                                          │
│  - 同时模拟 4096 个机器人                                       │
│  - PhysX 物理引擎，200Hz 仿真步长 (dt=0.005s)                    │
│  - 提供 GPU Tensor API，数据无需 CPU-GPU 拷贝                    │
└──────────────────────────┬───────────────────────────────────┘
                           │ 物理状态 (root_states, dof_state, contact_forces)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   legged_gym (ETH Zurich)                     │
│  环境侧：定义机器人的观测空间、奖励函数、控制器                     │
│  - compute_observations():   原始物理数据 → 48/235 维观测向量    │
│  - compute_reward():         设计奖励函数引导策略行为             │
│  - _compute_torques():       action → PD 控制器 → 关节力矩      │
│  - Domain Randomization:     摩擦/质量/推搡随机化                │
│  - Terrain Curriculum:       地形难度逐渐递增                    │
└──────────────────────────┬───────────────────────────────────┘
                           │ obs (48维), reward, done
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     rsl_rl (ETH Zurich)                       │
│  算法侧：PPO 神经网络的训练与更新                                │
│  - Actor MLP:  48 → 512 → 256 → 128 → 12 (动作均值)            │
│  - Critic MLP: 48 → 512 → 256 → 128 → 1  (状态价值)            │
│  - RolloutStorage: 收集 24步 × 4096环境 的 transition           │
│  - GAE 计算:     优势函数 + 回报                                │
│  - PPO update:   Clipped Surrogate Loss + Value Loss + Entropy │
└──────────────────────────────────────────────────────────────┘
```

**核心数据流**：`观测 → Actor/Critic 网络 → 动作 → PD 控制器 → 力矩 → Isaac Gym 仿真 → 下一个观测 + 奖励`

**环境侧决定"学什么"，算法侧决定"怎么学"。**

---

## 2. 强化学习基础与 PPO 算法原理

### 2.1 强化学习的基本框架

强化学习（Reinforcement Learning, RL）的核心是一个**智能体（Agent）与环境（Environment）交互**的过程：

```
         ┌─── action (a_t) ───►
Agent    │                       Environment
         ◄── state (s_t) ────
         ◄── reward (r_t) ───
```

- **状态 s_t**：时刻 t 机器人能感知到的信息（在 legged_gym 中就是 48 维观测向量）
- **动作 a_t**：智能体输出的决策（12 个关节的目标偏转角）
- **奖励 r_t**：环境对动作的反馈（正 = 鼓励，负 = 惩罚）
- **目标**：找到一个策略 π(a|s)，使得**累积折扣奖励最大化**：

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
```

其中 γ（gamma，折扣因子）= 0.99，表示未来的奖励要打折扣——越远的未来越不重要。

### 2.2 为什么选择 PPO

PPO（Proximal Policy Optimization，近端策略优化）是目前四足机器人强化学习训练中**最主流**的算法。相比其他算法：

| 算法 | 特点 | 在四足机器人中的适用性 |
|------|------|----------------------|
| **PPO** | On-policy，稳定，超参宽容度高 | 主流选择，仿真快可掩盖 on-policy 的数据效率低 |
| SAC | Off-policy，数据效率高 | 连续控制也可用，但在高维观测下不稳定 |
| TD3 | Off-policy，确定性策略 | 探索不足，不适合多模态的步态行为 |
| TRPO | On-policy，理论保证强 | 实现复杂，已被 PPO 替代 |

### 2.3 PPO 的核心思想

PPO 的核心创新是**限制策略更新的幅度**，防止一次更新步子迈得太大导致策略崩溃。

#### 重要性采样比率（Ratio）

用**旧策略 π_old** 收集数据，用**新策略 π_new** 来更新。ratio 衡量同一动作在新旧策略下概率的比值：

```
ratio = π_new(a|s) / π_old(a|s)
```

- ratio > 1：新策略更倾向于选这个动作
- ratio < 1：新策略不太喜欢这个动作
- ratio = 1：新旧策略对这个动作看法一样

#### Clipped Surrogate Objective（PPO 的核心损失函数）

PPO 的 Actor 损失函数如下（代码在 `ppo.py:296-302`）：

```python
ratio = exp(new_log_prob - old_log_prob)                    # 概率比值
surrogate = -advantages × ratio                             # 不加约束的目标
surrogate_clipped = -advantages × clip(ratio, 0.8, 1.2)     # 裁剪到 [0.8, 1.2]
surrogate_loss = max(surrogate, surrogate_clipped).mean()   # 取保守的
```

**直观理解 PPO Clip（clip_param = 0.2）：**

| 情况 | advantage | ratio 应该 | PPO 做了什么 |
|------|-----------|-----------|-------------|
| 好动作 | > 0 | 增大（提高概率） | 最多允许 ratio=1.2，再大就裁剪 |
| 坏动作 | < 0 | 减小（降低概率） | 最多允许 ratio=0.8，再小就裁剪 |

**为什么要裁剪？** 因为数据是用旧策略采集的，如果新策略变化太大，旧数据就不准确了（importance sampling 失效）。裁剪保证每一步更新都是"安全的"。

#### 价值函数损失（Value Loss）

Critic 负责估计 V(s)，它的损失是预测值和实际回报的均方误差（同样做了裁剪）：

```python
value_loss = max((V_new - Return)², (V_clipped - Return)²).mean()
```

#### 熵奖励（Entropy Bonus）

```python
entropy_loss = -entropy_coef × entropy.mean()    # entropy_coef = 0.01
```

分布熵越大 → 探索越随机。减去熵损失意味着**鼓励探索**，防止策略过早收敛到次优解。

#### 总损失

```
总损失 = surrogate_loss + value_loss_coef × value_loss - entropy_coef × entropy_loss
       = surrogate_loss + 1.0 × value_loss - 0.01 × entropy_loss
```

### 2.4 自适应学习率

PPO 通过监控新旧策略之间的 KL 散度来动态调整学习率（代码在 `ppo.py:260-294`）：

```
如果 KL > desired_kl × 2 (即 > 0.02)：
    → 策略变化太大，学习率 ÷ 1.5
如果 KL < desired_kl / 2 (即 < 0.005)：
    → 策略变化太小，学习率 × 1.5
```

`desired_kl = 0.01`，学习率范围限制在 `[1e-5, 1e-2]` 之间。

---

## 3. Actor-Critic 架构详解

### 3.1 什么是 Actor-Critic

Actor-Critic 是 PPO 使用的网络架构，它包含**两个独立的神经网络**：

```
                      ┌─────────────┐
         观测 ────────┤   Actor     ├──→ 动作 (12维关节目标)
         (48维)       │  (策略网络)  │
                      └─────────────┘

                      ┌─────────────┐
         观测 ────────┤   Critic    ├──→ 价值 V(s) (1个标量)
         (48维)       │  (价值网络)  │
                      └─────────────┘
```

**两者共享同一个输入**（观测向量），但**参数完全独立**，各自有自己的权重和梯度。

### 3.2 Actor（演员）—— 决定做什么

Actor 的目标是**输出动作**。它不直接输出确定性动作，而是输出一个**高斯分布**：

```
Actor(obs) → μ (均值, 12维)
           → σ (标准差, 12维, 可学习参数, 初始值 1.0)

动作 = 从 N(μ, σ) 中随机采样
```

**为什么输出分布而不是确定值？** 因为需要**探索**——同一个观测下采样到略有不同的动作，才能发现更好的策略。标准差 σ 的大小决定了探索的程度：
- σ 大 → 动作随机性强 → 探索多
- σ 小 → 动作接近均值 → 策略稳定

`σ` 是一个 `nn.Parameter`，**在训练中会被梯度下降优化**——策略越学越好，σ 会从 1.0 逐渐衰减。

#### 代码实现

`actor_critic.py:102-119`：
```python
def update_distribution(self, obs):
    mean = self.actor(obs)                     # MLP 前向传播得到均值
    std = self.std.expand_as(mean)             # 可学习参数扩展到 12 维
    self.distribution = Normal(mean, std)      # 构造高斯分布

def act(self, obs):
    self.update_distribution(obs)              # 构造分布
    return self.distribution.sample()          # 随机采样动作
```

### 3.3 Critic（评论家）—— 评判状态好坏

Critic 的目标是**估计状态的价值 V(s)**——当前局面下，从这一刻开始，机器人最终能拿到多少累计奖励。

```
Critic(obs) → V(s) (1个标量)
```

V(s) 的作用贯穿整个训练：

1. **计算 Advantage 优势**：`A(s,a) = r + γV(s') - V(s)`，衡量动作比平均水平好多少
2. **计算 Return 回报**：`Return = Advantage + V(s)`，用于 Critic 自己的训练目标
3. **Bootstrap 超时**：episode 因超时结束（不是摔倒），用 `γV(s')` 作为后续奖励的估计

### 3.4 Actor 和 Critic 为什么分开

这是 Actor-Critic 架构的核心设计理由：

- **Actor 不能自我评判**：如果 Actor 既做动作又评判自己，就像球员同时当裁判，无法客观认识自己的失误
- **Critic 只评估不决策**：Critic 专注学习"什么样的状态是好状态"，不需要探索动作空间
- **分离后各自专注**：两者各司其职，梯度更新互不干扰

**Critic 只在训练时存在**——训练完成后部署到机器人时，只需要 Actor 来输出动作，Critic 可以直接丢弃。

---

## 4. 神经网络结构 —— MLP

### 4.1 什么是 MLP

MLP（Multi-Layer Perceptron，多层感知机）是**最简单也是 legged_gym 中使用的神经网络结构**。它由全连接层（Linear）+ 激活函数交替堆叠而成。

一个层本质上是一个**线性变换**：

```
输入: x (dim_in 个数字)
输出: y = W·x + b   (dim_out 个数字)

其中 W 是权重矩阵 [dim_out × dim_in]，b 是偏置向量 [dim_out]
```

### 4.2 Go2 的 MLP 维度变化：为什么是 48→512→256→128→12

```
Actor MLP:
  输入层: Linear(48, 512)   + ELU    48→512  "展开找模式"
  隐藏层1: Linear(512, 256) + ELU    512→256  "提炼"
  隐藏层2: Linear(256, 128) + ELU    256→128  "再提炼"
  输出层: Linear(128, 12)   (无激活)  128→12   "输出关节目标"

Critic MLP:
  输入层: Linear(48, 512)   + ELU    48→512
  隐藏层1: Linear(512, 256) + ELU    512→256
  隐藏层2: Linear(256, 128) + ELU    256→128
  输出层: Linear(128, 1)    (无激活)  128→1    "输出状态价值"
```

**这种"先扩后缩"的形状不是随意的：**

| 阶段 | 维度变化 | 直觉理解 |
|------|---------|---------|
| **展开** (48→512) | 扩大约 10 倍 | 原始 48 个数字被组合成 512 个"特征检测器"，每个神经元从不同角度观察状态 |
| **提炼** (512→256→128) | 逐层压缩 | 强迫网络丢掉冗余，保留最重要的信息。类似于：细节→概念→抽象 |
| **决策** (128→12) | 压缩到动作维度 | 将 128 维的抽象理解映射为 12 个具体的关节偏转值 |

### 4.3 权重数量统计

| 层 | 计算 | 参数量 |
|----|------|--------|
| Linear(48, 512) | 48 × 512 + 512 | 25,088 |
| Linear(512, 256) | 512 × 256 + 256 | 131,328 |
| Linear(256, 128) | 256 × 128 + 128 | 32,896 |
| Linear(128, 12) | 128 × 12 + 12 | 1,548 |
| **Actor 合计** | | **~19 万** |
| Linear(128, 1) | 128 × 1 + 1 | 129 |
| **Critic 合计** | | **~19 万** |
| **Actor + Critic 总参数量** | | **~38 万** |

### 4.4 ELU 激活函数

配置中的 `activation = 'elu'`。ELU（Exponential Linear Unit）公式：

```
ELU(x) = x           (x > 0)
       = α(eˣ - 1)   (x ≤ 0)
```

**为什么 ELU 而不是 ReLU？**

| 激活函数 | 公式 | 优点 | 缺点 |
|---------|------|------|------|
| ReLU | max(0, x) | 计算快 | x<0 时梯度为 0，神经元"死亡" |
| ELU | 如上 | 负半轴有输出，均值接近 0，训练更稳定 | 计算稍慢 |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 输出在 [-1,1] | 梯度消失问题 |

在 RL 中 ELU 通常优于 ReLU，因为 ELU 的平滑梯度信号让策略更新更稳定。

### 4.5 MLP 代码结构

`mlp.py` 中的 MLP 继承自 `nn.Sequential`，按顺序注册层：

```python
class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="elu"):
        # 按顺序添加层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))  # 输入层
        layers.append(activation_mod)                         # 激活

        for i in range(len(hidden_dims) - 1):                 # 隐藏层
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(activation_mod)

        layers.append(nn.Linear(hidden_dims[-1], output_dim)) # 输出层 (无激活)
```

**注意：最后一层没有激活函数**——因为动作均值可以是任意实数，价值 V(s) 也可以是任意实数，不能有范围限制。

### 4.6 Actor 和 Critic 都是 MLP，不是"MLP + Actor-Critic"

一个常见的误解是把 MLP 当成独立于 Actor-Critic 的特征提取器。**实际上 MLP 就是 Actor/Critic 的实现形式**：

```python
# actor_critic.py:51,61
self.actor  = MLP(48, 12, [512, 256, 128], 'elu')   # Actor 就是一个 MLP
self.critic = MLP(48,  1, [512, 256, 128], 'elu')   # Critic 是另一个 MLP
```

特征提取和决策输出在**同一个前向传播**中完成，不存在"MLP 先提取特征→Actor-Critic 再做决策"的两阶段过程。

---

## 5. GAE 广义优势估计

### 5.1 为什么需要 Advantage（优势函数）

在 RL 中，我们需要评判"动作 a 比平均水平好多少"——仅靠 奖励 r_t 不够：

- r_t 高不一定是因为这个动作好，可能是因为状态本身就很好
- r_t 低不一定是因为动作差，可能是因为状态本身就不好

**Advantage 消去了状态本身的好坏**，只衡量动作的"增量贡献"：

```
A(s,a) = Q(s,a) - V(s)
```

如果 A > 0：这个动作比预期的好
如果 A < 0：这个动作不如预期
如果 A = 0：这个动作中规中矩

### 5.2 GAE 的计算过程

GAE（Generalized Advantage Estimation）用**指数加权**的方式组合未来多步的 TD 误差，在偏差（bias）和方差（variance）之间取得平衡。

代码在 `rollout_storage.py:127-149`：

```python
def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(24)):                    # 从最后一步向前遍历
        next_is_not_terminal = 1.0 - self.dones[step]   # 如果不是终点
        # TD 误差: r + γV(s') - V(s)
        delta = self.rewards[step] + gamma * next_values - self.values[step]
        # GAE 累加
        advantage = delta + gamma * lam * advantage
        # Return = Advantage + V(s)
        self.returns[step] = advantage + self.values[step]
    # 归一化 advantages
    self.advantages = (self.advantages - mean) / (std + 1e-8)
```

#### 逐步展开

**第 1 步：TD 误差**

```
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
```

- 如果 δ_t > 0：这一步的动作带来了超出预期的收益 → 好动作
- 如果 δ_t < 0：这一步的动作做得不如预期 → 坏动作

**第 2 步：GAE 累加**

```
A_t = δ_t + (γλ) × δ_{t+1} + (γλ)² × δ_{t+2} + (γλ)³ × δ_{t+3} + ...
```

λ（lambda）控制依赖多少未来信息：

| λ 值 | 含义 | 效果 |
|------|------|------|
| λ = 0 | A_t = δ_t（只用一步 TD 误差） | 低方差但高偏差 |
| λ = 1 | A_t = 所有未来 δ 的累加（等于 Monte Carlo） | 无偏但高方差 |
| λ = 0.95 | 在 bias 和 variance 间折中（最常用） | 大部分 RL 任务的默认选择 |

**第 3 步：归一化**

```python
advantages = (advantages - mean) / (std + 1e-8)
```

将优势变为均值 0、标准差 1 的分布，让训练信号在不同 iteration 之间保持一致尺度。

### 5.3 数值例子

假设一个简化场景（γ=0.9, λ=0.95，3 步轨迹）：

```
步骤:     t=0    t=1    t=2
r:        0      1      0
V(s):     0.5    0.3    0.0

δ_2 = 0 + 0.9×0 - 0 = 0
δ_1 = 1 + 0.9×0 - 0.3 = 0.7
δ_0 = 0 + 0.9×0.3 - 0.5 = -0.23

A_2 = 0
A_1 = 0.7 + 0.9×0.95×0 = 0.7
A_0 = -0.23 + 0.9×0.95×0.7 = 0.37

Return_2 = 0 + 0 = 0
Return_1 = 0.7 + 0.3 = 1.0
Return_0 = 0.37 + 0.5 = 0.87
```

注意：t=0 的动作虽然即时奖励是 0，但因为带来了 t=1 的好结果，GAE 给它分配了正优势（0.37）。

---

## 6. PPO 完整训练流程

### 6.1 一次完整迭代的详细步骤

```
Iteration 开始 (例如第 100 次迭代)
│
├─ 阶段 1: Rollout 收集 (torch.inference_mode, 不计算梯度)
│   └─ for step in range(24):
│       ├─ actor(obs) → μ(12维) → 配合可学习 std → 采样 action(12维)
│       ├─ critic(obs) → V(s) (1维标量)
│       ├─ action × 0.25 + default_angle → PD 控制器 × 4 次物理步
│       ├─ 物理仿真 → reward, done, next_obs
│       └─ 存入 RolloutStorage:
│           (obs, action, reward, value, log_prob, mu, sigma, done)
│           形状: 每个都是 [24, 4096, dim]
│
├─ 阶段 2: GAE 计算
│   ├─ 用 critic 计算最后一步的 bootstrap value
│   ├─ 从 t=23 反向遍历到 t=0
│   ├─ 计算 TD 误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
│   ├─ 累加 GAE:  A_t = δ_t + γλ·A_{t+1}
│   ├─ 计算 Return: R_t = A_t + V(s_t)
│   └─ 归一化 advantages → mean=0, std=1
│
├─ 阶段 3: PPO 更新 (计算梯度, 更新网络参数)
│   ├─ 98,304 条 transition 随机打乱
│   ├─ 分成 4 个 mini-batch, 每个 24,576 条
│   └─ for epoch in range(5):         # 5 轮
│       └─ for mini_batch in range(4): # 4 个小批量
│           ├─ 当前网络重新计算:
│           │   ├─ action_log_prob (新策略对旧动作的评价)
│           │   ├─ value (新的状态价值估计)
│           │   └─ entropy (当前策略的探索程度)
│           ├─ ratio = exp(new_log_prob - old_log_prob)
│           ├─ 计算三个损失:
│           │   ├─ surrogate_loss = max(-A·ratio, -A·clip(ratio, 0.8, 1.2))
│           │   ├─ value_loss = max((V_new-R)², (V_clipped-R)²)
│           │   └─ entropy_loss = -0.01 × entropy
│           ├─ loss = surrogate + value - entropy
│           ├─ loss.backward()  + 梯度裁剪(max=1.0) + optimizer.step()
│           └─ 累计 loss 用于日志
│   └─ 清空 storage
│
├─ 阶段 4: 自适应学习率
│   ├─ 计算新旧策略间的 KL 散度
│   ├─ 如果 KL > 0.02: lr ÷ 1.5
│   ├─ 如果 KL < 0.005: lr × 1.5
│   └─ 更新 optimizer 中的 learning_rate
│
└─ 阶段 5: 日志与保存
    ├─ TensorBoard 记录 loss / reward / std / fps
    └─ 每 50 次迭代保存一次模型
```

### 6.2 数据量统计

| 指标 | 数值 | 说明 |
|------|------|------|
| 每次迭代收集的 transitions | 24 × 4096 = 98,304 | num_steps_per_env × num_envs |
| 每次迭代的梯度更新次数 | 5 × 4 = 20 | num_learning_epochs × num_mini_batches |
| 每次梯度更新的 batch size | 98,304 / 4 = 24,576 | transitions / num_mini_batches |
| 总迭代次数 | 1500 | max_iterations |
| 总环境交互量 | 1500 × 98,304 ≈ 1.47 亿 | 总 transitions |
| 典型训练时间 (RTX 4090) | 2-4 小时 | 取决于 reward 函数复杂度 |

### 6.3 关键超参数速查表

| 参数 | 值 | 作用 | 调大后果 | 调小后果 |
|------|-----|------|---------|---------|
| clip_param | 0.2 | 策略更新保守度 | 更新慢，收敛慢 | 不稳定，可能崩溃 |
| gamma | 0.99 | 未来奖励折现 | 更看重长远 | 只看眼前 |
| lam | 0.95 | GAE bias-variance | 更高方差 | 更高偏差 |
| entropy_coef | 0.01 | 探索鼓励 | 策略过于随机 | 过早收敛 |
| learning_rate | 1e-3 | 梯度步长 | 不稳定 | 学习慢 |
| num_steps_per_env | 24 | 每次收集步数 | 更多数据 | 可能不够 |
| num_learning_epochs | 5 | 重复学习轮数 | 可能过拟合 | 学习不充分 |
| num_mini_batches | 4 | 梯度更新精细度 | batch 更小 | batch 更大 |

---

## 7. legged_gym 环境侧：从观测到力矩

### 7.1 观测空间 (Observation Space)

在 `legged_robot.py:209-219` 的 `compute_observations()` 中，48 维观测由以下组成：

```python
self.obs_buf = torch.cat((
    base_lin_vel × 2.0,           # 3维: 基座线速度 (自身坐标系)
    base_ang_vel × 0.25,          # 3维: 基座角速度
    projected_gravity,            # 3维: 重力在自身坐标系的方向
    commands[:, :3] × scale,      # 3维: 速度指令 (前向/横向/转向)
    (dof_pos - default) × 1.0,   # 12维: 关节位置偏差
    dof_vel × 0.05,              # 12维: 关节角速度
    self.actions                 # 12维: 上一时刻的动作
), dim=-1)
# 总计: 3+3+3+3+12+12+12 = 48维
```

**如果是粗糙地形（measure_heights = True）**，会再拼上 187 维高度采样（17×11 的网格，每个点的高度减去机器人当前高度），总共 235 维。

### 7.2 Decimation：策略频率与物理频率

`decimation = 4`，物理仿真频率 200Hz（dt=0.005s）：

```
策略频率: 200/4 = 50Hz (每 20ms 决策一次)
物理频率: 200Hz (每 5ms 更新一次力矩)
```

`legged_robot.py:89-91`：
```python
for _ in range(self.cfg.control.decimation):  # 循环 4 次
    torques = self._compute_torques(self.actions)  # action 不变，但 PD 每次用最新的关节状态
    self.gym.simulate(self.sim)                    # 物理仿真步进
```

注意：4 次物理步中 **action 不变**，但 PD 控制器每次都读取最新的关节位置和速度重新计算力矩，起到了插值 + 稳定的作用。

### 7.3 PD 控制器：从动作到力矩

`legged_robot.py:367-368`：
```python
# P 控制 (position control)
target_angle = action × 0.25 + default_angle
torque = 20.0 × (target_angle - current_angle) - 0.5 × current_velocity
```

**为什么神经网络不直接输出力矩？**

- 力矩空间数值大且不稳定，直接预测更难学
- 位置控制把问题分解为"高层做决策（神经网络说去哪）+ 底层做执行（PD 控制器用物理规律驱动力矩）"
- 类比生物：大脑给目标位置，脊髓反射做力矩调节

`action_scale = 0.25` 限制了动作幅度——action 是 [-1, 1] 之间的值，×0.25 后关节目标最多偏移 ±0.25 rad (~14°)，防止过于激进。

### 7.4 域随机化 (Domain Randomization)

在 `legged_robot_config.py` 中配置：

| 随机化项 | 范围 | 作用 |
|---------|------|------|
| 摩擦系数 | [0.5, 1.25] | 让策略适应不同地面（冰面→橡胶） |
| 推搡 | 每 15s 随机推 ≤1m/s | 让策略学会从扰动中恢复 |
| 质量随机化 | 可选的 [-1, +1] kg | 让策略适应不同负载 |

这些随机化**只在训练时做**，目的是让策略"见多识广"，部署到真机时能适应各种未见过的情况。

### 7.5 课程学习 (Curriculum Learning)

#### 地形课程

机器人走得越远 → 地形难度越高（从平地到楼梯、斜坡）；走不动了 → 退回简单地形。

#### 指令课程

当 tracking_lin_vel 奖励超过最大值的 80% → 逐渐增大指令速度范围。

---

## 8. 奖励函数设计

### 8.1 所有可用的奖励函数

代码在 `legged_robot.py:816-906`。奖励分为**正向激励**和**负向惩罚**两类：

#### 正向奖励（鼓励策略做某种行为，值域一般为 0~1）

| 函数 | 公式 | 激励的行为 |
|------|------|-----------|
| `_reward_tracking_lin_vel` | `exp(-(v_cmd - v_actual)² / 0.25)` | 精确跟踪速度指令 |
| `_reward_tracking_ang_vel` | `exp(-(ω_cmd - ω_actual)² / 0.25)` | 精确跟踪转向指令 |
| `_reward_feet_air_time` | `∑(悬空时间 - 0.5s) × 首次触地` | 鼓励大步幅，有腾空相位的步态 |

#### 负向惩罚（抑制策略做某种行为，值域一般为负值）

| 函数 | 公式 | 惩罚的行为 |
|------|------|-----------|
| `_reward_torques` | `-∑(torque²)` | 关节力矩过大（耗能） |
| `_reward_dof_pos_limits` | `-∑超出软限位的角度` | 关节接近 URDF 中的物理极限 |
| `_reward_dof_vel` | `-∑(vel²)` | 关节转动过快 |
| `_reward_dof_acc` | `-∑(acc²)` | 关节加速度过大（动作不平滑） |
| `_reward_action_rate` | `-∑(a_t - a_{t-1})²` | 动作变化过于剧烈（抽搐） |
| `_reward_collision` | 大腿/小腿接触地面 | 自碰撞 |
| `_reward_orientation` | `-∑重力在xy方向的分量` | 身体倾斜不稳定 |
| `_reward_lin_vel_z` | `-v_z²` | 身体上下晃动 |
| `_reward_base_height` | `-(h_actual - h_target)²` | 身体高度偏离目标 |
| `_reward_stand_still` | `-∑关节偏转 × is_zero_command` | 静止时乱动 |

### 8.2 reward 缩放机制

`legged_robot.py:544-562` 的 `_prepare_reward_function()`：

```python
for key in list(self.reward_scales.keys()):
    scale = self.reward_scales[key]
    if scale == 0:
        self.reward_scales.pop(key)          # scale 为 0 的项直接删除
    else:
        self.reward_scales[key] *= self.dt   # 非零项乘以 dt (0.005)
```

**scale 为 0 的奖励项会被完全移除**，其对应的函数不会被调用，节省计算。这就是为什么 Go2 配置中只设置了 torques 和 dof_pos_limits 两个非零值——其他的 scale 都是 0。

### 8.3 推荐的 Go2 奖励配置

如果只激活 torques 和 dof_pos_limits，策略只会学到"省力 + 别超关节限位"，不会走路。建议至少启用：

```python
class scales(LeggedRobotCfg.rewards.scales):
    tracking_lin_vel = 1.0      # 鼓励跟踪速度指令（核心）
    tracking_ang_vel = 0.5      # 鼓励跟踪转向指令（核心）
    torques = -0.0002           # 省力
    dof_pos_limits = -10.0      # 关节保护
    base_height = -30.0         # 保持目标高度
    lin_vel_z = -2.0            # 稳定行走
    ang_vel_xy = -0.05          # 身体不摇晃
    action_rate = -0.01         # 动作平滑
    orientation = -5.0          # 身体保持水平
    collision = -1.0            # 避免自碰撞
```

---

## 9. 训练监控与曲线解读

### 9.1 Loss 曲线（核心学习信号）

| 曲线 | 健康走势 | 危险信号 |
|------|---------|---------|
| **Loss/value_function** | 持续下降，最终稳定 | 一直不降 → Critic 容量不够；突然飙升 → 策略崩溃 |
| **Loss/surrogate** | 初期下降，中期小幅波动，长期稳定 | 持续为 0 → 策略停止学习；剧烈震荡 → 学习率太高 |
| **Loss/entropy** | 从高缓慢下降到低 | 快速降到接近 0 → 过早收敛到局部最优；一直很高 → 策略没学到东西 |
| **Loss/learning_rate** | 初期小幅波动，中期稳定在 1e-5~1e-2 之间 | 长期卡在最低值 → 几乎学不动 |

### 9.2 Policy 曲线

| 曲线 | 健康走势 | 危险信号 |
|------|---------|---------|
| **Policy/mean_noise_std** | 从 ~1.0 逐渐降到 0.3~0.6 | 降得比 entropy 还快 → 过度自信；一直 ~1.0 → 策略等于是纯随机 |

**mean_noise_std 是最直接的"策略是否在收敛"指标。**

### 9.3 Train 曲线（整体训练进展）

| 曲线 | 健康走势 | 危险信号 |
|------|---------|---------|
| **Train/mean_reward** | **持续上升** | 长期为负且不改善 → reward 设计问题；剧烈下降 → 策略崩溃 |
| **Train/mean_episode_length** | 初期可能短，逐渐增长到 20s | 长期低于 5s → 机器人一直摔倒，没学会站立 |

**这两个是"训练该不该继续"的核心判断指标。** 如果在 200-300 次迭代后 mean_reward 没有上升趋势，需要重新审视 reward 设计。

### 9.4 Episode 曲线（各项奖励分解）

- `Episode/rew_tracking_lin_vel`：跟踪指令速度的好程度。**越高越好**，0.8+ 表示跟踪很准
- `Episode/rew_torques`：关节力矩惩罚。**接近 0 越好**（允许小幅为负）
- `Episode/rew_dof_pos_limits`：关节限位惩罚。**理想值为 0**。持续恶化 → 关节频繁超限
- `Episode/rew_collision`：自碰撞计数。**理想值为 0**
- `Episode/rew_orientation`：身体倾斜。**为 0 最好**，持续恶化说明姿态不稳

### 9.5 健康的训练过程

```
阶段1 (0-100 iter):
  mean_reward 从负值快速上升
  episode_length 从 ~3s 增长到 10-15s
  entropy 从高开始下降
  mean_noise_std 从 1.0 开始缓慢下降

阶段2 (100-500 iter):
  mean_reward 继续上升但速度放缓
  episode_length 接近 20s（满时长）
  tracking_lin_vel 从 0 上升到 0.5-0.7

阶段3 (500-1500 iter):
  mean_reward 缓慢改善
  tracking_lin_vel 稳定在 0.7-0.9
  mean_noise_std 稳定在 0.3-0.5
  各项惩罚接近 0
```

---

## 10. Go2 配置实战

### 10.1 配置文件结构

`go2_config.py` 包含两个类：

```python
class Go2RoughCfg(LeggedRobotCfg):       # 环境配置
    class init_state: ...                # 机器人初始姿态
    class control: ...                   # PD 控制器参数
    class asset: ...                     # URDF 文件路径
    class rewards: ...                   # 奖励权重

class Go2RoughCfgPPO(LeggedRobotCfgPPO): # PPO 算法配置
    class algorithm: ...                 # PPO 超参数
    class runner: ...                    # 训练流程参数
```

### 10.2 Go2 关节名称与默认角度

Go2 有 4 条腿 × 3 个关节 = 12 个 DOF：

```python
default_joint_angles = {
    'FL_hip_joint': 0.1,     'RL_hip_joint': 0.1,
    'FR_hip_joint': -0.1,    'RR_hip_joint': -0.1,
    'FL_thigh_joint': 0.8,   'RL_thigh_joint': 1.0,
    'FR_thigh_joint': 0.8,   'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5,   'RL_calf_joint': -1.5,
    'FR_calf_joint': -1.5,   'RR_calf_joint': -1.5,
}
```

### 10.3 启动训练

```bash
PYTORCH_JIT=0 python legged_gym/scripts/train.py --task=go2
```

`PYTORCH_JIT=0` 禁用 TorchScript JIT 编译，避免 RTX 4090（compute capability 8.9）的 nvrtc 兼容性问题。

### 10.4 注册新任务

在 `legged_gym/envs/__init__.py` 中：

```python
from .go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO
task_registry.register("go2", LeggedRobot, Go2RoughCfg(), Go2RoughCfgPPO())
```

---

## 11. 关键代码文件索引

| 文件 | 作用 | 核心内容 |
|------|------|---------|
| `legged_gym/envs/base/legged_robot.py` | 环境主逻辑 | `step()`, `compute_observations()`, `compute_reward()`, `_compute_torques()`, `post_physics_step()` |
| `legged_gym/envs/base/legged_robot_config.py` | 配置类 | `LeggedRobotCfg`(环境参数), `LeggedRobotCfgPPO`(PPO参数) |
| `legged_gym/envs/go2/go2_config.py` | Go2 配置 | `Go2RoughCfg`, `Go2RoughCfgPPO` |
| `legged_gym/utils/task_registry.py` | 任务注册 | `task_registry.register()` |
| `legged_gym/utils/math.py` | 数学工具 | `torch_rand_float`, `quat_apply_yaw`, `wrap_to_pi` |
| `rsl_rl/modules/actor_critic.py` | Actor-Critic 网络 | `ActorCritic` 类, `update_distribution()`, `act()`, `evaluate()` |
| `rsl_rl/networks/mlp.py` | MLP 网络 | `MLP` 类 (继承自 `nn.Sequential`) |
| `rsl_rl/algorithms/ppo.py` | PPO 算法 | `PPO` 类, `act()`, `process_env_step()`, `compute_returns()`, `update()` |
| `rsl_rl/runners/on_policy_runner.py` | 训练主循环 | `OnPolicyRunner` 类, `learn()` |
| `rsl_rl/storage/rollout_storage.py` | 数据存储 | `RolloutStorage` 类, `compute_returns()`, `mini_batch_generator()` |

---

## 总结

legged_gym 的完整训练可以归结为一条主线：

```
观测(48维) → Actor/MLP → 动作均值 μ → + 可学习 std σ → 采样 → action(12维)
           → Critic/MLP → 状态价值 V(s)

action → PD 控制器 → 关节力矩 → Isaac Gym 仿真 → 下一个观测 + 奖励

奖励 → GAE 计算 Advantage → PPO 裁剪更新 → 策略逐步优化
```

**38 万个参数，4096 个并行环境，每次迭代 9.8 万条经验，1500 次迭代，总共约 1.47 亿次环境交互**——这让 Go2 从随机抽搐逐步学会稳定行走。

RL 训练的核心挑战不是网络结构（Actor-Critic + MLP 的架构相对固定），而是**奖励函数的设计**——你告诉策略"什么是好"，它就会往那个方向优化。奖励函数既是科学也是艺术，需要反复调整和观察训练曲线来迭代。
