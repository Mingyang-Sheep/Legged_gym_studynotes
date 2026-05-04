"""
导出 MoE-CTS 训练 checkpoint 为 TorchScript JIT 模型（供 RoboGauge 评估使用）

用法（在服务器上）:
  cd ~/lmy/RL/RoboGauge
  python export_jit.py

会自动从训练 log 目录加载 model_5500.pt，导出 policy_jit_5500.pt 到同目录
"""
import sys, os

# 把 rsl_rl 加到 Python 路径（不需要 import isaacgym，全程 CPU）
sys.path.insert(0, os.path.expanduser('~/lmy/RL/go2_rl_gym/go2_rl_gym-master/rsl_rl'))

import torch
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS

# ---- 配置 ----
CKPT_PATH = os.path.expanduser(
    '~/lmy/RL/go2_rl_gym/logs/go2_moe_cts/May03_12-19-00_/model_5500.pt'
)
HISTORY_LENGTH = 5   # 训练配置的 history_length

# ---- 重建网络并加载权重 ----
model = ActorCriticMoECTS(
    num_obs=45,
    num_critic_obs=263,
    num_actions=12,
    num_envs=8192,
    history_length=HISTORY_LENGTH,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    teacher_encoder_hidden_dims=[512, 256],
    student_encoder_hidden_dims=[512, 256, 256],
    expert_num=8,
    activation='elu',
    init_noise_std=1.0,
    latent_dim=32,
    norm_type='l2norm',
)

ckpt = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded checkpoint from {CKPT_PATH} (iter={ckpt["iter"]})')

# ---- TorchScript 导出包装器 ----
class Exporter(torch.nn.Module):
    """包装 Student MoE Encoder + Actor 为一个可导出的 TorchScript 模块。

    维护内部 5 帧观测历史缓存，每次前向传入 1 帧 45 维观测，
    返回 (12维动作, (8个专家权重, 32维隐状态))。
    """
    def __init__(self):
        super().__init__()
        self.student_moe_encoder = model.student_moe_encoder.cpu()
        self.actor = model.actor.cpu()
        self.register_buffer('history', torch.zeros(1, HISTORY_LENGTH, 45))

    def forward(self, x):
        # x: [1, 45] 单帧观测
        # 滑动窗口：丢掉最旧一帧，拼入新一帧
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        # Student MoE 编码：135 维历史 → 32 维隐状态
        latent, weights = self.student_moe_encoder(self.history.flatten(1))
        # Actor：32(隐状态) + 45(当前观测) = 77 维 → 12 维动作
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (weights, latent)

    @torch.jit.export
    def reset(self):
        """重置历史缓存（RoboGauge 在每次评估开始时会调用）"""
        self.history = torch.zeros_like(self.history)


# ---- 导出 ----
exporter = Exporter().eval().cpu()
traced = torch.jit.script(exporter)

out_path = os.path.join(os.path.dirname(CKPT_PATH), 'policy_jit_5500.pt')
traced.save(out_path)

print(f'JIT model saved to {out_path}')

# 验证可加载
test = torch.jit.load(out_path)
test.reset()
action, (weights, latent) = test(torch.randn(1, 45))
print(f'Verify OK - action shape: {action.shape}, weights: {weights.shape}, latent: {latent.shape}')
