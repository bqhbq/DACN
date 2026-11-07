"""
agents/dqn.py
-------------
DQN Trainer cho NodeAgent trong dự án FedMARL-ATSC.

Module này THAY THẾ hàm compute_dummy_loss() trong NodeAgent bằng
thuật toán DQN thực sự (Deep Q-Network, Mnih et al., 2015) với:
  - Target network (đồng bộ định kỳ)
  - Experience replay từ ReplayBuffer
  - Epsilon-greedy exploration với decay
  - Gradient clipping

Lớp này ĐỌC state từ NodeAgent nhưng KHÔNG tiếp xúc TrafficEnv hay FederatedServer.
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.node_agent    import NodeAgent
from agents.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNTrainer:
    """
    Trainer DQN cục bộ cho một NodeAgent.

    Thuật toán
    ----------
    1. select_action() → ε-greedy qua Q-network (online net)
    2. push(obs, action, reward, next_obs, done) → ReplayBuffer
    3. update() (khi buffer đủ):
       - Sample mini-batch
       - Tính TD target: y = r + γ · max_a' Q_target(s', a') · (1 − done)
       - Loss = MSE(Q_online(s,a), y)
       - Gradient step trên online net (backbone + head)
       - Soft / hard update target net
    4. get_shared_weights() → gửi lên FederatedServer

    Tham số
    -------
    agent           : NodeAgent sở hữu TrafficAgentNet
    buffer_capacity : Dung lượng replay buffer (mặc định 50 000)
    batch_size      : Kích thước mini-batch mỗi lần update (mặc định 64)
    gamma           : Hệ số chiết khấu (mặc định 0.99)
    target_sync_steps: Số bước giữa hai lần đồng bộ target net (mặc định 100)
    tau             : Soft update factor (nếu 1.0 → hard update)
    eps_start       : Epsilon khởi đầu (mặc định 1.0)
    eps_end         : Epsilon cuối (mặc định 0.05)
    eps_decay       : Số bước để epsilon giảm từ start → end (mặc định 5000)
    """

    def __init__(
        self,
        agent:            NodeAgent,
        buffer_capacity:  int   = 50_000,
        batch_size:       int   = 64,
        gamma:            float = 0.99,
        target_sync_steps:int   = 100,
        tau:              float = 1.0,
        eps_start:        float = 1.0,
        eps_end:          float = 0.05,
        eps_decay:        int   = 5_000,
    ) -> None:
        self.agent      = agent
        self.batch_size = batch_size
        self.gamma      = gamma
        self.target_sync_steps = target_sync_steps
        self.tau        = tau
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.device     = agent.device

        # Online network = agent.net (đã có)
        # Target network = bản sao đông lạnh, không train
        self.target_net = copy.deepcopy(agent.net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.buffer = ReplayBuffer(
            capacity = buffer_capacity,
            obs_dim  = agent.obs_dim,
        )

        self._total_steps:  int   = 0
        self._total_updates:int   = 0
        self._loss_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Epsilon Schedule
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Epsilon hiện tại theo lịch giảm tuyến tính."""
        progress = min(self._total_steps / self.eps_decay, 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    # ──────────────────────────────────────────────────────────────────────────
    # Chọn Hành động
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        """
        ε-greedy action selection với epsilon tự động giảm.

        Tham số
        -------
        obs : np.ndarray shape (obs_dim,) — observation từ TrafficEnv

        Trả về
        ------
        action_idx : int ∈ [0, num_phases)
        """
        self._total_steps += 1
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.agent.num_phases))

        self.agent.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q     = self.agent.net(obs_t)
        return int(q.squeeze(0).argmax().item())

    # ──────────────────────────────────────────────────────────────────────────
    # Lưu trữ Experience
    # ──────────────────────────────────────────────────────────────────────────

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Lưu một transition vào ReplayBuffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    # ──────────────────────────────────────────────────────────────────────────
    # DQN Update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Thực hiện một bước cập nhật gradient DQN.

        Trả về None nếu buffer chưa đủ dữ liệu.
        Trả về float (loss) nếu update thành công.
        """
        if not self.buffer.is_ready or len(self.buffer) < self.batch_size:
            return None

        self._total_updates += 1
        obs_t, act_t, rew_t, next_obs_t, done_t = self.buffer.sample(
            self.batch_size, device=self.device
        )

        # ── Q(s, a) từ online net ─────────────────────────────────────────
        self.agent.net.train()
        q_online = self.agent.net(obs_t)                   # (B, num_phases)
        q_sa     = q_online.gather(1, act_t.unsqueeze(1)).squeeze(1)  # (B,)

        # ── TD Target từ target net ───────────────────────────────────────
        with torch.no_grad():
            q_next_target = self.target_net(next_obs_t)    # (B, num_phases)
            max_q_next    = q_next_target.max(dim=1).values # (B,)
            td_target     = rew_t + self.gamma * max_q_next * (1.0 - done_t)

        # ── Huber Loss (ổn định hơn MSE với outlier) ─────────────────────
        loss = nn.functional.smooth_l1_loss(q_sa, td_target)

        self.agent.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.net.parameters(), max_norm=10.0)
        self.agent.optimizer.step()

        loss_val = float(loss.item())
        self._loss_history.append(loss_val)
        self.agent._loss_history.append(loss_val)

        # ── Đồng bộ Target Network ────────────────────────────────────────
        if self._total_updates % self.target_sync_steps == 0:
            self._sync_target_net()

        return loss_val

    def _sync_target_net(self) -> None:
        """
        Đồng bộ target network từ online network.
        Soft update: θ_target = τ·θ_online + (1−τ)·θ_target
        Hard update: τ=1.0 → copy toàn bộ.
        """
        if self.tau == 1.0:
            self.target_net.load_state_dict(copy.deepcopy(self.agent.net.state_dict()))
        else:
            for target_p, online_p in zip(
                self.target_net.parameters(), self.agent.net.parameters()
            ):
                target_p.data.mul_(1.0 - self.tau).add_(self.tau * online_p.data)
        logger.debug("[%s] Target net synced (τ=%.2f)", self.agent.intersection_id, self.tau)

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def loss_history(self) -> List[float]:
        return self._loss_history

    def stats(self) -> Dict:
        return {
            "intersection_id": self.agent.intersection_id,
            "total_steps":     self._total_steps,
            "total_updates":   self._total_updates,
            "epsilon":         round(self.epsilon, 4),
            "buffer_size":     len(self.buffer),
            "avg_loss_10":     float(np.mean(self._loss_history[-10:])) if self._loss_history else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"DQNTrainer(agent={self.agent.intersection_id!r}, "
            f"steps={self._total_steps}, eps={self.epsilon:.3f})"
        )
