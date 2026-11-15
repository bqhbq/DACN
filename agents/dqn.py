"""
agents/dqn.py
-------------
DQNAgent — một cài đặt off-policy đầy đủ trên bộ khung NodeAgent: replay
buffer, target network và ε giảm dần.

Đây là MỘT thuật toán mẫu, không phải mặc định của hệ thống. Nó minh hoạ
cách ghi đè `store_transition()` / `sample_batch()` để dùng bộ nhớ riêng.
Phần FL được thừa kế nguyên vẹn từ NodeAgent.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from agents.node_agent import NodeAgent
from agents.replay_buffer import DEFAULT_CAPACITY, ReplayBuffer, TransitionBatch

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
DEFAULT_GAMMA = 0.99
DEFAULT_REWARD_SCALE = 0.1
DEFAULT_TARGET_SYNC_STEPS = 100
DEFAULT_TAU = 1.0           # 1.0 → hard update target network
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY_STEPS = 5_000


class DQNAgent(NodeAgent):
    """Agent Deep Q-Network với experience replay và target network.

    Lưu ý về FL: `load_global_weights()` chỉ thay backbone của online network;
    target network bắt kịp ở lần đồng bộ kế tiếp (mỗi `target_sync_steps`
    bước cập nhật).

    Args:
        intersection_id: ID giao lộ.
        obs_dim: Kích thước vector state.
        num_phases: Số pha xanh khả dụng.
        buffer_capacity: Dung lượng replay buffer.
        batch_size: Kích thước mini-batch; chưa đủ thì chưa học.
        gamma: Hệ số chiết khấu γ.
        reward_scale: Hệ số nhân reward trước khi tính TD-target.
        target_sync_steps: Số bước cập nhật giữa hai lần đồng bộ target network.
        tau: Hệ số soft update (1.0 → sao chép toàn bộ).
        eps_start: ε ban đầu.
        eps_end: ε cuối.
        eps_decay_steps: Số lần chọn action để ε giảm tuyến tính từ đầu tới cuối.
        **kwargs: Tham số còn lại của `NodeAgent` (hidden_dim, lr, device, ...).
    """

    def __init__(
        self,
        intersection_id: str,
        obs_dim: int,
        num_phases: int,
        buffer_capacity: int = DEFAULT_CAPACITY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gamma: float = DEFAULT_GAMMA,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        target_sync_steps: int = DEFAULT_TARGET_SYNC_STEPS,
        tau: float = DEFAULT_TAU,
        eps_start: float = DEFAULT_EPS_START,
        eps_end: float = DEFAULT_EPS_END,
        eps_decay_steps: int = DEFAULT_EPS_DECAY_STEPS,
        **kwargs: Any,
    ) -> None:
        super().__init__(intersection_id, obs_dim, num_phases, **kwargs)
        self.batch_size = batch_size
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.target_sync_steps = target_sync_steps
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        # Target network: bản sao đông lạnh của online network, không train.
        self.target_net = copy.deepcopy(self.net).to(self.device)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self._action_steps = 0
        self._updates = 0

    @property
    def epsilon(self) -> float:
        """ε hiện tại theo lịch giảm tuyến tính."""
        progress = min(self._action_steps / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * progress

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Chọn action theo ε-greedy, ε tự giảm sau mỗi lần gọi.

        Args:
            state: Vector state shape (obs_dim,).
            greedy: True → luôn chọn argmax Q và không tính vào lịch ε.

        Returns:
            Index pha xanh mong muốn, thuộc [0, num_phases).
        """
        if greedy:
            return super().select_action(state, greedy=True)
        epsilon = self.epsilon
        self._action_steps += 1
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_phases))
        return super().select_action(state, greedy=True)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Đẩy transition vào replay buffer và tính nó vào n_k."""
        self.buffer.push(state, action, reward, next_state, done)
        self._register_local_samples()

    def sample_batch(self) -> Optional[TransitionBatch]:
        """Lấy ngẫu nhiên một mini-batch; None khi buffer chưa đủ `batch_size`."""
        if len(self.buffer) < self.batch_size:
            return None
        return self.buffer.sample(self.batch_size, device=self.device)

    def compute_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """TD loss với target network: Huber(Q(s, a), r·scale + γ·max Q_target(s', ·)).

        Args:
            batch: Mini-batch lấy từ replay buffer.

        Returns:
            Loss vô hướng.
        """
        q_sa = self.net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(batch.next_states).max(dim=1).values
            td_target = (
                batch.rewards * self.reward_scale
                + self.gamma * next_q * (1.0 - batch.dones)
            )
        return nn.functional.smooth_l1_loss(q_sa, td_target)

    def train_local_model(self) -> Optional[float]:
        """Một bước cập nhật DQN, kèm đồng bộ target network theo chu kỳ."""
        loss = super().train_local_model()
        if loss is not None:
            self._updates += 1
            if self._updates % self.target_sync_steps == 0:
                self._sync_target_net()
        return loss

    def _sync_target_net(self) -> None:
        """Cập nhật target network: θ_target ← τ·θ_online + (1 − τ)·θ_target."""
        if self.tau >= 1.0:
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            for target_param, online_param in zip(
                self.target_net.parameters(), self.net.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)
        logger.debug("[%s] Target net synced (τ=%.2f)", self.intersection_id, self.tau)

    def stats(self) -> Dict[str, Any]:
        """Bổ sung ε hiện tại và kích thước buffer vào số liệu tóm tắt."""
        return {
            **super().stats(),
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
        }
