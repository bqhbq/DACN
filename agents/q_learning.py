"""
agents/q_learning.py
--------------------
QLearningAgent — Q-learning một bước, ví dụ TỐI GIẢN về cách ráp một thuật
toán RL vào bộ khung NodeAgent.

Chỉ ghi đè hai hook: `select_action()` (ε-greedy) và `compute_loss()` (TD).
Không có replay buffer hay target network — bản đầy đủ hơn là
`agents/dqn.py`. Phần FL được thừa kế nguyên vẹn từ NodeAgent.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from agents.node_agent import NodeAgent
from agents.replay_buffer import TransitionBatch

DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.99        # hệ số chiết khấu γ (MDP_SPEC §1)
DEFAULT_REWARD_SCALE = 0.1  # co reward (đơn vị: số xe) về miền nhỏ để TD-target ổn định


class QLearningAgent(NodeAgent):
    """Agent value-based học ngay trên từng transition vừa quan sát.

    Args:
        intersection_id: ID giao lộ.
        obs_dim: Kích thước vector state.
        num_phases: Số pha xanh khả dụng.
        epsilon: Xác suất khám phá ε-greedy.
        gamma: Hệ số chiết khấu γ.
        reward_scale: Hệ số nhân reward trước khi tính TD-target.
        **kwargs: Tham số còn lại của `NodeAgent` (hidden_dim, lr, device, ...).
    """

    def __init__(
        self,
        intersection_id: str,
        obs_dim: int,
        num_phases: int,
        epsilon: float = DEFAULT_EPSILON,
        gamma: float = DEFAULT_GAMMA,
        reward_scale: float = DEFAULT_REWARD_SCALE,
        **kwargs: Any,
    ) -> None:
        super().__init__(intersection_id, obs_dim, num_phases, **kwargs)
        self.epsilon = epsilon
        self.gamma = gamma
        self.reward_scale = reward_scale

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Chọn action theo ε-greedy trên Q-values.

        Args:
            state: Vector state shape (obs_dim,).
            greedy: True → luôn chọn argmax Q.

        Returns:
            Index pha xanh mong muốn, thuộc [0, num_phases).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_phases))
        return super().select_action(state, greedy=True)

    def compute_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """TD loss một bước: Huber(Q(s, a), r·scale + γ·max_a' Q(s', a')).

        Args:
            batch: Lô transition vừa quan sát.

        Returns:
            Loss vô hướng.
        """
        q_sa = self.net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.net(batch.next_states).max(dim=1).values
            td_target = (
                batch.rewards * self.reward_scale
                + self.gamma * next_q * (1.0 - batch.dones)
            )
        return nn.functional.smooth_l1_loss(q_sa, td_target)
