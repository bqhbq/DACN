"""
agents/replay_buffer.py
-----------------------
Experience Replay Buffer cho DQN agent trong dự án FedMARL-ATSC.

Lớp này THUỘC VỀ NodeAgent — không được sử dụng bởi TrafficEnv hay FederatedServer.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Một bộ kinh nghiệm (s, a, r, s', done)."""
    obs:      np.ndarray   # shape (obs_dim,)
    action:   int
    reward:   float
    next_obs: np.ndarray   # shape (obs_dim,)
    done:     bool


class ReplayBuffer:
    """
    Experience Replay Buffer kiểu vòng (circular).

    Tham số
    -------
    capacity  : Số lượng transition tối đa được lưu (mặc định 50 000)
    obs_dim   : Kích thước vector observation
    seed      : Seed ngẫu nhiên để tái tạo kết quả
    """

    def __init__(
        self,
        capacity: int = 50_000,
        obs_dim:  int = 84,
        seed:     int = 42,
    ) -> None:
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        random.seed(seed)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Lưu một transition vào buffer."""
        self._buffer.append(Transition(
            obs      = np.array(obs,      dtype=np.float32),
            action   = int(action),
            reward   = float(reward),
            next_obs = np.array(next_obs, dtype=np.float32),
            done     = bool(done),
        ))

    def sample(
        self,
        batch_size: int,
        device:     Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lấy ngẫu nhiên một mini-batch.

        Trả về
        ------
        (obs, actions, rewards, next_obs, dones) — tất cả dưới dạng Tensor
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer chỉ có {len(self)} transitions, cần {batch_size}."
            )
        transitions = random.sample(self._buffer, batch_size)
        dev = device or torch.device("cpu")

        obs_t      = torch.tensor(np.stack([t.obs      for t in transitions]), device=dev)
        actions_t  = torch.tensor([t.action  for t in transitions], dtype=torch.long,  device=dev)
        rewards_t  = torch.tensor([t.reward  for t in transitions], dtype=torch.float32, device=dev)
        next_obs_t = torch.tensor(np.stack([t.next_obs for t in transitions]), device=dev)
        dones_t    = torch.tensor([t.done    for t in transitions], dtype=torch.float32, device=dev)

        return obs_t, actions_t, rewards_t, next_obs_t, dones_t

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True nếu buffer đã đủ dữ liệu để bắt đầu training."""
        return len(self._buffer) >= 64

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
