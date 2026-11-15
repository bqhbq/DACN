"""
agents/replay_buffer.py
-----------------------
Kiểu dữ liệu transition và Experience Replay Buffer cho các NodeAgent.

- `Transition`      : một bộ kinh nghiệm (s, a, r, s', done) dạng NumPy.
- `TransitionBatch` : một lô transition đã xếp thành tensor — đây là thứ
                      `NodeAgent.compute_loss()` nhận vào.
- `collate()`       : gom danh sách `Transition` thành `TransitionBatch`.
- `ReplayBuffer`    : bộ nhớ vòng cho các thuật toán off-policy.

Các lớp này THUỘC VỀ phía agent — `FederatedServer` và môi trường không dùng tới.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch

DEFAULT_CAPACITY = 50_000
DEFAULT_SEED = 42


@dataclass
class Transition:
    """Một bộ kinh nghiệm (s, a, r, s', done).

    Attributes:
        state: State s tại thời điểm ra quyết định, shape (obs_dim,).
        action: Action a đã chọn tại s.
        reward: Reward r môi trường trả về cho (s, a).
        next_state: State s' quan sát được ở lần ra quyết định kế tiếp.
        done: True nếu s' là trạng thái kết thúc episode.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool = False


class TransitionBatch(NamedTuple):
    """Một lô transition dạng tensor, B = kích thước lô.

    Attributes:
        states: float32, shape (B, obs_dim).
        actions: int64, shape (B,).
        rewards: float32, shape (B,).
        next_states: float32, shape (B, obs_dim).
        dones: float32 (0.0 / 1.0), shape (B,).
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


def collate(
    transitions: Sequence[Transition], device: Optional[torch.device] = None
) -> TransitionBatch:
    """Gom nhiều `Transition` thành một `TransitionBatch`.

    Args:
        transitions: Danh sách transition, không rỗng.
        device: Thiết bị đặt tensor; None → CPU.

    Returns:
        Lô tensor theo đúng thứ tự của `transitions`.

    Raises:
        ValueError: Nếu `transitions` rỗng.
    """
    if not transitions:
        raise ValueError("collate() nhận danh sách transition rỗng.")
    dev = device or torch.device("cpu")
    return TransitionBatch(
        states=torch.tensor(
            np.stack([t.state for t in transitions]), dtype=torch.float32, device=dev
        ),
        actions=torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=dev
        ),
        rewards=torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=dev
        ),
        next_states=torch.tensor(
            np.stack([t.next_state for t in transitions]),
            dtype=torch.float32,
            device=dev,
        ),
        dones=torch.tensor(
            [float(t.done) for t in transitions], dtype=torch.float32, device=dev
        ),
    )


class ReplayBuffer:
    """Experience Replay Buffer kiểu vòng: đầy thì transition cũ nhất bị loại.

    Args:
        capacity: Số transition tối đa được lưu.
        seed: Seed của bộ sinh ngẫu nhiên riêng của buffer (không đụng tới
            `random` toàn cục).
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, seed: int = DEFAULT_SEED) -> None:
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Lưu một transition vào buffer."""
        self._buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> TransitionBatch:
        """Lấy ngẫu nhiên (không lặp) một mini-batch.

        Args:
            batch_size: Số transition cần lấy.
            device: Thiết bị đặt tensor; None → CPU.

        Returns:
            Lô `batch_size` transition.

        Raises:
            ValueError: Nếu buffer có ít hơn `batch_size` transition.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(f"Buffer chỉ có {len(self)} transitions, cần {batch_size}.")
        return collate(self._rng.sample(list(self._buffer), batch_size), device)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
