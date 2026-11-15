"""
agents/baselines/actuated.py
----------------------------
Baseline #2: điều khiển thích ứng theo cảm biến (greedy longest-queue).

Đọc queue qua ``env.obs_layout.ObsLayout`` (bố cục dùng chung); làn xếp liên
tiếp theo hướng (7 làn / hướng), pha p phục vụ hướng p.
"""
from __future__ import annotations

import numpy as np

from env.obs_layout import ObsLayout

LANES_PER_APPROACH: int = 7
HYSTERESIS_QUEUE: float = 0.05   # chỉ đổi pha nếu hướng khác dài hơn ngưỡng này


class GreedyQueuePolicy:
    """Chọn pha phục vụ hướng có hàng đợi dài nhất (có hysteresis).

    Args:
        num_phases: Số pha xanh.
        num_lanes: Số làn vào.
        include_phase: State có khối [pha | elapsed] ở đầu không.
    """

    def __init__(self, num_phases: int = 4, num_lanes: int = 28, include_phase: bool = True) -> None:
        self.num_phases = num_phases
        self.num_lanes = num_lanes
        self.layout = ObsLayout(num_phases, num_lanes, include_phase)
        self._current = 0

    def reset(self) -> None:
        """Đặt lại pha hiện tại."""
        self._current = 0

    def select_action(self, obs: np.ndarray, greedy: bool = True) -> int:
        """Trả về pha có tổng queue lớn nhất."""
        queue = obs[self.layout.queue_slice]
        q = queue.reshape(self.num_phases, LANES_PER_APPROACH).mean(axis=1)
        best = int(q.argmax())
        if q[best] - q[self._current] > HYSTERESIS_QUEUE:
            self._current = best
        return self._current
