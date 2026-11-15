"""
agents/baselines/fixed_time.py
------------------------------
Baseline #1: chu kỳ đèn cố định, không phụ thuộc lưu lượng.
"""
from __future__ import annotations

import numpy as np

DEFAULT_GREEN_STEPS: int = 4   # 4 bước × 5 s = 20 s xanh mỗi pha


class FixedTimePolicy:
    """Luân phiên các pha theo chu kỳ cố định.

    Args:
        num_phases: Số pha xanh.
        green_steps: Số bước quyết định giữ mỗi pha.
    """

    def __init__(self, num_phases: int = 4, green_steps: int = DEFAULT_GREEN_STEPS) -> None:
        self.num_phases = num_phases
        self.green_steps = green_steps
        self._t = 0

    def reset(self) -> None:
        """Đặt lại bộ đếm bước."""
        self._t = 0

    def select_action(self, obs: np.ndarray, greedy: bool = True) -> int:
        """Trả về pha theo lịch cố định (bỏ qua ``obs``)."""
        a = (self._t // self.green_steps) % self.num_phases
        self._t += 1
        return int(a)
