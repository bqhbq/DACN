"""
env/safety.py
-------------
Bộ điều khiển an toàn pha đèn (C-01 … C-05, PROJECT_GUIDELINES.md §4).

Lớp ``PhaseSafetyController`` là state-machine thuần Python (không PyTorch,
không CityFlow) nên được dùng chung bởi ``MockTrafficEnv`` và wrapper CityFlow
thật. Agent chỉ "đề xuất" pha xanh; controller quyết định cái gì thực sự chạy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ── Mặc định (PROJECT_GUIDELINES §4.1) ───────────────────────────────────────
DEFAULT_NUM_PHASES: int = 4
DEFAULT_MIN_GREEN_S: int = 10
DEFAULT_YELLOW_S: int = 3
DEFAULT_ALL_RED_S: int = 1


class SignalState(Enum):
    """Trạng thái đèn của một giao lộ."""

    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass(frozen=True)
class SafetyConfig:
    """Cấu hình ràng buộc an toàn.

    Attributes:
        num_phases: Số pha xanh.
        min_green_s: Thời gian xanh tối thiểu (C-02), giây.
        yellow_s: Thời gian đèn vàng bắt buộc khi đổi pha (C-03), giây.
        all_red_s: Thời gian all-red sau vàng (C-04), 0 = tắt.
        strict_cycle: Nếu True, chỉ cho phép chuyển sang pha kế tiếp theo chu
            trình (C-01/C-05); đề xuất khác bị ép về pha kế tiếp.
    """

    num_phases: int = DEFAULT_NUM_PHASES
    min_green_s: int = DEFAULT_MIN_GREEN_S
    yellow_s: int = DEFAULT_YELLOW_S
    all_red_s: int = DEFAULT_ALL_RED_S
    strict_cycle: bool = False


@dataclass(frozen=True)
class RequestResult:
    """Kết quả xử lý một đề xuất hành động.

    Attributes:
        requested: Hành động agent đề xuất.
        applied: Pha xanh mục tiêu sau khi kiểm tra an toàn.
        overridden: True nếu đề xuất bị ghi đè (tính vào safety_violation_rate).
        reason: Mã lý do ghi đè ("invalid", "min_green", "cycle_order") hoặc None.
        switched: True nếu một chu kỳ chuyển pha (vàng → all-red) được bắt đầu.
    """

    requested: int
    applied: int
    overridden: bool
    reason: Optional[str]
    switched: bool


class PhaseSafetyController:
    """State-machine pha đèn cho một giao lộ.

    Quy trình mỗi giây mô phỏng: ``tick()`` trả về pha xanh đang phục vụ xe
    (hoặc ``None`` khi đang vàng / all-red). Ở đầu mỗi bước quyết định, gọi
    ``request(action)`` để agent đề xuất pha.

    Args:
        cfg: Cấu hình an toàn.
        initial_phase: Pha xanh ban đầu.
    """

    def __init__(self, cfg: SafetyConfig, initial_phase: int = 0) -> None:
        self.cfg = cfg
        self.phase: int = initial_phase
        self.state: SignalState = SignalState.GREEN
        self.green_elapsed: int = 0
        self._remaining: int = 0
        self._target: int = initial_phase

    def request(self, action: int) -> RequestResult:
        """Xử lý đề xuất pha của agent theo C-01 … C-05.

        Args:
            action: Chỉ số pha xanh mong muốn.

        Returns:
            ``RequestResult`` mô tả hành động thực sự được áp dụng.
        """
        # Đang chuyển pha: không nhận đề xuất mới, giữ mục tiêu hiện tại.
        if self.state is not SignalState.GREEN:
            return RequestResult(action, self._target, action != self._target, "in_transition", False)

        if not 0 <= action < self.cfg.num_phases:                       # C-05
            return RequestResult(action, self.phase, True, "invalid", False)
        if action == self.phase:
            return RequestResult(action, self.phase, False, None, False)

        if self.green_elapsed < self.cfg.min_green_s:                   # C-02
            return RequestResult(action, self.phase, True, "min_green", False)

        target, reason = action, None
        if self.cfg.strict_cycle:                                       # C-01
            nxt = (self.phase + 1) % self.cfg.num_phases
            if action != nxt:
                target, reason = nxt, "cycle_order"

        self._begin_transition(target)                                  # C-03
        return RequestResult(action, target, reason is not None, reason, True)

    def tick(self) -> Optional[int]:
        """Tiến 1 giây mô phỏng.

        Returns:
            Pha xanh đang phục vụ, hoặc ``None`` nếu đang vàng / all-red.
        """
        if self.state is SignalState.GREEN:
            self.green_elapsed += 1
            return self.phase

        self._remaining -= 1
        if self._remaining <= 0:
            if self.state is SignalState.YELLOW and self.cfg.all_red_s > 0:  # C-04
                self.state = SignalState.ALL_RED
                self._remaining = self.cfg.all_red_s
            else:
                self.phase = self._target
                self.state = SignalState.GREEN
                self.green_elapsed = 0
        return None

    def _begin_transition(self, target: int) -> None:
        self._target = target
        self.state = SignalState.YELLOW
        self._remaining = self.cfg.yellow_s


if __name__ == "__main__":
    c = PhaseSafetyController(SafetyConfig())
    for _ in range(3):
        c.tick()
    print(c.request(1))   # min_green -> overridden
    for _ in range(8):
        c.tick()
    print(c.request(1))   # switch allowed
    print([c.tick() for _ in range(6)])
    print("safety.py OK")
