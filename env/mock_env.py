"""
env/mock_env.py
---------------
MockTrafficEnv — môi trường giao thông xếp hàng (queue-based) thay thế CityFlow
để kiểm thử pipeline trên máy không có binary CityFlow (AGENTS.md §5.2).

Tuân thủ ranh giới lớp: KHÔNG import torch / cityflow, chứa toàn bộ logic an
toàn (qua ``PhaseSafetyController``) và reward. Mỗi giao lộ có 4 hướng tiếp cận
× 7 làn = 28 làn → obs_dim = 84 (+ 5 nếu có khối pha), giống ``sample_map_2``.

Tính không đồng nhất (heterogeneity) được mô phỏng bằng ``IntersectionProfile``
(tổng lưu lượng và độ lệch giữa các hướng khác nhau).

API đa-agent (khác chữ ký đơn-agent trong PROJECT_GUIDELINES §6.1 vì có nhiều
giao lộ): ``step`` nhận ``dict[id, action]`` và trả obs/reward theo từng id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from env.obs_layout import ObsLayout
from env.safety import PhaseSafetyController, SafetyConfig

# ── Hằng số môi trường ───────────────────────────────────────────────────────
NUM_APPROACHES: int = 4
LANES_PER_APPROACH: int = 7
NUM_LANES: int = NUM_APPROACHES * LANES_PER_APPROACH      # 28
MAX_QUEUE_PER_LANE: int = 40
SAT_FLOW_PER_LANE: float = 0.15        # xe/giây/làn khi xanh
GREEN_SPEED_NORM: float = 0.8
DELTA_T_S: int = 5                     # giây mô phỏng mỗi bước quyết định
EPISODE_S: int = 1800

# ── Hệ số reward (MDP_SPEC §4) ───────────────────────────────────────────────
ALPHA_QUEUE: float = 0.02
BETA_THROUGHPUT: float = 0.05
GAMMA_SWITCH: float = 0.5
OVERRIDE_PENALTY: float = 0.2


@dataclass(frozen=True)
class IntersectionProfile:
    """Đặc trưng lưu lượng của một giao lộ.

    Attributes:
        total_rate: Tổng xe đến (xe/giây) trên cả 4 hướng.
        approach_weights: Tỉ lệ lưu lượng giữa 4 hướng (tự chuẩn hoá).
    """

    total_rate: float
    approach_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


# Hồ sơ mặc định: hai nhóm giao lộ khác nhau để thấy negative transfer.
DEFAULT_PROFILES: tuple[IntersectionProfile, ...] = (
    IntersectionProfile(1.10, (3.0, 1.0, 3.0, 1.0)),   # trục chính Bắc-Nam, đông
    IntersectionProfile(1.00, (3.0, 1.0, 2.0, 1.0)),
    IntersectionProfile(0.70, (1.0, 2.0, 1.0, 2.0)),   # thưa, ưu thế Đông-Tây
    IntersectionProfile(0.60, (1.0, 3.0, 1.0, 2.0)),
)


@dataclass
class _Node:
    profile: IntersectionProfile
    ctrl: PhaseSafetyController
    queue: np.ndarray = field(default_factory=lambda: np.zeros(NUM_LANES, dtype=np.int64))
    credit: np.ndarray = field(default_factory=lambda: np.zeros(NUM_LANES, dtype=np.float64))
    arrivals: np.ndarray = field(default_factory=lambda: np.zeros(NUM_LANES, dtype=np.float64))


class MockTrafficEnv:
    """Môi trường đa giao lộ dựa trên hàng đợi.

    Args:
        num_intersections: Số giao lộ (hồ sơ lưu lượng lấy xoay vòng).
        profiles: Danh sách hồ sơ lưu lượng; mặc định ``DEFAULT_PROFILES``.
        safety: Cấu hình an toàn dùng chung.
        episode_s: Độ dài episode (giây).
        seed: Seed ban đầu.
        delta_t_s: Số giây mô phỏng mỗi bước quyết định.
        include_phase: Nếu True (mặc định), state bắt đầu bằng [one-hot pha |
            elapsed/min_green] (MDP_SPEC §2.1) → obs_dim = 89; False → 84.
            Bố cục đầy đủ: xem ``env/obs_layout.py``.
    """

    def __init__(
        self,
        num_intersections: int = 4,
        profiles: Optional[tuple[IntersectionProfile, ...]] = None,
        safety: Optional[SafetyConfig] = None,
        episode_s: int = EPISODE_S,
        seed: int = 0,
        include_phase: bool = True,
        delta_t_s: int = DELTA_T_S,
    ) -> None:
        profiles = profiles or DEFAULT_PROFILES
        self.safety = safety or SafetyConfig()
        self.episode_s = episode_s
        self.ids: list[str] = [f"intersection_{i + 1}_1" for i in range(num_intersections)]
        self._profiles = {iid: profiles[i % len(profiles)] for i, iid in enumerate(self.ids)}
        self.include_phase = include_phase
        self.delta_t_s = delta_t_s
        self.layout = ObsLayout(self.safety.num_phases, NUM_LANES, include_phase)
        self.obs_dim: int = self.layout.dim
        self.num_phases: int = self.safety.num_phases
        self._rng = np.random.default_rng(seed)
        self._nodes: dict[str, _Node] = {}
        self._t: int = 0
        self.reset(seed=seed)

    # ── Gym-like API ─────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None) -> tuple[dict[str, np.ndarray], dict]:
        """Khởi động lại episode.

        Args:
            seed: Seed mới (tuỳ chọn).

        Returns:
            (obs theo giao lộ, info).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._nodes = {
            iid: _Node(self._profiles[iid], PhaseSafetyController(self.safety))
            for iid in self.ids
        }
        return self.get_state(), {}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], bool, bool, dict[str, dict]]:
        """Chạy ``DELTA_T_S`` giây với hành động đã qua kiểm tra an toàn.

        Args:
            actions: Pha xanh đề xuất cho từng giao lộ.

        Returns:
            (obs, rewards, terminated, truncated, info) — obs/rewards/info theo id.
        """
        rewards: dict[str, float] = {}
        info: dict[str, dict] = {}
        for iid, node in self._nodes.items():
            res = node.ctrl.request(int(actions[iid]))
            served = 0
            for _ in range(self.delta_t_s):
                served += self._tick_node(node)
            queue_sum = float(node.queue.sum())
            r = (-ALPHA_QUEUE * queue_sum
                 + BETA_THROUGHPUT * served
                 - GAMMA_SWITCH * float(res.switched)
                 - OVERRIDE_PENALTY * float(res.overridden))
            rewards[iid] = r
            info[iid] = {
                "throughput": served,
                "queue_sum": queue_sum,
                "overridden": res.overridden,
                "override_reason": res.reason,
                "switched": res.switched,
                "phase": node.ctrl.phase,
            }
        self._t += self.delta_t_s
        truncated = self._t >= self.episode_s
        return self.get_state(), rewards, False, truncated, info

    def get_state(self) -> dict[str, np.ndarray]:
        """Vector quan sát chuẩn hoá [0,1] gồm queue, speed, density mỗi làn."""
        return {iid: self._obs(n) for iid, n in self._nodes.items()}

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _tick_node(self, node: _Node) -> int:
        w = np.asarray(node.profile.approach_weights, dtype=np.float64)
        w = w / w.sum()
        lam = np.repeat(node.profile.total_rate * w / LANES_PER_APPROACH, LANES_PER_APPROACH)
        arr = self._rng.poisson(lam)
        node.arrivals = arr.astype(np.float64)
        node.queue = np.minimum(node.queue + arr, MAX_QUEUE_PER_LANE)

        phase = node.ctrl.tick()
        if phase is None:
            return 0
        lanes = slice(phase * LANES_PER_APPROACH, (phase + 1) * LANES_PER_APPROACH)
        node.credit[lanes] += SAT_FLOW_PER_LANE
        out = np.minimum(np.floor(node.credit[lanes]).astype(np.int64), node.queue[lanes])
        node.queue[lanes] -= out
        node.credit[lanes] -= np.floor(node.credit[lanes])
        return int(out.sum())

    def _obs(self, node: _Node) -> np.ndarray:
        q = node.queue / MAX_QUEUE_PER_LANE
        green = np.zeros(NUM_LANES, dtype=bool)
        if node.ctrl.state.value == "green":
            p = node.ctrl.phase
            green[p * LANES_PER_APPROACH:(p + 1) * LANES_PER_APPROACH] = True
        speed = np.where(node.queue == 0, 1.0, np.where(green, GREEN_SPEED_NORM, 0.0))
        density = np.minimum(1.0, (node.queue + node.arrivals) / MAX_QUEUE_PER_LANE)
        parts = []
        if self.include_phase:
            onehot = np.zeros(self.safety.num_phases)
            elapsed = 0.0                    # đang vàng / all-red: không có pha xanh
            if node.ctrl.state.value == "green":
                onehot[node.ctrl.phase] = 1.0
                elapsed = min(node.ctrl.green_elapsed / max(self.safety.min_green_s, 1), 1.0)
            parts += [onehot, np.array([elapsed])]
        parts += [q, speed, density]
        return np.concatenate(parts).astype(np.float32)

    def close(self) -> None:
        """Giải phóng tài nguyên (không có gì với mock)."""


if __name__ == "__main__":
    env = MockTrafficEnv(num_intersections=2)
    obs, _ = env.reset(seed=1)
    done = False
    total = 0.0
    while not done:
        obs, rew, _, done, info = env.step({i: np.random.randint(4) for i in env.ids})
        total += sum(rew.values())
    print("obs_dim", obs[env.ids[0]].shape, "return", round(total, 2))
    print("mock_env.py OK")
