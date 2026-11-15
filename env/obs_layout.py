"""
env/obs_layout.py
-----------------
Hợp đồng DUY NHẤT về bố cục vector state (docs/MDP_SPEC.md §2.3).

    [ phase one-hot × P | elapsed × 1 | queue × L | speed × L | density × L ]

Chỉ chứa hằng số và phép tính chỉ số — không import torch hay cityflow — nên
môi trường, baseline và agent đều dùng chung được. ``agents.behavior_cloning.
ObservationLayout`` đọc đúng bố cục này (có test đối chiếu trong tests/).
"""
from __future__ import annotations

from dataclasses import dataclass

FEATURES_PER_LANE: int = 3          # queue, speed, density


@dataclass(frozen=True)
class ObsLayout:
    """Vị trí các nhóm đặc trưng trong vector state.

    Attributes:
        num_phases: P — số pha xanh.
        num_lanes: L — số làn vào giao lộ.
        include_phase: Có phần [phase one-hot | elapsed] ở đầu không.
    """

    num_phases: int
    num_lanes: int
    include_phase: bool = True

    @property
    def phase_dim(self) -> int:
        """Số chiều của khối [one-hot pha | elapsed]; 0 nếu không có."""
        return self.num_phases + 1 if self.include_phase else 0

    @property
    def dim(self) -> int:
        """Tổng số chiều của vector state."""
        return self.phase_dim + FEATURES_PER_LANE * self.num_lanes

    @property
    def phase_slice(self) -> slice:
        """Đoạn one-hot của pha hiện tại."""
        return slice(0, self.num_phases)

    @property
    def elapsed_index(self) -> int:
        """Chỉ số của thời gian xanh đã trôi qua (chuẩn hoá theo xanh tối thiểu)."""
        return self.num_phases

    @property
    def queue_slice(self) -> slice:
        """Đoạn queue của các làn vào."""
        return slice(self.phase_dim, self.phase_dim + self.num_lanes)

    @property
    def speed_slice(self) -> slice:
        """Đoạn tốc độ trung bình của các làn vào."""
        return slice(self.phase_dim + self.num_lanes, self.phase_dim + 2 * self.num_lanes)

    @property
    def density_slice(self) -> slice:
        """Đoạn mật độ của các làn vào."""
        return slice(self.phase_dim + 2 * self.num_lanes, self.dim)
