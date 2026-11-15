"""
agents/networks.py
------------------
Kiến trúc mạng Nơ-ron PyTorch cho NodeAgent trong dự án FedMARL-ATSC.

KIẾN TRÚC PHÂN RÃ BẮT BUỘC (PROJECT_GUIDELINES.md §6.2):
  shared_feature_extractor  → phần liên bang, gửi lên FederatedServer
  local_policy_head         → phần riêng tư, TUYỆT ĐỐI không rời khỏi node

Mạng này trung lập với thuật toán RL: đầu ra của `local_policy_head` là
"action scores" — thuật toán value-based đọc nó như Q-values, thuật toán
policy-based đọc nó như logits. Cách thay kiến trúc: xem
docs/RL_INTEGRATION_GUIDE.md §3.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_HIDDEN_DIM = 128
DEFAULT_BACKBONE_OUT = 64

# Tên thuộc tính cố định mà cơ chế FL dựa vào (xem NodeAgent.extract_weights()).
SHARED_MODULE_NAME = "shared_feature_extractor"
POLICY_HEAD_NAME = "local_policy_head"


class TrafficAgentNet(nn.Module):
    """Mạng Nơ-ron phân rã cho một Node-Agent điều khiển tín hiệu giao thông.

    Kiến trúc::

        Input (obs_dim,)
            │
            ▼
        shared_feature_extractor              ← Liên bang (FL) ✅
            Linear(obs_dim → hidden_dim) → LayerNorm → ReLU
            Linear(hidden_dim → hidden_dim) → LayerNorm → ReLU
            Linear(hidden_dim → backbone_out) → ReLU
            │
            ▼ features (backbone_out,)
            │
            ├──→ local_policy_head            ← Riêng tư 🔒
            │       Linear(backbone_out → num_phases)
            │       Output: action scores (Q-values hoặc logits)
            │
            └──→ local_value_head (tuỳ chọn)  ← Riêng tư 🔒
                    Linear(backbone_out → 1)
                    Output: state value V(s) cho Actor-Critic

    Args:
        obs_dim: Kích thước vector state.
        num_phases: Số pha xanh có thể chọn (kích thước action space).
        hidden_dim: Độ rộng lớp ẩn trong backbone.
        backbone_out: Kích thước embedding mà backbone trả ra.
        with_value: True → thêm `local_value_head` (critic) cho Actor-Critic.
    """

    def __init__(
        self,
        obs_dim: int,
        num_phases: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        backbone_out: int = DEFAULT_BACKBONE_OUT,
        with_value: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.backbone_out = backbone_out
        self.with_value = with_value

        # ── Backbone dùng chung (Liên bang) ──────────────────────────────────
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, backbone_out),
            nn.ReLU(),
        )

        # ── Policy head cục bộ (Riêng tư 🔒) ─────────────────────────────────
        self.local_policy_head = nn.Sequential(
            nn.Linear(backbone_out, num_phases),
        )

        # ── Value head cục bộ (tuỳ chọn, Riêng tư 🔒) ────────────────────────
        self.local_value_head: Optional[nn.Module] = (
            nn.Sequential(nn.Linear(backbone_out, 1)) if with_value else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số theo chuẩn orthogonal (ổn định huấn luyện RL)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def get_backbone_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Trả về embedding của backbone (trước mọi head).

        Args:
            obs: Tensor shape (B, obs_dim) hoặc (obs_dim,).

        Returns:
            Tensor shape (B, backbone_out).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.shared_feature_extractor(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward qua backbone rồi policy head.

        Args:
            obs: Tensor shape (B, obs_dim) hoặc (obs_dim,).

        Returns:
            Action scores shape (B, num_phases): Q-values hoặc logits tuỳ
            thuật toán của agent.
        """
        return self.local_policy_head(self.get_backbone_features(obs))

    def forward_with_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward trả về cả action scores lẫn state value (Actor-Critic).

        Args:
            obs: Tensor shape (B, obs_dim) hoặc (obs_dim,).

        Returns:
            (action scores shape (B, num_phases), value V(s) shape (B, 1)).

        Raises:
            RuntimeError: Nếu mạng được tạo với `with_value=False`.
        """
        if self.local_value_head is None:
            raise RuntimeError("Khởi tạo với with_value=True để dùng forward_with_value().")
        features = self.get_backbone_features(obs)
        return self.local_policy_head(features), self.local_value_head(features)

    def __repr__(self) -> str:
        return (
            f"TrafficAgentNet(obs={self.obs_dim}, hidden={self.hidden_dim}, "
            f"backbone_out={self.backbone_out}, phases={self.num_phases}, "
            f"with_value={self.with_value})"
        )


# ---------------------------------------------------------------------------
# Kiểm tra nhanh: python -m agents.networks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _OBS_DIM, _NUM_PHASES, _BATCH = 84, 4, 8

    net = TrafficAgentNet(obs_dim=_OBS_DIM, num_phases=_NUM_PHASES, with_value=True)
    print(net)

    dummy_obs = torch.randn(_BATCH, _OBS_DIM)
    scores, value = net.forward_with_value(dummy_obs)
    print(f"\nInput         : {tuple(dummy_obs.shape)}")
    print(f"Action scores : {tuple(scores.shape)}")
    print(f"State value   : {tuple(value.shape)}")
    print(f"Backbone out  : {tuple(net.get_backbone_features(dummy_obs).shape)}")

    shared_keys = set(net.shared_feature_extractor.state_dict())
    local_keys = {
        key for key in net.state_dict() if not key.startswith(SHARED_MODULE_NAME + ".")
    }
    print(f"\nBackbone tensors (liên bang) : {len(shared_keys)}")
    print(f"Local tensors (giữ tại node) : {len(local_keys)}")
    assert len(shared_keys) + len(local_keys) == len(net.state_dict())
    print("networks.py OK ✓")
