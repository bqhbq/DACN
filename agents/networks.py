"""
agents/networks.py
------------------
Định nghĩa kiến trúc mạng Nơ-ron PyTorch cho dự án FedMARL-ATSC.

KIẾN TRÚC PHÂN RÃ BẮT BUỘC (PROJECT_GUIDELINES.md §6.2):
  shared_feature_extractor  → phần liên bang, gửi lên FederatedServer
  local_policy_head         → phần riêng tư, TUYỆT ĐỐI không rời khỏi node
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class TrafficAgentNet(nn.Module):
    """
    Mạng Nơ-ron phân rã cho một Node-Agent điều khiển tín hiệu giao thông.

    Kiến trúc
    ----------
    Input (obs_dim,)
        │
        ▼
    shared_feature_extractor          ← Liên bang (FL) ✅
        Linear(obs_dim  → hidden_dim) → LayerNorm → ReLU
        Linear(hidden_dim → hidden_dim) → LayerNorm → ReLU
        Linear(hidden_dim → backbone_out) → ReLU
        │
        ▼ backbone_features (backbone_out,)
        │
        ├──→ local_policy_head (Q-network head)   ← Riêng tư 🔒
        │       Linear(backbone_out → num_phases)
        │       Output: Q-values shape (num_phases,)
        │
        └──→ local_value_head (optional, PPO baseline)  ← Riêng tư 🔒
                Linear(backbone_out → 1)
                Output: state value V(s)

    Tham số
    -------
    obs_dim      : Kích thước vector quan sát (84 với sample_map_2)
    hidden_dim   : Độ rộng lớp ẩn trong backbone (mặc định 128)
    backbone_out : Kích thước output backbone (mặc định 64)
    num_phases   : Số pha xanh có thể chọn (4 với sample_map_2)
    with_value   : Có thêm value head không (dùng cho PPO, mặc định False)
    """

    def __init__(
        self,
        obs_dim:      int  = 84,
        hidden_dim:   int  = 128,
        backbone_out: int  = 64,
        num_phases:   int  = 4,
        with_value:   bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim      = obs_dim
        self.hidden_dim   = hidden_dim
        self.backbone_out = backbone_out
        self.num_phases   = num_phases
        self.with_value   = with_value

        # ── Shared Backbone (Liên bang) ───────────────────────────────────────
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(obs_dim,     hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  backbone_out),
            nn.ReLU(),
        )

        # ── Local Policy Head (Q-values, Riêng tư 🔒) ────────────────────────
        self.local_policy_head = nn.Sequential(
            nn.Linear(backbone_out, num_phases),
        )

        # ── Local Value Head (PPO, tuỳ chọn, Riêng tư 🔒) ───────────────────
        if with_value:
            self.local_value_head: Optional[nn.Module] = nn.Sequential(
                nn.Linear(backbone_out, 1),
            )
        else:
            self.local_value_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Khởi tạo trọng số theo chuẩn orthogonal (ổn định huấn luyện RL)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua backbone + policy head.

        Tham số
        -------
        obs : Tensor shape (batch, obs_dim) hoặc (obs_dim,)

        Trả về
        ------
        q_values : Tensor shape (batch, num_phases) — Q-value thô
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.shared_feature_extractor(obs)
        q_values = self.local_policy_head(features)
        return q_values

    def forward_with_value(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass trả về cả Q-values và state value V(s) (dùng cho PPO).

        Yêu cầu with_value=True khi khởi tạo.
        """
        assert self.local_value_head is not None, \
            "Khởi tạo với with_value=True để dùng forward_with_value()."
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.shared_feature_extractor(obs)
        q_values = self.local_policy_head(features)
        value    = self.local_value_head(features)
        return q_values, value

    def get_backbone_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Trả về embedding từ backbone (trước policy head). Dùng cho neighbor embedding."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.shared_feature_extractor(obs)

    def __repr__(self) -> str:
        return (
            f"TrafficAgentNet("
            f"obs={self.obs_dim}, hidden={self.hidden_dim}, "
            f"backbone_out={self.backbone_out}, phases={self.num_phases}, "
            f"with_value={self.with_value})"
        )


# ---------------------------------------------------------------------------
# Kiểm tra nhanh (python -m agents.networks)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    net = TrafficAgentNet(obs_dim=84, hidden_dim=128, backbone_out=64, num_phases=4)
    print(net)

    dummy_obs = torch.randn(8, 84)                     # batch=8
    q_vals    = net(dummy_obs)
    backbone  = net.get_backbone_features(dummy_obs)

    print(f"\nInput        : {tuple(dummy_obs.shape)}")
    print(f"Q-values     : {tuple(q_vals.shape)}")
    print(f"Backbone out : {tuple(backbone.shape)}")

    # Kiểm tra phân tách backbone/head
    shared_keys = set(net.shared_feature_extractor.state_dict().keys())
    head_keys   = set(net.local_policy_head.state_dict().keys())
    all_keys    = set(net.state_dict().keys())
    print(f"\nBackbone keys : {len(shared_keys)}")
    print(f"Head keys     : {len(head_keys)}")
    print(f"Total keys    : {len(all_keys)}")
    print("networks.py OK ✓")
