"""
agents/node_agent.py
--------------------
Lớp NodeAgent — Agent RL tại một giao lộ, tuân thủ PROJECT_GUIDELINES.md §6.2

Kiến trúc Phân rã BẮT BUỘC:
  shared_feature_extractor  → phần liên bang (gửi lên FederatedServer)
  local_policy_head         → phần riêng tư (TUYỆT ĐỐI không rời khỏi node)

Lớp này chứa: mô hình PyTorch, optimizer, forward pass, dummy loss, trích xuất weights.
Lớp này KHÔNG chứa: logic CityFlow, replay buffer, lịch sử episode.
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import TrafficAgentNet

logger = logging.getLogger(__name__)


class NodeAgent:
    """
    Agent điều khiển tín hiệu tại một giao lộ duy nhất.

    Tham số
    -------
    intersection_id : ID của giao lộ (phải khớp với ID trong CityFlow roadnet)
    obs_dim         : Kích thước vector observation (num_incoming_lanes × 3)
    num_phases      : Số pha xanh khả dụng (action space size)
    hidden_dim      : Độ rộng lớp ẩn trong backbone (mặc định 128)
    backbone_out    : Output dim của backbone — chiều embedding được chia sẻ (mặc định 64)
    lr              : Learning rate cho optimizer cục bộ (mặc định 1e-3)
    epsilon         : Xác suất khám phá ε-greedy (mặc định 0.1)
    device          : torch.device để chạy model
    """

    def __init__(
        self,
        intersection_id: str,
        obs_dim:         int,
        num_phases:      int,
        hidden_dim:      int   = 128,
        backbone_out:    int   = 64,
        lr:              float = 1e-3,
        epsilon:         float = 0.1,
        device:          Optional[torch.device] = None,
    ) -> None:
        self.intersection_id = intersection_id
        self.obs_dim         = obs_dim
        self.num_phases      = num_phases
        self.epsilon         = epsilon
        self.device          = device or torch.device("cpu")

        # ── Mạng Nơ-ron phân rã (Decoupled Architecture) ─────────────────────
        self.net = TrafficAgentNet(
            obs_dim      = obs_dim,
            hidden_dim   = hidden_dim,
            backbone_out = backbone_out,
            num_phases   = num_phases,
        ).to(self.device)

        # ── Optimizer cục bộ — tối ưu hoá TOÀN BỘ network (cả backbone lẫn head) ─
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._loss_history: List[float] = []

        logger.info(
            "NodeAgent [%s] khởi tạo: obs_dim=%d, num_phases=%d, device=%s",
            intersection_id, obs_dim, num_phases, self.device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Chọn hành động (Action Selection)
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Chọn hành động theo chiến lược ε-greedy.

        Tham số
        -------
        obs    : Observation vector shape (obs_dim,)
        greedy : Nếu True → luôn chọn argmax Q (không khám phá)

        Trả về
        ------
        action_idx : int ∈ [0, num_phases)
        """
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_phases))

        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q_vals = self.net(obs_t)                 # shape (1, num_phases)
        return int(q_vals.squeeze(0).argmax().item())

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Trả về Q-values shape (num_phases,) dưới dạng numpy."""
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q = self.net(obs_t).squeeze(0)
        return q.cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Huấn luyện cục bộ (Local Training — Dummy Loss cho demo tích hợp)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_dummy_loss(self, obs: np.ndarray) -> float:
        """
        Tính toán một dummy MSE loss để kiểm tra luồng backward() hoạt động.

        Trong hệ thống thật: hàm này sẽ được thay bằng DQN TD-error hoặc PPO
        policy gradient do module agents/dqn.py hoặc agents/ppo.py cung cấp.

        Tham số
        -------
        obs : Observation vector shape (obs_dim,) lấy từ TrafficEnv.get_state()

        Trả về
        ------
        float : Giá trị loss sau bước backward()
        """
        self.net.train()
        self.optimizer.zero_grad()

        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.net(obs_t)                            # (1, num_phases)

        # Dummy target: zero Q-values (placeholder cho RL loss thực)
        dummy_target = torch.zeros_like(q_vals)
        loss = nn.functional.mse_loss(q_vals, dummy_target)

        loss.backward()
        # Gradient clipping — thực tiễn tốt cho RL
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self._loss_history.append(loss_val)
        return loss_val

    # ──────────────────────────────────────────────────────────────────────────
    # Federated Learning Interface (R-02, R-03 — quan trọng nhất)
    # ──────────────────────────────────────────────────────────────────────────

    def get_shared_weights(self) -> Dict[str, torch.Tensor]:
        """
        Trích xuất CHỈ trọng số shared_feature_extractor để gửi lên Server.

        ⚠ Quy tắc R-02: TUYỆT ĐỐI không bao gồm local_policy_head.
        ⚠ Quy tắc R-03: local_policy_head không được tham chiếu ở bất kỳ
                         đâu bên ngoài class NodeAgent.

        Trả về
        ------
        dict : state_dict của shared_feature_extractor (deep-copy, CPU tensors)
        """
        # R-02: Chỉ backbone — không có head
        shared_sd = self.net.shared_feature_extractor.state_dict()
        return {k: v.detach().cpu().clone() for k, v in shared_sd.items()}

    def set_shared_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """
        Nạp trọng số backbone đã tổng hợp từ FederatedServer.

        Chỉ cập nhật shared_feature_extractor — local_policy_head không thay đổi.
        Đây là bước Personalisation cốt lõi: head giữ kinh nghiệm cục bộ,
        backbone được cải thiện từ tri thức toàn cục.

        Tham số
        -------
        global_weights : state_dict từ FederatedServer.aggregate()
        """
        # Chuyển lên đúng device trước khi load
        weights_on_device = {k: v.to(self.device) for k, v in global_weights.items()}
        self.net.shared_feature_extractor.load_state_dict(weights_on_device, strict=True)
        logger.debug("[%s] Đã nạp global backbone weights.", self.intersection_id)

    def get_backbone_embedding(self, obs: np.ndarray) -> np.ndarray:
        """
        Trả về embedding từ backbone (trước head) — dùng cho neighbor embedding.
        Output shape: (backbone_out,)
        """
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            emb   = self.net.get_backbone_features(obs_t)
        return emb.squeeze(0).cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def metadata(self) -> Dict:
        """Metadata dùng cho clustering tại FederatedServer."""
        return {
            "intersection_id": self.intersection_id,
            "obs_dim":         self.obs_dim,
            "num_phases":      self.num_phases,
            "avg_loss":        float(np.mean(self._loss_history[-10:])) if self._loss_history else 0.0,
        }

    @property
    def loss_history(self) -> List[float]:
        return self._loss_history

    def __repr__(self) -> str:
        return (
            f"NodeAgent(id={self.intersection_id!r}, "
            f"obs_dim={self.obs_dim}, num_phases={self.num_phases}, "
            f"device={self.device})"
        )
