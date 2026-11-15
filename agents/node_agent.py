"""
agents/node_agent.py
--------------------
Lớp NodeAgent — bộ khung (skeleton) trung lập thuật toán cho agent tại một
giao lộ, tuân thủ PROJECT_GUIDELINES.md §6.2.

Lớp này gồm hai nhóm hàm ĐỘC LẬP với nhau:

1. Giao diện RL — vòng lặp chính (`main_fedmarl.py`) chỉ gọi ba hàm:

       action = agent.select_action(state)
       agent.store_transition(state, action, reward, next_state)
       loss = agent.train_local_model()

   Mỗi thuật toán RL là một lớp con ghi đè các hook `compute_loss()`,
   `select_action()`, `sample_batch()`, `build_network()` (xem
   docs/RL_INTEGRATION_GUIDE.md).

2. Giao diện FL — `FederatedServer` chỉ gọi:

       weights = agent.extract_weights()        # uplink: chỉ backbone
       agent.load_global_weights(weights)       # downlink: chỉ backbone

   Hai hàm này chỉ đụng tới `net.shared_feature_extractor`, không phụ thuộc
   thuật toán huấn luyện, và lớp con KHÔNG được ghi đè.

Lớp này KHÔNG chứa: logic CityFlow, luật an toàn đèn, hay logic riêng của
một thuật toán RL cụ thể.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import (
    DEFAULT_BACKBONE_OUT,
    DEFAULT_HIDDEN_DIM,
    POLICY_HEAD_NAME,
    SHARED_MODULE_NAME,
    TrafficAgentNet,
)
from agents.replay_buffer import Transition, TransitionBatch, collate

logger = logging.getLogger(__name__)

DEFAULT_LR = 1e-3
GRAD_CLIP_NORM = 10.0   # chặn chuẩn gradient mỗi bước cập nhật
METADATA_LOSS_WINDOW = 10  # số loss gần nhất lấy trung bình cho metadata()


class NodeAgent:
    """Agent điều khiển tín hiệu tại một giao lộ — lớp nền cho mọi thuật toán RL.

    Dùng trực tiếp lớp này thì `train_local_model()` sẽ báo lỗi vì chưa có
    `compute_loss()`; hãy dùng một lớp con (`QLearningAgent`, `DQNAgent`,
    `BehaviorCloningAgent`) hoặc tự viết lớp con cho thuật toán của bạn.

    Lưu ý cho lớp con: `build_network()` được gọi bên trong `__init__` của lớp
    này, nên mọi thuộc tính mà `build_network()` cần phải được gán TRƯỚC khi
    gọi `super().__init__()`.

    Args:
        intersection_id: ID giao lộ, khớp với ID trong CityFlow roadnet.
        obs_dim: Kích thước vector state.
        num_phases: Số pha xanh khả dụng (kích thước action space).
        hidden_dim: Độ rộng lớp ẩn của backbone.
        backbone_out: Kích thước embedding mà backbone trả ra.
        lr: Learning rate của optimizer cục bộ.
        device: Thiết bị chạy model; None → CPU.
        seed: Seed của `self.rng` (dùng cho khám phá); None → ngẫu nhiên.
        reset_optimizer_on_sync: True → dựng lại optimizer mỗi lần nhận global
            weights (xoá moment Adam). Mặc định False: giữ moment.

    Raises:
        TypeError: Nếu mạng do `build_network()` trả về thiếu
            `shared_feature_extractor` hoặc `local_policy_head`.
    """

    def __init__(
        self,
        intersection_id: str,
        obs_dim: int,
        num_phases: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        backbone_out: int = DEFAULT_BACKBONE_OUT,
        lr: float = DEFAULT_LR,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        reset_optimizer_on_sync: bool = False,
    ) -> None:
        self.intersection_id = intersection_id
        self.obs_dim = obs_dim
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim
        self.backbone_out = backbone_out
        self.lr = lr
        self.device = device or torch.device("cpu")
        # RNG riêng của agent (khám phá ε-greedy, ...) để chạy tái lập được.
        self.rng = np.random.default_rng(seed)
        self.reset_optimizer_on_sync = reset_optimizer_on_sync
        # FedProx (phía client): neo = backbone toàn cục nhận lần cuối.
        self.fedprox_mu = 0.0
        self._global_anchor: Optional[Dict[str, torch.Tensor]] = None

        self.net: nn.Module = self.build_network().to(self.device)
        self._check_decoupled_architecture(self.net)
        self.optimizer: optim.Optimizer = self.build_optimizer()

        self._pending: List[Transition] = []   # transition chờ được học
        self._loss_history: List[float] = []
        # n_k của FedAvg (MDP_SPEC §6.2): số mẫu cục bộ kể từ lần đồng bộ gần nhất.
        self._local_samples = 0

        logger.info(
            "%s [%s] khởi tạo: obs_dim=%d, num_phases=%d, device=%s",
            type(self).__name__, intersection_id, obs_dim, num_phases, self.device,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Hook dựng mô hình — ghi đè khi cần kiến trúc mạng / optimizer khác
    # ──────────────────────────────────────────────────────────────────────────

    def build_network(self) -> nn.Module:
        """Dựng mạng Nơ-ron của agent.

        Ghi đè để đổi kiến trúc (vd. thêm critic head). Mạng trả về BẮT BUỘC
        có hai sub-module tên `shared_feature_extractor` và `local_policy_head`;
        mọi sub-module khác được coi là riêng tư và ở lại node.

        Returns:
            Mạng chưa chuyển sang device (lớp nền tự `.to(device)`).
        """
        return TrafficAgentNet(
            obs_dim=self.obs_dim,
            num_phases=self.num_phases,
            hidden_dim=self.hidden_dim,
            backbone_out=self.backbone_out,
        )

    def build_optimizer(self) -> optim.Optimizer:
        """Dựng optimizer cục bộ, tối ưu toàn bộ mạng (backbone lẫn các head).

        Returns:
            Optimizer cho `self.net`.
        """
        return optim.Adam(self.net.parameters(), lr=self.lr)

    @staticmethod
    def _check_decoupled_architecture(net: nn.Module) -> None:
        """Dừng sớm nếu mạng không theo kiến trúc phân rã backbone / head."""
        for name in (SHARED_MODULE_NAME, POLICY_HEAD_NAME):
            if not isinstance(getattr(net, name, None), nn.Module):
                raise TypeError(
                    f"Mạng {type(net).__name__} thiếu sub-module '{name}' — "
                    "vi phạm kiến trúc Decoupled (PROJECT_GUIDELINES.md §6.2)."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Giao diện RL — ba hàm vòng lặp chính gọi
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Chọn action cho state hiện tại.

        Mặc định: action có score cao nhất. Lớp con ghi đè để thêm chiến lược
        khám phá (ε-greedy, lấy mẫu từ phân phối policy, ...).

        Args:
            state: Vector state shape (obs_dim,).
            greedy: True → không khám phá (dùng khi đánh giá).

        Returns:
            Index pha xanh mong muốn, thuộc [0, num_phases).
        """
        return int(np.argmax(self.action_scores(state)))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Ghi nhận một transition (s, a, r, s') vừa quan sát được.

        Mặc định: xếp vào hàng chờ, `train_local_model()` kế tiếp sẽ học rồi
        xoá. Lớp con ghi đè để dùng bộ nhớ khác (vd. replay buffer) và phải
        gọi `_register_local_samples()` cho mỗi transition được nhận.

        Args:
            state: State s lúc ra quyết định, shape (obs_dim,).
            action: Action a đã chọn tại s.
            reward: Reward r do môi trường tính.
            next_state: State s' ở lần ra quyết định kế tiếp.
            done: True nếu s' kết thúc episode.
        """
        self._pending.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )
        self._register_local_samples()

    def train_local_model(self) -> Optional[float]:
        """Cập nhật mô hình cục bộ một bước từ dữ liệu đã lưu.

        Khuôn mẫu chung cho mọi thuật toán:
            batch = sample_batch() → loss = compute_loss(batch) → backward →
            chặn gradient → optimizer.step().

        Gradient đi qua cả backbone lẫn head, nên `shared_feature_extractor`
        của mỗi node lệch dần theo dữ liệu riêng — đó là thứ FL tổng hợp.

        Returns:
            Giá trị loss của bước này; None nếu chưa đủ dữ liệu để học.
        """
        batch = self.sample_batch()
        if batch is None:
            return None

        self.net.train()
        loss = self.compute_loss(batch)
        proximal = self.proximal_term()
        if proximal is not None:
            loss = loss + proximal
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        loss_value = float(loss.item())
        self._loss_history.append(loss_value)
        return loss_value

    # ──────────────────────────────────────────────────────────────────────────
    # Hook thuật toán — nơi đội RL viết code
    # ──────────────────────────────────────────────────────────────────────────

    def sample_batch(self) -> Optional[TransitionBatch]:
        """Lấy lô dữ liệu cho bước cập nhật kế tiếp.

        Mặc định (kiểu on-policy): toàn bộ transition đang chờ, dùng một lần
        rồi xoá. Thuật toán off-policy ghi đè để lấy mẫu từ replay buffer.

        Returns:
            Lô transition; None nếu chưa có gì để học.
        """
        if not self._pending:
            return None
        batch = collate(self._pending, self.device)
        self._pending.clear()
        return batch

    def compute_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """Tính loss của thuật toán RL trên một lô transition.

        ĐÂY LÀ HÀM CHÍNH MÀ MỖI THUẬT TOÁN PHẢI GHI ĐÈ.

        Args:
            batch: Lô transition do `sample_batch()` trả về.

        Returns:
            Tensor vô hướng còn gắn đồ thị tính toán (để `backward()`).

        Raises:
            NotImplementedError: Lớp nền không gắn với thuật toán nào.
        """
        raise NotImplementedError(
            f"{type(self).__name__} chưa cài compute_loss(). "
            "Xem docs/RL_INTEGRATION_GUIDE.md §2."
        )

    def _register_local_samples(self, count: int = 1) -> None:
        """Cộng `count` vào n_k — số mẫu cục bộ dùng làm trọng số FedAvg."""
        self._local_samples += count

    # ──────────────────────────────────────────────────────────────────────────
    # Giao diện FL — độc lập với thuật toán huấn luyện, KHÔNG ghi đè
    # ──────────────────────────────────────────────────────────────────────────

    def extract_weights(self) -> Dict[str, torch.Tensor]:
        """Trích trọng số gửi lên FederatedServer: CHỈ `shared_feature_extractor`.

        Mọi phần khác của mạng (`local_policy_head`, critic head, ...) không
        bao giờ xuất hiện trong kết quả (quy tắc R-02).

        Returns:
            Bản sao tách rời (CPU) của state_dict backbone.
        """
        shared = getattr(self.net, SHARED_MODULE_NAME).state_dict()
        return {key: tensor.detach().cpu().clone() for key, tensor in shared.items()}

    def load_global_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Nạp backbone toàn cục do FederatedServer phát về.

        Chỉ `shared_feature_extractor` bị ghi đè; các head giữ nguyên kinh
        nghiệm cục bộ (personalisation). Bộ đếm n_k được đặt lại cho vòng FL
        kế tiếp.

        Args:
            global_weights: state_dict backbone từ `FederatedServer.aggregate()`.

        Raises:
            ValueError: Nếu `global_weights` không khớp cấu trúc backbone.
        """
        on_device = {key: tensor.to(self.device) for key, tensor in global_weights.items()}
        try:
            getattr(self.net, SHARED_MODULE_NAME).load_state_dict(on_device, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"[{self.intersection_id}] global weights không khớp cấu trúc "
                f"{SHARED_MODULE_NAME}: {exc}"
            ) from exc
        self._local_samples = 0
        self._global_anchor = {key: tensor.detach().clone() for key, tensor in on_device.items()}
        if self.reset_optimizer_on_sync:
            self.optimizer = self.build_optimizer()
        self.on_global_weights_loaded()
        logger.debug("[%s] Đã nạp global backbone weights.", self.intersection_id)

    def on_global_weights_loaded(self) -> None:
        """Hook gọi sau mỗi lần nhận global weights; mặc định không làm gì.

        Thuật toán có trạng thái phụ thuộc backbone (vd. target network của
        DQN) ghi đè hàm này để đồng bộ lại. KHÔNG ghi đè `load_global_weights`.
        """

    def set_fedprox_mu(self, mu: float) -> None:
        """Bật/tắt FedProx phía client.

        Args:
            mu: Hệ số proximal μ ≥ 0; 0 = FedAvg thuần.

        Raises:
            ValueError: Nếu μ < 0.
        """
        if mu < 0:
            raise ValueError(f"fedprox_mu phải ≥ 0, nhận {mu}.")
        self.fedprox_mu = float(mu)

    def proximal_term(self) -> Optional[torch.Tensor]:
        """Số hạng FedProx μ/2·‖w_backbone − w_global‖² (có gắn đồ thị tính toán).

        Returns:
            Tensor vô hướng; None nếu μ = 0 hoặc chưa nhận global weights.
        """
        if self.fedprox_mu <= 0.0 or self._global_anchor is None:
            return None
        shared = getattr(self.net, SHARED_MODULE_NAME)
        squared = sum(
            ((param - self._global_anchor[name]) ** 2).sum()
            for name, param in shared.named_parameters()
            if name in self._global_anchor
        )
        return 0.5 * self.fedprox_mu * squared

    def get_shared_weights(self) -> Dict[str, torch.Tensor]:
        """Tên cũ của `extract_weights()`, giữ cho notebook và tài liệu hiện có."""
        return self.extract_weights()

    def set_shared_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Tên cũ của `load_global_weights()`, giữ cho notebook và tài liệu hiện có."""
        self.load_global_weights(global_weights)

    @property
    def num_local_samples(self) -> int:
        """n_k của FedAvg: số mẫu cục bộ kể từ lần nhận global weights gần nhất."""
        return self._local_samples

    def local_weights_norm(self) -> float:
        """Chuẩn L2 của mọi tham số KHÔNG thuộc backbone (các head cục bộ).

        Là "dấu vân tay" để kiểm chứng phần riêng tư không đổi qua một vòng FL,
        mà không để trọng số head rời khỏi node.

        Returns:
            Chuẩn L2 của toàn bộ tham số cục bộ.
        """
        prefix = SHARED_MODULE_NAME + "."
        squared = sum(
            float((param.detach() ** 2).sum())
            for name, param in self.net.named_parameters()
            if not name.startswith(prefix)
        )
        return math.sqrt(squared)

    # ──────────────────────────────────────────────────────────────────────────
    # Tiện ích
    # ──────────────────────────────────────────────────────────────────────────

    def action_scores(self, state: np.ndarray) -> np.ndarray:
        """Đầu ra của policy head cho một state (Q-values hoặc logits).

        Args:
            state: Vector state shape (obs_dim,).

        Returns:
            Mảng shape (num_phases,).
        """
        self.net.eval()
        with torch.no_grad():
            scores = self.net(self._to_tensor(state))
        return scores.squeeze(0).cpu().numpy()

    def get_backbone_embedding(self, state: np.ndarray) -> np.ndarray:
        """Embedding của backbone cho một state — dùng làm neighbor embedding.

        Args:
            state: Vector state shape (obs_dim,).

        Returns:
            Mảng shape (backbone_out,).
        """
        self.net.eval()
        with torch.no_grad():
            features = getattr(self.net, SHARED_MODULE_NAME)(
                self._to_tensor(state).unsqueeze(0)
            )
        return features.squeeze(0).cpu().numpy()

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Đổi một state NumPy thành tensor float32 trên device của agent."""
        return torch.as_tensor(state, dtype=torch.float32, device=self.device)

    @property
    def loss_history(self) -> List[float]:
        """Loss của mọi bước `train_local_model()` đã chạy, theo thứ tự thời gian."""
        return self._loss_history

    @property
    def last_loss(self) -> Optional[float]:
        """Loss của bước huấn luyện gần nhất; None nếu chưa huấn luyện bước nào."""
        return self._loss_history[-1] if self._loss_history else None

    def metadata(self) -> Dict[str, Any]:
        """Metadata phục vụ clustering tại FederatedServer (không chứa dữ liệu giao thông)."""
        recent = self._loss_history[-METADATA_LOSS_WINDOW:]
        return {
            "intersection_id": self.intersection_id,
            "obs_dim": self.obs_dim,
            "num_phases": self.num_phases,
            "avg_loss": float(np.mean(recent)) if recent else 0.0,
        }

    def stats(self) -> Dict[str, Any]:
        """Số liệu tóm tắt để in cuối buổi chạy; lớp con bổ sung số liệu riêng."""
        return {"train_steps": len(self._loss_history)}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.intersection_id!r}, "
            f"obs_dim={self.obs_dim}, num_phases={self.num_phases}, "
            f"device={self.device})"
        )
