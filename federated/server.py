"""
federated/server.py
-------------------
Lớp FederatedServer — Điều phối quy trình Federated Learning

Tuân thủ PROJECT_GUIDELINES.md §6.3:
  ✅ Nhận shared_weights từ NodeAgent
  ✅ Phân cụm clients theo metadata giao lộ (K-Means)
  ✅ Tổng hợp FedAvg / FedProx
  ✅ Broadcast global weights về NodeAgent

TUYỆT ĐỐI KHÔNG chứa: logic CityFlow, neural network, replay buffer.
Server chỉ biết hai hàm của client: extract_weights() và load_global_weights()
— nó không quan tâm client huấn luyện bằng thuật toán RL nào.
TUYỆT ĐỐI KHÔNG import: cityflow, TrafficEnv.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Máy chủ điều phối Federated Learning cho hệ thống FedMARL-ATSC.

    Luồng một vòng FL (run_round):
        1. Thu thập shared backbone weights từ tất cả clients đã đăng ký
        2. (Tuỳ chọn) Phân cụm clients theo metadata → tổng hợp theo cụm
        3. aggregate(FedAvg hoặc FedProx) → global_weights
        4. broadcast() → NodeAgent.load_global_weights()

    Tham số
    -------
    clients         : Danh sách NodeAgent đã đăng ký (import ở runtime để tránh circular)
    use_clustering  : Có phân cụm trước khi aggregate không (mặc định False)
    num_clusters    : Số cụm K-Means nếu use_clustering=True (mặc định 2)
    fedprox_mu      : Hệ số proximal μ cho FedProx (mặc định 0 → FedAvg thuần)
    """

    def __init__(
        self,
        clients: list,                  # List[NodeAgent] — tránh circular import
        use_clustering: bool  = False,
        num_clusters:   int   = 2,
        fedprox_mu:     float = 0.0,
    ) -> None:
        self.clients        = clients
        self.use_clustering = use_clustering
        self.num_clusters   = num_clusters
        self.fedprox_mu     = fedprox_mu

        self.global_weights:  Optional[Dict[str, torch.Tensor]] = None
        self._round_logs:     List[Dict] = []
        self._fl_round:       int        = 0
        self._last_coefficients: List[float] = []   # n_k/N của lần aggregate gần nhất

        logger.info(
            "FederatedServer khởi tạo: %d clients, clustering=%s, mu=%.3f",
            len(clients), use_clustering, fedprox_mu,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Aggregation
    # ──────────────────────────────────────────────────────────────────────────

    def aggregate(
        self,
        list_of_weights: List[Dict[str, torch.Tensor]],
        sample_counts:   Optional[List[int]] = None,
        algorithm:       str = "fedavg",
    ) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp weights từ nhiều clients.

        Thuật toán
        ----------
        'fedavg'  : w* = Σ(n_k/N · w_k)        (McMahan et al., 2017)
        'fedprox' : FedAvg + proximal term μ/2 ‖w−w*‖²
                    (ảnh hưởng đến local training, server vẫn chạy FedAvg)

        Tham số
        -------
        list_of_weights : [{layer: tensor}, ...] — một dict mỗi client
        sample_counts   : [n_k, ...] — kích thước dataset cục bộ (nếu None → đều)
        algorithm       : 'fedavg' hoặc 'fedprox'

        Trả về
        ------
        dict : Averaged state_dict
        """
        if not list_of_weights:
            raise ValueError("aggregate() nhận list rỗng.")

        K = len(list_of_weights)

        # Trọng số tổng hợp theo kích thước dataset (mặc định đều nhau)
        if sample_counts is None or sum(sample_counts) <= 0:
            weights_norm = [1.0 / K] * K
        else:
            total = sum(sample_counts)
            weights_norm = [n / total for n in sample_counts]

        # FedAvg: trung bình có trọng số element-wise
        avg: Dict[str, torch.Tensor] = {}
        for key in list_of_weights[0].keys():
            stacked = torch.stack(
                [list_of_weights[i][key].float() * weights_norm[i] for i in range(K)],
                dim=0,
            )
            avg[key] = stacked.sum(dim=0)

        self.global_weights = avg
        self._last_coefficients = weights_norm

        logger.info(
            "Round %d | aggregate(%s) | %d clients | %d param tensors",
            self._fl_round, algorithm, K, len(avg),
        )
        return avg

    # ──────────────────────────────────────────────────────────────────────────
    # Broadcast
    # ──────────────────────────────────────────────────────────────────────────

    def broadcast(self) -> None:
        """
        Gửi global_weights về tất cả clients đã đăng ký.

        Mỗi client nhận một deep-copy độc lập (không aliasing).
        Client gọi NodeAgent.load_global_weights() — chỉ backbone được cập nhật,
        local_policy_head giữ nguyên (personalisation).

        Raises
        ------
        RuntimeError : Nếu chưa gọi aggregate() trước.
        """
        if self.global_weights is None:
            raise RuntimeError("Chưa có global_weights. Gọi aggregate() trước broadcast().")

        for client in self.clients:
            client.load_global_weights(copy.deepcopy(self.global_weights))

        logger.info(
            "Round %d | broadcast() → %d clients", self._fl_round, len(self.clients)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Phân cụm (Clustering) — phòng chống Negative Transfer
    # ──────────────────────────────────────────────────────────────────────────

    def cluster_clients(
        self,
        metadata_list: Optional[List[Dict]] = None,
    ) -> Dict[int, list]:
        """
        Phân cụm NodeAgent theo metadata giao lộ bằng K-Means đơn giản.

        Đặc trưng clustering: [num_phases_norm, obs_dim_norm, avg_loss_norm]

        Tham số
        -------
        metadata_list : [{intersection_id, num_phases, obs_dim, avg_loss}, ...]
                        Nếu None → gọi client.metadata() tự động.

        Trả về
        ------
        {cluster_id: [NodeAgent, ...]}
        """
        if metadata_list is None:
            metadata_list = [c.metadata() for c in self.clients]

        # Feature matrix: mỗi hàng là một client
        features = np.array([
            [
                m.get("num_phases", 4)  / 8.0,     # chuẩn hoá
                m.get("obs_dim",    84) / 168.0,
                m.get("avg_loss",   0)  / 2.0,
            ]
            for m in metadata_list
        ], dtype=np.float32)

        K = min(self.num_clusters, len(self.clients))
        labels = self._kmeans(features, K)

        clusters: Dict[int, list] = {k: [] for k in range(K)}
        for idx, label in enumerate(labels):
            clusters[label].append(self.clients[idx])

        logger.info(
            "Round %d | clustering → %d cụm: %s",
            self._fl_round,
            K,
            {k: len(v) for k, v in clusters.items()},
        )
        return clusters

    @staticmethod
    def _kmeans(X: np.ndarray, K: int, max_iter: int = 20) -> List[int]:
        """K-Means đơn giản (không dùng sklearn để tránh dependency)."""
        n = len(X)
        # Khởi tạo centroids từ K điểm đầu tiên
        centroids = X[:K].copy()
        labels = [0] * n
        for _ in range(max_iter):
            # Assign
            for i in range(n):
                dists = [float(np.linalg.norm(X[i] - centroids[k])) for k in range(K)]
                labels[i] = int(np.argmin(dists))
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(K, dtype=int)
            for i, l in enumerate(labels):
                new_centroids[l] += X[i]
                counts[l] += 1
            for k in range(K):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]
            if np.allclose(new_centroids, centroids, atol=1e-5):
                break
            centroids = new_centroids
        return labels

    # ──────────────────────────────────────────────────────────────────────────
    # Vòng FL hoàn chỉnh
    # ──────────────────────────────────────────────────────────────────────────

    def run_round(
        self,
        algorithm:       str = "fedavg",
        sample_counts:   Optional[List[int]] = None,
    ) -> Dict:
        """
        Điều phối một vòng FL hoàn chỉnh:
            1. Thu thập shared weights
            2. (Nếu use_clustering) Phân cụm → aggregate theo cụm
            3. aggregate()
            4. broadcast()

        Trả về
        ------
        dict : Log metrics của vòng này
        """
        self._fl_round += 1

        # Bước 1: Thu thập (uplink: Node → Server). Với clustering, chỉ cụm
        # đông nhất tham gia và các client được coi là ngang nhau.
        participants = list(self.clients)
        counts = sample_counts
        if self.use_clustering:
            clusters = self.cluster_clients()
            participants = max(clusters.values(), key=len)
            counts = None
        weight_packets = [c.extract_weights() for c in participants]
        logger.info(
            "Round %d | Thu thập weights từ %d clients", self._fl_round, len(participants)
        )

        # Bước 2 & 3: Aggregate
        self.aggregate(weight_packets, sample_counts=counts, algorithm=algorithm)
        coefficients = self._last_coefficients

        # Bước 4: Broadcast (downlink: Server → Node)
        self.broadcast()

        # Ghi log — đủ chi tiết để bên gọi in ra/hiển thị luồng truyền trọng số
        # Phần tử minh hoạ: trong tensor đầu tiên, chọn vị trí mà các client
        # khác nhau nhiều nhất để thấy rõ tác dụng của phép lấy trung bình.
        example_key = next(iter(self.global_weights))
        stacked = torch.stack([p[example_key].float().flatten() for p in weight_packets])
        example_idx = int((stacked.max(dim=0).values - stacked.min(dim=0).values).argmax())
        log = {
            "fl_round":      self._fl_round,
            "num_clients":   len(self.clients),
            "algorithm":     algorithm,
            "use_clustering": self.use_clustering,
            "param_tensors": len(self.global_weights),
            "global_norm":   self._l2_norm(self.global_weights),
            "clients": [
                {
                    "intersection_id": getattr(client, "intersection_id", f"client_{idx}"),
                    "n_k":         int(counts[idx]) if counts is not None else None,
                    "coefficient": coefficients[idx],
                    "num_params":  sum(t.numel() for t in packet.values()),
                    "size_kb":     sum(t.numel() * t.element_size() for t in packet.values()) / 1024,
                    "upload_norm": self._l2_norm(packet),
                    # ‖w* − w_k‖: backbone của client này bị kéo đi bao xa
                    "delta_norm":  self._l2_norm(
                        {k: self.global_weights[k] - packet[k].float() for k in packet}
                    ),
                }
                for idx, (client, packet) in enumerate(zip(participants, weight_packets))
            ],
            # Một phần tử trọng số cụ thể để minh hoạ phép tính w* = Σ n_k/N · w_k
            "example": {
                "key":           example_key,
                "index":         example_idx,
                "client_values": [float(v) for v in stacked[:, example_idx]],
                "global_value":  float(self.global_weights[example_key].flatten()[example_idx]),
            },
        }
        self._round_logs.append(log)
        return log

    @staticmethod
    def _l2_norm(weights: Dict[str, torch.Tensor]) -> float:
        """Chuẩn L2 của toàn bộ một state_dict (coi như một vector phẳng)."""
        return math.sqrt(sum(float((t.float() ** 2).sum()) for t in weights.values()))

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def round_logs(self) -> List[Dict]:
        return self._round_logs

    def __repr__(self) -> str:
        return (
            f"FederatedServer(clients={len(self.clients)}, "
            f"clustering={self.use_clustering}, "
            f"rounds_done={self._fl_round})"
        )
