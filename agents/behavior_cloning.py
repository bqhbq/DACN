"""
agents/behavior_cloning.py
--------------------------
BehaviorCloningAgent — mô hình "Thầy - Trò" trên bộ khung NodeAgent.

  Thầy (`LLMExpert`)      đọc state dạng văn bản → KEEP (0) / CHANGE (1)
  Trò  (`TrafficAgentNet`) học bắt chước bằng CrossEntropyLoss trên nhãn của thầy

Đây là giai đoạn đầu của lộ trình (CLAUDE.md §4): dùng để demo và kiểm chứng
luồng FL trước khi có thuật toán RL thật. Agent này bỏ qua reward — "loss"
của nó là sai khác giữa dự đoán của mạng và quyết định của thầy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from agents.llm_expert import ACTION_CHANGE, LLMExpert
from agents.node_agent import NodeAgent
from agents.replay_buffer import TransitionBatch

ELAPSED_READY = 1.0  # elapsed_time ≥ ngưỡng này ⇔ đã đủ thời gian xanh tối thiểu


@dataclass(frozen=True)
class ObservationLayout:
    """Vị trí các nhóm đặc trưng trong vector state (docs/MDP_SPEC.md §2.3).

    Layout: [phase one-hot × P, elapsed_time, queue × L, speed × L, density × L].
    Phải khớp với nơi dựng state ở phía môi trường.

    Attributes:
        num_phases: P — số pha xanh.
        num_lanes: L — số làn đi vào giao lộ.
        queue_scale: Ngưỡng chuẩn hoá queue phía môi trường (queue thật =
            giá trị trong state × queue_scale).
    """

    num_phases: int
    num_lanes: int
    queue_scale: float

    def current_phase(self, state: np.ndarray) -> int:
        """Index pha xanh hiện tại, đọc từ phần one-hot."""
        return int(np.argmax(state[: self.num_phases]))

    def elapsed_time(self, state: np.ndarray) -> float:
        """Thời gian pha hiện tại đã duy trì, chuẩn hoá theo xanh tối thiểu."""
        return float(state[self.num_phases])

    def queues(self, state: np.ndarray) -> List[int]:
        """Số xe đang chờ trên từng làn vào, đã khử chuẩn hoá."""
        start = self.num_phases + 1
        scaled = np.asarray(state[start : start + self.num_lanes]) * self.queue_scale
        return [int(round(float(value))) for value in scaled]


class BehaviorCloningAgent(NodeAgent):
    """Agent học bắt chước quyết định của một LLM.

    Khi huấn luyện, action điều khiển đèn là action của THẦY; mạng chỉ học
    theo. `select_action(greedy=True)` trả về dự đoán của riêng mạng.

    Args:
        intersection_id: ID giao lộ.
        obs_dim: Kích thước vector state.
        num_phases: Số pha xanh khả dụng.
        layout: Vị trí các đặc trưng trong state, để dịch state cho LLM.
        llm_provider: "gemini" hoặc "openai".
        approach_names: Tên các hướng vào giao lộ, theo thứ tự làn trong state.
        phase_descriptions: Mô tả ngắn từng pha xanh.
        **kwargs: Tham số còn lại của `NodeAgent` (hidden_dim, lr, device, ...).

    Raises:
        ValueError: Nếu `layout` không khớp `obs_dim` / `num_phases`.
    """

    def __init__(
        self,
        intersection_id: str,
        obs_dim: int,
        num_phases: int,
        layout: ObservationLayout,
        llm_provider: str = "gemini",
        approach_names: Optional[Sequence[str]] = None,
        phase_descriptions: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        if layout.num_phases != num_phases or layout.num_phases + 1 + layout.num_lanes > obs_dim:
            raise ValueError(
                f"ObservationLayout {layout} không khớp obs_dim={obs_dim}, num_phases={num_phases}."
            )
        super().__init__(intersection_id, obs_dim, num_phases, **kwargs)
        self.layout = layout
        self.expert = LLMExpert(
            provider=llm_provider,
            name=intersection_id,
            num_phases=num_phases,
            approach_names=approach_names,
            phase_descriptions=phase_descriptions,
        )
        self._criterion = nn.CrossEntropyLoss()
        # True ⇔ action vừa trả về đến từ thầy, tức là một nhãn hợp lệ để học.
        self._has_expert_label = False

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Hỏi thầy nên giữ hay đổi pha, trả về pha xanh đích.

        Không gọi LLM khi chưa đủ thời gian xanh tối thiểu (đọc từ
        `elapsed_time` trong state) vì lúc đó không đổi pha được. Khi không
        có quyết định của thầy (chưa đủ xanh, LLM lỗi), agent giữ pha hiện
        tại và transition tương ứng KHÔNG được dùng để học.

        Args:
            state: Vector state shape (obs_dim,).
            greedy: True → trả về dự đoán của mạng, không hỏi thầy.

        Returns:
            Index pha xanh đích, thuộc [0, num_phases).
        """
        if greedy:
            return super().select_action(state, greedy=True)

        current_phase = self.layout.current_phase(state)
        self._has_expert_label = False
        if self.layout.elapsed_time(state) < ELAPSED_READY:
            return current_phase

        prompt = self.expert.describe_state(self.layout.queues(state), current_phase)
        expert_action = self.expert.decide(prompt)
        if expert_action is None:
            return current_phase

        self._has_expert_label = True
        if expert_action == ACTION_CHANGE:
            return (current_phase + 1) % self.num_phases
        return current_phase

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Lưu (state, nhãn của thầy); bỏ qua nếu action không đến từ thầy."""
        if not self._has_expert_label:
            return
        self._has_expert_label = False
        super().store_transition(state, action, reward, next_state, done)

    def compute_loss(self, batch: TransitionBatch) -> torch.Tensor:
        """CrossEntropy giữa logits của mạng và pha đích thầy đã chọn.

        Args:
            batch: Lô (state, nhãn); reward và next_state không được dùng.

        Returns:
            Loss vô hướng.
        """
        return self._criterion(self.net(batch.states), batch.actions)

    def stats(self) -> Dict[str, Any]:
        """Bổ sung số lần gọi LLM và số lần lỗi vào số liệu tóm tắt."""
        return {
            **super().stats(),
            "llm_calls": self.expert.calls,
            "llm_failures": self.expert.failures,
        }
