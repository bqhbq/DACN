"""
main_demo_fl.py
---------------
Demo tích hợp FedMARL-ATSC: CityFlow + NodeAgent + FederatedServer.

Luồng chạy:
    1. Sinh roadnet/flow dạng lưới và config.json, bật replay logging.
    2. Mỗi DECISION_INTERVAL giây, từng NodeAgent đọc state và huấn luyện cục bộ.
       Có hai chế độ (`--trainer`):
         - `local` (mặc định): Q-learning một bước với reward Max-Pressure
           (docs/MDP_SPEC.md §4) — KHÔNG cần API bên ngoài.
         - `llm`: Behavior Cloning theo LLM (Gemini/OpenAI), cần API key
           trong `.env`.
    3. `SignalController` áp action xuống CityFlow bằng `eng.set_tl_phase()`,
       tự chèn pha vàng và giữ thời gian xanh tối thiểu.
    4. Mỗi FL_INTERVAL giây, FederatedServer tổng hợp backbone (FedAvg có trọng
       số n_k) và broadcast; console in rõ uplink → aggregate → downlink.
    5. Ghi replay + FL log + chart để xem bằng `cityflow/frontend/index.html`.

Cách chạy (từ thư mục gốc repo, trong môi trường đã cài CityFlow):
    python main_demo_fl.py
    python main_demo_fl.py --steps 1800 --fl-interval 100
    python main_demo_fl.py --trainer llm --llm-provider openai
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
CITYFLOW_DIR = ROOT_DIR / "cityflow"

# Các generator của nhóm mô phỏng nằm trong package `src` bên dưới cityflow/.
for _path in (ROOT_DIR, CITYFLOW_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cityflow  # noqa: E402

from src.generators.config_gen import gen_config  # noqa: E402
from src.generators.roadnet_gen import generate_roadnet  # noqa: E402

from agents.node_agent import NodeAgent  # noqa: E402
from federated.server import FederatedServer  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────────────────────────────────────

SCENARIO_DIR = CITYFLOW_DIR / "sample" / "demo_fl"
CONFIG_NAME = "config.json"
ROADNET_NAME = "roadnet.json"
FLOW_NAME = "temp_flow.json"  # sinh kèm khi tạo roadnet
REPLAY_LOG_FILE = "render/replay.txt"
ROADNET_LOG_FILE = "render/replay_roadnet.json"
FL_LOG_FILE = "render/fl_log.json"    # sự kiện FL cho frontend (ô "FL Log File")
FL_CHART_FILE = "render/fl_chart.txt"  # độ lệch backbone theo thời gian (ô "Chart File")
FL_CHART_TITLE = "Backbone divergence giữa các node (max |Δw|) — rơi về 0 sau mỗi vòng FL"

# Ngưỡng chuẩn hoá state theo docs/MDP_SPEC.md §2.3
Q_MAX = 50.0
V_MAX = 16.67
RHO_MAX = 50.0
FEATURES_PER_LANE = 3  # queue, speed, density

# Trọng số Composite Reward theo docs/MDP_SPEC.md §4.2
REWARD_ALPHA = 0.6          # α — Max-Pressure penalty
REWARD_BETA = 0.4           # β — Throughput bonus
REWARD_PHASE_PENALTY = 0.05  # γ_phase — phạt mỗi lần đổi pha

TRAINER_LOCAL = "local"  # Q-learning cục bộ, không cần API
TRAINER_LLM = "llm"      # Behavior Cloning theo LLM

RIGHT_TURN = "turn_right"  # luồng rẽ phải luôn được đi, kể cả ở pha vàng
MOVEMENT_NAMES = {"go_straight": "straight", "turn_left": "left turn"}


@dataclass(frozen=True)
class DemoConfig:
    """Tham số của buổi demo.

    Attributes:
        rows: Số hàng giao lộ của lưới.
        cols: Số cột giao lộ của lưới.
        total_steps: Tổng số bước mô phỏng (1 bước = 1 giây).
        decision_interval: Δt giữa hai lần agent ra quyết định (giây).
        fl_interval: N bước giữa hai vòng FL.
        min_green: Thời gian xanh tối thiểu trước khi được đổi pha (giây).
        yellow_time: Thời gian pha vàng chèn giữa hai pha xanh (giây).
        epsilon: Xác suất khám phá ε-greedy của agent.
        lr: Learning rate của optimizer cục bộ.
        seed: Seed cho CityFlow, NumPy và PyTorch.
        trainer: Cách huấn luyện cục bộ: "local" (Q-learning) hoặc "llm".
        llm_provider: LLM dùng khi trainer="llm": "gemini" hoặc "openai".
    """

    rows: int = 1
    cols: int = 2
    total_steps: int = 600
    decision_interval: int = 10
    fl_interval: int = 100
    min_green: int = 10
    yellow_time: int = 3  # C-03 trong MDP_SPEC §3.3
    epsilon: float = 0.3
    lr: float = 1e-3
    seed: int = 42
    trainer: str = TRAINER_LOCAL
    llm_provider: str = "gemini"


# ──────────────────────────────────────────────────────────────────────────────
# Bước 1 — Chuẩn bị dữ liệu và cấu hình CityFlow
# ──────────────────────────────────────────────────────────────────────────────


def ensure_replay_enabled(config_path: Path) -> dict[str, Any]:
    """Đọc config CityFlow và bật các cờ mà demo cần, ghi đè file nếu phải sửa.

    Các cờ được đảm bảo:
        - `saveReplay = true` và có `replayLogFile`, `roadnetLogFile`
          để frontend có dữ liệu hiển thị.
        - `rlTrafficLight = true`; thiếu cờ này CityFlow sẽ bỏ qua
          `set_tl_phase()` và tự chạy chu kỳ đèn cố định.

    Args:
        config_path: Đường dẫn tới file config.json.

    Returns:
        Nội dung config sau khi đã chuẩn hoá.
    """
    with config_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)

    required: dict[str, Any] = {"saveReplay": True, "rlTrafficLight": True}
    defaults: dict[str, Any] = {
        "replayLogFile": REPLAY_LOG_FILE,
        "roadnetLogFile": ROADNET_LOG_FILE,
    }

    changed = False
    for key, value in required.items():
        if config.get(key) != value:
            config[key] = value
            changed = True
    for key, value in defaults.items():
        if not config.get(key):
            config[key] = value
            changed = True

    # CityFlow không tự tạo thư mục chứa file replay.
    base_dir = Path(config["dir"])
    for key in defaults:
        (base_dir / config[key]).parent.mkdir(parents=True, exist_ok=True)

    if changed:
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
        print(f"[Setup] Đã ghi đè {config_path.name}: bật saveReplay và rlTrafficLight.")
    return config


def prepare_scenario(cfg: DemoConfig) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Sinh roadnet, flow và config.json cho demo.

    Args:
        cfg: Tham số demo.

    Returns:
        Bộ ba (đường dẫn config.json, nội dung config, nội dung roadnet).
    """
    roadnet_path = generate_roadnet(
        rows=cfg.rows,
        cols=cfg.cols,
        output_dir=SCENARIO_DIR,
        roadnet_name=ROADNET_NAME,
        tl_plan=True,
    )
    # Config luôn được ghi lại vì trường "dir" là đường dẫn tuyệt đối,
    # khác nhau giữa máy host và container.
    gen_config(
        dir=SCENARIO_DIR,
        config_name=CONFIG_NAME,
        roadnet_name=ROADNET_NAME,
        flow_name=FLOW_NAME,
        seed=cfg.seed,
        save_replay=True,
        rl_traffic_light=True,
    )
    config_path = SCENARIO_DIR / CONFIG_NAME
    config = ensure_replay_enabled(config_path)

    with roadnet_path.open("r", encoding="utf-8") as file:
        roadnet: dict[str, Any] = json.load(file)
    return config_path, config, roadnet


# ──────────────────────────────────────────────────────────────────────────────
# Lớp cầu nối giữa agent và CityFlow cho MỘT giao lộ
# ──────────────────────────────────────────────────────────────────────────────


class SignalController:
    """Đọc state và điều khiển đèn của một giao lộ trên CityFlow Engine.

    Lớp này đóng vai trò `TrafficEnv` rút gọn cho demo: nó không chứa PyTorch,
    và agent không biết gì về pha vàng hay thời gian xanh tối thiểu.

    Args:
        engine: CityFlow Engine đang chạy.
        intersection: Phần tử tương ứng trong `roadnet["intersections"]`.
        roads: Bảng tra road theo id, lấy từ `roadnet["roads"]`.
        min_green: Thời gian xanh tối thiểu (giây).
        yellow_time: Thời gian pha vàng (giây).
    """

    def __init__(
        self,
        engine: cityflow.Engine,
        intersection: dict[str, Any],
        roads: dict[str, dict[str, Any]],
        min_green: int,
        yellow_time: int,
    ) -> None:
        self.engine = engine
        self.intersection_id: str = intersection["id"]
        self.min_green = min_green
        self.yellow_time = yellow_time

        # Làn đi vào giao lộ, theo quy ước đặt tên của CityFlow: "<road_id>_<k>".
        incoming_roads = [
            road_id
            for road_id in intersection["roads"]
            if roads[road_id]["endIntersection"] == self.intersection_id
        ]
        self.incoming_lanes: list[str] = [
            f"{road_id}_{lane_idx}"
            for road_id in incoming_roads
            for lane_idx in range(len(roads[road_id]["lanes"]))
        ]
        # Làn đi ra khỏi giao lộ — cần cho áp lực Max-Pressure (MDP_SPEC §4.1).
        self.outgoing_lanes: list[str] = [
            f"{road_id}_{lane_idx}"
            for road_id in intersection["roads"]
            if roads[road_id]["startIntersection"] == self.intersection_id
            for lane_idx in range(len(roads[road_id]["lanes"]))
        ]
        self._prev_incoming_vehicles: set[str] = set()
        self._switched_since_reward = False

        # Hướng mà xe đi tới (West/South/East/North), cùng thứ tự với incoming_lanes;
        # dùng để LLM đọc state bằng ngôn ngữ tự nhiên.
        approach_of = {
            road_id: self._approach_name(roads[road_id]) for road_id in incoming_roads
        }
        self.approach_names: list[str] = [approach_of[r] for r in incoming_roads]

        # Pha xanh là pha cho phép ít nhất một luồng không phải rẽ phải;
        # pha còn lại (chỉ rẽ phải) được dùng làm pha vàng.
        road_links = intersection["roadLinks"]
        phases = intersection["trafficLight"]["lightphases"]
        self.green_phases: list[int] = []
        self.yellow_phase: int | None = None
        for phase_id, phase in enumerate(phases):
            is_green = any(
                road_links[link]["type"] != RIGHT_TURN
                for link in phase["availableRoadLinks"]
            )
            if is_green:
                self.green_phases.append(phase_id)
            elif self.yellow_phase is None:
                self.yellow_phase = phase_id
        if not self.green_phases:
            raise ValueError(f"{self.intersection_id}: không tìm thấy pha xanh nào.")

        # Mô tả từng pha xanh, vd. "West/East straight".
        self.phase_descriptions: list[str] = []
        for phase_id in self.green_phases:
            movements: dict[str, list[str]] = {}
            for link in phases[phase_id]["availableRoadLinks"]:
                link_type = road_links[link]["type"]
                start_road = road_links[link]["startRoad"]
                if link_type != RIGHT_TURN and start_road in approach_of:
                    movements.setdefault(link_type, []).append(approach_of[start_road])
            self.phase_descriptions.append(
                ", ".join(
                    f"{'/'.join(sides)} {MOVEMENT_NAMES.get(kind, kind)}"
                    for kind, sides in movements.items()
                )
            )

        self.current_green: int = 0  # index trong green_phases
        self._pending_green: int | None = None
        self._yellow_left: int = 0
        self._green_elapsed: int = 0
        self.switch_count: int = 0

        self._set_phase(self.green_phases[self.current_green])

    @property
    def obs_dim(self) -> int:
        """Kích thước vector state."""
        return len(self.incoming_lanes) * FEATURES_PER_LANE

    @property
    def num_phases(self) -> int:
        """Số pha xanh mà agent được chọn."""
        return len(self.green_phases)

    @property
    def can_switch(self) -> bool:
        """True nếu lúc này được phép đổi pha (không ở pha vàng, đã đủ xanh tối thiểu)."""
        return self._pending_green is None and self._green_elapsed >= self.min_green

    @staticmethod
    def _approach_name(road: dict[str, Any]) -> str:
        """Tên hướng mà xe trên `road` đi tới giao lộ (ngược với hướng xe chạy)."""
        start, end = road["points"][0], road["points"][-1]
        dx, dy = end["x"] - start["x"], end["y"] - start["y"]
        if abs(dx) >= abs(dy):
            return "West" if dx > 0 else "East"
        return "South" if dy > 0 else "North"

    def _set_phase(self, phase_id: int) -> None:
        """Áp pha đèn xuống CityFlow. Đây là lời gọi làm đèn đổi màu trên UI."""
        self.engine.set_tl_phase(self.intersection_id, phase_id)

    def get_state(self) -> np.ndarray:
        """Đọc state cục bộ: [queue, speed, density] cho mỗi làn vào, đã chuẩn hoá.

        Returns:
            Vector float32 shape (obs_dim,), mọi phần tử trong [0, 1].
        """
        waiting = self.engine.get_lane_waiting_vehicle_count()
        counts = self.engine.get_lane_vehicle_count()
        lane_vehicles = self.engine.get_lane_vehicles()
        speeds = self.engine.get_vehicle_speed()

        queue, speed, density = [], [], []
        for lane in self.incoming_lanes:
            vehicles = lane_vehicles.get(lane, [])
            # Làn trống được coi là thông thoáng (tốc độ tối đa).
            mean_speed = (
                float(np.mean([speeds[v] for v in vehicles])) if vehicles else V_MAX
            )
            queue.append(waiting.get(lane, 0) / Q_MAX)
            speed.append(mean_speed / V_MAX)
            density.append(counts.get(lane, 0) / RHO_MAX)

        state = np.array(queue + speed + density, dtype=np.float32)
        return np.clip(state, 0.0, 1.0)

    def get_reward(self) -> float:
        """Tính Composite Reward của khoảng thời gian từ lần gọi trước tới giờ.

        Theo docs/MDP_SPEC.md §4.2:
            R = −α·P + β·ΔThroughput − γ_phase·1[đổi pha]
        với P = |Σ q_in − Σ q_out| (dạng tuyệt đối, §4.1) tính bằng số xe đang
        chờ, và ΔThroughput là số xe đã rời các làn vào (qua vạch dừng).

        Returns:
            Reward cục bộ R của giao lộ này.
        """
        waiting = self.engine.get_lane_waiting_vehicle_count()
        queue_in = sum(waiting.get(lane, 0) for lane in self.incoming_lanes)
        queue_out = sum(waiting.get(lane, 0) for lane in self.outgoing_lanes)
        pressure = abs(queue_in - queue_out)

        lane_vehicles = self.engine.get_lane_vehicles()
        incoming_now = {
            vehicle
            for lane in self.incoming_lanes
            for vehicle in lane_vehicles.get(lane, [])
        }
        throughput = len(self._prev_incoming_vehicles - incoming_now)
        phase_changed = float(self._switched_since_reward)

        self._prev_incoming_vehicles = incoming_now
        self._switched_since_reward = False
        return (
            -REWARD_ALPHA * pressure
            + REWARD_BETA * throughput
            - REWARD_PHASE_PENALTY * phase_changed
        )

    def apply_action(self, action: int) -> str:
        """Nhận action của agent và bắt đầu chuyển pha nếu hợp lệ.

        Args:
            action: Index pha xanh mong muốn, thuộc [0, num_phases).

        Returns:
            Mô tả ngắn kết quả: "keep", "switch" hoặc "blocked".
        """
        if action == self.current_green and self._pending_green is None:
            return "keep"
        # Đang ở pha vàng hoặc chưa đủ thời gian xanh tối thiểu: giữ nguyên.
        if self._pending_green is not None or self._green_elapsed < self.min_green:
            return "blocked"

        self.switch_count += 1
        self._switched_since_reward = True
        if self.yellow_phase is None or self.yellow_time <= 0:
            self._start_green(action)
        else:
            self._pending_green = action
            self._yellow_left = self.yellow_time
            self._set_phase(self.yellow_phase)
        return "switch"

    def tick(self) -> None:
        """Cập nhật đồng hồ pha; gọi đúng một lần sau mỗi `engine.next_step()`."""
        if self._pending_green is None:
            self._green_elapsed += 1
            return
        self._yellow_left -= 1
        if self._yellow_left <= 0:
            self._start_green(self._pending_green)

    def _start_green(self, green_idx: int) -> None:
        """Bật pha xanh mới và đặt lại đồng hồ."""
        self.current_green = green_idx
        self._pending_green = None
        self._green_elapsed = 0
        self._set_phase(self.green_phases[green_idx])


# ──────────────────────────────────────────────────────────────────────────────
# Federated Learning — log ra console
# ──────────────────────────────────────────────────────────────────────────────


def backbone_divergence(agents: list[NodeAgent]) -> float:
    """Tính độ lệch backbone lớn nhất giữa các node so với node đầu tiên.

    Args:
        agents: Danh sách NodeAgent.

    Returns:
        Giá trị |Δw| lớn nhất trên mọi tham số backbone; bằng 0 khi đã đồng bộ.
    """
    reference = agents[0].get_shared_weights()
    max_diff = 0.0
    for agent in agents[1:]:
        weights = agent.get_shared_weights()
        for key, tensor in reference.items():
            max_diff = max(max_diff, float((tensor - weights[key]).abs().max()))
    return max_diff


def run_fl_round(
    server: FederatedServer, agents: list[NodeAgent], node_names: dict[str, str]
) -> dict[str, Any]:
    """Chạy một vòng FL và in log từng chặng: uplink → aggregate → downlink.

    Args:
        server: FederatedServer (Central Coordinator) đã đăng ký các agent.
        agents: Danh sách NodeAgent tham gia.
        node_names: Bảng ánh xạ intersection_id → tên hiển thị.

    Returns:
        Log của vòng FL (từ `FederatedServer.run_round()`), bổ sung tên node,
        độ lệch backbone trước/sau và kết quả kiểm tra head cục bộ.
    """
    before = backbone_divergence(agents)
    head_before = [agent.local_head_norm() for agent in agents]
    log = server.run_round(
        algorithm="fedavg",
        sample_counts=[agent.num_local_samples for agent in agents],
    )
    after = backbone_divergence(agents)
    head_kept = [
        agent.local_head_norm() == norm for agent, norm in zip(agents, head_before)
    ]

    for client, kept in zip(log["clients"], head_kept):
        client["name"] = node_names[client["intersection_id"]]
        client["head_unchanged"] = kept
    log["divergence_before"] = before
    log["divergence_after"] = after

    tag = f"[FL Round {log['fl_round']}]"
    clients = log["clients"]
    print("─" * 72)
    print(f"{tag} ① UPLINK  Node Agent ──▶ Central Coordinator (chỉ backbone)")
    for c in clients:
        print(
            f"   {c['name']} ──▶ Server | {log['param_tensors']} tensors, "
            f"{c['num_params']:,} tham số ({c['size_kb']:.1f} KB) | "
            f"n_k={c['n_k']} mẫu | ‖w_k‖={c['upload_norm']:.4f}"
        )

    print(f"{tag} ② AGGREGATE tại Central Coordinator: w* = Σ (n_k/N)·w_k  (FedAvg)")
    print(
        "   hệ số n_k/N: "
        + ", ".join(f"{c['name']}={c['coefficient']:.3f}" for c in clients)
    )
    example = log["example"]
    terms = " + ".join(
        f"{c['coefficient']:.3f}×({value:+.5f})"
        for c, value in zip(clients, example["client_values"])
    )
    print(
        f"   ví dụ phần tử [{example['index']}] của '{example['key']}': {terms} "
        f"= {example['global_value']:+.5f}"
    )
    print(f"   ‖w*‖={log['global_norm']:.4f}")

    print(f"{tag} ③ DOWNLINK Central Coordinator ──▶ Node Agent (broadcast w*)")
    for c in clients:
        head = "giữ nguyên ✓" if c["head_unchanged"] else "BỊ THAY ĐỔI ✗"
        print(
            f"   Server ──▶ {c['name']} | backbone dịch ‖w*−w_k‖={c['delta_norm']:.4f} | "
            f"local_policy_head {head}"
        )
    print(f"{tag} độ lệch backbone giữa các node: {before:.6f} → {after:.6f}")
    print("─" * 72)
    return log


# ──────────────────────────────────────────────────────────────────────────────
# Huấn luyện cục bộ tại một node — hai chế độ
# ──────────────────────────────────────────────────────────────────────────────


def local_decision(
    agent: NodeAgent,
    controller: SignalController,
    memory: dict[str, tuple[np.ndarray, int]],
) -> tuple[str, dict[str, Any]]:
    """Một decision point ở chế độ `local` (docs/MDP_SPEC.md §6.1).

    Quan sát s', nhận reward r của hành động trước, học từ transition
    (s, a, r, s') rồi chọn hành động mới bằng ε-greedy. Safety wrapper của
    `SignalController` có thể chặn hành động (C-02) — agent không cần biết.

    Args:
        agent: NodeAgent của giao lộ.
        controller: SignalController của cùng giao lộ.
        memory: (s, a) của lần quyết định trước, theo intersection_id.

    Returns:
        (chuỗi tóm tắt để in, dict số liệu để ghi vào FL log).
    """
    obs = controller.get_state()
    reward = controller.get_reward()

    loss: float | None = None
    previous = memory.get(agent.intersection_id)
    if previous is not None:
        loss = agent.train_local_step(previous[0], previous[1], reward, obs)

    action = agent.select_action(obs)            # pha xanh mong muốn (§3.1)
    result = controller.apply_action(action)     # → eng.set_tl_phase()
    memory[agent.intersection_id] = (obs, action)

    loss_text = f"loss={loss:.4f}" if loss is not None else "loss=—"
    text = f"a={action} ({result}), r={reward:+.2f}, {loss_text}"
    info = {"action": action, "result": result, "reward": reward, "loss": loss}
    return text, info


def llm_decision(
    agent: NodeAgent, controller: SignalController
) -> tuple[str, dict[str, Any]]:
    """Một decision point ở chế độ `llm`: LLM quyết định KEEP/CHANGE, mạng học theo.

    Args:
        agent: NodeAgent của giao lộ.
        controller: SignalController của cùng giao lộ.

    Returns:
        (chuỗi tóm tắt để in, dict số liệu để ghi vào FL log).
    """
    reward = controller.get_reward()
    # Đang pha vàng / chưa đủ xanh tối thiểu: không đổi pha được,
    # nên không tốn một lời gọi LLM.
    if not controller.can_switch:
        info = {"action": None, "result": "hold", "reward": reward, "loss": None}
        return "hold", info

    obs = controller.get_state()
    phase = controller.current_green
    student = agent.select_action(obs, greedy=True)  # mạng đoán trước khi học
    steps_before = len(agent.loss_history)
    expert_action = agent.train_step(obs, current_phase=phase)
    target = agent.expert_action_to_phase(expert_action, phase)
    result = controller.apply_action(target)         # → eng.set_tl_phase()

    loss = agent.last_loss if len(agent.loss_history) > steps_before else None
    loss_text = f"loss={loss:.4f}" if loss is not None else "no-train"
    text = (
        f"LLM={expert_action}→phase {target} ({result}), student={student}, {loss_text}"
    )
    info = {"action": target, "result": result, "reward": reward, "loss": loss}
    return text, info


# ──────────────────────────────────────────────────────────────────────────────
# Vòng lặp mô phỏng chính
# ──────────────────────────────────────────────────────────────────────────────


def run_demo(cfg: DemoConfig) -> None:
    """Chạy toàn bộ demo.

    Args:
        cfg: Tham số demo.
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 1. Dữ liệu + config (đã bật replay)
    config_path, config, roadnet = prepare_scenario(cfg)
    engine = cityflow.Engine(str(config_path), thread_num=1)

    # 2. Mỗi giao lộ thật có một SignalController và một NodeAgent
    roads = {road["id"]: road for road in roadnet["roads"]}
    controllers: dict[str, SignalController] = {}
    agents: list[NodeAgent] = []
    node_names: dict[str, str] = {}
    for intersection in roadnet["intersections"]:
        if intersection["virtual"]:
            continue
        controller = SignalController(
            engine, intersection, roads, cfg.min_green, cfg.yellow_time
        )
        agent = NodeAgent(
            intersection_id=controller.intersection_id,
            obs_dim=controller.obs_dim,
            num_phases=controller.num_phases,
            lr=cfg.lr,
            epsilon=cfg.epsilon,
            llm_provider=cfg.llm_provider if cfg.trainer == TRAINER_LLM else "none",
            approach_names=controller.approach_names,
            phase_descriptions=controller.phase_descriptions,
        )
        controllers[controller.intersection_id] = controller
        agents.append(agent)
        node_names[controller.intersection_id] = f"Node_{len(agents)}"

    # Kiểm tra kiến trúc Decoupled (AGENTS.md §5.3): head không được rời node.
    for agent in agents:
        shared = agent.get_shared_weights()
        assert not any("local_policy_head" in key for key in shared), "R-02 vi phạm!"

    server = FederatedServer(clients=agents)
    # Vòng 0 của FL: mọi node xuất phát từ CÙNG một backbone w⁰ do server phát,
    # nếu không vòng 1 sẽ phải lấy trung bình hai mạng khởi tạo ngẫu nhiên khác nhau.
    server.aggregate([agents[0].get_shared_weights()])
    server.broadcast()

    print("=" * 72)
    print(" FedMARL-ATSC — Demo FL trên CityFlow")
    print("=" * 72)
    for agent in agents:
        print(
            f" {node_names[agent.intersection_id]} ↔ {agent.intersection_id} | "
            f"obs_dim={agent.obs_dim} | num_phases={agent.num_phases}"
        )
    print(
        f" Mô phỏng {cfg.total_steps} bước | quyết định mỗi {cfg.decision_interval}s | "
        f"FL mỗi {cfg.fl_interval} bước"
    )
    trainer_text = (
        f"LLM [{cfg.llm_provider}] — Behavior Cloning"
        if cfg.trainer == TRAINER_LLM
        else "local — Q-learning một bước, reward Max-Pressure (không dùng API)"
    )
    print(f" Huấn luyện cục bộ: {trainer_text}")
    print(
        f" [FL Round 0] Central Coordinator ──▶ {len(agents)} node: phát backbone "
        f"khởi tạo chung w⁰ (độ lệch = {backbone_divergence(agents):.6f})"
    )
    print("=" * 72)

    # 3. Vòng lặp mô phỏng
    memory: dict[str, tuple[np.ndarray, int]] = {}   # (s, a) lần trước của mỗi node
    decisions: list[dict[str, Any]] = []             # ghi cho frontend
    rounds: list[dict[str, Any]] = []
    chart_values: list[float] = []
    divergence = backbone_divergence(agents)

    for step in range(cfg.total_steps):
        if step % cfg.decision_interval == 0:
            summary = []
            record: dict[str, Any] = {}
            for agent in agents:
                controller = controllers[agent.intersection_id]
                if cfg.trainer == TRAINER_LLM:
                    text, info = llm_decision(agent, controller)
                else:
                    text, info = local_decision(agent, controller, memory)
                info["n_k"] = agent.num_local_samples
                info["phase"] = controller.current_green
                record[agent.intersection_id] = info
                summary.append(f"{node_names[agent.intersection_id]}: {text}")
            divergence = backbone_divergence(agents)
            decisions.append({"step": step, "divergence": divergence, "nodes": record})
            print(
                f"[Step {step:04d}] xe={engine.get_vehicle_count():3d} | "
                + " | ".join(summary)
            )

        engine.next_step()
        for controller in controllers.values():
            controller.tick()

        # Sau mỗi N bước: tổng hợp backbone tại FL Server
        if (step + 1) % cfg.fl_interval == 0:
            round_log = run_fl_round(server, agents, node_names)
            round_log["step"] = step
            rounds.append(round_log)
            divergence = round_log["divergence_after"]

        chart_values.append(divergence)

    # Ghi file phụ cho frontend: FL log (JSON) và chart (text, mỗi bước một dòng).
    (SCENARIO_DIR / FL_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    fl_log = {
        "trainer": cfg.trainer,
        "total_steps": cfg.total_steps,
        "fl_interval": cfg.fl_interval,
        "nodes": node_names,
        "decisions": decisions,
        "rounds": rounds,
    }
    with (SCENARIO_DIR / FL_LOG_FILE).open("w", encoding="utf-8") as file:
        json.dump(fl_log, file, ensure_ascii=False, indent=1)
    with (SCENARIO_DIR / FL_CHART_FILE).open("w", encoding="utf-8") as file:
        file.write(FL_CHART_TITLE + "\n")
        # CityFlow ghi replay nhiều hơn số bước một khung hình, nên lặp lại
        # giá trị cuối để chart luôn đủ dòng cho frontend.
        file.writelines(f"{value:.6f}\n" for value in [*chart_values, chart_values[-1]])

    # 4. Tổng kết
    print("=" * 72)
    print(f" Hoàn tất {cfg.total_steps} bước, {len(server.round_logs)} vòng FL.")
    print(f" Average travel time: {engine.get_average_travel_time():.1f}s")
    for agent in agents:
        controller = controllers[agent.intersection_id]
        llm_text = (
            f" | LLM gọi {agent.llm_calls} lần, lỗi {agent.llm_failures}"
            if cfg.trainer == TRAINER_LLM
            else ""
        )
        print(
            f" {node_names[agent.intersection_id]}: đổi pha {controller.switch_count} lần | "
            f"{len(agent.loss_history)} bước huấn luyện cục bộ{llm_text}"
        )
    # In đường dẫn tương đối với gốc repo để dùng được cả trên host lẫn trong container.
    replay_dir = SCENARIO_DIR.relative_to(ROOT_DIR)
    print(" File nạp vào frontend (cityflow/frontend/index.html):")
    print(f"   Roadnet File: {replay_dir / config['roadnetLogFile']}")
    print(f"   Replay File : {replay_dir / config['replayLogFile']}")
    print(f"   Chart File  : {replay_dir / FL_CHART_FILE}  (tuỳ chọn)")
    print(f"   FL Log File : {replay_dir / FL_LOG_FILE}  (tuỳ chọn)")
    print("=" * 72)

    del engine  # đóng Engine để CityFlow ghi nốt replay ra đĩa


def parse_args() -> DemoConfig:
    """Đọc tham số dòng lệnh và trả về DemoConfig."""
    defaults = DemoConfig()
    parser = argparse.ArgumentParser(description="Demo FedMARL-ATSC trên CityFlow.")
    parser.add_argument("--rows", type=int, default=defaults.rows)
    parser.add_argument("--cols", type=int, default=defaults.cols)
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument(
        "--decision-interval", type=int, default=defaults.decision_interval
    )
    parser.add_argument("--fl-interval", type=int, default=defaults.fl_interval)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--trainer", choices=(TRAINER_LOCAL, TRAINER_LLM), default=defaults.trainer
    )
    parser.add_argument(
        "--llm-provider", choices=("gemini", "openai"), default=defaults.llm_provider
    )
    args = parser.parse_args()
    return DemoConfig(
        rows=args.rows,
        cols=args.cols,
        total_steps=args.steps,
        decision_interval=args.decision_interval,
        fl_interval=args.fl_interval,
        seed=args.seed,
        trainer=args.trainer,
        llm_provider=args.llm_provider,
    )


if __name__ == "__main__":
    # Hiện log của NodeAgent, vd. "LLM [Gemini] decided Action: 1".
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Log nội bộ của FederatedServer bị ẩn vì run_fl_round() đã in chi tiết hơn.
    logging.getLogger("federated.server").setLevel(logging.WARNING)
    run_demo(parse_args())
