"""Kiểm thử bản port main_fedmarl.py sang API agents mới, KHÔNG cần CityFlow thật.

Dùng một `cityflow` giả (module + Engine giả) và roadnet tối thiểu 2 giao lộ để
chạy được SignalController, decision_point và run_simulation trên host.
"""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ── Module cityflow giả: phải có trong sys.modules TRƯỚC khi import main_fedmarl ──
_fake = types.ModuleType("cityflow")
_fake.Engine = object  # type: ignore[attr-defined]
sys.modules.setdefault("cityflow", _fake)

import main_fedmarl as mf  # noqa: E402

DIRS = {"W": (-100, 0), "E": (100, 0), "S": (0, -100), "N": (0, 100)}


class FakeEngine:
    """Engine giả: hàng đợi phụ thuộc bước, xe rời làn khi có xanh."""

    def __init__(self, lanes: list[str]) -> None:
        self.lanes, self.t, self.phases = lanes, 0, {}
        self.rng = np.random.default_rng(0)

    def set_tl_phase(self, iid: str, phase: int) -> None:
        self.phases[iid] = phase

    def next_step(self) -> None:
        self.t += 1

    def get_lane_waiting_vehicle_count(self) -> dict[str, int]:
        return {lane: (self.t // 3 + i) % 6 for i, lane in enumerate(self.lanes)}

    def get_lane_vehicle_count(self) -> dict[str, int]:
        return {lane: (self.t // 4 + i) % 9 for i, lane in enumerate(self.lanes)}

    def get_lane_vehicles(self) -> dict[str, list[str]]:
        return {lane: [f"{lane}-{k}" for k in range((self.t // 5 + i) % 3)]
                for i, lane in enumerate(self.lanes)}

    def get_vehicle_speed(self) -> dict[str, float]:
        return {f"{lane}-{k}": 5.0 for lane in self.lanes for k in range(3)}

    def get_vehicle_count(self) -> int:
        return 10 + self.t % 5

    def get_average_travel_time(self) -> float:
        return 42.0


def _intersection(iid: str) -> tuple[dict, dict]:
    """Giao lộ 4 hướng, mỗi hướng 1 làn vào; 1 pha chỉ-rẽ-phải (vàng) + 4 pha xanh."""
    roads, links = {}, []
    for d, (x, y) in DIRS.items():
        roads[f"{iid}_in_{d}"] = {
            "id": f"{iid}_in_{d}", "startIntersection": f"v_{iid}_{d}", "endIntersection": iid,
            "lanes": [{}], "points": [{"x": x, "y": y}, {"x": 0, "y": 0}],
        }
        roads[f"{iid}_out_{d}"] = {
            "id": f"{iid}_out_{d}", "startIntersection": iid, "endIntersection": f"v_{iid}_{d}",
            "lanes": [{}], "points": [{"x": 0, "y": 0}, {"x": x, "y": y}],
        }
    for kind in ("go_straight", "turn_left", "turn_right"):
        for d in DIRS:
            links.append({"type": kind, "startRoad": f"{iid}_in_{d}"})
    idx = {(l["type"], l["startRoad"].rsplit("_", 1)[-1]): i for i, l in enumerate(links)}
    rights = [idx[("turn_right", d)] for d in DIRS]
    phase = lambda ds, kind: {"availableRoadLinks": [idx[(kind, d)] for d in ds] + rights}  # noqa: E731
    inter = {
        "id": iid, "virtual": False, "roads": list(roads),
        "roadLinks": links,
        "trafficLight": {"lightphases": [
            {"availableRoadLinks": rights},
            phase("WE", "go_straight"), phase("WE", "turn_left"),
            phase("SN", "go_straight"), phase("SN", "turn_left"),
        ]},
    }
    return inter, roads


@pytest.fixture()
def world():
    inters, roads = [], {}
    for iid in ("i1", "i2"):
        inter, r = _intersection(iid)
        inters.append(inter)
        roads.update(r)
    lanes = [f"{rid}_0" for rid in roads]
    return FakeEngine(lanes), inters, roads


def _ctrl(world, i=0, min_green=10):
    eng, inters, roads = world
    return mf.SignalController(eng, inters[i], roads, min_green, 3)


def test_state_layout_phase_block_and_dim(world) -> None:
    c = _ctrl(world)
    assert (c.num_phases, len(c.incoming_lanes)) == (4, 4)
    assert c.obs_dim == 4 + 1 + 3 * 4
    s = c.get_state()
    assert s.shape == (c.obs_dim,) and s.min() >= 0 and s.max() <= 1
    assert s[c.layout.phase_slice].tolist() == [1, 0, 0, 0] and s[c.layout.elapsed_index] == 0
    for _ in range(10):
        c.tick()
    assert c.get_state()[c.layout.elapsed_index] == 1.0


def test_yellow_zeroes_phase_block_and_counts_blocks(world) -> None:
    c = _ctrl(world)
    assert c.apply_action(1) == "blocked" and c.blocked_counts["min_green"] == 1
    for _ in range(10):
        c.tick()
    assert c.apply_action(1) == "switch"
    s = c.get_state()
    assert s[c.layout.phase_slice].sum() == 0 and s[c.layout.elapsed_index] == 0
    assert c.apply_action(2) == "blocked" and c.blocked_counts["in_transition"] == 1
    assert c.proposals == 3


@pytest.mark.parametrize("trainer,cls", [("local", "QLearningAgent"), ("dqn", "DQNAgent")])
def test_decision_point_trains_and_penalises_blocked(world, trainer, cls) -> None:
    c = _ctrl(world)
    agent = mf.build_agent(mf.DemoConfig(trainer=trainer), c, 0)
    assert type(agent).__name__ == cls and agent.obs_dim == c.obs_dim
    memory: mf.Memory = {}
    losses = []
    for step in range(60):
        _, info = mf.decision_point(agent, c, memory)
        losses.append(info["loss"])
        for _ in range(5):
            world[0].next_step()
            c.tick()
    assert agent.num_local_samples == 59            # transition đầu chưa có (s, a)
    assert any(l is not None for l in losses)
    assert not any("local_policy_head" in k for k in agent.extract_weights())


def test_blocked_action_gets_override_penalty(world) -> None:
    c = _ctrl(world)

    class Spy:
        intersection_id = c.intersection_id
        num_local_samples = 0
        rewards: list[float] = []

        def store_transition(self, s, a, r, ns, done):
            self.rewards.append(r)

        def train_local_model(self):
            return None

        def select_action(self, obs):
            return 1                                 # luôn đòi đổi pha → bị chặn lúc đầu

    spy, memory = Spy(), {}
    mf.decision_point(spy, c, memory)               # blocked (min_green chưa đủ)
    assert memory[c.intersection_id][2] is True
    base = c.get_reward() * 0                        # reset trạng thái reward
    mf.decision_point(spy, c, memory)
    assert spy.rewards and spy.rewards[0] <= -mf.REWARD_OVERRIDE_PENALTY + 1e-9 + base


@pytest.mark.parametrize("kwargs,fl", [
    ({"trainer": "dqn"}, True),
    ({"trainer": "dqn", "fl": False}, False),
    ({"trainer": "dqn", "fedprox_mu": 0.01}, True),
    ({"trainer": "local"}, True),
])
def test_run_simulation_end_to_end(world, tmp_path, monkeypatch, kwargs, fl) -> None:
    eng, inters, roads = world
    monkeypatch.setattr(mf, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(mf, "SCENARIO_DIR", tmp_path / "sc")
    cfg = mf.DemoConfig(total_steps=80, decision_interval=5, fl_interval=40, **kwargs)
    roadnet = {"roads": list(roads.values()), "intersections": inters}
    config = {"roadnetLogFile": "render/rn.json", "replayLogFile": "render/rp.txt"}
    mf.run_simulation(eng, cfg, config, roadnet)

    log = json.loads((tmp_path / "sc" / mf.FL_LOG_FILE).read_text(encoding="utf-8"))
    assert len(log["rounds"]) == (2 if fl else 0)
    assert log["kpi_summary"]["proposals"] > 0
    if fl:
        assert all(c["head_unchanged"] for r in log["rounds"] for c in r["clients"])
        assert log["rounds"][0]["divergence_after"] == 0.0
        assert log["rounds"][0]["algorithm"] == ("fedprox" if "fedprox_mu" in kwargs else "fedavg")
    rows = list(csv.DictReader((tmp_path / "sc" / mf.KPI_CSV_FILE).open(encoding="utf-8")))
    assert len(rows) == 16 and {"step", "waiting", "divergence"} <= set(rows[0])
