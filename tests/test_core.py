"""Kiểm thử kiến trúc Decoupled (R-02), lớp an toàn (C-01…C-05) và FL pipeline."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.node_agent import NodeAgent
from env.mock_env import MockTrafficEnv
from env.safety import PhaseSafetyController, SafetyConfig, SignalState
from federated.server import FederatedServer


def _agent(i: str = "a") -> NodeAgent:
    return NodeAgent(i, obs_dim=84, num_phases=4)


def test_shared_weights_exclude_policy_head() -> None:
    sw = _agent().get_shared_weights()
    assert not any("local_policy_head" in k for k in sw), "R-02 VI PHẠM"
    assert any(k for k in sw), "Backbone rỗng"


def test_set_shared_weights_leaves_head_untouched() -> None:
    a, b = _agent("a"), _agent("b")
    head_before = {k: v.clone() for k, v in a.net.local_policy_head.state_dict().items()}
    a.set_shared_weights(b.get_shared_weights())
    for k, v in a.net.local_policy_head.state_dict().items():
        assert torch.equal(v, head_before[k])
    for k, v in b.get_shared_weights().items():
        assert torch.equal(a.get_shared_weights()[k], v)


def test_fedavg_is_mean_and_broadcast_syncs_backbone() -> None:
    agents = [_agent(str(i)) for i in range(3)]
    expected = {
        k: torch.stack([a.get_shared_weights()[k] for a in agents]).mean(0)
        for k in agents[0].get_shared_weights()
    }
    FederatedServer(agents).run_round()
    for a in agents:
        for k, v in a.get_shared_weights().items():
            assert torch.allclose(v, expected[k], atol=1e-6)


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError):
        FederatedServer([]).aggregate([])


def test_min_green_enforced() -> None:
    c = PhaseSafetyController(SafetyConfig(min_green_s=10))
    for _ in range(5):
        c.tick()
    r = c.request(2)
    assert r.overridden and r.reason == "min_green" and c.phase == 0


def test_yellow_and_all_red_inserted_before_new_green() -> None:
    cfg = SafetyConfig(min_green_s=2, yellow_s=3, all_red_s=1)
    c = PhaseSafetyController(cfg)
    for _ in range(2):
        c.tick()
    assert c.request(1).switched
    seq = [c.tick() for _ in range(5)]
    assert seq[:4] == [None] * 4            # 3 s vàng + 1 s all-red, không xe đi
    assert c.state is SignalState.GREEN and c.phase == 1


def test_invalid_action_rejected() -> None:
    c = PhaseSafetyController(SafetyConfig())
    assert c.request(99).reason == "invalid" and c.request(-1).overridden


def test_strict_cycle_forces_next_phase() -> None:
    c = PhaseSafetyController(SafetyConfig(min_green_s=0, strict_cycle=True))
    r = c.request(3)
    assert r.overridden and r.reason == "cycle_order" and r.applied == 1


def test_env_obs_shape_range_and_determinism() -> None:
    def rollout() -> float:
        env = MockTrafficEnv(2, episode_s=100)
        obs, _ = env.reset(seed=7)
        total, done = 0.0, False
        while not done:
            obs, rew, _, done, _ = env.step({i: 0 for i in env.ids})
            total += sum(rew.values())
        for o in obs.values():
            assert o.shape == (89,) and o.min() >= 0.0 and o.max() <= 1.0
        return total

    assert rollout() == rollout()
