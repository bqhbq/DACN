"""Kiểm thử DQNAgent + biên giới FL (target resync, FedProx, n_k, seed, layout)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.behavior_cloning import ObservationLayout
from agents.dqn import DQNAgent
from agents.networks import SHARED_MODULE_NAME
from env.mock_env import MockTrafficEnv
from federated.server import FederatedServer

OBS, PH = 89, 4


def _agent(seed: int = 0, **kw) -> DQNAgent:
    torch.manual_seed(seed)
    return DQNAgent(f"a{seed}", OBS, PH, batch_size=8, seed=seed, **kw)


def _fill(a: DQNAgent, n: int = 16) -> None:
    rng = np.random.default_rng(0)
    for _ in range(n):
        a.store_transition(rng.random(OBS), int(rng.integers(PH)), -1.0, rng.random(OBS), False)


def test_n_k_counts_stored_transitions_and_resets_on_sync() -> None:
    a = _agent()
    _fill(a, 5)
    assert a.num_local_samples == 5
    a.load_global_weights(a.extract_weights())
    assert a.num_local_samples == 0


def test_train_returns_none_until_batch_ready() -> None:
    a = _agent()
    _fill(a, 4)
    assert a.train_local_model() is None
    _fill(a, 8)
    assert isinstance(a.train_local_model(), float)


def test_target_backbone_resynced_after_global_weights() -> None:
    a, b = _agent(0), _agent(1)
    a.load_global_weights(b.extract_weights())
    online = getattr(a.net, SHARED_MODULE_NAME).state_dict()
    target = getattr(a.target_net, SHARED_MODULE_NAME).state_dict()
    for k, v in online.items():
        assert torch.equal(v, target[k])


def test_fedprox_pulls_backbone_toward_anchor() -> None:
    a = _agent()
    a.load_global_weights(a.extract_weights())
    assert a.proximal_term() is None                       # μ = 0
    a.set_fedprox_mu(1.0)
    assert a.proximal_term().item() == pytest.approx(0.0)  # chưa lệch
    shared = getattr(a.net, SHARED_MODULE_NAME)
    with torch.no_grad():
        for p in shared.parameters():
            p.add_(0.1)
    term = a.proximal_term()
    assert term.item() > 0
    for p in shared.parameters():
        p.grad = None
    term.backward()
    p0 = next(shared.parameters())
    assert (p0.grad > 0).all()          # gradient hướng về neo khi w > anchor


def test_negative_mu_rejected() -> None:
    with pytest.raises(ValueError):
        _agent().set_fedprox_mu(-0.1)


def test_same_seed_same_actions() -> None:
    obs = np.random.default_rng(1).random(OBS).astype(np.float32)
    a1, a2 = _agent(3), _agent(3)
    assert [a1.select_action(obs) for _ in range(30)] == [a2.select_action(obs) for _ in range(30)]


def test_head_not_in_shared_weights_after_dqn_training() -> None:
    a = _agent()
    _fill(a, 16)
    a.train_local_model()
    assert not any("local_policy_head" in k for k in a.extract_weights())


def test_fedavg_round_with_dqn_agents_syncs_backbone() -> None:
    agents = [_agent(i) for i in range(3)]
    for a in agents:
        _fill(a, 3)
    FederatedServer(agents).run_round(sample_counts=[a.num_local_samples for a in agents])
    ref = agents[0].extract_weights()
    for a in agents[1:]:
        for k, v in a.extract_weights().items():
            assert torch.equal(v, ref[k])


def test_mock_obs_matches_behavior_cloning_layout() -> None:
    env = MockTrafficEnv(1, episode_s=60)
    obs, _ = env.reset(seed=0)
    o = obs[env.ids[0]]
    lay = ObservationLayout(num_phases=PH, num_lanes=28, queue_scale=1.0)
    assert o.shape == (env.obs_dim,) == (OBS,)
    assert lay.current_phase(o) == 0 and lay.elapsed_time(o) == 0.0
    assert lay.queues(o) == [0] * 28
    env.step({env.ids[0]: 0})
    o2 = env.get_state()[env.ids[0]]
    assert sum(lay.queues(o2 * 1.0)) >= 0 and env.layout.queue_slice.start == PH + 1
