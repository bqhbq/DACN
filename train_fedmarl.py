"""
train_fedmarl.py
----------------
Thí nghiệm so sánh trên MockTrafficEnv (không cần CityFlow / Docker):

    fixed_time   : baseline chu kỳ cố định
    greedy_queue : baseline thích ứng (hàng đợi dài nhất)
    independent  : DQN độc lập, không chia sẻ gì
    fedavg       : DQN + FedAvg trên backbone (head giữ riêng)
    fedprox      : như fedavg, thêm số hạng proximal μ/2‖w−w_global‖² ở client

``--hetero 0`` cho mọi giao lộ cùng một hồ sơ lưu lượng; ``--hetero 1`` dùng
các hồ sơ khác nhau (nơi negative transfer có thể xuất hiện).

Ví dụ ablation:
    python train_fedmarl.py --hetero 1 --episodes 40 --tag het
    python train_fedmarl.py --hetero 0 --episodes 40 --tag hom

Ghi ``results/<tag>_{train,eval,summary}.csv``.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from agents.baselines.actuated import GreedyQueuePolicy
from agents.baselines.fixed_time import FixedTimePolicy
from agents.dqn import DQNAgent
from env.mock_env import DEFAULT_PROFILES, DELTA_T_S, MockTrafficEnv
from federated.server import FederatedServer

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
ALL_MODES = ("fixed_time", "greedy_queue", "independent", "fedavg", "fedprox")
FL_MODES = ("fedavg", "fedprox")
LEARNED_MODES = ("independent",) + FL_MODES
OVERRIDE_REASONS = ("min_green", "invalid", "cycle_order", "in_transition")
EVAL_SEED_BASE = 1_000_000
CONVERGENCE_WINDOW = 3          # số episode làm trơn khi tính episodes_to_target

Policy = Callable[[str, np.ndarray], int]


@dataclass
class TrainConfig:
    """Hyperparameter của một lần chạy."""

    episodes: int = 100
    fl_every: int = 2               # chạy 1 vòng FL sau mỗi N episode
    eval_episodes: int = 3
    num_intersections: int = 4
    episode_s: int = 1800
    seed: int = 0
    hetero: int = 1                 # 1 = hồ sơ lưu lượng khác nhau, 0 = giống nhau
    include_phase: int = 1          # 1 = state có khối [one-hot pha | elapsed]
    decision_s: int = 10            # giây mô phỏng mỗi bước quyết định (10 s >> 5 s: xem sweep)
    fixed_green_s: int = 20         # xanh mỗi pha của baseline fixed_time (giây)
    fedprox_mu: float = 0.001       # μ cho mode fedprox (0.01 quá mạnh ở sweep đầu)
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    target_sync_steps: int = 50
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4     # tỉ lệ tổng số bước để ε giảm hết
    reward_scale: float = 0.1       # chuẩn hoá reward cho DQN
    train_every: int = 1            # số bước env giữa hai lần update


def make_env(cfg: TrainConfig) -> MockTrafficEnv:
    """Dựng môi trường theo cấu hình (đồng nhất hay không đồng nhất)."""
    profiles = DEFAULT_PROFILES if cfg.hetero else (DEFAULT_PROFILES[0],)
    return MockTrafficEnv(
        cfg.num_intersections, profiles=profiles, episode_s=cfg.episode_s,
        seed=cfg.seed, include_phase=bool(cfg.include_phase),
        delta_t_s=cfg.decision_s,
    )


def build_learners(env: MockTrafficEnv, cfg: TrainConfig) -> dict[str, DQNAgent]:
    """Tạo một DQNAgent (NodeAgent con) cho từng giao lộ."""
    steps_total = cfg.episodes * (cfg.episode_s // cfg.decision_s)
    agents: dict[str, DQNAgent] = {}
    for i, iid in enumerate(env.ids):
        torch.manual_seed(cfg.seed + i)             # khởi tạo trọng số
        agents[iid] = DQNAgent(
            iid, env.obs_dim, env.num_phases,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            reward_scale=cfg.reward_scale,
            target_sync_steps=cfg.target_sync_steps,
            eps_start=cfg.eps_start,
            eps_end=cfg.eps_end,
            eps_decay_steps=int(steps_total * cfg.eps_decay_frac),
            seed=cfg.seed * 1000 + i,               # ε-greedy + replay
        )
    return agents


def backbone_divergence(agents: list[DQNAgent]) -> float:
    """Độ lệch backbone giữa các agent: max |Δw| trên mọi trọng số."""
    packets = [a.extract_weights() for a in agents]
    worst = 0.0
    for key in packets[0]:
        stacked = torch.stack([p[key].float() for p in packets])
        worst = max(worst, float((stacked.max(0).values - stacked.min(0).values).max()))
    return worst


def run_episode(
    env: MockTrafficEnv,
    act: Policy,
    seed: int,
    agents: Optional[dict[str, DQNAgent]] = None,
    cfg: Optional[TrainConfig] = None,
) -> dict[str, float]:
    """Chạy một episode; nếu ``agents`` được cho thì huấn luyện online.

    Args:
        env: Môi trường.
        act: Hàm ``(iid, obs) -> action``.
        seed: Seed episode (cố định để so sánh công bằng).
        agents: DQNAgent theo giao lộ (None = chỉ đánh giá).
        cfg: Cấu hình (bắt buộc khi huấn luyện).

    Returns:
        KPI trung bình của episode.
    """
    obs, _ = env.reset(seed=seed)
    n = len(env.ids)
    done, step = False, 0
    ret = thr = overrides = switches = 0.0
    reasons = {r: 0 for r in OVERRIDE_REASONS}
    queue_acc = np.zeros(n)
    while not done:
        actions = {iid: act(iid, obs[iid]) for iid in env.ids}
        nxt, rew, _, done, info = env.step(actions)
        for k, iid in enumerate(env.ids):
            ret += rew[iid]
            thr += info[iid]["throughput"]
            overrides += info[iid]["overridden"]
            switches += info[iid]["switched"]
            if info[iid]["override_reason"] in reasons:
                reasons[info[iid]["override_reason"]] += 1
            queue_acc[k] += info[iid]["queue_sum"]
            if agents is not None:
                assert cfg is not None
                # truncated ≠ terminal → bootstrap (done=False)
                agents[iid].store_transition(obs[iid], actions[iid], rew[iid], nxt[iid], False)
                if step % cfg.train_every == 0:
                    agents[iid].train_local_model()
        obs, step = nxt, step + 1
    mean_q = queue_acc / step
    total = step * n
    out = {
        "return": ret / n,
        "mean_queue": float(mean_q.mean()),
        "worst_queue": float(mean_q.max()),
        "throughput": thr,
        "violation_rate": overrides / total,
        "switch_rate": switches / total,
    }
    out.update({f"viol_{r}": reasons[r] / total for r in OVERRIDE_REASONS})
    return out


def run_mode(mode: str, cfg: TrainConfig) -> tuple[list[dict], list[dict]]:
    """Huấn luyện (nếu cần) và đánh giá một chế độ.

    Returns:
        (log train theo episode, log eval theo episode).
    """
    env = make_env(cfg)
    train_log: list[dict] = []
    agents: Optional[dict[str, DQNAgent]] = None
    server: Optional[FederatedServer] = None
    pols: dict[str, object] = {}

    if mode in LEARNED_MODES:
        agents = build_learners(env, cfg)
        if mode in FL_MODES:
            server = FederatedServer(list(agents.values()))
            if mode == "fedprox":
                for a in agents.values():
                    a.set_fedprox_mu(cfg.fedprox_mu)
            # Vòng 0: phát CHUNG một backbone w⁰ (không lấy trung bình các init khác nhau)
            server.aggregate([next(iter(agents.values())).extract_weights()])
            server.broadcast()
        t0 = time.time()
        for ep in range(cfg.episodes):
            div_before = backbone_divergence(list(agents.values())) if server else 0.0
            m = run_episode(
                env, lambda i, o: agents[i].select_action(o),
                seed=cfg.seed * 10_000 + ep, agents=agents, cfg=cfg,
            )
            m["divergence"] = backbone_divergence(list(agents.values())) if server else 0.0
            if server is not None and (ep + 1) % cfg.fl_every == 0:
                server.run_round(
                    algorithm=mode,
                    sample_counts=[a.num_local_samples for a in agents.values()],
                )
            m.update(mode=mode, episode=ep, divergence_start=div_before,
                     epsilon=next(iter(agents.values())).epsilon, wall_s=time.time() - t0)
            train_log.append(m)
            print(f"[{mode:12s}] ep {ep:3d}  q={m['mean_queue']:6.1f}  thr={m['throughput']:6.0f}  "
                  f"ret={m['return']:8.1f}  viol={m['violation_rate']:.2f}  eps={m['epsilon']:.2f}")
        policy: Policy = lambda i, o: agents[i].select_action(o, greedy=True)  # noqa: E731
    else:
        phase = bool(cfg.include_phase)
        pols = {iid: (FixedTimePolicy(env.num_phases, max(cfg.fixed_green_s // cfg.decision_s, 1))
                      if mode == "fixed_time"
                      else GreedyQueuePolicy(env.num_phases, include_phase=phase))
                for iid in env.ids}
        policy = lambda i, o: pols[i].select_action(o)  # noqa: E731

    eval_log: list[dict] = []
    for k in range(cfg.eval_episodes):
        for p in pols.values():
            p.reset()
        m = run_episode(env, policy, seed=EVAL_SEED_BASE + k)       # seed eval cố định
        m.update(mode=mode, episode=k)
        eval_log.append(m)
    return train_log, eval_log


def episodes_to_target(train_rows: list[dict], target_queue: float) -> Optional[int]:
    """Episode đầu tiên mà mean_queue (làm trơn) ≤ target; None nếu chưa đạt."""
    q = [r["mean_queue"] for r in train_rows]
    for i in range(len(q)):
        window = q[max(0, i - CONVERGENCE_WINDOW + 1): i + 1]
        if len(window) == CONVERGENCE_WINDOW and float(np.mean(window)) <= target_queue:
            return i
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    """Ghi list dict ra CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(modes: list[str], train: list[dict], evals: list[dict]) -> list[dict]:
    """Bảng tóm tắt theo mode (trung bình eval + episodes_to_target)."""
    target = next(
        (float(np.mean([r["mean_queue"] for r in evals if r["mode"] == "fixed_time"]))
         for m in modes if m == "fixed_time"), None)
    rows = []
    for mode in modes:
        ev = [r for r in evals if r["mode"] == mode]
        row = {"mode": mode}
        for key in ("mean_queue", "worst_queue", "throughput", "violation_rate", "switch_rate"):
            row[key] = float(np.mean([r[key] for r in ev]))
        tr = [r for r in train if r["mode"] == mode]
        row["episodes_to_fixed_time"] = (
            episodes_to_target(tr, target) if (tr and target is not None) else None
        )
        rows.append(row)
    return rows


def main() -> None:
    """Điểm vào CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=ALL_MODES)
    p.add_argument("--tag", default="run")
    for k, v in asdict(TrainConfig()).items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    logging.basicConfig(level=logging.WARNING)

    all_train: list[dict] = []
    all_eval: list[dict] = []
    for mode in args.modes:
        tr, ev = run_mode(mode, cfg)
        all_train += tr
        all_eval += ev
    summary = summarize(list(args.modes), all_train, all_eval)
    write_csv(RESULTS_DIR / f"{args.tag}_train.csv", all_train)
    write_csv(RESULTS_DIR / f"{args.tag}_eval.csv", all_eval)
    write_csv(RESULTS_DIR / f"{args.tag}_summary.csv", summary)

    print(f"\n=== EVAL (hetero={cfg.hetero}, fixed eval seeds) ===")
    print(f"{'mode':14s}{'mean_queue':>12s}{'worst_queue':>13s}{'throughput':>12s}"
          f"{'violation':>11s}{'ep_to_fixed':>12s}")
    for r in summary:
        ep = "-" if r["episodes_to_fixed_time"] is None else str(r["episodes_to_fixed_time"])
        print(f"{r['mode']:14s}{r['mean_queue']:12.1f}{r['worst_queue']:13.1f}"
              f"{r['throughput']:12.0f}{r['violation_rate']:11.3f}{ep:>12s}")


if __name__ == "__main__":
    main()
