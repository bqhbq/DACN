# Báo cáo tiến độ phần MARL — khung RL cho FedMARL-ATSC

> **Ngày**: 2026-11-15 · **Nhánh**: `marl/rl-frame` · **Phạm vi**: phần MARL (nhóm 3 người: CityFlow / FL / MARL)
> **Trạng thái**: khung RL chạy được và có test trên môi trường giả lập; bản port CityFlow **chưa chạy thật** (Docker).

---

## 1. Mục tiêu giai đoạn

Dựng một **khung** (không cần hoàn hảo) cho phần RL: các agent DQN huấn luyện được, có đúng các hook ở ranh giới FL,
một bộ thí nghiệm so sánh có thể lặp lại, và demo CityFlow dùng cùng mã agent đó.

## 2. Hiện trạng ban đầu (sau lần pull `5e46d4b`)

- `agents/` đã được tái cấu trúc thành `NodeAgent` trung lập thuật toán, với `QLearningAgent`, `DQNAgent`,
  `BehaviorCloningAgent` là lớp con; giao diện FL là `extract_weights()` / `load_global_weights()`, `n_k = num_local_samples`.
- `main_fedmarl.py` (demo CityFlow) **không chạy được** với API mới: gọi `NodeAgent(epsilon=..., llm_provider=...)`,
  `train_local_step`, `expert_action_to_phase` — các hàm này không còn.
- Phía MARL của ranh giới FL còn thiếu: target network không đồng bộ lại sau FL, FedProx chưa có số hạng proximal ở
  client, seed chưa tái lập, observation thiếu pha hiện tại và thời gian xanh đã trôi (MDP_SPEC §2.1).
- Chưa có `env/`, `tests/`, baseline, `docs/RL_INTEGRATION_GUIDE.md` (code đã tham chiếu tới nó).

## 3. Những gì đã làm

### 3.1 Hook FL trong `NodeAgent` (trung lập thuật toán)
| Thay đổi | File |
|---|---|
| FedProx phía client: `set_fedprox_mu(mu)`; `train_local_model` cộng `μ/2·‖w_backbone − w_global‖²` cho mọi thuật toán | `agents/node_agent.py` |
| `load_global_weights` lưu neo `w_global` và gọi hook `on_global_weights_loaded()` (không ghi đè hàm FL) | `agents/node_agent.py` |
| `DQNAgent` ghi đè hook để chép backbone mới sang target net | `agents/dqn.py` |
| Tuỳ chọn `reset_optimizer_on_sync`; RNG riêng mỗi agent (`seed`); replay buffer nhận seed | `agents/node_agent.py`, `dqn.py`, `q_learning.py` |

### 3.2 Môi trường, an toàn, baseline
- `env/obs_layout.py` — **hợp đồng bố cục state duy nhất**: `[one-hot pha × P | elapsed | queue × L | speed × L | density × L]`.
  Mock env, baseline greedy, `SignalController` (CityFlow) và `behavior_cloning.ObservationLayout` cùng dùng, có test đối chiếu.
- `env/safety.py` — `PhaseSafetyController`: C-01…C-05 (xanh tối thiểu, vàng, all-red, pha không hợp lệ, thứ tự chu trình tuỳ chọn);
  đếm lý do ghi đè để tính `safety_violation_rate`.
- `env/mock_env.py` — `MockTrafficEnv`: mô phỏng hàng đợi nhiều giao lộ, 4 hồ sơ lưu lượng khác nhau (heterogeneity),
  bước quyết định cấu hình được (`delta_t_s`). Không import torch/cityflow.
- `agents/baselines/` — `fixed_time.py` và `actuated.py` (greedy longest-queue).

### 3.3 Bộ thí nghiệm
- `train_fedmarl.py` — so sánh `fixed_time`, `greedy_queue`, `independent`, `fedavg`, `fedprox` trên lưu lượng
  giống nhau (`--hetero 0`) hoặc khác nhau (`--hetero 1`). Vòng 0 phát **một** backbone w⁰ chung; ghi KPI, độ lệch backbone,
  đếm vi phạm theo lý do; chạy lại với cùng seed cho kết quả giống hệt.
- `aggregate_results.py` — gộp nhiều seed thành bảng mean ± std (`results/multiseed_summary.csv`).

### 3.4 Port demo CityFlow (`main_fedmarl.py`)
- `build_agent()` chọn lớp con theo `--trainer` (`local` → `QLearningAgent`, `dqn` → `DQNAgent`, `llm` → `BehaviorCloningAgent`).
- `decision_point()` dùng chung cho mọi thuật toán: `get_state` → `get_reward` → `store_transition` + `train_local_model` → `select_action` → `apply_action`.
  Hành động bị lớp an toàn chặn vẫn được lưu (hành động đề xuất) kèm phạt `REWARD_OVERRIDE_PENALTY`.
- State có khối [pha | elapsed]; `SignalController` đếm số đề xuất và số lần bị chặn theo lý do.
- Thêm `--no-fl`, `--fedprox-mu`; ghi `render/kpi.csv` và `kpi_summary` trong `fl_log.json`; engine luôn được giải phóng trong `finally`.

### 3.5 Kiểm thử và tài liệu
- 27 test (`tests/`): kiến trúc Decoupled (không rò rỉ `local_policy_head`), FedAvg, các luật an toàn, DQN + biên giới FL
  (n_k, target resync, FedProx kéo về neo, seed), bố cục state, và **bản port CityFlow chạy trên engine giả** (4 cấu hình end-to-end).
- `docs/RL_INTEGRATION_GUIDE.md` (§2 viết thuật toán mới, §3 thay mạng, biên giới FL, seed, bố cục state, khác biệt với PPO).
- Cập nhật `CLAUDE.md`, `README.md`; `.gitignore` thêm `!/env/` (luật `ENV/` đang ignore nhầm thư mục `env/` trên Windows).

## 4. Tinh chỉnh DQN

| Thay đổi | Tác động (hàng đợi TB, lưu lượng khác nhau, 1 seed) |
|---|---|
| Bước quyết định 5 s → **10 s** | 190 → ~140 (đòn bẩy lớn nhất: ở 5 s, đổi pha tốn 4 s vàng + all-red) |
| 100 episode, cập nhật mỗi bước, target sync 50, ε decay 0.4 | ~140 → 125–137 |
| FedProx μ 0.01 → **0.001** | 334 → ~118 (μ=0.01 quá mạnh) |

## 5. Kết quả cuối (hàng đợi TB ± std, 3 seed 0–2, thấp hơn là tốt hơn)

| Chế độ | Lưu lượng khác nhau | Lưu lượng giống nhau |
|---|---|---|
| fixed_time | 196 | 350 |
| greedy_queue | **114** | 221 |
| independent DQN | 120 ± 3 | 239 ± 9 |
| FedAvg | 133 ± 13 | 219 ± 13 |
| FedProx (μ=0.001) | **118 ± 4** | **215 ± 5** |

- DQN giờ **ngang baseline greedy** (trước tinh chỉnh: 190 so với 118).
- **Lưu lượng giống nhau:** FL có lợi — FedAvg/FedProx tốt hơn independent khoảng 8–10%.
- **Lưu lượng khác nhau:** FedAvg kém independent (133 vs 120), đúng chiều của *negative transfer*; FedProx loại bỏ phần lớn thiệt hại (118).
- **Lưu ý:** chỉ 3 seed; chênh lệch FedAvg ở lưu lượng khác nhau xấp xỉ độ lệch chuẩn của nó. Kết quả nhất quán nhưng
  chưa kết luận chắc chắn. Baseline không có std vì không huấn luyện và dùng seed đánh giá cố định.
- Dữ liệu: `results/a_*`, `results/s1_*`, `results/s2_*`, `results/multiseed_summary.csv`.

## 6. Chưa xong / hạn chế

- **Bản port CityFlow chưa từng chạy thật** (Docker daemon không chạy; ổ C: của máy phát triển đầy 0 GB nên build image lỗi).
  Lần chạy thật đầu tiên: `docker compose exec cityflow python main_fedmarl.py --trainer dqn --steps 200 --fl-interval 100`.
- Demo CityFlow chưa có chế độ `--eval` (tham lam, không học).
- ~12% đề xuất vẫn bị chặn (sau đổi pha chỉ còn ~6 s xanh so với xanh tối thiểu 10 s).
- `main_demo_fedmarl.py` là bản sao y hệt của file demo cũ — cần xác nhận với người phụ trách trước khi xoá.
- Chưa làm: neighbor embedding (MDP_SPEC §2.2), PPO, baseline Centralized-MARL, action masking.
- `metadata()` của agent chưa có thống kê lưu lượng để phân cụm.

## 7. Việc cần các thành viên còn lại

- **Người phụ trách FL:** sửa phân cụm (hiện chỉ tổng hợp cụm đông nhất nhưng `broadcast()` gửi cho mọi client); xác nhận FedProx
  nằm ở client (đã có hook) nên `server.fedprox_mu` không cần dùng.
- **Người phụ trách CityFlow:** thống nhất bố cục state (`env/obs_layout.py`); cung cấp quan hệ kề giữa các giao lộ cho neighbor embedding;
  topology/lưu lượng không đồng nhất (generator đã hỗ trợ số làn/khoảng cách, `main_fedmarl.py` mới chỉ dùng `rows/cols`).

## 8. Cách chạy lại

```bash
python -m pytest tests -q                                   # 27 test, không cần CityFlow
python train_fedmarl.py --hetero 1 --episodes 100 --tag het  # độc lập / FedAvg / FedProx + 2 baseline
python train_fedmarl.py --hetero 0 --episodes 100 --tag hom
python aggregate_results.py                                  # gộp nhiều seed

# CityFlow thật (trong container, xem cityflow/docker-compose.yml)
docker compose exec cityflow python main_fedmarl.py --trainer dqn --steps 200 --fl-interval 100
docker compose exec cityflow python main_fedmarl.py --trainer dqn --no-fl           # đối chứng
docker compose exec cityflow python main_fedmarl.py --trainer dqn --fedprox-mu 0.001
```
