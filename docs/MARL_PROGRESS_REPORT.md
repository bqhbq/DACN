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

## 8. Hướng dẫn chạy cho thành viên khác (đặc biệt người phụ trách FL)

> **Lưu ý quan trọng:** demo CityFlow trên nhánh này **chưa từng chạy với CityFlow thật** (người viết không build được Docker).
> Bản port chỉ mới được kiểm chứng với engine giả. Người chạy đầu tiên vì vậy là người thử nghiệm đầu tiên — nếu có lỗi,
> hãy gửi lại toàn bộ thông báo lỗi để sửa tiếp trên cùng nhánh.
> Bản port chỉ dùng lại đúng các lời gọi `engine.*` của demo cũ và không đổi phần sinh roadnet/flow, nên rủi ro được đánh giá là thấp.

### 8.1 Lấy nhánh về

```bash
git fetch origin
git checkout marl/rl-frame
```

### 8.2 Phần chạy không cần Docker / CityFlow (xác nhận nhanh môi trường và agent)

Cần Python ≥ 3.10, `numpy`, `torch` (bản CPU) và `pytest`. **`torch` và `pytest` không có trong `requirements.txt`**
(Dockerfile cài `torch` riêng; `pytest` chưa có ở đâu) nên cần cài tay:

```bash
pip install numpy pandas
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest

python -m pytest tests -q                                      # kỳ vọng: 27 passed
python train_fedmarl.py --hetero 1 --episodes 100 --tag het    # independent / FedAvg / FedProx + 2 baseline (vài phút)
python train_fedmarl.py --hetero 0 --episodes 100 --tag hom    # đối chứng: giao lộ giống nhau
python aggregate_results.py                                    # gộp nhiều seed -> results/multiseed_summary.csv
```

Để thử `FederatedServer` mới của mình, chỉ cần thay `federated/server.py` rồi chạy lại hai lệnh `train_fedmarl.py` ở trên và
so sánh `results/*_summary.csv`. Agent chỉ dùng ba hàm của server: `extract_weights()`, `load_global_weights()`, `metadata()`
(chỉ khi bật phân cụm).

### 8.3 Demo CityFlow thật (trong Docker, Python 3.10)

```bash
cd cityflow
docker compose up -d --build        # lần đầu mất vài phút (build CityFlow từ source)

# chạy thử nhanh (vài giây)
docker compose exec cityflow python main_fedmarl.py --trainer dqn --steps 200 --fl-interval 100
```

**Kết quả đúng sẽ có:**
- Các khối `[FL Round 1]`, `[FL Round 2]` với ba chặng UPLINK → AGGREGATE → DOWNLINK.
- Dòng `local_policy_head giữ nguyên ✓` ở chặng DOWNLINK.
- Độ lệch backbone `... → 0.000000` sau mỗi vòng.
- Cuối buổi chạy: `Average travel time`, `Hàng đợi TB`, tỷ lệ `vi phạm/bị chặn`, và đường dẫn các file replay.

**Các biến thể:**

```bash
# đầy đủ hơn
docker compose exec cityflow python main_fedmarl.py --trainer dqn --steps 1800 --fl-interval 100

# đối chứng không FL
docker compose exec cityflow python main_fedmarl.py --trainer dqn --no-fl --steps 1800

# FedProx phía client
docker compose exec cityflow python main_fedmarl.py --trainer dqn --fedprox-mu 0.001 --steps 1800

# lưới lớn hơn (2x2 giao lộ)
docker compose exec cityflow python main_fedmarl.py --trainer dqn --rows 2 --cols 2 --steps 600

# Q-learning một bước (agent đơn giản nhất); '--trainer llm' cần API key trong .env, KHÔNG dùng để thử lần đầu
docker compose exec cityflow python main_fedmarl.py --trainer local --steps 200
```

Dừng container khi xong: `docker compose down`.

### 8.4 Xem kết quả

- KPI: `cityflow/sample/demo_fl/render/kpi.csv` (hàng đợi, số xe, reward TB, độ lệch backbone theo từng lần quyết định) và
  khối `kpi_summary` trong `render/fl_log.json`.
- Replay trên trình duyệt (chạy trên **host**, không phải trong container):

```bash
cd cityflow/frontend
python -m http.server 8000      # mở http://localhost:8000
```

Nạp `replay_roadnet.json` và `replay.txt` từ `cityflow/sample/demo_fl/render/`; tuỳ chọn thêm `fl_chart.txt` và `fl_log.json`.
Mỗi lần chạy **ghi đè** thư mục `cityflow/sample/demo_fl/`, nên hãy sao lưu `render/` nếu muốn giữ kết quả cũ.

### 8.5 Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Read-only file system`, `Input/output error` khi build | Ổ chứa ổ đĩa ảo của Docker (mặc định ổ C:) đã đầy. Giải phóng ≥ 20 GB hoặc chuyển *Disk image location* của Docker Desktop sang ổ khác; rồi `docker builder prune -a -f` và `docker compose build --no-cache`. |
| `dpkg-preconfigure: ... not found` khi build | Cache build đã hỏng do ổ đầy: `docker builder prune -a -f`, sau đó build lại với `--no-cache`. |
| Container không thấy `main_fedmarl.py` / `/workspace` rỗng | Đường dẫn repo có ký tự không phải ASCII (ví dụ "Đ"): sao chép repo sang đường dẫn không dấu rồi chạy lại. |
| `No module named pytest` / `torch` | Cài tay như mục 8.2 (hai gói này chưa nằm trong `requirements.txt`). |
| `TypeError` / `AttributeError` khi tạo agent trong demo | Đang chạy nhầm `main_demo_fedmarl.py` (bản cũ, không còn khớp API `agents/`). Dùng `main_fedmarl.py`. |
| Lỗi khác trong demo CityFlow | Gửi lại toàn bộ thông báo lỗi (kèm vài dòng log phía trên nó). |
