# CLAUDE.md — Project Memory & Rules

**FedMARL-ATSC** · Federated Multi-Agent RL for Adaptive Traffic Signal Control

File này tóm tắt các ràng buộc phải nhớ trong mọi phiên làm việc. Chi tiết nằm ở:

| Tài liệu | Nội dung | Khi nào đọc |
|---|---|---|
| [AGENTS.md](AGENTS.md) | Quy tắc kiến trúc, coding conventions, quy trình làm việc | Trước mọi task |
| [PROJECT_GUIDELINES.md](PROJECT_GUIDELINES.md) | Mục tiêu nghiên cứu, baseline, KPIs, interface 3 lớp | Trước mọi task viết code |
| [docs/MDP_SPEC.md](docs/MDP_SPEC.md) | Đặc tả State / Action / Reward, timing, công thức FedAvg | Khi đụng tới env, reward, vòng FL |

Nếu file này mâu thuẫn với các tài liệu trên về **môi trường chạy** (mục Tech Stack Constraints), ưu tiên file này.

---

## 1. Project Overview

- Hệ thống **điều khiển tín hiệu giao thông thích ứng (ATSC)** phân tán: mỗi giao lộ là một `NodeAgent` học tăng cường, phối hợp với nhau qua **Federated Learning** (cơ chế RL-FedMARL).
- Mô phỏng bằng **CityFlow** (engine C++ có Python binding).
- Bài toán nghiên cứu trọng tâm: **Negative Transfer** giữa các giao lộ không đồng nhất. Hướng giải quyết: personalisation (giữ policy head tại node), clustering và selective collaboration ở server.
- Điểm vào demo CityFlow: `python main_fedmarl.py` (xem docstring đầu file cho `--trainer`, `--steps`, `--fl-interval`, `--no-fl`, `--fedprox-mu`). Thí nghiệm nhanh không cần CityFlow: `python train_fedmarl.py` (xem `docs/RL_INTEGRATION_GUIDE.md`).

**Trạng thái code hiện tại** (kiểm tra lại trước khi dựa vào):

- Đã có: `agents/` (`node_agent.py`, `networks.py`, `q_learning.py`, `dqn.py`, `behavior_cloning.py`, `llm_expert.py`, `replay_buffer.py`, `baselines/`), `federated/server.py`, `main_fedmarl.py`, `env/` (`MockTrafficEnv`, `PhaseSafetyController`, `obs_layout.py`), `train_fedmarl.py`, `tests/`.
- Chưa có: lớp `TrafficEnv` bọc CityFlow. Trong demo CityFlow, ràng buộc an toàn (pha vàng, xanh tối thiểu) do `SignalController` trong `main_fedmarl.py` đảm nhiệm; `env/safety.py` là bản đầy đủ hơn (C-01…C-05) dùng cho `MockTrafficEnv`.
- Chưa có: cấu hình linter/formatter, CI. Bản đồ mã nguồn ở §6, lệnh chạy ở §7, các chỗ code còn lệch quy tắc ở §8.

---

## 2. Architectural Principles

### 2.1 Decoupled — ba lớp tách biệt trách nhiệm

| Lớp | Trách nhiệm | Cấm |
|---|---|---|
| `TrafficEnv` | Bao bọc CityFlow, dựng observation, tính reward, **tự thực thi ngầm** ràng buộc an toàn C-01…C-05 (thứ tự pha, xanh tối thiểu, vàng, all-red) | Import PyTorch, chứa `nn.Module`, optimizer hay loss |
| `NodeAgent` | Chứa mạng PyTorch và logic học cục bộ; `select_action()`, `get_shared_weights()`, `set_shared_weights()` | Gọi trực tiếp CityFlow; tự xử lý luật an toàn đèn |
| `FederatedServer` | `aggregate()` (FedAvg / FedProx), `broadcast()`, phân cụm client | Import `cityflow`, chứa logic RL, giữ replay buffer |

`TrafficEnv` và `NodeAgent` chỉ trao đổi qua observation (`np.ndarray`), action (`int`) và reward (`float`).

### 2.2 Mạng nơ-ron tách đôi

`TrafficAgentNet` có đúng hai sub-module với tên cố định:

- `shared_feature_extractor` — backbone, **tham gia** FL.
- `local_policy_head` — head, **giữ tại node**, không bao giờ rời khỏi node.

Không gộp hai phần thành một `nn.Sequential`.

### 2.3 Luồng FL chỉ tổng hợp trọng số

- Thứ duy nhất đi giữa node và server là **state_dict của `shared_feature_extractor`**.
- **Không** truyền state, action, reward, transition hay dữ liệu giao thông thô lên server.
- Tổng hợp theo FedAvg có trọng số số mẫu cục bộ `n_k` (MDP_SPEC §6.2).

Kiểm tra bắt buộc sau mọi thay đổi liên quan `NodeAgent` / `TrafficAgentNet`:

```python
sw = agent.get_shared_weights()
assert not any("local_policy_head" in k for k in sw), "Policy head bị leak!"
assert any("shared_feature_extractor" in k for k in sw), "Backbone rỗng!"
```

---

## 3. Tech Stack Constraints

- **Python 3.10** — bắt buộc đúng bản này: pybind11 đi kèm CityFlow không build được với Python ≥ 3.11.
- **PyTorch CPU-only** — cài từ `--index-url https://download.pytorch.org/whl/cpu`. Không cài wheel CUDA, không viết code giả định có GPU (`.cuda()`, `device="cuda"` cứng).
- **Chạy hoàn toàn trong Docker / Dev Container** — không cài CityFlow hay dependency lên máy host, không tạo `.venv` riêng.
  - Image: [cityflow/Dockerfile](cityflow/Dockerfile) (`python:3.10-slim-bookworm`, build CityFlow từ source, torch CPU).
  - Khởi động: `cd cityflow && docker compose up -d --build`, hoặc mở repo bằng Dev Container ([.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)).
  - Repo được mount vào `/workspace`; interpreter là `/usr/local/bin/python`.
- Dependency mới: thêm vào [requirements.txt](requirements.txt) rồi build lại image; không `pip install` tạm trong container rồi bỏ quên.
- Thư viện khác: Flower (`flwr`), `wandb`, NumPy, Pandas; `google-genai` / `openai` cho LLM expert.

---

## 4. Algorithm Strategy

Lộ trình hai giai đoạn:

1. **Behavior Cloning (LLM Thầy–Trò)** — giai đoạn hiện tại. LLM (Gemini / OpenAI) đóng vai "thầy" đưa ra action; `NodeAgent` là "trò" học bắt chước. Mục đích là **demo và kiểm chứng luồng FL** (uplink backbone → aggregate → downlink), chưa phải tối ưu chính sách. Chế độ `--trainer local` là phương án không cần API.
2. **Thuật toán RL thực thụ** — tích hợp sau khi luồng FL đã ổn định; thuật toán cụ thể chưa chốt (value-based hoặc policy-based).

**Nghiêm cấm hardcode logic chỉ dành riêng cho DQN** (hay bất kỳ thuật toán đơn lẻ nào) vào thành phần dùng chung:

- `NodeAgent`, `TrafficAgentNet`, `TrafficEnv`, `FederatedServer` và vòng lặp huấn luyện phải **trung lập với thuật toán**.
- Logic đặc thù (target network, ε-greedy, replay buffer, TD-target, clipping, GAE…) chỉ nằm trong module trainer của thuật toán đó: `agents/<rl_algo>.py`. `agents/dqn.py` là **một** cài đặt, không phải mặc định của hệ thống.
- Không giả định đầu ra của `local_policy_head` luôn là Q-values; coi nó là "action scores" (Q-values hoặc logits tuỳ trainer).
- Không giả định mọi agent đều có replay buffer; `n_k` cho FedAvg lấy từ số mẫu cục bộ, không đọc thẳng `len(buffer)`.
- Trong tài liệu và tên biến/hàm dùng chung, viết "RL" hoặc "cơ chế RL-FedMARL" thay vì "DQN".

---

## 5. Coding Conventions

### 5.1 Xử lý lỗi nghiêm ngặt (try-except)

Mọi lời gọi ra ngoài tiến trình Python thuần đều phải được bọc `try-except`.

**API bên ngoài (LLM, wandb, mạng):**

- Bọc toàn bộ lời gọi; lường trước timeout, rớt mạng, hết quota, sai API key, phản hồi không parse được.
- Luôn có **fallback xác định** (ví dụ: giữ pha hiện tại) để vòng mô phỏng không dừng giữa chừng vì một lần gọi lỗi.
- Retry có giới hạn và có giãn cách; không retry vô hạn.
- Không ghi API key hay nội dung `.env` vào log.

**CityFlow (engine C++ và các file của nó):**

- Bọc `cityflow.Engine(...)`, `next_step()`, `set_tl_phase()`, các hàm `get_*` và thao tác đọc/ghi `config.json`, `roadnet.json`, `flow.json`, replay log.
- **Kiểm tra trước khi gọi engine**: file tồn tại, JSON hợp lệ, ID giao lộ và chỉ số pha nằm trong roadnet. Lỗi phía C++ có thể làm sập cả tiến trình thay vì ném exception Python, nên `try-except` không thay được bước validate này.
- Giải phóng engine trong `finally` (hoặc `close()`), kể cả khi lỗi.

**Quy tắc chung:**

- Bắt exception cụ thể khi biết loại lỗi; chỉ dùng `except Exception` ở ranh giới với hệ thống ngoài. Không dùng `except:` trần, không `pass` im lặng.
- Ghi log bằng `logging` kèm ngữ cảnh (intersection ID, step, đường dẫn file), rồi fallback hoặc `raise ... from exc`.
- Lỗi cấu hình (thiếu file, sai schema) phải dừng sớm với thông báo rõ ràng, không âm thầm chạy tiếp.

### 5.2 Quy ước khác (chi tiết ở AGENTS.md §4–§5)

- Type hints đầy đủ cho mọi hàm; docstring Google style.
- Đường dẫn dùng `pathlib`, không hardcode chuỗi tuyệt đối.
- Hyperparameter đặt ở đầu file hoặc trong config (`dataclass` / `TypedDict`), không rải số vào giữa hàm.
- Thứ tự import: standard library → third-party → project modules.
- Kiểm tra pipeline bằng dummy data trước khi nối CityFlow thật; mỗi thay đổi đi kèm smoke test.

---

## 6. Codebase Map

```
main_fedmarl.py         Điểm vào demo CityFlow: sinh kịch bản → vòng mô phỏng → vòng FL → log + KPI (main_demo_fedmarl.py là bản sao y hệt)
train_fedmarl.py         Thí nghiệm so sánh trên MockTrafficEnv (independent / fedavg / fedprox / baselines) → results/*.csv
env/                     MockTrafficEnv, PhaseSafetyController (C-01…C-05), obs_layout.py (hợp đồng bố cục state) — không import torch
tests/                   pytest: kiến trúc Decoupled, an toàn, DQN+FL, bản port CityFlow (engine giả)
agents/
  networks.py            TrafficAgentNet (backbone + head; value head tuỳ chọn qua with_value)
  node_agent.py          NodeAgent trung lập thuật toán: giao diện RL + giao diện FL (extract/load_global_weights), FedProx, hook sau FL
  q_learning.py, dqn.py, behavior_cloning.py   Mỗi thuật toán là một lớp con của NodeAgent (llm_expert.py: client LLM)
  replay_buffer.py       Transition, TransitionBatch, ReplayBuffer
  baselines/             fixed_time.py, actuated.py (greedy longest-queue)
federated/server.py      FederatedServer: aggregate / broadcast / cluster_clients / run_round
cityflow/
  Dockerfile, docker-compose.yml   Môi trường chạy (service `cityflow`)
  src/generators/        generate_roadnet(), gen_config(), FlowGenerator — package `src`
  src/vehicles/          VehicleType, VehicleFleet (dùng khi sinh flow)
  tools/                 Script gốc của CityFlow (generator lưới, converter SUMO, debug C++)
  frontend/              Trang replay tĩnh (index.html + script.js), đã mở rộng để hiện FL log
  sample/                Dữ liệu sinh ra khi chạy — bị gitignore, không sửa tay
docs/                    MDP_SPEC.md, RL_INTEGRATION_GUIDE.md (viết thuật toán RL mới), FL_REPORT.md (phân tích một lần chạy demo)
notebooks/               Notebook minh hoạ FL và tích hợp CityFlow
```

### 6.1 Luồng chạy của `main_fedmarl.py`

1. `prepare_scenario()` gọi `generate_roadnet()` (chạy `cityflow/tools/generator/generate_grid_scenario.py` bằng subprocess) và `gen_config()`, ghi vào `cityflow/sample/demo_fl/`. Mọi file ở đây được **ghi đè mỗi lần chạy**; `config.json` chứa `dir` là đường dẫn tuyệt đối. `ensure_replay_enabled()` ép `saveReplay` và `rlTrafficLight` (thiếu cờ sau thì CityFlow bỏ qua `set_tl_phase()`).
2. Mỗi giao lộ không `virtual` có một `SignalController` (đọc state, tính reward, giữ luật pha) và một agent do `build_agent()` dựng theo `--trainer` (`local` → `QLearningAgent`, `dqn` → `DQNAgent`, `llm` → `BehaviorCloningAgent`). `obs_dim` và `num_phases` lấy từ controller, không hardcode.
3. Vòng FL 0 (bỏ qua nếu `--no-fl`): server phát backbone của agent đầu tiên cho mọi node để cùng xuất phát từ `w⁰`.
4. Mỗi `decision_interval` bước: `decision_point()` (dùng chung cho mọi thuật toán): `get_state` → `get_reward` → `store_transition` + `train_local_model` → `select_action` → `apply_action`. Sau mỗi `engine.next_step()` phải gọi `controller.tick()` đúng một lần.
5. Mỗi `fl_interval` bước: `run_fl_round()` → `FederatedServer.run_round()` với `n_k = agent.num_local_samples`; `set_shared_weights()` đặt `n_k` về 0.
6. Kết thúc: ghi `render/fl_log.json`, `render/fl_chart.txt`, `render/kpi.csv` cạnh replay của CityFlow; `run_demo()` luôn `del engine` trong `finally` để CityFlow ghi nốt replay.

### 6.2 Điều cần biết khi sửa

- **Observation**: `[one-hot pha × P, elapsed, queue × L, speed × L, density × L]` (L = số làn vào), chuẩn hoá về `[0, 1]`. Hợp đồng duy nhất là `env/obs_layout.py::ObsLayout`; `MockTrafficEnv`, `SignalController`, `actuated.py` và `behavior_cloning.ObservationLayout` phải khớp (có test đối chiếu). `elapsed` = giây xanh đã trôi / `min_green`, cắt ở 1; khi đang vàng thì one-hot toàn 0. `Q_MAX` khai báo trong `main_fedmarl.py` (prompt LLM khử chuẩn hoá queue bằng nó).
- **Action**: index trong `SignalController.green_phases`, không phải chỉ số pha của CityFlow. Pha "xanh" là pha có ít nhất một luồng không phải rẽ phải; pha chỉ-rẽ-phải đầu tiên được dùng làm pha vàng. `apply_action()` trả `keep` / `switch` / `blocked`.
- **Reward**: `−α·|Σq_in − Σq_out| + β·throughput − γ_phase·1[đổi pha]`, hằng số ở đầu `main_fedmarl.py`. `get_reward()` có trạng thái (tập xe lần trước), gọi đúng một lần mỗi decision point. Hành động bị lớp an toàn chặn vẫn được lưu (hành động đề xuất) kèm phạt `REWARD_OVERRIDE_PENALTY`.
- **Chế độ `llm`**: `BehaviorCloningAgent.select_action()` hỏi LLM KEEP(0)/CHANGE(1) rồi đổi sang pha đích. LLM lỗi hoặc chưa đủ xanh tối thiểu → giữ pha và transition đó **không** được học. Gemini xoay vòng qua `GEMINI_FREE_MODELS` với cooldown; trạng thái cooldown và throttle là biến lớp, dùng chung cho mọi `NodeAgent`. Key đọc từ `.env` ở gốc repo (`GEMINI_API_KEY`, `OPENAI_API_KEY`; tuỳ chọn `GEMINI_MODEL`, `OPENAI_MODEL`).
- **FedProx** chạy ở **client**: `agent.set_fedprox_mu(mu)` (hoặc `--fedprox-mu`); `NodeAgent.train_local_model` cộng `μ/2‖w_backbone − w_global‖²`. Server vẫn chỉ chạy FedAvg và `algorithm="fedprox"` chỉ là nhãn log. Sau mỗi `load_global_weights`, hook `on_global_weights_loaded()` chạy (DQN chép backbone sang target net).
- **`FederatedServer`**: `fedprox_mu` của server không được dùng. Khi `use_clustering=True`, chỉ cụm đông nhất được tổng hợp (trọng số đều) nhưng `broadcast()` vẫn gửi cho **mọi** client.
- `import agents` kéo theo `torch`; `main_fedmarl.py` tự chèn `cityflow/` vào `sys.path` để import `src.generators`.

---

## 7. Commands

Chạy trong container (thư mục làm việc `/workspace`). Từ host: thêm tiền tố `docker compose exec cityflow` khi đứng ở `cityflow/`.

```bash
python main_fedmarl.py                                  # demo mặc định: lưới 1×2, 600 bước, trainer local
python main_fedmarl.py --trainer dqn --steps 200 --fl-interval 100   # chạy nhanh để kiểm tra (vài giây)
python main_fedmarl.py --trainer dqn --no-fl            # đối chứng không FL
python main_fedmarl.py --trainer llm --llm-provider openai
python -m agents.networks                               # smoke test TrafficAgentNet, không cần CityFlow
python -m pytest tests -q                               # test không cần CityFlow (engine giả)
python train_fedmarl.py --hetero 1 --episodes 40 --tag het   # so sánh independent / fedavg / fedprox trên MockTrafficEnv
```

Tham số khác: `--rows`, `--cols`, `--decision-interval`, `--seed`, `--no-fl`, `--fedprox-mu`. `min_green`, `yellow_time`, `epsilon`, `lr` chỉ đổi được trong `DemoConfig`.

Smoke test `NodeAgent` + `FederatedServer` không cần CityFlow hay API key — tạo agent với `llm_provider="none"`:

```python
agents = [QLearningAgent(f"i{k}", obs_dim=89, num_phases=4) for k in range(2)]
FederatedServer(agents).run_round(sample_counts=[1, 3])
```

Xem replay: trên **host** chạy `python3 -m http.server 8000` trong `cityflow/frontend/`, mở <http://localhost:8000>, nạp `replay_roadnet.json` và `replay.txt` từ `cityflow/sample/demo_fl/render/` (tuỳ chọn thêm `fl_chart.txt`, `fl_log.json`; KPI ở `kpi.csv`). Chi tiết ở README mục "Demo FL và Giao diện Replay".

Sau mỗi thay đổi: chạy `python -m pytest tests -q` (đã gồm kiểm tra §2.3) và demo 200 bước ở trên; log vòng FL phải báo `local_policy_head giữ nguyên ✓` và độ lệch backbone về `0.000000`.

---

## 8. Known Gaps — code hiện tại còn lệch quy tắc

Đừng coi các chỗ dưới đây là mẫu để làm theo; khi sửa vùng code liên quan thì đưa về đúng quy tắc.

| Quy tắc | Hiện trạng |
|---|---|
| §2.1 — `TrafficEnv` riêng | Chưa có bản bọc CityFlow; `SignalController` nằm trong `main_fedmarl.py` và chỉ thực thi xanh tối thiểu + vàng (không có all-red, thứ tự pha). `env/safety.py` đã có đủ C-01…C-05 cho `MockTrafficEnv` nhưng chưa được `SignalController` dùng |
| §4 — `NodeAgent` trung lập thuật toán | Đã đạt: thuật toán nằm ở lớp con (`q_learning.py`, `dqn.py`, `behavior_cloning.py`). `TrafficAgentNet` chỉ có một `Linear` làm policy head (không dueling/distributional) và `build_network()` mặc định không bật `with_value` — thuật toán cần khác phải override (xem `docs/RL_INTEGRATION_GUIDE.md` §3) |
| Chưa có / chưa làm | Baseline Centralized MARL (`agents/shared_policy.py`), neighbor embedding (MDP_SPEC §2.2), action masking, `metadata()` chưa có thống kê lưu lượng cho clustering, `ReplayBuffer` không có n-step/prioritized |
| §5.1 — bọc `try-except` và validate CityFlow | `main_fedmarl.py` đã bọc việc tạo Engine và giải phóng trong `finally`, nhưng `next_step()`, `set_tl_phase()`, `get_*` và đọc/ghi JSON vẫn chưa có `try-except`/validate ID giao lộ và chỉ số pha |
| §5.1 — dùng `logging` | Demo và nhánh lỗi LLM trong `NodeAgent` dùng `print` |
| §5.2 — hyperparameter không rải trong hàm | `cluster_clients()` chuẩn hoá bằng số cứng (8, 168, 2); `max_norm=10.0` lặp lại trong `dqn.py` |
| §3 — chỉ chạy trong Docker | README mục "Installation" và nửa sau `cityflow/cityflow.md` vẫn hướng dẫn tạo `.venv` trên host |
| §3 — thư viện | `flwr`, `wandb`, `pandas`, `scikit-learn` có trong `requirements.txt` nhưng chưa file `.py` nào import |
