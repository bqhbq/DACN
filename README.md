# Federated Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control (FedMARL-ATSC)

> **Phase 1 — Project Scaffold & Architecture Design**

---

## Overview

This research project investigates **Federated Multi-Agent Reinforcement Learning (FedMARL)** as a privacy-preserving, scalable approach to **Adaptive Traffic Signal Control (ATSC)** across heterogeneous urban intersections.

Classical centralised ATSC systems require raw traffic data to be transmitted to a central server — raising privacy, bandwidth, and single-point-of-failure concerns. This work instead trains local RL agents at each intersection and coordinates them through a **federated aggregation protocol**, so only model parameters (not raw sensor data) ever leave the local edge device.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Federated Server                        │
│   • Aggregates local model updates (FedAvg / FedProx)       │
│   • Manages global communication rounds                     │
│   • Handles client selection & stragglers                   │
└────────────────┬──────────────────────────────┬─────────────┘
                 │  Global model weights (↓)    │
                 │  Local gradients / Δw  (↑)   │
     ┌───────────▼───────────┐       ┌───────────▼───────────┐
     │   Intersection A      │  ...  │   Intersection N      │
     │  (high-throughput)    │       │  (pedestrian-heavy)   │
     │                       │       │                       │
     │  ┌─────────────────┐  │       │  ┌─────────────────┐  │
     │  │   Local Agent   │  │       │  │   Local Agent   │  │
     │  │   (PPO / DQN)   │  │       │  │   (PPO / DQN)   │  │
     │  └────────┬────────┘  │       │  └────────┬────────┘  │
     │           │           │       │           │           │
     │  ┌────────▼────────┐  │       │  ┌────────▼────────┐  │
     │  │  CityFlow / SUMO│  │       │  │  CityFlow / SUMO│  │
     │  │  Env Wrapper    │  │       │  │  Env Wrapper    │  │
     │  └─────────────────┘  │       │  └─────────────────┘  │
     └───────────────────────┘       └───────────────────────┘
```

### Handling Intersection Heterogeneity

Real urban networks contain intersections that differ in:

| Dimension | Examples |
|---|---|
| **Topology** | 3-way T-junction vs. 4-way cross vs. complex multi-lane |
| **Phase space** | 2 phases (simple) up to 8+ phases (complex) |
| **Traffic volume** | Low-density suburban vs. high-density arterial |
| **Sensor modality** | Loop detectors, cameras, V2X beacons |

This project addresses heterogeneity via:
1. **Local adaptation layers** — a small intersection-specific head sits on top of a shared backbone, so the federated server only aggregates the backbone weights.
2. **Personalised FL (pFedMe / FedProx)** — adds a proximal regularisation term to prevent local models from drifting too far from the global model.
3. **Role-based client clustering** — intersections are clustered by topology type before aggregation so that structurally similar agents share stronger inductive biases.

---

## Repository Structure

```
FedMARL-ATSC/
├── agents/             # RL agent implementations (PPO, DQN, …)
│   └── __init__.py
├── data/               # CityFlow / SUMO simulation data (populated by sim team)
├── env/                # Traffic environment wrappers
│   └── __init__.py
├── federated/          # FL server & client logic (Flower-based)
│   └── __init__.py
├── notebooks/          # EDA and prototyping notebooks
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU recommended (CPU training is supported but slower)
- [CityFlow](https://cityflow.readthedocs.io/) or SUMO installed separately (see `env/` README once populated)

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/FedMARL-ATSC.git
cd FedMARL-ATSC
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Open the notebooks

JupyterLab is not part of `requirements.txt`. Open the `.ipynb` files in `notebooks/` with the VS Code Jupyter extension and select the project's Python environment as the kernel (`ipykernel` is installed by `requirements.txt`).

To run CityFlow on macOS Apple Silicon, use the Docker setup described in [`cityflow/cityflow.md`](cityflow/cityflow.md).

---

## Luồng Dữ liệu và Thực thi CityFlow

Mục này mô tả cách thư mục `cityflow/` sinh dữ liệu mô phỏng và nạp vào simulator CityFlow.

### Tóm tắt

- **Không có dữ liệu tĩnh trong repo.** Toàn bộ `roadnet.json`, `flow.json`, `config.json` đều do script sinh ra lúc chạy, không file nào được commit sẵn.
- **Entry point là notebook** [`notebooks/cityflow/simulate.ipynb`](notebooks/cityflow/simulate.ipynb). Thư mục `cityflow/` không còn script khởi chạy riêng (`main.py` và `src/simulate.py` đã bị xoá).
- **Thư mục output mặc định** là `cityflow/sample/test/test_1/` (biến `MAIN_DIR` trong notebook).

### Các thành phần chính

| File | Vai trò |
|---|---|
| `notebooks/cityflow/simulate.ipynb` | Entry point: gọi lần lượt các bước sinh dữ liệu rồi chạy simulator |
| `cityflow/src/generators/roadnet_gen.py` | Hàm `generate_roadnet()`: sinh roadnet dạng lưới `rows x cols` |
| `cityflow/src/generators/flow_gen.py` | Lớp `FlowGenerator`: sinh flow từ roadnet (đang thử nghiệm) |
| `cityflow/src/generators/config_gen.py` | Hàm `gen_config()`: ghi `config.json` cho CityFlow Engine |
| `cityflow/src/vehicles/vehicle.py` | `VehicleType` và `VehicleFleet` (`MOTORCYCLE`, `CAR`, `TRUCK`): thông số xe hardcode |
| `cityflow/tools/generator/` | Script gốc của CityFlow: `generate_grid_scenario.py`, `generate_json_from_grid.py` |
| `cityflow/tools/converter/` | Chuyển roadnet SUMO (`.net.xml`) sang định dạng CityFlow; pipeline hiện tại chưa dùng |
| `cityflow/frontend/` | Trang web xem replay (`index.html`) |
| `cityflow/cityflow.md` | Hướng dẫn cài CityFlow từ source (Linux) |

### Dữ liệu được sinh ra từ đâu?

**Roadnet (map)** được sinh bằng thuật toán, không lấy từ bản đồ thật:

1. `generate_roadnet()` gọi `tools/generator/generate_grid_scenario.py` qua `subprocess`.
2. Script này gọi `gridToRoadnet()` để dựng lưới giao lộ và ghi ra `roadnet.json`.
3. Các nút ở rìa lưới được đánh dấu `virtual: true`; đây là điểm xe đi vào và rời khỏi mạng lưới.
4. Với `tl_plan=True`, mỗi giao lộ thật có 8 pha đèn: 4 pha xanh 30 giây xen kẽ 4 pha 5 giây.

Ví dụ với cấu hình trong notebook (`rows=2`, `cols=3`): 6 giao lộ thật, 10 nút virtual, 34 road.

**Flow (traffic demand)** có hai nguồn, cả hai đều do script sinh:

| File | Nguồn | Đặc điểm |
|---|---|---|
| `temp_flow.json` | Sinh kèm khi tạo roadnet (`generate_grid_scenario.py`) | Tuyến đi thẳng xuyên lưới, thêm 8 tuyến rẽ zig-zag nếu `turn=True`. Một loại xe duy nhất, `interval` 2 giây. Ví dụ trên cho 18 flow. |
| `flow.json` | `FlowGenerator.generate()` | Một flow cho mỗi cặp (road đi ra từ nút virtual A, road đi vào nút virtual B). Loại xe chọn ngẫu nhiên theo trọng số (mặc định 80% `CAR`, 20% `TRUCK`), `interval` 10 giây. Ví dụ trên cho 90 flow. |

Lưu ý:

- Notebook hiện chạy mô phỏng với **`temp_flow.json`**. `flow.json` vẫn được sinh ra nhưng chưa được dùng.
- Trong `flow.json`, mỗi `route` chỉ gồm road đầu và road cuối; CityFlow tự tìm đường đi giữa hai road này.
- Flow chỉ mang tính ngẫu nhiên ở khâu chọn loại xe. Lưu lượng là hằng số (`interval` cố định, `endTime = -1` tức chạy suốt mô phỏng).

### Execution flow

```
simulate.ipynb
  │
  ├─ 0. Tìm thư mục gốc project, chdir vào cityflow/, thêm vào sys.path
  │
  ├─ 1. generate_roadnet(rows, cols, output_dir=MAIN_DIR, ...)
  │       └─ subprocess: generate_grid_scenario.py
  │             └─ gridToRoadnet()  →  roadnet.json + temp_flow.json
  │
  ├─ 2. FlowGenerator(roadnet.json, flow.json).generate(...)   (thử nghiệm)
  │       └─ đọc roadnet.json, tìm nút virtual  →  flow.json
  │
  ├─ 3. gen_config(dir=MAIN_DIR, roadnet_name, flow_name, ...)
  │       └─ kiểm tra roadnet/flow tồn tại, tạo render/  →  config.json
  │
  ├─ 4. cityflow.Engine("config.json", thread_num=1)
  │       └─ đọc config.json, ghép "dir" + tên file để load roadnet và flow
  │
  └─ 5. Lặp engine.next_step() 1000 bước, in số xe và average travel time
          └─ ghi replay vào render/replay.txt và render/replay_roadnet.json
```

Sau khi chạy xong, `MAIN_DIR` chứa:

```
cityflow/sample/test/test_1/
├── roadnet.json
├── temp_flow.json
├── flow.json
├── config.json
└── render/
    ├── replay_roadnet.json
    └── replay.txt
```

Để xem lại mô phỏng, mở `cityflow/frontend/index.html` và nạp hai file trong `render/`.

### Cách chạy

1. Cài CityFlow theo [`cityflow/cityflow.md`](cityflow/cityflow.md): dùng Docker trên macOS ARM64, hoặc build từ source trên Linux.
2. Mở `notebooks/cityflow/simulate.ipynb` bằng VS Code (với Docker thì attach VS Code vào container trước) và chạy các cell từ trên xuống.

Muốn đổi kích thước lưới, số làn hoặc thư mục output thì sửa tham số ở cell `generate_roadnet(...)` và biến `MAIN_DIR`. Muốn mô phỏng bằng flow tự sinh thì đổi `flow_name="temp_flow.json"` thành `"flow.json"` ở cell `gen_config(...)`.

---

## Demo FL và Giao diện Replay

Script [`main_fedmarl.py`](main_fedmarl.py) chạy một buổi demo đầy đủ: CityFlow mô phỏng giao thông, mỗi giao lộ có một `NodeAgent` điều khiển đèn, và `FederatedServer` định kỳ tổng hợp backbone. Kết quả mô phỏng được ghi thành replay để xem lại trên trình duyệt.

Mỗi node huấn luyện cục bộ theo một trong hai chế độ, chọn bằng `--trainer`:

| Chế độ | Cách học | Cần gì |
|---|---|---|
| `local` (mặc định) | Q-learning một bước qua `NodeAgent.train_local_step()`, reward Composite Max-Pressure và action chọn pha theo [`docs/MDP_SPEC.md`](docs/MDP_SPEC.md) §3.1, §4.2 | Không cần API |
| `llm` | Behavior Cloning qua `NodeAgent.train_step()`: LLM (Gemini hoặc OpenAI) quyết định KEEP/CHANGE, mạng học bắt chước | `GEMINI_API_KEY` / `OPENAI_API_KEY` trong `.env`; chọn bằng `--llm-provider` |

Chế độ `local` là bản rút gọn để minh hoạ luồng FL (không có replay buffer hay target network); bản DQN đầy đủ nằm ở `agents/dqn.py`.

### 1. Chạy demo

Với Docker (xem [`cityflow/cityflow.md`](cityflow/cityflow.md)), chạy từ thư mục `cityflow/`:

```bash
docker compose up -d
docker compose exec cityflow python main_fedmarl.py
```

Nếu CityFlow đã cài trực tiếp trên máy, chạy từ thư mục gốc repo:

```bash
python main_fedmarl.py
```

Tham số tuỳ chọn:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--steps` | 600 | Tổng số bước mô phỏng (1 bước = 1 giây) |
| `--decision-interval` | 10 | Số giây giữa hai lần agent chọn pha |
| `--fl-interval` | 100 | Số bước giữa hai vòng FL |
| `--rows`, `--cols` | 1, 2 | Kích thước lưới giao lộ |
| `--seed` | 42 | Seed cho CityFlow, NumPy và PyTorch |
| `--trainer` | `local` | Cách huấn luyện cục bộ: `local` hoặc `llm` |
| `--llm-provider` | `gemini` | LLM dùng khi `--trainer llm`: `gemini` hoặc `openai` |

Mỗi lần chạy, script thực hiện:

1. Sinh `roadnet.json`, `temp_flow.json` và `config.json` vào `cityflow/sample/demo_fl/`. Config được bật `saveReplay` và `rlTrafficLight` (thiếu `rlTrafficLight` thì CityFlow bỏ qua lệnh đổi pha).
2. Trước khi mô phỏng, server phát một backbone khởi tạo chung cho mọi node (vòng 0).
3. Mỗi 10 giây, từng agent đọc state, học từ transition vừa rồi và chọn pha xanh. Pha này được áp xuống mô phỏng bằng `eng.set_tl_phase()`, có chèn pha vàng 3 giây và giữ xanh tối thiểu 10 giây.
4. Mỗi 100 bước, FL Server chạy một vòng FedAvg có trọng số `n_k` (số mẫu mỗi node đã học từ vòng trước) và in log từng chặng:

```
[FL Round 1] ① UPLINK  Node Agent ──▶ Central Coordinator (chỉ backbone)
   Node_1 ──▶ Server | 10 tensors, 30,016 tham số (117.2 KB) | n_k=9 mẫu | ‖w_k‖=21.9976
   Node_2 ──▶ Server | 10 tensors, 30,016 tham số (117.2 KB) | n_k=9 mẫu | ‖w_k‖=21.9921
[FL Round 1] ② AGGREGATE tại Central Coordinator: w* = Σ (n_k/N)·w_k  (FedAvg)
   hệ số n_k/N: Node_1=0.500, Node_2=0.500
   ví dụ phần tử [84] của '0.weight': 0.500×(-0.07473) + 0.500×(-0.07072) = -0.07272
[FL Round 1] ③ DOWNLINK Central Coordinator ──▶ Node Agent (broadcast w*)
   Server ──▶ Node_1 | backbone dịch ‖w*−w_k‖=0.2753 | local_policy_head giữ nguyên ✓
   Server ──▶ Node_2 | backbone dịch ‖w*−w_k‖=0.2753 | local_policy_head giữ nguyên ✓
[FL Round 1] độ lệch backbone giữa các node: 0.013601 → 0.000000
```

5. Ghi replay, `fl_log.json` và `fl_chart.txt` vào `cityflow/sample/demo_fl/render/` cho giao diện.

### 2. Mở giao diện replay

Chạy trên máy host (không phải trong container, vì container không mở cổng nào):

```bash
cd cityflow/frontend
python3 -m http.server 8000
```

Mở trình duyệt tại <http://localhost:8000>. Trang cần Internet để tải thư viện từ CDN.

### 3. Nạp file replay

Trong bảng điều khiển của trang:

1. Ô **Roadnet File**: chọn `cityflow/sample/demo_fl/render/replay_roadnet.json`.
2. Ô **Replay File**: chọn `cityflow/sample/demo_fl/render/replay.txt`.
3. (Tuỳ chọn) Ô **Chart File**: chọn `render/fl_chart.txt` để xem đồ thị độ lệch backbone giữa các node — tăng dần khi các node học cục bộ, rơi về 0 sau mỗi vòng FL.
4. (Tuỳ chọn) Ô **FL Log File**: chọn `render/fl_log.json` để xem luồng Federated Learning:
   - Trên bản đồ có hộp **FL Server** nối với từng node. Ngay sau mỗi vòng FL, gói trọng số chạy từ node lên server (cam), server tổng hợp (tím), rồi chạy ngược về node (xanh lá).
   - Bảng ở góc phải hiện số liệu học cục bộ của từng node (pha, action, reward, loss, `n_k`) và số liệu vòng FL gần nhất (hệ số `n_k/N`, chuẩn trọng số gửi lên, phép tính FedAvg mẫu, độ lệch trước → sau).
5. Bấm **Start**.

Lưu ý:

- Ô Roadnet File nhận file **`replay_roadnet.json`** do CityFlow ghi ra khi mô phỏng, không phải `roadnet.json` đầu vào.
- Replay chỉ có hai màu đèn là đỏ và xanh. Trong pha vàng, các hướng đi thẳng và rẽ trái hiển thị màu đỏ.
- Có thể kéo thanh trượt để tua, và chỉnh tốc độ phát ngay trên trang.

---

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. Log in before running any training:

```bash
wandb login
```

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| **1** | Project scaffold, architecture design, environment setup | 🔄 In Progress |
| **2** | CityFlow env wrapper, single-agent DQN baseline | ⏳ Planned |
| **3** | Multi-agent extension, local PPO agents | ⏳ Planned |
| **4** | Federated aggregation (FedAvg, FedProx) | ⏳ Planned |
| **5** | Heterogeneity handling, personalised FL | ⏳ Planned |
| **6** | Benchmarking, ablation studies, paper writing | ⏳ Planned |

---

## Citation

> *Work in progress. Citation will be added upon publication.*

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
