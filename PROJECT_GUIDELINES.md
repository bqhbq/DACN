# 📋 PROJECT_GUIDELINES.md

> **Version**: 1.0 · **Ngày tạo**: 2026-11-07  

---

## Mục lục

1. [Mục tiêu Nghiên cứu Cốt lõi](#1-mục-tiêu-nghiên-cứu-cốt-lõi)
2. [Yêu cầu & Kết quả Giai đoạn 1](#2-yêu-cầu--kết-quả-giai-đoạn-1)
3. [Mô hình hóa Agent (MDP)](#3-mô-hình-hóa-agent-mdp)
4. [Ràng buộc An toàn & Vận hành](#4-ràng-buộc-an-toàn--vận-hành)
5. [KPIs Đánh giá](#5-kpis-đánh-giá)
6. [Kiến trúc Phần mềm OOP](#6-kiến-trúc-phần-mềm-oop)
7. [Cấu trúc Thư mục Dự án](#7-cấu-trúc-thư-mục-dự-án)
8. [Quy tắc Sinh Code Bắt buộc](#8-quy-tắc-sinh-code-bắt-buộc)

---

## 1. Mục tiêu Nghiên cứu Cốt lõi

Dự án nghiên cứu **Federated Multi-Agent Reinforcement Learning (FedMARL)** ứng dụng vào bài toán **Điều khiển Tín hiệu Giao thông Thích ứng (Adaptive Traffic Signal Control — ATSC)** trong môi trường phân tán.

### Vấn đề trọng tâm cần giải quyết

> **"Negative Transfer"** (Chuyển giao tri thức tiêu cực) là hiện tượng xảy ra khi tri thức học được từ một giao lộ được đồng bộ hóa sang một giao lộ khác có đặc điểm **không đồng nhất** (khác biệt về topology cấu trúc, lưu lượng xe, số pha đèn), dẫn đến suy giảm hiệu suất thay vì cải thiện.

### Hướng tiếp cận giải quyết

Dự án giải quyết bài toán Negative Transfer thông qua các cơ chế:

| Cơ chế | Mô tả |
|---|---|
| **Personalization** | Mỗi Node-Agent giữ lại một `local_policy_head` riêng biệt, không tham gia vào quá trình học liên bang. Chỉ có `shared_feature_extractor` được chia sẻ toàn cục. |
| **Clustering** | Server phân cụm (K-Means / phân cụm phân cấp) các giao lộ theo metadata (topology, lưu lượng trung bình) trước khi tổng hợp — các giao lộ tương đồng mới được phép học chung. |
| **Selective Collaboration** | Server lựa chọn tập con các client có phân phối dữ liệu tương đồng để tham gia mỗi vòng liên bang, tránh cưỡng bức kết hợp các agent quá dị biệt. |

---

## 2. Yêu cầu & Kết quả Giai đoạn 1

### 2.1 Môi trường Mô phỏng

- **Simulator**: CityFlow (đã cài đặt tại `cityflow/externals/CityFlow/`)
- **Các topology bắt buộc phải có**:

| Topology | Mô tả | Mức độ ưu tiên |
|---|---|---|
| **Grid** | Lưới ô vuông N×M giao lộ — cơ sở để so sánh baseline | 🔴 Bắt buộc |
| **Arterial** | Tuyến đường chính 1 chiều/2 chiều với nhiều giao lộ nối tiếp | 🔴 Bắt buộc |
| **Corridor** | Hành lang giao thông kết hợp nhiều loại giao lộ | 🟡 Khuyến nghị |

- **Các kịch bản lưu lượng (Traffic Demand Scenarios)**:
  - `Uniform`: Lưu lượng đều, dùng để kiểm tra baseline
  - `Demand Shift`: Thay đổi lưu lượng đột ngột theo giờ cao điểm
  - `Congestion`: Lưu lượng vượt ngưỡng, kiểm tra khả năng phục hồi
  - `Incident`: Mô phỏng sự cố (lane closure), kiểm tra tính thích ứng

### 2.2 Các Baseline Bắt buộc Cài đặt

> [!IMPORTANT]
> Tất cả baseline phải được cài đặt trước khi triển khai FedMARL để đảm bảo tính so sánh khoa học.

| STT | Baseline | Mô tả | Module |
|---|---|---|---|
| 1 | **Fixed-time** | Chu kỳ đèn cố định, không phụ thuộc lưu lượng | `agents/baselines/fixed_time.py` |
| 2 | **Actuated Control** | Điều chỉnh theo cảm biến vòng lặp (loop detector) | `agents/baselines/actuated.py` |
| 3 | **Independent RL** | Mỗi giao lộ chạy PPO/DQN độc lập, không chia sẻ | `agents/dqn.py`, `agents/ppo.py` |
| 4 | **Centralized MARL** | Tất cả agent dùng chung một policy (shared weights) | `agents/shared_policy.py` |

---

## 3. Mô hình hóa Agent (MDP)

Mỗi giao lộ được mô hình hóa như một **Node-Agent** với bài toán ra quyết định theo quy trình Markov (MDP).

### 3.1 Không gian Trạng thái — State Space

Trạng thái của Agent tại bước thời gian $t$ bao gồm **hai thành phần**:

#### Thành phần 1: Dữ liệu giao thông cục bộ (Local Traffic Data)

| Đặc trưng | Ký hiệu | Mô tả | Chuẩn hóa |
|---|---|---|---|
| Chiều dài hàng đợi | $q_l$ | Số xe chờ trên mỗi làn xe vào | $[0, 1]$ theo max queue |
| Lưu lượng | $f_l$ | Số xe đi qua làn trong $\Delta t$ giây | $[0, 1]$ theo max flow |
| Mật độ | $\rho_l$ | Xe / mét làn xe | $[0, 1]$ |
| Pha đèn hiện tại | $p$ | One-hot encoding pha đèn đang hoạt động | $\{0,1\}^{N_{phases}}$ |
| Thời gian pha còn lại | $t_{rem}$ | Giây còn lại của pha hiện tại | $[0, 1]$ |

#### Thành phần 2: Thông tin nhúng từ Agent lân cận (Neighbor Embedding)

- Nhận vector đặc trưng đã được nén (`backbone output`) từ các giao lộ kề cận.
- **Mục đích**: Cho phép Agent nhận biết tình trạng ùn tắc sắp lan đến từ thượng nguồn.
- **Lưu ý triển khai**: Sử dụng cơ chế attention hoặc mean-pooling trên tập neighbor embeddings.

### 3.2 Không gian Hành động — Action Space

Dự án hỗ trợ **hai dạng** không gian hành động, có thể cấu hình qua `config.yaml`:

| Dạng | Mô tả | Ưu điểm |
|---|---|---|
| **Phase-based** | Agent chọn trực tiếp ID pha đèn tiếp theo | Đơn giản, phù hợp DQN |
| **Keep/Change** | Agent quyết định giữ nguyên hoặc chuyển sang pha kế tiếp | Phù hợp PPO, kiểm soát tần suất chuyển pha |

> [!WARNING]
> Lớp `TrafficEnv` PHẢI chặn hoàn toàn các hành động không hợp lệ (chuyển pha không theo thứ tự, vi phạm thời gian xanh tối thiểu) trước khi truyền xuống CityFlow. Agent không được có cơ hội quan sát trạng thái hệ quả của hành động bất hợp lệ.

### 3.3 Hàm Phần thưởng — Reward Function

Dự án ưu tiên sử dụng công thức dựa trên **Max-Pressure** kết hợp với **giảm thiểu hàng đợi**:

$$r_t = -\alpha \cdot \sum_{l \in \mathcal{L}_{in}} q_l(t) + \beta \cdot \text{Throughput}(t) - \gamma \cdot \mathbb{1}[\text{phase\_change}]$$

Trong đó:
- $q_l(t)$: Chiều dài hàng đợi làn $l$ tại bước $t$
- $\text{Throughput}(t)$: Số xe rời giao lộ trong bước $t$
- $\mathbb{1}[\text{phase\_change}]$: Phạt khi chuyển pha (tránh dao động quá mức)
- $\alpha, \beta, \gamma$: Trọng số điều chỉnh thực nghiệm

**Phương án thay thế**: Max-Pressure reward (Wei et al., 2019):
$$r_t = -\sum_{l \in \mathcal{L}} |q_l^{in}(t) - q_l^{out}(t)|$$

---

## 4. Ràng buộc An toàn & Vận hành

> [!CAUTION]
> Đây là các ràng buộc **không thể thương lượng**. Vi phạm bất kỳ ràng buộc nào dưới đây sẽ dẫn đến kết quả thực nghiệm không hợp lệ và mô phỏng không an toàn.

### 4.1 Danh sách Ràng buộc Bắt buộc

| ID | Ràng buộc | Giá trị mặc định | Nơi thực thi |
|---|---|---|---|
| `C-01` | **Thứ tự pha hợp lệ** | Phải tuân theo chu trình pha đã định nghĩa trong roadnet | `TrafficEnv.step()` |
| `C-02` | **Thời gian xanh tối thiểu** | ≥ 10 giây (cấu hình được) | `TrafficEnv.step()` |
| `C-03` | **Chu kỳ đèn vàng** | Bắt buộc 3–5 giây giữa hai pha xanh | `TrafficEnv._apply_yellow()` |
| `C-04` | **Chu kỳ All-Red** | Tùy chọn 1–2 giây sau đèn vàng | `TrafficEnv._apply_all_red()` |
| `C-05` | **Cấm chuyển pha không an toàn** | Không được nhảy cóc qua pha trung gian | `TrafficEnv._validate_action()` |

### 4.2 Cơ chế Thực thi

**Phương án 1: Hard Constraints** (Mặc định cho Giai đoạn 1)
- `TrafficEnv` ghi đè hành động của Agent nếu vi phạm.
- Agent nhận penalty reward khi hành động bị ghi đè.

**Phương án 2: Action Masking** (Nâng cao)
- `TrafficEnv` trả về `action_mask: np.ndarray` cùng với `observation`.
- Agent chỉ được phép lựa chọn trong tập hành động hợp lệ.
- Phù hợp với PPO (sử dụng masked softmax).

---

## 5. KPIs Đánh giá

### 5.1 Chỉ số Hiệu suất Giao thông

| KPI | Ký hiệu | Đơn vị | Mục tiêu |
|---|---|---|---|
| Thời gian chờ trung bình | $\bar{W}$ | giây/xe | ↓ Giảm thiểu |
| Chiều dài hàng đợi trung bình | $\bar{Q}$ | xe/làn | ↓ Giảm thiểu |
| Queue Spillback | — | % episode có spillback | ↓ Giảm thiểu |
| Throughput | $\Phi$ | xe/giờ | ↑ Tối đa hóa |
| Thời gian di chuyển trung bình | $\bar{T}$ | giây | ↓ Giảm thiểu |

### 5.2 Chỉ số Công bằng & Ổn định

| KPI | Mô tả | Công thức |
|---|---|---|
| **Fairness Index** | Mức độ công bằng phân phối thời gian chờ giữa các hướng | Jain's Fairness Index |
| **Policy Stability** | Tần suất chuyển pha không cần thiết | Số lần thay đổi / tổng bước |
| **Convergence Speed** | Số vòng FL để đạt 95% hiệu suất baseline | Vòng FL |

### 5.3 Chỉ số An toàn Vận hành

| KPI | Mô tả |
|---|---|
| `safety_violation_rate` | % bước thời gian mà Agent đề xuất hành động vi phạm (trước khi bị TrafficEnv chặn) |
| `yellow_compliance_rate` | % lần chuyển pha có đủ chu kỳ đèn vàng |
| `min_green_violation_rate` | % lần vi phạm thời gian xanh tối thiểu |

---

## 6. Kiến trúc Phần mềm OOP

> [!IMPORTANT]
> Đây là ràng buộc kiến trúc **không thể thỏa hiệp**. Ba lớp dưới đây phải hoàn toàn tách biệt về trách nhiệm. Không được phép viết logic của lớp này vào bên trong lớp khác.

### Sơ đồ Kiến trúc Tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│                        FederatedServer                           │
│                                                                  │
│  cluster_clients()  →  aggregate(FedAvg/FedProx)  →  broadcast() │
│                                                                  │
│  Nhận: shared_weights từ NodeAgent                               │
│  Trả về: global_shared_weights → NodeAgent                       │
└───────────────────┬──────────────────────────────────────────────┘
                    │  global backbone weights (↓)
                    │  local backbone weights  (↑)
         ┌──────────▼───────────┐      ┌──────────────────────────┐
         │      NodeAgent A     │  ... │      NodeAgent N         │
         │                      │      │                          │
         │ ┌──────────────────┐ │      │ ┌──────────────────────┐ │
         │ │shared_feature_   │ │      │ │shared_feature_       │ │
         │ │extractor (FL)    │ │      │ │extractor (FL)        │ │
         │ ├──────────────────┤ │      │ ├──────────────────────┤ │
         │ │local_policy_head │ │      │ │local_policy_head     │ │
         │ │   (Private)      │ │      │ │   (Private)          │ │
         │ └──────────────────┘ │      │ └──────────────────────┘ │
         └──────────┬───────────┘      └──────────┬───────────────┘
                    │  actions                    │  actions
         ┌──────────▼─────────────────────────────▼───────────────┐
         │                      TrafficEnv                        │
         │                                                        │
         │  reset() → step(actions) → get_state() → get_reward()  │
         │                                                        │
         │  [Thực thi tự động: yellow phase, min green, all-red]  │
         └──────────────────────────┬─────────────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   CityFlow Engine  │
                          │  (Simulator Core)  │
                          └────────────────────┘
```

---

### 6.1 Lớp `TrafficEnv` — Môi trường Mô phỏng

**Vị trí**: `env/traffic_env.py`  
**Vai trò**: Bao bọc CityFlow. Hoạt động như một **data pipeline an toàn** giữa Agent và Simulator.

#### Trách nhiệm DUY NHẤT của lớp này

- Khởi tạo và quản lý vòng đời CityFlow Engine
- Thực thi **tự động và ngầm** mọi ràng buộc an toàn (C-01 đến C-05)
- Xây dựng vector `observation` từ dữ liệu CityFlow thô
- Tính toán `reward` theo công thức đã định nghĩa
- **KHÔNG** chứa bất kỳ logic RL nào (không có optimizer, loss, neural network)

#### Interface Bắt buộc

```python
class TrafficEnv:
    def reset(self, *, seed=None) -> tuple[np.ndarray, dict]:
        """Khởi động lại episode. Trả về (observation, info)."""

    def step(self, actions: dict[str, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Nhận hành động từ tất cả Node-Agent.
        1. Gọi _validate_action() — chặn hành động vi phạm
        2. Gọi _apply_yellow() nếu cần chuyển pha
        3. Gọi engine.next_step() lặp green_duration lần
        4. Gọi get_state() và get_reward()
        Trả về (obs, reward, terminated, truncated, info)
        """

    def get_state(self) -> np.ndarray:
        """Trả về vector observation đã chuẩn hóa."""

    def get_reward(self) -> float:
        """Tính và trả về scalar reward."""

    def _validate_action(self, intersection_id: str, action: int) -> int:
        """Kiểm tra và hiệu chỉnh hành động nếu vi phạm ràng buộc."""

    def _apply_yellow(self, intersection_id: str) -> None:
        """Chèn chu kỳ đèn vàng bắt buộc khi chuyển pha."""

    def close(self) -> None:
        """Giải phóng tài nguyên CityFlow Engine."""
```

---

### 6.2 Lớp `NodeAgent` — Agent Học Tăng cường

**Vị trí**: `agents/node_agent.py`  
**Vai trò**: Chứa mô hình PyTorch và logic RL. Là đơn vị học tập tại mỗi giao lộ.

#### Ràng buộc Kiến trúc Mạng Nơ-ron — **TUYỆT ĐỐI KHÔNG ĐƯỢC VI PHẠM**

```
Input: observation vector
        │
        ▼
┌───────────────────────────────┐
│   shared_feature_extractor    │  ← Tham số này được gửi lên FederatedServer
│   (Linear → ReLU → Linear)    │    để tổng hợp FedAvg
└───────────────────┬───────────┘
                    │ features
                    ▼
┌───────────────────────────────┐
│      local_policy_head        │  ← Tham số này TUYỆT ĐỐI GIỮ LẠI tại node
│   (Linear → Q-values/logits)  │    KHÔNG BAO GIỜ gửi lên Server
└───────────────────────────────┘
        │
        ▼
Output: action logits / Q-values
```

#### Interface Bắt buộc

```python
class NodeAgent:
    def __init__(self, intersection_id: str, obs_dim: int, num_phases: int):
        self.shared_feature_extractor = nn.Sequential(...)   # Phần liên bang
        self.local_policy_head        = nn.Sequential(...)   # Phần riêng tư

    def forward_pass(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward qua cả hai sub-module."""

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Tính RL loss (TD-error cho DQN, policy gradient cho PPO)."""

    def get_shared_weights(self) -> dict[str, torch.Tensor]:
        """
        CHỈ trả về state_dict của shared_feature_extractor.
        TUYỆT ĐỐI KHÔNG bao gồm local_policy_head.
        """
        return copy.deepcopy(self.shared_feature_extractor.state_dict())

    def set_shared_weights(self, global_weights: dict[str, torch.Tensor]) -> None:
        """Nạp weights backbone đã tổng hợp từ FederatedServer."""
        self.shared_feature_extractor.load_state_dict(global_weights)
```

---

### 6.3 Lớp `FederatedServer` — Máy chủ Điều phối Liên bang

**Vị trí**: `federated/server.py`  
**Vai trò**: Điều phối toàn bộ quy trình Federated Learning. **KHÔNG** tiếp xúc với môi trường hay dữ liệu giao thông thô.

#### Trách nhiệm DUY NHẤT của lớp này

- Nhận `shared_weights` từ các `NodeAgent`
- Thực hiện phân cụm (clustering) trước khi tổng hợp (tránh Negative Transfer)
- Chạy thuật toán tổng hợp `aggregate()` (FedAvg / FedProx)
- Broadcast weights đã cập nhật trở lại các `NodeAgent`
- **KHÔNG** chứa logic RL, không tiếp xúc CityFlow, không có replay buffer

#### Interface Bắt buộc

```python
class FederatedServer:
    def __init__(self, clients: list[NodeAgent]):
        self.clients = clients
        self.global_weights = None

    def cluster_clients(self, metadata: list[dict]) -> dict[int, list[NodeAgent]]:
        """
        Phân cụm NodeAgent theo metadata giao lộ (topology, avg_flow, num_lanes).
        Sử dụng K-Means hoặc phân cụm phân cấp.
        Trả về {cluster_id: [NodeAgent, ...]}
        """

    def aggregate(
        self,
        list_of_weights: list[dict[str, torch.Tensor]],
        algorithm: str = "fedavg",
    ) -> dict[str, torch.Tensor]:
        """
        Tổng hợp weights:
        - 'fedavg' : Trung bình đơn giản (McMahan et al., 2017)
        - 'fedprox': FedAvg + proximal regularization (Li et al., 2020)
        """

    def broadcast(self) -> None:
        """Gửi global_weights về tất cả NodeAgent đã đăng ký."""

    def run_round(self, round_idx: int) -> dict:
        """
        Điều phối một vòng FL hoàn chỉnh:
        1. Thu thập shared_weights từ clients
        2. cluster_clients() (nếu bật)
        3. aggregate()
        4. broadcast()
        5. Trả về log metrics của vòng này
        """
```

---

## 7. Cấu trúc Thư mục Dự án

```
FedMARL-ATSC/                        ← Workspace root
│
├── PROJECT_GUIDELINES.md            ← 📋 File này (tham chiếu cố định)
├── README.md                        ← Tổng quan dự án
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── env/                             ← Lớp TrafficEnv
│   ├── __init__.py
│   ├── base_env.py                  ← Abstract base class (Gymnasium interface)
│   ├── traffic_env.py               ← [CẦN TẠO] Wrapper CityFlow chính thức
│   └── cityflow_env.py              ← Prototype env (tham khảo)
│
├── agents/                          ← Lớp NodeAgent & các Baseline
│   ├── __init__.py
│   ├── node_agent.py                ← [CẦN TẠO] NodeAgent với Decoupled Architecture
│   ├── networks.py                  ← [CẦN TẠO] TrafficAgentNet PyTorch module
│   ├── replay_buffer.py             ← [CẦN TẠO] Experience replay
│   ├── dqn.py                       ← [CẦN TẠO] DQN trainer
│   ├── ppo.py                       ← [CẦN TẠO] PPO trainer
│   └── baselines/
│       ├── fixed_time.py            ← [CẦN TẠO]
│       └── actuated.py              ← [CẦN TẠO]
│
├── federated/                       ← Lớp FederatedServer
│   ├── __init__.py
│   └── server.py                    ← [CẦN TẠO] FL server & aggregation
│
├── notebooks/                       ← Jupyter notebooks thực nghiệm
│   ├── 01_federated_learning_demo.ipynb   ← ✅ Hoàn thành (FL demo với mock data)
│   └── FL_demo.ipynb                      ← Bản sao (JupyterLab auto-save)
│
├── cityflow/                        ← CityFlow installation & map data
│   ├── externals/CityFlow/          ← CityFlow source & Python binding
│   ├── build/generator_cityflow/    ← Các bản đồ mô phỏng đã tạo sẵn
│   │   ├── sample_map/              ← 1 giao lộ đơn
│   │   ├── sample_map_1/            ← Mạng lưới nhỏ
│   │   ├── sample_map_2/            ← 2 giao lộ có đèn, 8 pha ⭐ Dùng để dev
│   │   └── sample_map_3/            ← Mạng lớn hơn
│   └── src/                         ← Utilities sinh bản đồ
│
└── data/                            ← Dữ liệu mô phỏng (do simulation team cung cấp)
```

### Bản đồ Tham chiếu Chính cho Giai đoạn 1

> **`cityflow/build/generator_cityflow/sample_map_2/`** là bản đồ mặc định cho mọi thực nghiệm Giai đoạn 1.

| Thuộc tính | Giá trị |
|---|---|
| Config file | `build/generator_cityflow/sample_map_2/config.json` |
| Số giao lộ có đèn | 2 (`intersection_1_1`, `intersection_2_1`) |
| Số pha / giao lộ | 8 (4 pha xanh + 4 pha vàng xen kẽ) |
| Số làn xe vào / giao lộ | 28 (4 hướng × 7 làn) |
| Kích thước obs vector | 28 × 3 = **84** (queue, speed, density / làn) |
| Kích thước action space | **4** (4 pha xanh có thể chọn) |

---

## 8. Quy tắc Sinh Code Bắt buộc

> [!NOTE]
> Phần này là **cam kết kỹ thuật** áp dụng cho mọi tác vụ sinh code trong toàn bộ vòng đời dự án.

### 8.1 Quy tắc về Kiến trúc

| ID | Quy tắc | Mức độ |
|---|---|---|
| `R-01` | Mọi logic an toàn giao thông phải nằm trong `TrafficEnv`, không để lọt vào `NodeAgent` hay `FederatedServer` | 🔴 Bắt buộc |
| `R-02` | `NodeAgent.get_shared_weights()` chỉ được phép trả về `state_dict` của `shared_feature_extractor` | 🔴 Bắt buộc |
| `R-03` | `local_policy_head` không được tham chiếu ở bất kỳ đâu bên ngoài class `NodeAgent` | 🔴 Bắt buộc |
| `R-04` | `FederatedServer` không được import hoặc khởi tạo bất kỳ đối tượng CityFlow nào | 🔴 Bắt buộc |
| `R-05` | Mọi hyperparameter (learning rate, num_phases, hidden_dim, ...) phải đặt ở đầu file hoặc trong `config.yaml`, không được hardcode vào giữa hàm | 🟡 Khuyến nghị mạnh |

### 8.2 Quy tắc về Chất lượng Code

- **Ngôn ngữ**: Python ≥ 3.10 với type hints đầy đủ
- **Docstring**: Mọi class và public method phải có docstring theo chuẩn Google style
- **Tên biến**: Dùng tiếng Anh cho code, comment/docstring có thể song ngữ Anh–Việt
- **Test**: Mỗi module phải có thể chạy độc lập với `python -m module_name` để kiểm tra nhanh

### 8.3 Cam kết của AI Assistant

> **Đối với bất kỳ yêu cầu viết code nào trong tương lai, tôi sẽ tham chiếu nghiêm ngặt đến các hướng dẫn trong file này, ĐẶC BIỆT LÀ kiến trúc phân rã mạng Nơ-ron (Decoupled Architecture — `shared_feature_extractor` liên bang + `local_policy_head` riêng tư) và sự tách biệt tuyệt đối nhiệm vụ giữa ba lớp cốt lõi `TrafficEnv`, `NodeAgent`, và `FederatedServer`. Bất kỳ đoạn code nào vi phạm các ràng buộc này sẽ bị từ chối và viết lại trước khi trình bày với người dùng.**

---

*Tài liệu này được tạo ngày 2026-11-07. Mọi thay đổi phải được ghi chú rõ ràng với ngày tháng và lý do.*
