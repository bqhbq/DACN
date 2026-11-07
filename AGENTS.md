# AGENTS.md — System Context & Guidelines for AI Coding Agents
# FedMARL-ATSC · Federated Multi-Agent RL for Adaptive Traffic Signal Control

---

## 1. Project Context

Dự án xây dựng hệ thống **điều khiển tín hiệu giao thông thích ứng (ATSC)** phân tán sử dụng **Federated Multi-Agent Reinforcement Learning (FedMARL)**.

### Vấn đề trọng tâm
Các hệ thống MARL tập trung truyền thống yêu cầu chia sẻ dữ liệu giao thông thô giữa các giao lộ, gây ra:
- **Vi phạm quyền riêng tư** dữ liệu hành trình của người dùng
- **"Negative Transfer"**: tri thức học từ giao lộ A làm *giảm* hiệu suất giao lộ B nếu hai giao lộ có cấu trúc/lưu lượng không đồng nhất (heterogeneous intersections)
- Khó mở rộng quy mô (scalability bottleneck tại server tập trung)

### Giải pháp
Thay vì chia sẻ dữ liệu, mỗi giao lộ chỉ gửi **model weights của phần backbone** lên Federated Server. Phần `local_policy_head` được giữ lại tại biên để **cá nhân hoá (personalisation)** theo đặc thù từng giao lộ.

---

## 2. Tech Stack

| Thành phần | Công nghệ | Phiên bản tối thiểu |
|---|---|---|
| Ngôn ngữ | Python | 3.10+ |
| Deep Learning | PyTorch | 2.2.0 |
| Federated Learning | Flower (`flwr`) | 1.8.0 |
| Traffic Simulation | CityFlow | Xem `cityflow/externals/` |
| Experiment Tracking | Weights & Biases (`wandb`) | 0.17.0 |
| Data / Notebook | NumPy, Pandas, JupyterLab | Xem `requirements.txt` |

### Môi trường phát triển
- **macOS Apple Silicon (ARM64)**: PyTorch phải cài từ `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` hoặc native Apple Silicon wheel. **Không dùng CUDA wheel** trên máy này.
- CityFlow binary ở `cityflow/externals/CityFlow/` — đã build sẵn cho Linux x86_64. Trên macOS ARM64, dùng `MockTrafficEnv` hoặc Docker.
- Luôn làm việc trong virtual environment (`.venv/`).

---

## 3. Architecture Rules — TUYỆT ĐỐI TUÂN THỦ

### 3.1 Decoupled Neural Network Architecture

Mạng Nơ-ron của mỗi Agent **PHẢI** chia làm đúng 2 phần với tên thuộc tính chính xác:

```python
class TrafficAgentNet(nn.Module):
    def __init__(self):
        # ✅ ĐÚNG — Hai sub-module có tên cố định
        self.shared_feature_extractor = nn.Sequential(...)   # Phần liên bang
        self.local_policy_head        = nn.Sequential(...)   # Phần riêng tư
```

| Sub-module | Tên thuộc tính | Liên bang? | Mục đích |
|---|---|---|---|
| Backbone | `shared_feature_extractor` | ✅ Gửi lên Server | Học đặc trưng giao thông chung |
| Policy Head | `local_policy_head` | ❌ Giữ tại node | Quyết định pha đèn cục bộ |

### 3.2 Separation of Concerns — 3 lớp độc lập

```
TrafficEnv          NodeAgent           FederatedServer
─────────────       ─────────────       ───────────────
Bao bọc CityFlow    Chứa PyTorch net    Điều phối FL
Thực thi safety     select_action()     aggregate()
get_state() ──────► forward_pass()      broadcast()
get_reward()        get_shared_weights()──► [nhận backbone]
                    set_shared_weights()◄── [gửi backbone]
```

**Quy tắc vi phạm nghiêm cấm:**
- ❌ `TrafficEnv` KHÔNG được import PyTorch hoặc chứa `nn.Module`
- ❌ `FederatedServer` KHÔNG được import `cityflow` hoặc chứa logic RL
- ❌ `NodeAgent.get_shared_weights()` KHÔNG được trả về key của `local_policy_head`
- ❌ Không được merge/flatten backbone và head thành một `nn.Sequential` duy nhất

### 3.3 Safety Wrapper (Ràng buộc An toàn)

`TrafficEnv.step()` phải xử lý **ngầm** (không lộ ra Agent):
1. Validate action → chặn phase jump bất hợp lệ (C-01)
2. Enforce minimum green time (C-02)
3. Tự chèn yellow interval khi chuyển pha (C-03)
4. Tự chèn all-red interval nếu cấu hình (C-04)

---

## 4. Coding Conventions

### 4.1 OOP Structure
```
agents/
  node_agent.py    → class NodeAgent
  networks.py      → class TrafficAgentNet
  dqn.py           → class DQNTrainer
  replay_buffer.py → class ReplayBuffer
env/
  cityflow_wrapper.py → class TrafficEnv
federated/
  server.py        → class FederatedServer
```

### 4.2 Type Hinting
```python
# ✅ ĐÚNG
def step(self, actions: dict[str, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
    ...

# ❌ SAI — Thiếu type hints
def step(self, actions):
    ...
```

### 4.3 Docstrings (Google Style)
```python
def aggregate(
    self,
    list_of_weights: list[dict[str, torch.Tensor]],
    algorithm: str = "fedavg",
) -> dict[str, torch.Tensor]:
    """
    Tổng hợp weights từ nhiều clients bằng FedAvg.

    Args:
        list_of_weights: Danh sách state_dict backbone từ mỗi client.
        algorithm: Thuật toán tổng hợp ('fedavg' hoặc 'fedprox').

    Returns:
        Dict chứa averaged weights, cùng cấu trúc với input.

    Raises:
        ValueError: Nếu list_of_weights rỗng.
    """
```

### 4.4 Path Management (pathlib — không hardcode)
```python
# ✅ ĐÚNG
from pathlib import Path
CONFIG = Path(__file__).resolve().parent.parent / "cityflow/build/sample_map_2/config.json"

# ❌ SAI — hardcode string path
CONFIG = "/Users/buiquanghuy/Documents/ĐACN/cityflow/build/sample_map_2/config.json"
```

### 4.5 Constants & Config
- Tất cả hyperparameter (lr, gamma, batch_size, hidden_dim) phải đặt ở đầu file hoặc trong `config.yaml`.
- Không hardcode giá trị số vào giữa hàm.
- Sử dụng `dataclass` hoặc `TypedDict` cho cấu hình phức tạp.

---

## 5. AI Agent Rules

### 5.1 Quy trình bắt buộc khi nhận yêu cầu mới

```
1. ĐỌC file này (AGENTS.md) ← bạn đang ở đây
2. ĐỌC docs/MDP_SPEC.md để hiểu State/Action/Reward
3. ĐỌC PROJECT_GUIDELINES.md để xem KPIs và roadmap
4. KHẢO SÁT code hiện có (env/, agents/, federated/)
5. Viết code — tuân thủ Decoupled Architecture
6. Viết smoke test hoặc dùng dummy data để xác nhận pipeline
```

### 5.2 Ưu tiên Testing với Mock/Dummy Data

Luôn ưu tiên kiểm tra **data pipeline** trước khi tích hợp RL thật:

```python
# ✅ ĐÚNG — Test pipeline với dummy obs
obs = np.random.rand(84).astype(np.float32)
action = agent.select_action(obs)
loss = agent.compute_dummy_loss(obs)
shared_w = agent.get_shared_weights()
server.aggregate([shared_w])

# Chỉ sau khi pipeline xác nhận OK mới kết nối CityFlow thật
```

### 5.3 Không được phá vỡ kiến trúc Decoupled

Trước khi commit bất kỳ thay đổi nào liên quan đến `NodeAgent` hoặc `TrafficAgentNet`, phải chạy kiểm tra:

```python
agent = NodeAgent(...)
sw = agent.get_shared_weights()
assert not any("local_policy_head" in k for k in sw), "R-02 VI PHẠM: Policy head bị leak!"
assert any("shared_feature_extractor" in k for k in sw), "Backbone rỗng!"
```

### 5.4 Map CityFlow mặc định cho Development

Dùng `sample_map_2` (2 giao lộ có đèn, symmetric 4-way, 4 green phases):
```
cityflow/build/generator_cityflow/sample_map_2/config.json
  → intersection_1_1: obs_dim=84, num_phases=4
  → intersection_2_1: obs_dim=84, num_phases=4
```

### 5.5 Quy tắc Import

```python
# Thứ tự import chuẩn
# 1. Standard library
import copy, json, logging
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn

# 3. Project modules (theo dependency order)
from env.cityflow_wrapper import TrafficEnv
from agents.networks      import TrafficAgentNet
from agents.node_agent    import NodeAgent
from federated.server     import FederatedServer
```

---

## 6. File Index Nhanh

| File | Mô tả | Khi nào đọc |
|---|---|---|
| `AGENTS.md` | File này — System context | Luôn luôn |
| `PROJECT_GUIDELINES.md` | Kiến trúc OOP, KPIs, roadmap | Trước mọi task |
| `docs/MDP_SPEC.md` | Đặc tả toán học State/Action/Reward | Khi viết env hoặc reward |
| `env/cityflow_wrapper.py` | `TrafficEnv` chính | Khi sửa môi trường |
| `agents/node_agent.py` | `NodeAgent` với Decoupled Net | Khi sửa agent logic |
| `federated/server.py` | `FederatedServer` FedAvg/FedProx | Khi sửa FL pipeline |
| `notebooks/02_cityflow_fl_integration.ipynb` | Demo tích hợp đầy đủ | Khi cần ví dụ chạy được |

---

*Cập nhật lần cuối: 2026-11-07*
