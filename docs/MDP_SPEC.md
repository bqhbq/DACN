# docs/MDP_SPEC.md
# Đặc tả Mô hình Toán học RL — FedMARL-ATSC

> **Phiên bản**: 1.0 · **Ngày**: 2026-11-07
> Tài liệu này định nghĩa chính xác không gian trạng thái, hành động và hàm phần thưởng
> cho mọi agent trong hệ thống. **Mọi implementation phải tuân thủ spec này.**

---

## 1. Tổng quan — Decentralized MDP

Mỗi giao lộ $i$ được mô hình hóa như một **Node-Agent** trong bài toán **Decentralized Partially Observable MDP (Dec-POMDP)**:

$$\mathcal{M}_i = \langle \mathcal{S}_i,\ \mathcal{A}_i,\ \mathcal{T}_i,\ \mathcal{R}_i,\ \gamma \rangle$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{S}_i$ | Không gian trạng thái cục bộ + lân cận của giao lộ $i$ |
| $\mathcal{A}_i$ | Không gian hành động (rời rạc) của giao lộ $i$ |
| $\mathcal{T}_i$ | Hàm chuyển trạng thái (CityFlow Engine) |
| $\mathcal{R}_i$ | Hàm phần thưởng cục bộ |
| $\gamma \in [0,1)$ | Hệ số chiết khấu (mặc định $\gamma = 0.99$) |

Agent $i$ **không quan sát** trực tiếp trạng thái toàn hệ thống — chỉ thấy giao lộ mình và các giao lộ lân cận (partial observability).

---

## 2. Không gian Trạng thái — $\mathcal{S}$

Vector trạng thái $s_i^t$ tại bước thời gian $t$ được ghép từ hai thành phần:

$$s_i^t = \left[ s_i^{\text{local}},\ s_i^{\text{neighbor}} \right]$$

### 2.1 Local State — $s_i^{\text{local}}$

| Đặc trưng | Ký hiệu | Mô tả | Chuẩn hoá | Dim |
|---|---|---|---|---|
| **Phase encoding** | $\phi_i^t$ | One-hot vector pha đèn hiện tại | $\{0,1\}^{N_\phi}$ | $N_\phi$ |
| **Elapsed time** | $\tau_i^t$ | Thời gian đã trôi qua của pha hiện tại | $[0, 1]$ so với max cycle | $1$ |
| **Queue length** | $q_l^t$ | Số xe chờ trên mỗi làn xe vào $l$ | $[0, 1]$ so với $Q_{\max}$ | $N_{\text{lanes}}$ |
| **Approach flow** | $f_l^t$ | Số xe tiếp cận mỗi làn trong $\Delta t$ giây | $[0, 1]$ so với $F_{\max}$ | $N_{\text{lanes}}$ |
| **Average speed** | $v_l^t$ | Tốc độ trung bình xe trên làn $l$ | $[0, 1]$ so với $v_{\max}$ | $N_{\text{lanes}}$ |

Với `sample_map_2` (28 làn, 4 pha):
$$\dim(s_i^{\text{local}}) = 4 + 1 + 28 + 28 + 28 = 89$$

> **Implementation hiện tại** (Phase 1 — simplified): Dùng 3 đặc trưng mỗi làn (queue, speed, density), bỏ qua elapsed time và approach flow.
> $$\dim(s_i^{\text{local}}) = 28 \times 3 = 84$$

### 2.2 Neighboring State — $s_i^{\text{neighbor}}$

Để Agent $i$ có "tầm nhìn toàn cục" và phòng chống **Queue Spillback** (ùn tắc lan truyền từ giao lộ này sang giao lộ khác), agent nhận **embedding** từ các giao lộ lân cận:

$$s_i^{\text{neighbor}} = \text{Aggregate}\left( \left\\{ e_j^t \mid j \in \mathcal{N}(i) \right\\} \right)$$

Trong đó:
- $\mathcal{N}(i)$: Tập giao lộ kề cận (chia sẻ làn đường trực tiếp)
- $e_j^t = \text{Backbone}(s_j^{\text{local}})$: Embedding từ `shared_feature_extractor` của agent $j$
- $\text{Aggregate}(\cdot)$: Hàm tổng hợp — Mean-pooling hoặc Attention

> **Phase 1 — Chưa triển khai**: `s_i^{\text{neighbor}}` được đặt bằng vector 0 hoặc bỏ qua. Sẽ triển khai ở Phase 2.

### 2.3 Tóm tắt State Vector

```
s_i^t = [
    # Local (84 dims — Phase 1)
    queue_norm[0..27],       # 28 dims — chiều dài hàng đợi mỗi làn
    speed_norm[0..27],       # 28 dims — tốc độ TB mỗi làn
    density_norm[0..27],     # 28 dims — mật độ mỗi làn

    # Neighbor (Phase 2 — TBD)
    # neighbor_embedding[0..63]  # 64 dims — backbone output của giao lộ lân cận
]
```

**Giá trị chuẩn hoá**: Mọi feature trong $[0, 1]$ bằng min-max scaling với ngưỡng:
- $Q_{\max} = 50$ xe/làn
- $v_{\max} = 16.67$ m/s (~60 km/h)
- $\rho_{\max} = 50$ xe/làn

---

## 3. Không gian Hành động — $\mathcal{A}$

### 3.1 Phase-based Action (Hiện tại — Phase 1)

Agent chọn **trực tiếp** pha xanh muốn kích hoạt:

$$a_i^t \in \{0, 1, 2, \ldots, N_\phi - 1\}$$

Với `sample_map_2`: $|\mathcal{A}_i| = 4$ (4 pha xanh, index CityFlow: 0, 2, 4, 6).

### 3.2 Binary Keep/Change Action (Tuỳ chọn — Phase 2)

$$a_i^t \in \{0, 1\}$$

| Action | Ý nghĩa |
|---|---|
| $0$ | **Keep** — Giữ nguyên pha hiện tại thêm một step |
| $1$ | **Change** — Chuyển sang pha kế tiếp trong chu trình |

### 3.3 Safety Wrapper — Ràng buộc An toàn Bắt buộc

> [!IMPORTANT]
> Mạng Nơ-ron **KHÔNG** trực tiếp điều khiển đèn vàng hay đèn đỏ toàn hướng.
> `TrafficEnv` tự động xử lý các chuyển tiếp này.

Các ràng buộc được `TrafficEnv` thực thi **ngầm** (transparent to Agent):

| ID | Ràng buộc | Giá trị mặc định | Cơ chế |
|---|---|---|---|
| C-01 | **Valid Phase Order** | Phải theo chu trình đã định | Clamp action vào $[0, N_\phi)$ |
| C-02 | **Minimum Green Time** | $T_{\min} = 10$ giây | Action Masking — giữ pha nếu chưa đủ |
| C-03 | **Yellow Interval** | $T_{\text{yellow}} = 3$ giây | Tự chèn trước khi đổi pha |
| C-04 | **All-Red Interval** | $T_{\text{allred}} = 1$ giây (tuỳ chọn) | Tự chèn sau đèn vàng |
| C-05 | **No Phase Skipping** | Chỉ chuyển tuần tự | Clamped action index |

**Sơ đồ chuyển tiếp pha** (ví dụ chuyển từ pha 0 → pha 2):

```
Agent đề xuất: a = 1 (pha xanh 1, CF index 2)
                │
                ▼
TrafficEnv kiểm tra C-02: Đã đủ min_green?
  Chưa đủ ─────────────► Giữ pha 0, bỏ qua yêu cầu
  Đủ rồi ──────────────► Tiếp tục
                │
                ▼
Chèn Yellow Phase (CF index 1):  3 giây × engine.next_step()
                │
                ▼
Set Green Phase 2 (CF index 2): 30 giây × engine.next_step()
                │
                ▼
Trả về (obs, reward, ...) cho Agent
```

---

## 4. Hàm Phần thưởng — $\mathcal{R}$

### 4.1 Max-Pressure Reward (Công thức chính)

Dựa trên lý thuyết Max-Pressure Control (Varaiya, 2013; Wei et al., KDD 2019):

**Áp lực giao lộ** $P_i^t$ được định nghĩa là tổng hiệu số hàng đợi vào và ra:

$$P_i^t = \sum_{l \in \mathcal{L}_i^{\text{in}}} q_l^t - \sum_{l \in \mathcal{L}_i^{\text{out}}} q_l^t$$

Hoặc dạng tuyệt đối (robust hơn với outgoing lanes có thể không quan sát được):

$$P_i^t = \left| \sum_{l \in \mathcal{L}_i^{\text{in}}} q_l^t - \sum_{l \in \mathcal{L}_i^{\text{out}}} q_l^t \right|$$

**Reward cơ bản** (cực tiểu hoá áp lực ≡ tối đa hoá throughput):

$$\boxed{R_i^t = -P_i^t}$$

**Lý do chọn Max-Pressure**: Thuật toán Max-Pressure control được chứng minh là tối ưu throughput-maximal trong điều kiện lý tưởng. Dùng nó làm reward signal giúp Agent học được chính sách gần tối ưu ngay cả với deep RL.

### 4.2 Composite Reward (Khuyến nghị)

Reward đầy đủ bao gồm thêm các thành phần phụ:

$$R_i^t = \underbrace{-\alpha \cdot P_i^t}_{\text{Max-Pressure}} + \underbrace{\beta \cdot \Delta \text{Throughput}_i^t}_{\text{Throughput bonus}} - \underbrace{\gamma_{\text{phase}} \cdot \mathbb{1}[\text{phase changed}]}_{\text{Phase-change penalty}}$$

| Thành phần | Ký hiệu | Vai trò | Giá trị mặc định |
|---|---|---|---|
| **Max-Pressure penalty** | $-\alpha \cdot P_i^t$ | Chính — cực tiểu hoá áp lực | $\alpha = 0.6$ |
| **Throughput bonus** | $\beta \cdot \Delta\text{Through}$ | Khuyến khích thông xe | $\beta = 0.4$ |
| **Phase-change penalty** | $-\gamma \cdot \mathbb{1}[\text{change}]$ | Tránh đổi pha quá nhanh | $\gamma = 0.05$ |

**Lý do thêm Phase-change penalty:**
- Mỗi lần đổi pha tốn $T_{\text{yellow}} = 3$ giây không thông xe
- Agent cần học khi nào nên chịu chi phí chuyển pha để thu về lợi ích lớn hơn
- Tránh "phase hunting" — oscillation liên tục giữa các pha

### 4.3 Phần thưởng Toàn hệ thống (Global Reward — dùng để đánh giá)

Mặc dù mỗi agent tối ưu hoá reward cục bộ $R_i^t$, KPI đánh giá toàn hệ thống là:

$$R_{\text{global}}^t = \frac{1}{N} \sum_{i=1}^{N} R_i^t$$

Giảm thiểu tổng thời gian chờ toàn hệ thống:

$$\text{AWT} = \frac{1}{|V|} \sum_{v \in V} W_v \quad \text{(Average Waiting Time)}$$

---

## 5. Chu kỳ Mô phỏng — Timing Model

```
Mỗi action step của Agent = 1 "decision point"

Timeline:
  t=0        t=3s       t=33s      t=36s      t=66s
   │          │          │          │          │
   ├──Green₀──┼──Yellow──┼──Green₁──┼──Yellow──┼──...
   │  30 sim  │  3 sim   │  30 sim  │  3 sim   │
   │  steps   │  steps   │  steps   │  steps   │
   │          │          │          │          │
   Agent      Env auto   Agent      Env auto
   action     inserts    action     inserts
   a₀=0       yellow     a₁=1       yellow
```

| Tham số | Giá trị | Đơn vị |
|---|---|---|
| Simulation interval | 1.0 | giây/step CityFlow |
| Green duration | 30 | giây (mặc định) |
| Yellow duration | 3 | giây (C-03) |
| All-red duration | 0 | giây (tắt mặc định) |
| Episode length | 3600 | giây (1 giờ mô phỏng) |
| Max action steps / episode | ≈ 109 | steps (3600 / 33) |

---

## 6. Federated Learning Integration

### 6.1 Thời điểm FL

```
for episode in range(num_episodes):
    for step in range(max_steps):
        obs = env.get_state()
        actions = {i: agent_i.select_action(obs_i) for i in intersections}
        obs', reward, ... = env.step(actions)
        buffer.push(obs, actions, reward, obs')
        loss = dqn_trainer.update()     ← Local RL update

    if episode % FL_INTERVAL == 0:     ← FL round sau mỗi N episodes
        weights = [a.get_shared_weights() for a in agents]
        server.aggregate(weights)       ← FedAvg
        server.broadcast()              ← Cập nhật backbone
```

### 6.2 Công thức FedAvg

$$w^* = \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j} \cdot w_k^{\text{backbone}}$$

Trong đó:
- $K$: Số agents tham gia vòng FL
- $n_k$: Số transition trong replay buffer của agent $k$ (ước lượng dataset size)
- $w_k^{\text{backbone}}$: Trọng số `shared_feature_extractor` của agent $k$

---

## 7. Các KPIs Đánh giá

### 7.1 KPIs Hiệu suất Giao thông

| KPI | Công thức | Đơn vị | Mục tiêu |
|---|---|---|---|
| **Average Waiting Time (AWT)** | $\frac{1}{\|V\|}\sum_{v} W_v$ | giây/xe | ↓ |
| **Average Queue Length (AQL)** | $\frac{1}{N \cdot L}\sum_{i,l} q_{il}$ | xe/làn | ↓ |
| **Throughput** | $\frac{\|V_{\text{exit}}\|}{\Delta T}$ | xe/giờ | ↑ |
| **Average Travel Time** | $\bar{T} = \frac{1}{\|V\|}\sum_v T_v$ | giây | ↓ |

### 7.2 KPIs An toàn Vận hành

| KPI | Mô tả |
|---|---|
| `safety_violation_rate` | % bước Agent đề xuất action vi phạm (trước khi TrafficEnv chặn) |
| `yellow_compliance_rate` | % lần chuyển pha có đủ yellow interval |
| `min_green_violation_rate` | % lần vi phạm minimum green time |

### 7.3 KPIs Federated Learning

| KPI | Mô tả |
|---|---|
| `convergence_rounds` | Số vòng FL để đạt 95% hiệu suất baseline |
| `negative_transfer_rate` | % agents có hiệu suất giảm sau FL round |
| `backbone_divergence` | Độ khác biệt backbone weights giữa agents trước/sau FL |

---

## 8. Tài liệu Tham khảo

1. **FedAvg**: McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
2. **FedProx**: Li et al. (2020). *Federated Optimization in Heterogeneous Networks.* MLSys.
3. **Max-Pressure**: Varaiya, P. (2013). *Max pressure control of a network of signalized intersections.* Transportation Research Part C.
4. **PressLight**: Wei et al. (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network.* KDD.
5. **CityFlow**: Zhang et al. (2019). *CityFlow: A Multi-Agent Traffic Simulation Platform for Large-Scale and Realistic Traffic Scenarios.* WWW.
6. **CoLight**: Wei et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control.* CIKM.

---

*Tài liệu này là "source of truth" cho mọi implementation liên quan đến MDP. Mọi thay đổi phải được review và cập nhật đồng thời.*
