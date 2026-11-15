# Hướng dẫn tích hợp thuật toán RL vào bộ khung FedMARL-ATSC

> Dành cho người viết thuật toán RL (phần MARL). Mã nguồn tham chiếu số mục `§2` và `§3` của tài liệu này
> (`agents/node_agent.py`, `agents/networks.py`, `agents/__init__.py`) — giữ nguyên đánh số.

## 1. Tổng quan

Ba lớp tách biệt; thuật toán RL chỉ nằm ở lớp con của `NodeAgent`:

```
TrafficEnv / MockTrafficEnv  ──obs, reward──►  NodeAgent (lớp con = thuật toán)  ──extract_weights()──►  FederatedServer
        ▲                                         select_action / store_transition /                         │
        └──────────── action ◄────────────────── train_local_model                                           │
                                                 ◄──────────── load_global_weights() ◄───────────────────────┘
```

Vòng lặp chính chỉ gọi ba hàm của agent:

```python
action = agent.select_action(state)
agent.store_transition(state, action, reward, next_state, done)
loss = agent.train_local_model()          # None nếu chưa đủ dữ liệu
```

Mẫu tối giản: `agents/q_learning.py`. Mẫu đầy đủ (replay + target net + ε giảm dần): `agents/dqn.py`.

## 2. Viết một thuật toán mới

Tạo `agents/<ten_thuat_toan>.py` với lớp con của `NodeAgent`.

| Hook | Bắt buộc? | Ghi chú |
|---|---|---|
| `compute_loss(batch) -> Tensor` | **Có** | Loss vô hướng còn gắn đồ thị tính toán. Lớp nền ném `NotImplementedError`. |
| `select_action(state, greedy)` | Nên | Thêm khám phá; nhánh tham lam gọi `super().select_action(state, greedy=True)`. Dùng `self.rng` (không dùng `np.random` toàn cục). |
| `store_transition(...)`, `sample_batch()` | Nếu off-policy | Mặc định là hàng chờ on-policy dùng một lần. Override `store_transition` thì **phải** gọi `self._register_local_samples()` cho mỗi transition, nếu không `n_k = 0`. |
| `build_network()`, `build_optimizer()` | Tuỳ chọn | Xem §3. |
| `train_local_model()` | Chỉ để bọc thêm | Gọi `super()`. Lớp nền đã làm: sample → loss → **số hạng FedProx** → backward → clip → step. |
| `on_global_weights_loaded()` | Nếu có trạng thái phụ thuộc backbone | Ví dụ target network (xem §4). |

**Không được override:** `extract_weights`, `load_global_weights`, `get_shared_weights`, `set_shared_weights`,
`num_local_samples`, `local_weights_norm`. (Không có cơ chế ép buộc; đây là quy ước — R-02.)

Bẫy thường gặp:

- `build_network()` được gọi **bên trong** `NodeAgent.__init__`. Thuộc tính mà `build_network`/`build_optimizer` cần phải
  gán **trước** `super().__init__()`. Các tham số thuật toán khác (ε, γ, ...) gán sau.
- **`done` và truncation.** API chỉ có một cờ `done`, và DQN dùng `(1 − done)` khi bootstrap. Episode kết thúc vì hết giờ
  (truncated) thì truyền `done=False`; chỉ truyền `True` khi trạng thái là terminal thật.
- **Hành động hiệu lực.** Lớp an toàn có thể ghi đè hành động của agent (xanh tối thiểu, pha vàng, ...). Với off-policy,
  lưu hành động thực sự đã chạy hoặc xử lý trường hợp bị chặn — đừng gán thưởng cho hành động chưa từng xảy ra.
- `TransitionBatch` cố định: `states, actions, rewards, next_states, dones`. Thuật toán cần thêm (log-prob, value, mask)
  phải tự giữ bộ nhớ qua `store_transition`/`sample_batch`.
- Hằng số thuật toán (`reward_scale`, ε, γ) nằm ở lớp con, không đặt vào lớp nền.

## 3. Thay kiến trúc mạng

`TrafficAgentNet` có đúng hai sub-module với tên cố định: `shared_feature_extractor` (liên bang) và
`local_policy_head` (riêng tư). Override `build_network()` để đổi kiến trúc nhưng:

- Mạng trả về **bắt buộc** có hai sub-module trên (kiểm tra ở `_check_decoupled_architecture`). Mọi sub-module khác
  (critic, value head, ...) tự động là riêng tư.
- Không gộp backbone và head vào một `nn.Sequential`.
- Không dùng `BatchNorm` trong backbone: `running_mean/var` và `num_batches_tracked` sẽ bị FedAvg ép sang float. Dùng `LayerNorm`.
- Đầu ra `local_policy_head` coi là "action scores" (Q-values hoặc logits tuỳ thuật toán).
- Actor-critic: `TrafficAgentNet(with_value=True)` và gọi `forward_with_value`; `action_scores` chỉ dùng policy head.

Kiểm tra bắt buộc sau mọi thay đổi liên quan mạng/agent:

```python
sw = agent.extract_weights()
assert not any("local_policy_head" in k for k in sw)
assert sw, "Backbone rỗng"
```

## 4. Biên giới FL — phần MARL chịu trách nhiệm

| Chủ đề | Quy ước hiện tại |
|---|---|
| **n_k** | Số transition đã lưu kể từ lần nhận global weights gần nhất (`num_local_samples`); `load_global_weights` đặt lại về 0. Truyền `sample_counts=[a.num_local_samples ...]` vào `FederatedServer.run_round`. |
| **Target network** | `load_global_weights` gọi hook `on_global_weights_loaded()`. `DQNAgent` ghi đè để chép backbone mới sang target net. Thuật toán khác có trạng thái phụ thuộc backbone cũng phải ghi đè hook này. |
| **Optimizer** | Mặc định giữ moment Adam qua vòng FL. Đặt `reset_optimizer_on_sync=True` để dựng lại optimizer sau mỗi lần nhận global weights. |
| **FedProx** | Server chỉ chạy FedAvg. Số hạng proximal `μ/2·‖w_backbone − w_global‖²` nằm ở client: gọi `agent.set_fedprox_mu(mu)`; lớp nền cộng nó vào loss ở `train_local_model` cho mọi thuật toán. Neo `w_global` được lưu mỗi lần `load_global_weights`. |
| **Khởi tạo vòng 0** | Phát **một** backbone chung cho mọi agent (aggregate một client rồi `broadcast`), không lấy trung bình các khởi tạo ngẫu nhiên khác nhau. |
| **metadata()** | Hiện gồm `obs_dim`, `num_phases`, `avg_loss`. `avg_loss` phụ thuộc thuật toán/thang reward nên chưa phù hợp để phân cụm; cần bổ sung thống kê lưu lượng (việc chung với người phụ trách FL). |

## 5. Tái lập kết quả

- Mỗi agent có `self.rng = np.random.default_rng(seed)` cho khám phá; `DQNAgent(seed=...)` cũng seed replay buffer.
- Khởi tạo trọng số: gọi `torch.manual_seed(...)` **trước** khi tạo agent.
- Môi trường: `env.reset(seed=...)`. Đánh giá dùng seed cố định (`EVAL_SEED_BASE` trong `train_fedmarl.py`).
- Cùng cấu hình + cùng seed ⇒ cùng kết quả (có test).

## 6. Bố cục vector state

Một hợp đồng duy nhất — `env/obs_layout.py::ObsLayout`:

```
[ one-hot pha × P | elapsed × 1 | queue × L | speed × L | density × L ]
```

`elapsed` = thời gian xanh đã trôi qua / xanh tối thiểu, cắt ở 1. Khi đang vàng / all-red: one-hot toàn 0 và elapsed = 0.
`agents.behavior_cloning.ObservationLayout` đọc đúng bố cục này (có test đối chiếu). Phía CityFlow phải dựng state theo
cùng thứ tự. Neighbor embedding (MDP_SPEC §2.2) chưa triển khai; sẽ nối thêm vào cuối.

## 7. Thí nghiệm và kiểm thử

```bash
python -m pytest tests -q
python train_fedmarl.py --hetero 1 --episodes 40 --tag het   # independent / fedavg / fedprox + 2 baseline
python train_fedmarl.py --hetero 0 --episodes 40 --tag hom   # đối chứng: giao lộ giống nhau
```

Kết quả ghi `results/<tag>_{train,eval,summary}.csv`. So sánh `hetero=1` với `hetero=0`: nếu FL chỉ kém độc lập khi
`hetero=1`, đó là bằng chứng negative transfer.

## 8. Thuật toán policy-gradient (PPO) khác gì

- Cần value head (`with_value=True`) và bộ nhớ on-policy riêng giữ `log_prob`, `value` (override `store_transition`/`sample_batch`).
- Đầu ra head là logits; `select_action` lấy mẫu từ phân phối thay vì ε-greedy; cân nhắc action masking từ lớp an toàn.
- Critic nằm ngoài backbone nên ở lại node, nhưng phụ thuộc backbone chung: sau `load_global_weights` có thể cần
  fine-tune critic ngắn (dùng hook `on_global_weights_loaded`).
