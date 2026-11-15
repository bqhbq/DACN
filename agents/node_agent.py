"""
agents/node_agent.py
--------------------
Lớp NodeAgent — Agent RL tại một giao lộ, tuân thủ PROJECT_GUIDELINES.md §6.2

Kiến trúc Phân rã BẮT BUỘC:
  shared_feature_extractor  → phần liên bang (gửi lên FederatedServer)
  local_policy_head         → phần riêng tư (TUYỆT ĐỐI không rời khỏi node)

Lớp này chứa: mô hình PyTorch, optimizer, forward pass, huấn luyện Behavior
Cloning theo mô hình "Thầy - Trò" (LLM làm thầy), trích xuất weights.
Lớp này KHÔNG chứa: logic CityFlow, replay buffer, lịch sử episode.

Mô hình Thầy - Trò:
  Thầy (LLM: Gemini / OpenAI) đọc state dạng văn bản → trả về KEEP (0) / CHANGE (1)
  Trò  (TrafficAgentNet)      học bắt chước bằng CrossEntropyLoss trên nhãn của thầy
"""
from __future__ import annotations

import copy
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import TrafficAgentNet

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình LLM
# ──────────────────────────────────────────────────────────────────────────────

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

LLM_PROVIDERS: Dict[str, str] = {"gemini": "Gemini", "openai": "OpenAI"}  # id → tên hiển thị
LLM_API_KEY_ENV: Dict[str, str] = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
# Model mặc định; ghi đè bằng biến môi trường GEMINI_MODEL / OPENAI_MODEL trong .env.
LLM_MODEL_ENV: Dict[str, str] = {"gemini": "GEMINI_MODEL", "openai": "OPENAI_MODEL"}
# Các model Gemini miễn phí, xoay vòng theo thứ tự ưu tiên: model đầu là model
# chính; khi nó hết quota (429) thì chuyển ngay sang model kế tiếp.
# GEMINI_MODEL trong .env (nếu có) được đặt lên đầu danh sách.
GEMINI_FREE_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]
LLM_DEFAULT_MODEL: Dict[str, str] = {"gemini": GEMINI_FREE_MODELS[0], "openai": "gpt-4o-mini"}
LLM_TIMEOUT_S = 15.0  # không để một lời gọi API treo vòng lặp mô phỏng

# Mã lỗi HTTP khiến một model Gemini bị coi là tạm thời không dùng được:
# 429 hết quota, 404 model không tồn tại với key này, 5xx/504 server quá tải.
GEMINI_ROTATE_ON_CODES = frozenset({429, 404, 500, 503, 504})
GEMINI_MODEL_COOLDOWN_S = 60.0   # sau lỗi trên, bỏ qua model này trong bấy nhiêu giây
LLM_MIN_CALL_INTERVAL_S = 1.0    # giãn cách tối thiểu giữa hai lời gọi API (mọi agent)

ACTION_KEEP = 0    # giữ pha hiện tại
ACTION_CHANGE = 1  # chuyển sang pha kế tiếp trong chu trình
FALLBACK_ACTION = ACTION_KEEP  # dùng khi LLM lỗi / trả sai định dạng

# Bóc tách câu trả lời của LLM
HIDDEN_CHARS_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\x00-\x08\x0b-\x1f]")  # zero-width, BOM, control
ACTION_DIGIT_RE = re.compile(r"(?<!\d)[01](?!\d)")  # chữ số 0/1 không dính vào số khác

# Layout của observation (docs/MDP_SPEC.md §2): [queue × L, speed × L, density × L]
FEATURES_PER_LANE = 3
Q_MAX = 50.0  # ngưỡng chuẩn hoá queue — phải khớp với phía môi trường

GRAD_CLIP_NORM = 10.0

# Huấn luyện cục bộ không cần LLM (Q-learning một bước, docs/MDP_SPEC.md §1, §4)
GAMMA = 0.99          # hệ số chiết khấu γ (MDP_SPEC §1)
REWARD_SCALE = 0.1    # co reward (đơn vị: số xe) về miền nhỏ để TD-target ổn định
LLM_DISABLED = "none" # giá trị llm_provider khi không dùng LLM


class NodeAgent:
    """
    Agent điều khiển tín hiệu tại một giao lộ duy nhất.

    Tham số
    -------
    intersection_id : ID của giao lộ (phải khớp với ID trong CityFlow roadnet)
    obs_dim         : Kích thước vector observation (num_incoming_lanes × 3)
    num_phases      : Số pha xanh khả dụng (action space size)
    hidden_dim      : Độ rộng lớp ẩn trong backbone (mặc định 128)
    backbone_out    : Output dim của backbone — chiều embedding được chia sẻ (mặc định 64)
    lr              : Learning rate cho optimizer cục bộ (mặc định 1e-3)
    epsilon         : Xác suất khám phá ε-greedy (mặc định 0.1)
    device          : torch.device để chạy model
    llm_provider    : LLM đóng vai "Thầy": "gemini" (mặc định) hoặc "openai"
    approach_names  : Tên các hướng vào giao lộ theo đúng thứ tự làn trong obs
                      (vd. ["West", "South", "East", "North"]); các làn được chia
                      đều cho các hướng. None → prompt liệt kê theo từng làn.
    phase_descriptions : Mô tả ngắn từng pha xanh (vd. "West/East straight"),
                      giúp LLM hiểu pha nào phục vụ hướng nào. None → bỏ qua.
    """

    # Trạng thái dùng CHUNG cho mọi NodeAgent, vì quota tính theo API key chứ
    # không theo từng giao lộ.
    _gemini_cooldown_until: Dict[str, float] = {}   # model → mốc time.monotonic() được dùng lại
    _last_llm_call_ts: float = 0.0                  # mốc lời gọi API gần nhất

    def __init__(
        self,
        intersection_id: str,
        obs_dim:         int,
        num_phases:      int,
        hidden_dim:      int   = 128,
        backbone_out:    int   = 64,
        lr:              float = 1e-3,
        epsilon:         float = 0.1,
        device:          Optional[torch.device] = None,
        llm_provider:    str   = "gemini",
        approach_names:  Optional[List[str]] = None,
        phase_descriptions: Optional[List[str]] = None,
    ) -> None:
        if llm_provider not in LLM_PROVIDERS and llm_provider != LLM_DISABLED:
            raise ValueError(
                f"llm_provider phải thuộc {sorted(LLM_PROVIDERS)} hoặc {LLM_DISABLED!r}, "
                f"nhận {llm_provider!r}."
            )
        num_lanes = obs_dim // FEATURES_PER_LANE
        if approach_names and num_lanes % len(approach_names) != 0:
            raise ValueError(
                f"{num_lanes} làn không chia đều được cho {len(approach_names)} hướng."
            )
        if phase_descriptions and len(phase_descriptions) != num_phases:
            raise ValueError("phase_descriptions phải có đúng num_phases phần tử.")

        self.intersection_id = intersection_id
        self.obs_dim         = obs_dim
        self.num_phases      = num_phases
        self.epsilon         = epsilon
        self.device          = device or torch.device("cpu")

        # ── Mạng Nơ-ron phân rã (Decoupled Architecture) ─────────────────────
        self.net = TrafficAgentNet(
            obs_dim      = obs_dim,
            hidden_dim   = hidden_dim,
            backbone_out = backbone_out,
            num_phases   = num_phases,
        ).to(self.device)

        # ── Optimizer cục bộ — tối ưu hoá TOÀN BỘ network (cả backbone lẫn head) ─
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self._loss_history: List[float] = []

        # ── LLM cho Behavior Cloning ──────────────────────────────────
        self.llm_provider       = llm_provider
        self.approach_names     = approach_names
        self.phase_descriptions = phase_descriptions
        self._num_lanes         = num_lanes
        self._bc_criterion      = nn.CrossEntropyLoss()
        # n_k trong FedAvg (MDP_SPEC §6.2): số mẫu đã học kể từ lần đồng bộ gần nhất
        self._local_samples     = 0
        self._llm_ok            = False   # lời gọi LLM gần nhất có thành công không
        self.llm_calls          = 0
        self.llm_failures       = 0
        self._llm_model_name    = ""
        # Chỉ client của provider đang chọn được tạo; client còn lại là None.
        self.gemini_client: Any = None   # google.genai.Client
        self.openai_client: Any = None   # openai.OpenAI
        self._init_llm_client()

        logger.info(
            "NodeAgent [%s] khởi tạo: obs_dim=%d, num_phases=%d, device=%s, llm=%s",
            intersection_id, obs_dim, num_phases, self.device, llm_provider,
        )

    def _init_llm_client(self) -> None:
        """
        Nạp API key từ `.env` và tạo client cho provider đã chọn.

        SDK chỉ được import ở đây (lazy) nên chỉ cần cài thư viện của provider
        đang dùng. Mọi lỗi (thiếu key, thiếu thư viện) đều không làm crash:
        agent vẫn chạy, `_get_llm_action()` sẽ trả về hành động dự phòng.

        Kết quả nằm ở `self.gemini_client` (SDK `google-genai`) hoặc
        `self.openai_client`; giữ nguyên None nếu không khởi tạo được.
        """
        if self.llm_provider == LLM_DISABLED:
            return  # chế độ huấn luyện cục bộ: không cần API key hay SDK nào

        from dotenv import find_dotenv, load_dotenv

        # Ưu tiên .env ở gốc repo (suy ra từ vị trí file này, vd. /workspace/.env
        # trong container); nếu không có thì dò ngược từ thư mục đang chạy.
        env_path = ENV_FILE if ENV_FILE.is_file() else Path(find_dotenv(usecwd=True) or ENV_FILE)
        loaded = load_dotenv(dotenv_path=env_path)
        logger.debug("[%s] load_dotenv(%s) → %s", self.intersection_id, env_path, loaded)
        display = LLM_PROVIDERS[self.llm_provider]
        key_name = LLM_API_KEY_ENV[self.llm_provider]
        api_key = os.getenv(key_name)
        self._llm_model_name = os.getenv(
            LLM_MODEL_ENV[self.llm_provider], LLM_DEFAULT_MODEL[self.llm_provider]
        )
        # Thứ tự xoay vòng model Gemini: model chính trước, bỏ trùng lặp.
        self._gemini_models: List[str] = list(
            dict.fromkeys([self._llm_model_name, *GEMINI_FREE_MODELS])
        )
        if not api_key:
            logger.warning(
                "[%s] Thiếu %s trong %s — LLM [%s] bị vô hiệu, dùng hành động dự phòng.",
                self.intersection_id, key_name, ENV_FILE.name, display,
            )
            return

        try:
            if self.llm_provider == "gemini":
                from google import genai
                from google.genai import types

                self.gemini_client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_S * 1000)),  # ms
                )
            else:
                from openai import OpenAI

                self.openai_client = OpenAI(
                    api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=1
                )
        except Exception as exc:  # thiếu thư viện, key sai định dạng, ...
            print(f"[LLM Error - {self.intersection_id}] {type(exc).__name__}: {str(exc)}")
            logger.warning(
                "[%s] Không khởi tạo được LLM [%s].", self.intersection_id, display
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Chọn hành động (Action Selection)
    # ──────────────────────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Chọn hành động theo chiến lược ε-greedy.

        Tham số
        -------
        obs    : Observation vector shape (obs_dim,)
        greedy : Nếu True → luôn chọn argmax Q (không khám phá)

        Trả về
        ------
        action_idx : int ∈ [0, num_phases)
        """
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_phases))

        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q_vals = self.net(obs_t)                 # shape (1, num_phases)
        return int(q_vals.squeeze(0).argmax().item())

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Trả về Q-values shape (num_phases,) dưới dạng numpy."""
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            q = self.net(obs_t).squeeze(0)
        return q.cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Huấn luyện cục bộ (Local Training — Dummy Loss cho demo tích hợp)
    # ──────────────────────────────────────────────────────────────────────────

    def compute_dummy_loss(self, obs: np.ndarray) -> float:
        """
        Tính toán một dummy MSE loss để kiểm tra luồng backward() hoạt động.

        Trong hệ thống thật: hàm này sẽ được thay bằng DQN TD-error hoặc PPO
        policy gradient do module agents/dqn.py hoặc agents/ppo.py cung cấp.

        Tham số
        -------
        obs : Observation vector shape (obs_dim,) lấy từ TrafficEnv.get_state()

        Trả về
        ------
        float : Giá trị loss sau bước backward()
        """
        self.net.train()
        self.optimizer.zero_grad()

        obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.net(obs_t)                            # (1, num_phases)

        # Dummy target: zero Q-values (placeholder cho RL loss thực)
        dummy_target = torch.zeros_like(q_vals)
        loss = nn.functional.mse_loss(q_vals, dummy_target)

        loss.backward()
        # Gradient clipping — thực tiễn tốt cho RL
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self._loss_history.append(loss_val)
        return loss_val

    # ──────────────────────────────────────────────────────────────────────────
    # Behavior Cloning với LLM
    # ──────────────────────────────────────────────────────────────────────────

    def _state_to_text(self, state_tensor: torch.Tensor, current_phase: int = 0) -> str:
        """
        Dịch state thành một prompt tiếng Anh ngắn gọn cho LLM.

        Chỉ dùng phần queue (số xe đang chờ) của state, đã khử chuẩn hoá và
        cộng dồn theo từng hướng vào giao lộ.

        Args:
            state_tensor: Tensor shape (obs_dim,), layout [queue, speed, density] × làn.
            current_phase: Index pha xanh hiện tại, thuộc [0, num_phases).

        Returns:
            Prompt yêu cầu LLM trả về đúng một chữ số: 0 (KEEP) hoặc 1 (CHANGE).
        """
        queues = (state_tensor.flatten()[: self._num_lanes] * Q_MAX).round().int().tolist()

        if self.approach_names:
            per_approach = self._num_lanes // len(self.approach_names)
            waiting = ", ".join(
                f"{name}: {sum(queues[i * per_approach:(i + 1) * per_approach])}"
                for i, name in enumerate(self.approach_names)
            )
        else:
            waiting = ", ".join(f"lane {i}: {q}" for i, q in enumerate(queues))

        next_phase = (current_phase + 1) % self.num_phases
        if self.phase_descriptions:
            phase_text = (
                f"Phase is {current_phase} (green for {self.phase_descriptions[current_phase]}); "
                f"next phase is {next_phase} (green for {self.phase_descriptions[next_phase]})."
            )
        else:
            phase_text = f"Phase is {current_phase}."

        return (
            f"Intersection state: {phase_text} Waiting cars in {waiting}. "
            "Should I KEEP current phase (reply 0) or CHANGE to next phase (reply 1)? "
            "ONLY output a single digit 0 or 1, no explanation."
        )

    def _get_llm_action(self, prompt: str) -> int:
        """
        Hỏi LLM nên KEEP hay CHANGE.

        Toàn bộ lời gọi API được bọc try-except: rớt mạng, hết quota hay LLM trả
        sai định dạng đều KHÔNG làm crash vòng lặp mô phỏng — hàm trả về
        FALLBACK_ACTION (KEEP) và đặt `self._llm_ok = False` để `train_step()`
        biết nhãn này không đến từ LLM.

        Args:
            prompt: Chuỗi sinh bởi `_state_to_text()`.

        Returns:
            0 (KEEP) hoặc 1 (CHANGE).
        """
        display = LLM_PROVIDERS.get(self.llm_provider, self.llm_provider)
        self._llm_ok = False
        self.llm_calls += 1
        raw_text: Optional[str] = None  # khác None ⇔ API đã trả lời thành công
        try:
            client = self.gemini_client if self.llm_provider == "gemini" else self.openai_client
            if client is None:
                raise RuntimeError("LLM client chưa được khởi tạo (thiếu API key hoặc thư viện)")

            if self.llm_provider == "gemini":
                raw_text = self._call_gemini(prompt)
            else:
                self._throttle()
                response = self.openai_client.chat.completions.create(
                    model=self._llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = response.choices[0].message.content or ""

            action = self._parse_llm_reply(raw_text)
        except Exception as exc:
            self.llm_failures += 1
            # In thẳng ra console để thấy lỗi thật ngay cả khi logging chưa được cấu hình.
            print(f"[LLM Error - {self.intersection_id}] {type(exc).__name__}: {str(exc)}")
            if raw_text is not None:  # API gọi được nhưng không bóc ra được 0/1
                print(f"[LLM Format Warning] Không thể parse kết quả: {raw_text}")
            logger.warning(
                "[%s] LLM [%s] lỗi → dùng hành động dự phòng %d (KEEP).",
                self.intersection_id, display, FALLBACK_ACTION,
            )
            return FALLBACK_ACTION

        self._llm_ok = True
        logger.info(
            "[%s] LLM [%s] decided Action: %d (model=%s)",
            self.intersection_id, display, action, self._llm_model_name,
        )
        return action

    def _call_gemini(self, prompt: str) -> str:
        """
        Gọi Gemini với chiến thuật xoay vòng model (fallback).

        Thử lần lượt các model trong `self._gemini_models`. Model nào trả lỗi
        thuộc GEMINI_ROTATE_ON_CODES (vd. 429 hết quota) sẽ bị đưa vào cooldown
        GEMINI_MODEL_COOLDOWN_S giây và hàm chuyển NGAY sang model kế tiếp,
        không chờ. Hết cooldown, model ưu tiên cao hơn được thử lại.

        Args:
            prompt: Chuỗi sinh bởi `_state_to_text()`.

        Returns:
            Văn bản thô Gemini trả về (chuỗi rỗng nếu câu trả lời bị chặn).

        Raises:
            Exception: Lỗi không thuộc diện xoay vòng (vd. sai API key), hoặc
                lỗi cuối cùng khi mọi model đều thất bại / đang cooldown.
        """
        last_exc: Optional[Exception] = None
        for model in self._gemini_models:
            if time.monotonic() < NodeAgent._gemini_cooldown_until.get(model, 0.0):
                continue  # model này vừa lỗi, chưa hết thời gian chờ

            self._throttle()
            try:
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": 0.0},
                )
            except Exception as exc:
                code = getattr(exc, "code", None)
                if code not in GEMINI_ROTATE_ON_CODES:
                    raise
                NodeAgent._gemini_cooldown_until[model] = (
                    time.monotonic() + GEMINI_MODEL_COOLDOWN_S
                )
                print(
                    f"[LLM Fallback - {self.intersection_id}] {model} lỗi {code} "
                    f"→ tạm nghỉ {GEMINI_MODEL_COOLDOWN_S:.0f}s, chuyển sang model kế tiếp."
                )
                last_exc = exc
                continue

            self._llm_model_name = model
            # response.text là None khi câu trả lời bị chặn / rỗng.
            return response.text or ""

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "Mọi model Gemini đang trong thời gian chờ (cooldown) sau lỗi quota/server"
        )

    @staticmethod
    def _throttle() -> None:
        """Ngủ nếu cần để hai lời gọi API liên tiếp cách nhau ≥ LLM_MIN_CALL_INTERVAL_S."""
        wait = LLM_MIN_CALL_INTERVAL_S - (time.monotonic() - NodeAgent._last_llm_call_ts)
        if wait > 0:
            time.sleep(wait)
        NodeAgent._last_llm_call_ts = time.monotonic()

    @staticmethod
    def _parse_llm_reply(raw_text: str) -> int:
        """
        Bóc quyết định 0/1 ra khỏi câu trả lời thô của LLM.

        Chịu được khoảng trắng thừa, ký tự ẩn (zero-width, BOM) và markdown
        (`1`, **1**, "Answer: 1."). Lấy chữ số 0/1 ĐỨNG RIÊNG đầu tiên, nên
        "10" hay "phase 12" không bị hiểu nhầm thành 1.

        Args:
            raw_text: Nội dung văn bản LLM trả về.

        Returns:
            0 (KEEP) hoặc 1 (CHANGE).

        Raises:
            ValueError: Nếu không tìm thấy chữ số 0/1 đứng riêng nào.
        """
        cleaned = HIDDEN_CHARS_RE.sub("", raw_text).strip()
        match = ACTION_DIGIT_RE.search(cleaned)
        if match is None:
            raise ValueError(f"LLM trả về sai định dạng: {raw_text!r}")
        return int(match.group())

    def train_local_step(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
    ) -> float:
        """
        Một bước huấn luyện cục bộ KHÔNG cần LLM: Q-learning một bước.

        Đây là phần "local RL update" trong MDP_SPEC §6.1, ở dạng đơn giản nhất
        (không replay buffer, không target network — bản đầy đủ là agents/dqn.py):

            y    = r · REWARD_SCALE + γ · max_a' Q(s', a')      (TD target)
            loss = Huber(Q(s, a), y)

        Gradient đi qua cả backbone lẫn head, nên `shared_feature_extractor`
        của mỗi node lệch dần theo dữ liệu giao thông riêng của node đó — chính
        là thứ FederatedServer sẽ tổng hợp.

        Args:
            obs: State s tại thời điểm ra quyết định, shape (obs_dim,).
            action: Pha xanh a đã chọn tại s, thuộc [0, num_phases) (MDP_SPEC §3.1).
            reward: Reward r do môi trường tính (Max-Pressure, MDP_SPEC §4).
            next_obs: State s' quan sát được ở lần ra quyết định kế tiếp.

        Returns:
            Giá trị loss của bước này.
        """
        obs_t      = torch.tensor(obs, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        self.net.train()
        q_sa = self.net(obs_t)[0, action]                   # Q(s, a)
        with torch.no_grad():
            td_target = reward * REWARD_SCALE + GAMMA * self.net(next_obs_t).max()
        loss = nn.functional.smooth_l1_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        self._loss_history.append(loss.item())
        self._local_samples += 1
        return loss.item()

    def train_step(self, obs: np.ndarray, current_phase: int = 0) -> int:
        """
        Một bước Behavior Cloning: LLM gán nhãn, mạng học bắt chước.

        Luồng:
            1. state → `_state_to_text()` → `_get_llm_action()` → expert_action (0/1).
            2. Đổi expert_action sang pha đích trong không gian hành động của
               mạng (MDP_SPEC §3.1): KEEP → pha hiện tại, CHANGE → pha kế tiếp.
            3. logits = net(state); loss = CrossEntropyLoss(logits, pha đích).
            4. zero_grad → backward → optimizer.step sinh trọng số mới.

        Nếu LLM lỗi, bước cập nhật trọng số bị BỎ QUA (không học theo nhãn dự
        phòng giả), nhưng hàm vẫn trả về KEEP để mô phỏng chạy tiếp.

        Args:
            obs: Observation vector shape (obs_dim,) do môi trường cung cấp.
            current_phase: Index pha xanh hiện tại, thuộc [0, num_phases).

        Returns:
            expert_action: 0 (KEEP) hoặc 1 (CHANGE) — bên gọi dùng để điều khiển đèn.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)

        # 1. Hỏi LLM
        prompt = self._state_to_text(obs_t, current_phase)
        expert_action = self._get_llm_action(prompt)
        if not self._llm_ok:
            return expert_action

        # 2. Nhãn = pha đích mà LLM muốn bật
        target_phase = self.expert_action_to_phase(expert_action, current_phase)
        label = torch.tensor([target_phase], dtype=torch.long, device=self.device)

        # 3-4. Trò dự đoán và học theo nhãn
        self.net.train()
        logits = self.net(obs_t)                            # (1, num_phases)
        loss = self._bc_criterion(logits, label)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=GRAD_CLIP_NORM)
        self.optimizer.step()

        self._loss_history.append(loss.item())
        self._local_samples += 1
        logger.debug(
            "[%s] BC step: target_phase=%d, loss=%.4f",
            self.intersection_id, target_phase, loss.item(),
        )
        return expert_action

    def expert_action_to_phase(self, expert_action: int, current_phase: int) -> int:
        """
        Đổi hành động KEEP/CHANGE của LLM thành index pha xanh đích.

        Args:
            expert_action: 0 (KEEP) hoặc 1 (CHANGE).
            current_phase: Index pha xanh hiện tại.

        Returns:
            Index pha xanh đích, thuộc [0, num_phases).
        """
        if expert_action == ACTION_CHANGE:
            return (current_phase + 1) % self.num_phases
        return current_phase

    @property
    def last_loss(self) -> Optional[float]:
        """Loss của bước huấn luyện gần nhất; None nếu chưa huấn luyện bước nào."""
        return self._loss_history[-1] if self._loss_history else None

    # ──────────────────────────────────────────────────────────────────────────
    # Federated Learning Interface (R-02, R-03 — quan trọng nhất)
    # ──────────────────────────────────────────────────────────────────────────

    def get_shared_weights(self) -> Dict[str, torch.Tensor]:
        """
        Trích xuất CHỈ trọng số shared_feature_extractor để gửi lên Server.

        ⚠ Quy tắc R-02: TUYỆT ĐỐI không bao gồm local_policy_head.
        ⚠ Quy tắc R-03: local_policy_head không được tham chiếu ở bất kỳ
                         đâu bên ngoài class NodeAgent.

        Trả về
        ------
        dict : state_dict của shared_feature_extractor (deep-copy, CPU tensors)
        """
        # R-02: Chỉ backbone — không có head
        shared_sd = self.net.shared_feature_extractor.state_dict()
        return {k: v.detach().cpu().clone() for k, v in shared_sd.items()}

    def set_shared_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """
        Nạp trọng số backbone đã tổng hợp từ FederatedServer.

        Chỉ cập nhật shared_feature_extractor — local_policy_head không thay đổi.
        Đây là bước Personalisation cốt lõi: head giữ kinh nghiệm cục bộ,
        backbone được cải thiện từ tri thức toàn cục.

        Tham số
        -------
        global_weights : state_dict từ FederatedServer.aggregate()
        """
        # Chuyển lên đúng device trước khi load
        weights_on_device = {k: v.to(self.device) for k, v in global_weights.items()}
        self.net.shared_feature_extractor.load_state_dict(weights_on_device, strict=True)
        self._local_samples = 0  # bắt đầu đếm n_k cho vòng FL kế tiếp
        logger.debug("[%s] Đã nạp global backbone weights.", self.intersection_id)

    def get_backbone_embedding(self, obs: np.ndarray) -> np.ndarray:
        """
        Trả về embedding từ backbone (trước head) — dùng cho neighbor embedding.
        Output shape: (backbone_out,)
        """
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            emb   = self.net.get_backbone_features(obs_t)
        return emb.squeeze(0).cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def num_local_samples(self) -> int:
        """n_k của FedAvg: số mẫu đã học cục bộ kể từ lần nhận global weights gần nhất."""
        return self._local_samples

    def local_head_norm(self) -> float:
        """
        Chuẩn L2 của `local_policy_head` — "dấu vân tay" để kiểm chứng rằng
        head KHÔNG đổi qua một vòng FL, mà không để trọng số head rời node (R-02).
        """
        params = self.net.local_policy_head.parameters()
        return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in params)))

    def metadata(self) -> Dict:
        """Metadata dùng cho clustering tại FederatedServer."""
        return {
            "intersection_id": self.intersection_id,
            "obs_dim":         self.obs_dim,
            "num_phases":      self.num_phases,
            "avg_loss":        float(np.mean(self._loss_history[-10:])) if self._loss_history else 0.0,
        }

    @property
    def loss_history(self) -> List[float]:
        return self._loss_history

    def __repr__(self) -> str:
        return (
            f"NodeAgent(id={self.intersection_id!r}, "
            f"obs_dim={self.obs_dim}, num_phases={self.num_phases}, "
            f"device={self.device})"
        )
