"""
agents/llm_expert.py
--------------------
LLMExpert — "Thầy" trong mô hình Behavior Cloning Thầy - Trò.

Thầy (LLM: Gemini / OpenAI) đọc state dạng văn bản và trả lời KEEP (0) hay
CHANGE (1). Module này gói toàn bộ phần gọi API: nạp key từ `.env`, xoay
vòng model Gemini khi hết quota, giãn cách lời gọi và bóc tách câu trả lời.

Mọi lời gọi API đều được bọc try-except: lỗi mạng, hết quota hay trả lời sai
định dạng KHÔNG làm dừng mô phỏng — `decide()` trả về None để bên gọi tự
chọn hành động dự phòng.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

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
GEMINI_TEMPERATURE = 0.0
OPENAI_MAX_RETRIES = 1

# Mã lỗi HTTP khiến một model Gemini bị coi là tạm thời không dùng được:
# 429 hết quota, 404 model không tồn tại với key này, 5xx/504 server quá tải.
GEMINI_ROTATE_ON_CODES = frozenset({429, 404, 500, 503, 504})
GEMINI_MODEL_COOLDOWN_S = 60.0   # sau lỗi trên, bỏ qua model này trong bấy nhiêu giây
LLM_MIN_CALL_INTERVAL_S = 1.0    # giãn cách tối thiểu giữa hai lời gọi API (mọi agent)

ACTION_KEEP = 0    # giữ pha hiện tại
ACTION_CHANGE = 1  # chuyển sang pha kế tiếp trong chu trình

# Bóc tách câu trả lời của LLM
HIDDEN_CHARS_RE = re.compile(r"[​-‏⁠﻿\x00-\x08\x0b-\x1f]")  # zero-width, BOM, control
ACTION_DIGIT_RE = re.compile(r"(?<!\d)[01](?!\d)")  # chữ số 0/1 không dính vào số khác


class LLMExpert:
    """LLM đóng vai chuyên gia quyết định KEEP / CHANGE cho một giao lộ.

    Args:
        provider: "gemini" hoặc "openai".
        name: Nhãn dùng trong log (thường là intersection_id).
        num_phases: Số pha xanh của giao lộ.
        approach_names: Tên các hướng vào giao lộ theo đúng thứ tự làn (vd.
            ["West", "South", "East", "North"]); các làn được chia đều cho các
            hướng. None → prompt liệt kê theo từng làn.
        phase_descriptions: Mô tả ngắn từng pha xanh (vd. "West/East
            straight"). None → bỏ qua.

    Raises:
        ValueError: Nếu `provider` không được hỗ trợ hoặc `phase_descriptions`
            không có đúng `num_phases` phần tử.
    """

    # Trạng thái dùng CHUNG cho mọi LLMExpert, vì quota tính theo API key chứ
    # không theo từng giao lộ.
    _gemini_cooldown_until: Dict[str, float] = {}   # model → mốc time.monotonic() được dùng lại
    _last_call_ts: float = 0.0                      # mốc lời gọi API gần nhất

    def __init__(
        self,
        provider: str,
        name: str,
        num_phases: int,
        approach_names: Optional[Sequence[str]] = None,
        phase_descriptions: Optional[Sequence[str]] = None,
    ) -> None:
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                f"llm_provider phải thuộc {sorted(LLM_PROVIDERS)}, nhận {provider!r}."
            )
        if phase_descriptions and len(phase_descriptions) != num_phases:
            raise ValueError("phase_descriptions phải có đúng num_phases phần tử.")

        self.provider = provider
        self.name = name
        self.num_phases = num_phases
        self.approach_names = list(approach_names) if approach_names else None
        self.phase_descriptions = list(phase_descriptions) if phase_descriptions else None

        self.calls = 0
        self.failures = 0
        self.model_name = ""
        self._gemini_models: List[str] = []
        self._client: Any = None  # google.genai.Client hoặc openai.OpenAI; None nếu chưa sẵn sàng
        self._init_client()

    @property
    def display_name(self) -> str:
        """Tên hiển thị của provider, vd. "Gemini"."""
        return LLM_PROVIDERS[self.provider]

    def _init_client(self) -> None:
        """Nạp API key từ `.env` và tạo client cho provider đã chọn.

        SDK chỉ được import ở đây (lazy) nên chỉ cần cài thư viện của provider
        đang dùng. Thiếu key hay thiếu thư viện đều không làm crash:
        `self._client` giữ None và `decide()` sẽ trả về None.
        """
        try:
            from dotenv import find_dotenv, load_dotenv

            # Ưu tiên .env ở gốc repo; nếu không có thì dò ngược từ thư mục đang chạy.
            env_path = (
                ENV_FILE if ENV_FILE.is_file() else Path(find_dotenv(usecwd=True) or ENV_FILE)
            )
            load_dotenv(dotenv_path=env_path)
        except ImportError:
            logger.warning("[%s] Thiếu python-dotenv — chỉ đọc biến môi trường sẵn có.", self.name)

        key_name = LLM_API_KEY_ENV[self.provider]
        api_key = os.getenv(key_name)
        self.model_name = os.getenv(
            LLM_MODEL_ENV[self.provider], LLM_DEFAULT_MODEL[self.provider]
        )
        # Thứ tự xoay vòng model Gemini: model chính trước, bỏ trùng lặp.
        self._gemini_models = list(dict.fromkeys([self.model_name, *GEMINI_FREE_MODELS]))
        if not api_key:
            logger.warning(
                "[%s] Thiếu %s trong %s — LLM [%s] bị vô hiệu, dùng hành động dự phòng.",
                self.name, key_name, ENV_FILE.name, self.display_name,
            )
            return

        try:
            if self.provider == "gemini":
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_S * 1000)),  # ms
                )
            else:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=OPENAI_MAX_RETRIES
                )
        except Exception as exc:  # ranh giới với SDK ngoài: thiếu thư viện, key sai định dạng, ...
            logger.warning(
                "[%s] Không khởi tạo được LLM [%s]: %s: %s",
                self.name, self.display_name, type(exc).__name__, exc,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # State → prompt
    # ──────────────────────────────────────────────────────────────────────────

    def describe_state(self, queues: Sequence[int], current_phase: int) -> str:
        """Dịch tình trạng giao lộ thành một prompt tiếng Anh ngắn gọn.

        Args:
            queues: Số xe đang chờ trên từng làn vào (đã khử chuẩn hoá).
            current_phase: Index pha xanh hiện tại, thuộc [0, num_phases).

        Returns:
            Prompt yêu cầu LLM trả về đúng một chữ số: 0 (KEEP) hoặc 1 (CHANGE).
        """
        if self.approach_names and len(queues) % len(self.approach_names) == 0:
            per_approach = len(queues) // len(self.approach_names)
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

    # ──────────────────────────────────────────────────────────────────────────
    # Gọi LLM
    # ──────────────────────────────────────────────────────────────────────────

    def decide(self, prompt: str) -> Optional[int]:
        """Hỏi LLM nên KEEP hay CHANGE.

        Args:
            prompt: Chuỗi sinh bởi `describe_state()`.

        Returns:
            0 (KEEP) hoặc 1 (CHANGE); None nếu lời gọi lỗi hoặc câu trả lời
            không bóc tách được (đã ghi log, bên gọi tự chọn hành động dự phòng).
        """
        self.calls += 1
        raw_text: Optional[str] = None  # khác None ⇔ API đã trả lời
        try:
            if self._client is None:
                raise RuntimeError("LLM client chưa được khởi tạo (thiếu API key hoặc thư viện)")
            raw_text = self._call_gemini(prompt) if self.provider == "gemini" else self._call_openai(prompt)
            action = self._parse_reply(raw_text)
        except Exception as exc:  # ranh giới với API ngoài: mạng, quota, sai định dạng, ...
            self.failures += 1
            logger.warning(
                "[%s] LLM [%s] lỗi (%s: %s)%s → bên gọi dùng hành động dự phòng.",
                self.name, self.display_name, type(exc).__name__, exc,
                f"; câu trả lời thô: {raw_text!r}" if raw_text is not None else "",
            )
            return None

        logger.info(
            "[%s] LLM [%s] decided Action: %d (model=%s)",
            self.name, self.display_name, action, self.model_name,
        )
        return action

    def _call_openai(self, prompt: str) -> str:
        """Gọi OpenAI Chat Completions và trả về văn bản thô."""
        self._throttle()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _call_gemini(self, prompt: str) -> str:
        """Gọi Gemini với chiến thuật xoay vòng model.

        Thử lần lượt các model trong `self._gemini_models`. Model nào trả lỗi
        thuộc GEMINI_ROTATE_ON_CODES (vd. 429 hết quota) sẽ bị đưa vào cooldown
        GEMINI_MODEL_COOLDOWN_S giây và hàm chuyển NGAY sang model kế tiếp.
        Hết cooldown, model ưu tiên cao hơn được thử lại.

        Args:
            prompt: Chuỗi sinh bởi `describe_state()`.

        Returns:
            Văn bản thô Gemini trả về (chuỗi rỗng nếu câu trả lời bị chặn).

        Raises:
            Exception: Lỗi không thuộc diện xoay vòng (vd. sai API key), hoặc
                lỗi cuối cùng khi mọi model đều thất bại.
            RuntimeError: Khi mọi model đều đang trong cooldown.
        """
        last_exc: Optional[Exception] = None
        for model in self._gemini_models:
            if time.monotonic() < LLMExpert._gemini_cooldown_until.get(model, 0.0):
                continue  # model này vừa lỗi, chưa hết thời gian chờ

            self._throttle()
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"temperature": GEMINI_TEMPERATURE},
                )
            except Exception as exc:  # SDK ném nhiều loại lỗi; phân loại theo mã HTTP
                code = getattr(exc, "code", None)
                if code not in GEMINI_ROTATE_ON_CODES:
                    raise
                LLMExpert._gemini_cooldown_until[model] = (
                    time.monotonic() + GEMINI_MODEL_COOLDOWN_S
                )
                logger.warning(
                    "[%s] %s lỗi %s → tạm nghỉ %.0fs, chuyển sang model kế tiếp.",
                    self.name, model, code, GEMINI_MODEL_COOLDOWN_S,
                )
                last_exc = exc
                continue

            self.model_name = model
            return response.text or ""  # response.text là None khi câu trả lời bị chặn / rỗng

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "Mọi model Gemini đang trong thời gian chờ (cooldown) sau lỗi quota/server"
        )

    @staticmethod
    def _throttle() -> None:
        """Ngủ nếu cần để hai lời gọi API liên tiếp cách nhau ≥ LLM_MIN_CALL_INTERVAL_S."""
        wait = LLM_MIN_CALL_INTERVAL_S - (time.monotonic() - LLMExpert._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        LLMExpert._last_call_ts = time.monotonic()

    @staticmethod
    def _parse_reply(raw_text: str) -> int:
        """Bóc quyết định 0/1 ra khỏi câu trả lời thô của LLM.

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
