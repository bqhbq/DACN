# Môi trường chạy CityFlow cho FedMARL-ATSC.
# Build native trên linux/arm64 (Apple Silicon) lẫn linux/amd64.
#
# Build context là thư mục gốc của repo (xem docker-compose.yml).

# Python 3.10: bản pybind11 đi kèm CityFlow không build được với Python >= 3.11.
FROM python:3.10-slim-bookworm

ARG CITYFLOW_REPO=https://github.com/cityflow-project/CityFlow.git
ARG CITYFLOW_REF=master

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

# Build CityFlow từ source. Đặt trước requirements.txt để layer này được cache
# khi chỉ thay đổi dependency Python.
RUN python -m pip install --upgrade pip wheel setuptools \
 && git clone --recursive --depth 1 --branch "${CITYFLOW_REF}" "${CITYFLOW_REPO}" /opt/CityFlow \
 && pip install /opt/CityFlow \
 && rm -rf /opt/CityFlow \
 && python -c "import cityflow; print('CityFlow', cityflow.__version__)"

# Cài đặt PyTorch bản CPU-only TRƯỚC để chặn việc tự động tải bộ thư viện NVIDIA CUDA 5GB
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Mã nguồn không được copy vào image; repo được mount vào /workspace lúc chạy.
WORKDIR /workspace

# Giữ container chạy ngầm để VS Code attach vào và dùng Python trong container.
CMD ["sleep", "infinity"]