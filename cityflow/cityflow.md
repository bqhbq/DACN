# Cityflow Guide

Có hai cách cài CityFlow:

- **Docker** (khuyến nghị trên macOS Apple Silicon / ARM64): xem mục [Setup bằng Docker](#setup-bằng-docker).
- **Cài trực tiếp trên Linux**: xem mục [Setup](#setup).

## Setup bằng Docker

CityFlow không có wheel dựng sẵn cho macOS ARM64 và phần hướng dẫn cài trực tiếp bên dưới dùng `apt-get`. Trên máy Mac, chạy CityFlow trong container Linux ARM64 là cách gọn nhất: image được build native, không cần giả lập x86.

### Các file liên quan

| File | Vai trò |
|---|---|
| `cityflow/Dockerfile` | Image Python 3.10, build CityFlow từ source, cài `requirements.txt` của project |
| `cityflow/docker-compose.yml` | Build image, mount toàn bộ repo vào `/workspace`, giữ container chạy ngầm |
| `cityflow/Dockerfile.dockerignore` | Giới hạn build context, chỉ gửi `requirements.txt` vào image |

Mã nguồn không được copy vào image. Repo được mount vào container, nên mọi thay đổi code trên máy host có hiệu lực ngay, và dữ liệu sinh ra trong container (`roadnet.json`, `flow.json`, replay...) xuất hiện luôn trong thư mục repo.

### Yêu cầu

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) cho Mac (bản Apple Silicon), đang chạy.

### 1. Build image

Chạy từ thư mục `cityflow/`:

```bash
cd cityflow
docker compose build
```

Lần build đầu mất vài phút vì phải biên dịch CityFlow và tải PyTorch. Các lần sau dùng lại cache.

### 2. Kiểm tra CityFlow

```bash
docker compose run --rm cityflow python -c "import cityflow; print(cityflow.__version__)"
# Expect: 0.1
```

### 3. Khởi động container

```bash
docker compose up -d
```

Container chạy ngầm và không mở cổng nào. Kiểm tra bằng `docker compose ps`, dừng và dọn bằng `docker compose down`.

### 4. Chạy code và notebook bằng VS Code

Image không chứa JupyterLab. Script và notebook được chạy bằng VS Code, dùng Python bên trong container. Repo có sẵn `.devcontainer/devcontainer.json` trỏ tới service `cityflow` ở trên.

1. Cài extension **Dev Containers** (`ms-vscode-remote.remote-containers`) trên VS Code.
2. Mở thư mục gốc của repo bằng VS Code.
3. Mở Command Palette (`Cmd+Shift+P`), chọn **Dev Containers: Reopen in Container**. VS Code tự khởi động container (không cần chạy `docker compose up -d` trước), mở `/workspace` và cài sẵn extension **Python** và **Jupyter**.
4. Chạy script bằng nút Run như bình thường; interpreter mặc định là `/usr/local/bin/python` (Python 3.10).
5. Với notebook, mở file `.ipynb`, bấm **Select Kernel > Python Environments** và chọn `/usr/local/bin/python`.

Để quay lại môi trường host, chọn **Dev Containers: Reopen Folder Locally**. Đóng cửa sổ VS Code sẽ dừng container.

Muốn chạy lệnh trực tiếp thì dùng terminal của cửa sổ VS Code đó, hoặc từ máy host:

```bash
docker compose exec cityflow bash
# Thư mục hiện tại là /workspace (gốc repo)
```

### 5. Xem replay

Sau khi chạy mô phỏng, replay nằm trong `cityflow/sample/test/test_1/render/` trên máy host. Mở `cityflow/frontend/index.html` bằng trình duyệt trên máy host và nạp `replay_roadnet.json` cùng `replay.txt`.

### Lưu ý

- `config.json` do `gen_config()` sinh trong container chứa đường dẫn tuyệt đối dạng `/workspace/...`. File này chỉ dùng được bên trong container; chạy ở môi trường khác thì cần sinh lại.
- Image dùng Python 3.10 vì bản pybind11 đi kèm CityFlow không build được với Python từ 3.11 trở lên.
- Khi `requirements.txt` thay đổi, chạy lại `docker compose up -d --build`.
- Muốn build CityFlow từ một nhánh hoặc tag khác: `docker compose build --build-arg CITYFLOW_REF=<ref>`.

## Setup
### Active .venv
```bash
# python3 -m venv .venv
source .venv/bin/activate # macOS / Linux
# python -m pip install -r requirements.txt
python -m pip install --upgrade pip wheel setuptools
```

### Installation
```bash
apt-get update
apt-get install -y build-essential cmake

mkdir -p .temp && cd .temp

git clone --recursive https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .

cd ../..


sudo apt update
sudo apt install sumo sumo-tools

```

### Check
```bash
python -c "import cityflow; print(cityflow.__version__)"
# Expect: 0.1.0
```