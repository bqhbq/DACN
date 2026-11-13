# Cityflow Guide

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