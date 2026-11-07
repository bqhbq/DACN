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

### 4. Launch JupyterLab

```bash
jupyter lab
```

Navigate to `notebooks/` to begin exploratory analysis.

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
