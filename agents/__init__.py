"""
agents/ — các lớp Agent RL cho dự án FedMARL-ATSC.

Bộ khung (trung lập thuật toán):
    node_agent.py        NodeAgent — giao diện RL + giao diện FL
    networks.py          TrafficAgentNet — mạng phân rã backbone / head
    replay_buffer.py     Transition, TransitionBatch, ReplayBuffer

Thuật toán mẫu (mỗi thuật toán một module, là lớp con của NodeAgent):
    q_learning.py        QLearningAgent — ví dụ tối giản
    dqn.py               DQNAgent — replay buffer + target network
    behavior_cloning.py  BehaviorCloningAgent — học theo LLM (llm_expert.py)

Cách thêm thuật toán mới: docs/RL_INTEGRATION_GUIDE.md.
"""
from agents.networks import TrafficAgentNet
from agents.node_agent import NodeAgent

__all__ = ["NodeAgent", "TrafficAgentNet"]
