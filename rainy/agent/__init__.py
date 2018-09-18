from .base import Agent
from .dqn import DqnAgent

del base
del dqn

def run_agent(agent: Agent):
    max_steps = agent.config.max_steps
    while True:
        if max_steps and agent.total_steps > max_steps:
            break
        agent.episode()

