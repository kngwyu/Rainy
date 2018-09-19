from rainy import agent, Config
from rainy.agent import Agent


def run_agent(ag: Agent, train: bool = True):
    max_steps = ag.config.max_steps
    while True:
        if max_steps and ag.total_steps > max_steps:
            break
        print(ag.episode(train=train))
    ag.save("saved-example")


def run():
    c = Config()
    c.max_steps = 100000
    a = agent.DqnAgent(c)
    # a.load("saved-example")
    run_agent(a)


if __name__ == '__main__':
    run()
