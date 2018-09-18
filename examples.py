from rainy import agent, Config

def run():
    c = Config()
    a = agent.DqnAgent(c)
    agent.run_agent(a)

if __name__ == '__main__':
    run()
