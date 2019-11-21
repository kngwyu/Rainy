import os
import rainy
from rainy.utils.cli import run_cli
from rainy.envs import MultiProcEnv
from torch.optim import Adam


def config() -> rainy.Config:
    c = rainy.Config()
    c.max_steps = int(1e6)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: Adam(params, lr=0.001))
    c.grad_clip = 0.1
    c.value_loss_weight = 0.2
    c.use_gae = False
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 10
    c.entropy_weight = 0.001
    # c.set_net_fn('actor-critic', rainy.net.actor_critic.fc_shared(rnn=rainy.net.GruBlock))
    return c


if __name__ == "__main__":
    run_cli(config, rainy.agents.A2CAgent, script_path=os.path.realpath(__file__))
