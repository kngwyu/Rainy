import os
import rainy
from rainy.utils.cli import run_cli
from rainy.envs import MultiProcEnv
from torch.optim import Adam


def config() -> rainy.Config:
    c = rainy.Config()
    c.max_steps = int(1e6)
    c.nworkers = 8
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.value_loss_weight = 0.1
    c.network_log_freq = 20
    c.grad_clip = 0.1
    c.gae_tau = 0.95
    c.nsteps = 32
    c.ppo_minibatch_size = 64
    c.use_gae = True
    c.lr_decay = False
    c.clip_decay = False
    c.eval_freq = None
    # c.set_net_fn('actor-critic', rainy.net.actor_critic.fc_shared(rnn=rainy.net.GruBlock))
    return c


if __name__ == '__main__':
    run_cli(config(), rainy.agents.PpoAgent, script_path=os.path.realpath(__file__))
