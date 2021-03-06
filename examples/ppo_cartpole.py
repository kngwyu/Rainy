import os

from torch.optim import Adam

import rainy
from rainy.envs import ClassicControl, MultiProcEnv


@rainy.main(rainy.agents.PPOAgent, script_path=os.path.realpath(__file__))
def main(
    envname: str = "CartPole-v0", use_rnn: bool = False, use_separated: bool = False
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(1e5)
    c.nworkers = 8
    c.nsteps = 32
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.value_loss_weight = 0.2
    c.entropy_weight = 0.001
    c.grad_clip = 0.1
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 64
    c.use_gae = True
    c.ppo_clip = 0.2
    c.eval_freq = 5000
    c.eval_times = 1
    c.episode_log_freq = 100
    if use_rnn:
        c.set_net_fn(
            "actor-critic", rainy.net.actor_critic.fc_shared(rnn=rainy.net.GruBlock)
        )
    elif use_separated:
        c.set_net_fn("actor-critic", rainy.net.actor_critic.fc_separated())
    return c


if __name__ == "__main__":
    main()
