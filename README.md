# Rainy
[![Actions Status](https://github.com/kngwyu/Rainy/workflows/Tests/badge.svg)](https://github.com/kngwyu/Rainy/actions)
[![PyPI version](https://img.shields.io/pypi/v/Rainy?style=flat-square)](https://pypi.org/project/rainy/)
[![Black](https://img.shields.io/badge/code%20style-black-000.svg)](https://github.com/psf/black)

Reinforcement learning utilities and algrithm implementations using PyTorch.

## API documentation
COMING SOON

## Supported python version
Python >= 3.6.1

## Implementation Status

|**Algorithm** |**Multi Worker(Sync)**|**Recurrent**                   |**Discrete Action** |**Continuous Action**|**MPI support**   |
| ------------ | -------------------- | ------------------------------ | ------------------ | ------------------- | ---------------- |
|DQN/Double DQN|:x:                   |:x:                             |:heavy_check_mark:  |:x:                  |:x:               |
|BootDQN/RPF   |:x:                   |:x:                             |:heavy_check_mark:  |:x:                  |:x:               |
|DDPG          |:x:                   |:x:                             |:x:                 |:heavy_check_mark:   |:x:               |
|TD3           |:x:                   |:x:                             |:x:                 |:heavy_check_mark:   |:x:               |
|SAC           |:x:                   |:x:                             |:x:                 |:heavy_check_mark:   |:x:               |
|PPO           |:heavy_check_mark:    |:heavy_check_mark:<sup>(1)</sup>|:heavy_check_mark:  |:heavy_check_mark:   |:heavy_check_mark:|
|A2C           |:heavy_check_mark:    |:heavy_check_mark:              |:heavy_check_mark:  |:heavy_check_mark:   |:x:               |
|ACKTR         |:heavy_check_mark:    |:x:<sup>(2)</sup>               |:heavy_check_mark:  |:heavy_check_mark:   |:x:               |
|AOC           |:heavy_check_mark:    |:x:                             |:heavy_check_mark:  |:heavy_check_mark:   |:x:               |

<sup><sup>(1): It's very unstable </sup></sup><br>
<sup><sup>(2): Needs https://openreview.net/forum?id=HyMTkQZAb implemented </sup></sup><br>

## Sub packages

- [intrinsic-rewards](https://github.com/kngwyu/intrinsic-rewards)
  - Contains an implementation of RND(Random Network Distillation)

## References

### DQN (Deep Q Network)
- https://www.nature.com/articles/nature14236/

### DDQN (Double DQN)
- https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389

### Bootstrapped DQN
- https://arxiv.org/abs/1602.04621

### RPF(Randomized Prior Functions)
- https://arxiv.org/abs/1806.03335

### DDPQ(Deep Deterministic Policy Gradient)
- https://arxiv.org/abs/1509.02971

### TD3(Twin Delayed Deep Deterministic Policy Gradient)
- https://arxiv.org/abs/1802.09477

### SAC(Soft Actor Critic)
- https://arxiv.org/abs/1812.05905

### A2C (Advantage Actor Critic)
- http://proceedings.mlr.press/v48/mniha16.pdf , https://arxiv.org/abs/1602.01783 (A3C, original version)
- https://blog.openai.com/baselines-acktr-a2c/ (A2C, synchronized version)

### ACKTR (Actor Critic using Kronecker-Factored Trust Region)
- https://papers.nips.cc/paper/7112-scalable-trust-region-method-for-deep-reinforcement-learning-using-kronecker-factored-approximation

### PPO (Proximal Policy Optimization)
- https://arxiv.org/abs/1707.06347

### AOC (Advantage Option Critic)
- https://arxiv.org/abs/1609.05140 (DQN-like option critic)
- https://arxiv.org/abs/1709.04571 (A3C-like option critic called A2OC)

## Implementaions I referenced
Thank you!

https://github.com/openai/baselines

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

https://github.com/ShangtongZhang/DeepRL

https://github.com/chainer/chainerrl

https://github.com/Thrandis/EKFAC-pytorch (for ACKTR)

https://github.com/jeanharb/a2oc_delib (for AOC)

https://github.com/sfujim/TD3 (for DDPG and TD3)

https://github.com/vitchyr/rlkit (for SAC)

## License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).


