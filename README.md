# Rainy
[![Build Status](https://travis-ci.org/kngwyu/Rainy.svg?branch=master)](https://travis-ci.org/kngwyu/Rainy)

Reinforcement learning utilities and algrithm implementations using PyTorch.

# API documentation
COMING SOON

# Supported python version
Python >= 3.6.1

# Run examples
Though this project is still WIP, all examples are verified to work.

First, clone this repository and create a virtual env.
```bash
git clone https://github.com/kngwyu/Rainy.git
cd Rainy
mkdir .rainy-env
virtualenv --system-site-packages -p python3 .rainy-env
# if you use bash/zsh
. .rainy_env/bin/activate
# or if you use fish
. .rainy_env/bin/activate.fish
pip install -e .
```

Then run your favorite example.
```bash
cd examples
python a2c_cartpole.py train
```

# Implemented Algorithms

## DQN (Deep Q Network)
- https://www.nature.com/articles/nature14236/

## A2C (Advantage Actor Critic)
- http://proceedings.mlr.press/v48/mniha16.pdf (Original version)
- https://blog.openai.com/baselines-acktr-a2c/ (Synchronized version)

## PPO (Proximal Policy Optimization)
- https://arxiv.org/abs/1707.06347

# Implementaions I referenced
I referenced mainly openai baselines, but all these pacakages were useful.

Thanks!

https://github.com/openai/baselines

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

https://github.com/ShangtongZhang/DeepRL

https://github.com/chainer/chainerrl

# License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).


