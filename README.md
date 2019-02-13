# Rainy
[![Build Status](https://travis-ci.org/kngwyu/Rainy.svg?branch=master)](https://travis-ci.org/kngwyu/Rainy)

Reinforcement learning utilities and algrithm implementations using PyTorch.

# API documentation
COMING SOON

# Supported python version
Python >= 3.6.1

# Run examples
Though this project is still WIP, all examples are verified to work.

First, install [pipenv](https://pipenv.readthedocs.io/en/latest/).
E.g. you can install it via
``` bash
pip install pipenv --user
```

Then, clone this repository and create a virtual environment in it.
```bash
git clone https://github.com/kngwyu/Rainy.git
cd Rainy
pipenv --site-packages --three
pipenv install --dev
```

Now you are ready to start!

You can run examples via `pipenv shell` or `pipenv run`.
```bash
pipenv shell
cd examples
python a2c_cart_pole.py train
```

After training, you can run learned agents.

Please replace `(log-directory)` in the below command with your real log file.
It should be named like `a2c_cart_pole-190213-225317-1bdac1d6`.
``` bash
python a2c_cart_pole.py eval (log-directory) --render
```

You can also plot training results in your log directory.
This command opens a ipython shell with your log file.
``` bash
python a2c_cart_pole.py ipython --log-dir=(log-directory)
```

``` python
import matplotlib.pyplot as plt
x, y = 'update-steps', 'reward-mean'
plt.plot(tlog[x], tlog[y])
plt.title('a2c cart pole')
plt.xlabel(x); plt.ylabel(y)
plt.show()
```
![A2C cart pole](./pictures/a2c-cart-pole.png)

# Implemented Algorithms

## DQN (Deep Q Network)
- https://www.nature.com/articles/nature14236/

## A2C (Advantage Actor Critic)
- http://proceedings.mlr.press/v48/mniha16.pdf , https://arxiv.org/abs/1602.01783 (Original version)
- https://blog.openai.com/baselines-acktr-a2c/ (Synchronized version)

## ACKTR (Actor Critic using Kronecker-Factored Trust Region)
- https://papers.nips.cc/paper/7112-scalable-trust-region-method-for-deep-reinforcement-learning-using-kronecker-factored-approximation

## PPO (Proximal Policy Optimization)
- https://arxiv.org/abs/1707.06347

# Implementaions I referenced
I referenced mainly openai baselines, but all these pacakages were useful.

Thanks!

https://github.com/openai/baselines

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

https://github.com/ShangtongZhang/DeepRL

https://github.com/chainer/chainerrl

https://github.com/Thrandis/EKFAC-pytorch (ACKTR)

# License
This project is licensed under Apache License, Version 2.0
([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).


