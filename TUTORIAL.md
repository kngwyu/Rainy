# Rainy tutorial

## Run CLI examples
Though this project is still WIP, all examples are verified to work.

First, install [pipenv](https://pipenv.readthedocs.io/en/latest/).
E.g. you can install it via
``` bash
pip3 install pipenv --user
```

Then, clone this repository and create a virtual environment in it.
```bash
git clone https://github.com/kngwyu/Rainy.git
cd Rainy
pipenv --site-packages --three install
```

Now you are ready to start!

```bash
pipenv run python examples/acktr_cart_pole.py train
```

After training, you can run learned agents.

Please replace `(log-directory)` in the below command with your real log directory.
It should be named like `Results/acktr_cart_pole`.
``` bash
pipenv run python acktr_cart_pole.py eval (log-directory) --render
```

You can also plot training results in your log directory.
This command opens an ipython shell with your log file.
``` bash
pipenv run python -m rainy.ipython
```
Then you can plot training rewards via
```python
log = open_log('log-directory')
log.plot_reward(12 * 20, max_steps=int(4e5), title='ACKTR cart pole')
```
![ACKTR cart pole](./pictures/acktr-cart-pole.png)

## MPI/NCCL support
Distributed training is supported via [horovod](https://horovod.readthedocs.io/en/latest/).

E.g., if you want to use it with NCLL, you can install it via
```bash
sudo env HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir horovod
```

You can run the training script using `horovodrun` command.

E.g., if you want to use two hosts(localhost and anotherhost) and run `ppo_atari.py`, use
```bash
horovodrun -np 2 -H localhost:1,anotherhost:1 pipenv run python examples/ppo_atari.py train
```
