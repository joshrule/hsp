# Hierarchical Self-Play (HSP)

This is a code for running experiments in the paper [Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1811.09083).

## Setup

### With `nix`
With `nix`, it's as simple as:
```
nix-shell
```
### With `pipenv`
With `pipenv`, run the following:
```
pipenv --python 3.9
pipenv install pandas pygame numpy progressbar2 torch visdom gym[classic_control]
```
### Multi-threading
The code is multi-threaded, so make sure each thread will only use a single CPU core:
```
export OMP_NUM_THREADS=1
```
For the plotting to work, you need to have a Visdom server running (see [this](https://github.com/facebookresearch/visdom#usage) for details).

## Algorithms
The code implements three different algorithms.

### 1. Reinforce
By default, the code will use vanilla Reinforce for training. For example, run the following to train a Reinforce agent on CartPole:
```
python main.py --max_steps 100 --num_epochs 200 --num_steps 500 --mode reinforce --verbose 1
```

### 2. Asymmetric Self-play
We also implement a form of asymmetric self-play from
[https://arxiv.org/abs/1703.05407](https://arxiv.org/abs/1703.05407).
```
python main.py --max_steps 100 --num_epochs 50 --num_steps 500 --num_threads 1 \
    --mode self-play --verbose 1 --sp_steps 60 --sp_state_thresh_0 0.3333      \
    --sp_state_thresh_1 0.000001 --sp_state_thresh_factor 0.95 --sp_mode repeat 
```
