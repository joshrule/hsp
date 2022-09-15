import argparse
import gym
import os
import sys
import time
import torch
import models
import progressbar
from self_play import SPModel
from action_utils import parse_env_args
from env_wrappers import GymWrapper
from multi_threading import ThreadedTrainer
from self_play import SelfPlayWrapper
from trainer import ReinforceTrainer, SelfPlayTrainer, Trainer

def init_arg_parser():
    """Initialize the argument parser."""
    parser = argparse.ArgumentParser(description='Reinforce, Asymmetric Self-Play, Hiearchical Self-Play')
    # Training: total number of steps = num_epochs x num_batches x num_steps x num_threads
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--num_batches', type=int, default=10, help='number of batches per epoch')
    parser.add_argument('--num_steps', type=int, default=500, help='number of steps per batch (per thread)')
    parser.add_argument('--num_threads', type=int, default=16, help='How many threads to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed (might not work when nthreads > 0)')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='scale reward before backprop')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor between steps')
    parser.add_argument('--normalize_rewards', action='store_true', default=False, help='normalize rewards in each batch')
    parser.add_argument('--value_coeff', type=float, default=0.05, help='coeff for value loss term')
    parser.add_argument('--entr', type=float, default=0, help='entropy regularization coeff')
    parser.add_argument('--freeze', default=False, action='store_true', help='freeze the model (no learning)')

    # Model
    parser.add_argument('--hid_size', default=64, type=int, help='hidden layer size')
    parser.add_argument('--mode', default='', type=str, help='model mode: play | self-play | reinforce')

    # Environment
    parser.add_argument('--max_steps', default=20, type=int, help='forcibly end episode after this many steps')

    # Self-Play
    parser.add_argument('--sp_goal_diff', default=False, action='store_true', help='encode goal as difference (e.g. g=enc(s*)-enc(s_t))')
    parser.add_argument('--sp_goal_dim', default=2, type=int, help='goal representation dimension')
    parser.add_argument('--sp_mode', default='reverse', type=str, help='self-play mode: reverse | repeat')
    parser.add_argument('--sp_persist', default=0, type=int, help='start next self-play episode from previous one for K episodes')
    parser.add_argument('--sp_persist_discount', default=1.0, type=float, help='discount coeff between persist episodes')
    parser.add_argument('--sp_persist_separate', default=False, action='store_true', help='keep Alice and Bob trajectory separate')
    parser.add_argument('--sp_persist_success', default=False, action='store_true', help='only persist if prev success')
    parser.add_argument('--sp_state_thresh_0', default=0, type=float, help='initial threshold of success for Bob')
    parser.add_argument('--sp_state_thresh_1', default=1, type=float, help='final threshold of success for Bob')
    parser.add_argument('--sp_state_thresh_factor', default=1, type=float, help='final threshold of success for Bob')
    parser.add_argument('--sp_steps', default=5, type=int, help='maximum self-play length')
    parser.add_argument('--sp_test_rate', default=0, type=float, help='percentage of target task episodes')

    # Ergonomics
    parser.add_argument('--progress', action='store_true', default=False, help='Display progress bar.')
    parser.add_argument('--display', action='store_true', default=False, help='Display episode from policy after each epoch.')
    parser.add_argument('--save', default='', type=str, help='save the model after training')
    parser.add_argument('--load', default='', type=str, help='load the model')
    parser.add_argument('--verbose', default=0, type=int, help='verbose output')

    return parser

def configure_torch(args):
    """Set parameters and random seeds for `torch`."""
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.set_default_tensor_type('torch.DoubleTensor')

def init_env(args):
    """Initialize the environment."""
    base_env = GymWrapper(gym.make('CartPole-v1', render_mode=None, new_step_api=True))
    if args.mode == "play":
        return PlayWrapper(args, base_env)
    elif args.mode == "self-play":
        return SelfPlayWrapper(args, base_env)
    else:
        return base_env

def init_policy(args):
    if args.mode == 'play':
        args.num_inputs = args.input_dim
        args.input_dim = env.env.observation_dim # actual observation dim
        policy_net = PlayModel(args)
    elif args.mode == 'self-play':
        args.num_inputs = args.input_dim
        args.input_dim = env.env.observation_dim # actual observation dim
        policy_net = SPModel(args)
    else:
        # REINFORCE with Multi-Layer Perceptron
        policy_net = models.MLP(args)

    # share parameters among threads, but not gradients
    for p in policy_net.parameters():
        p.data.share_memory_()
    return policy_net

def init_trainer(args, policy_net):
    """Initialize the trainer."""
    if args.mode == 'play':
        f = lambda: Trainer(args, policy_net, init_env(args))
    elif args.mode == 'self-play':
        f = lambda: SelfPlayTrainer(args, policy_net, init_env(args))
    else:
        f = lambda: ReinforceTrainer(args, policy_net, init_env(args))

    if args.num_threads > 1:
        return ThreadedTrainer(args, f)
    else:
        return f()

def load(args, policy_net, trainer):
    if args.load != '':
        d = torch.load(args.path)
        policy_net.load_state_dict(d['policy_net'])
        trainer.load_state_dict(d['trainer'])

def save(args, policy_net, trainer):
    if args.save != '':
        d = dict()
        d['policy_net'] = policy_net.state_dict()
        # d['log'] = log
        d['trainer'] = trainer.state_dict()
        d['args'] = args
        torch.save(d, args.save)

def visualize_policy(args, trainer):
    if args.display:
        if args.num_threads > 1:
            trainer.trainer.display = True
        else:
            trainer.display = True
        trainer.get_episode()
        if args.num_threads > 1:
            trainer.trainer.display = False
        else:
            trainer.display = False

def run(args, policy, trainer):
    reward = 0
    for epoch in range(args.num_epochs):
        print(f'Begin Epoch {epoch}')
        epoch_reward = 0
        epoch_begin_time = time.time()
        if args.progress:
            progress = progressbar.ProgressBar(max_value=args.num_batches, redirect_stdout=True).start()

        for n in range(args.num_batches):
            print(f'    Begin Batch {n}')
            if args.progress:
                progress.update(n+1)
            stat = trainer.train_batch()
            epoch_reward += stat['batch_reward']
            print(f'    End Batch {n} (Reward: {stat["batch_reward"]:.2f})')
        if args.progress:
            progress.finish()

        epoch_time = time.time() - epoch_begin_time
        reward += epoch_reward
        print(f'End Epoch {epoch} (Reward {epoch_reward:.2f}\tTime {epoch_time:.2f}s)')

        save(args, policy, trainer)

        visualize_policy(args, trainer)
    print(f'End Run (Reward: {reward:.2f})')

    return policy, trainer

if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()
    print(args, end="\n\n")

    configure_torch(args)

    env = init_env(args)
    parse_env_args(args, env)
    print(args, end="\n\n")

    policy_net = init_policy(args)

    trainer = init_trainer(args, policy_net)

    load(args, policy_net, trainer)

    policy_net, trainer = run(args, policy_net, trainer)

    save(args, policy_net, trainer)

    if sys.flags.interactive == 0 and args.num_threads > 1:
        trainer.quit()
        os._exit(0)
