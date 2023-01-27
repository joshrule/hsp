import argparse
import gym
import numpy as np
import os
import sys
import time
import torch
import models
from action_utils import parse_env_args
from env_wrappers import GymWrapper, ResetableTimeLimit
from multi_threading import ThreadedTrainer
# from play import PlayWrapper
from self_play import SelfPlayWrapper, SelfPlayPPO
# from trainer import SelfPlayRunner, PlayRunner
from algorithms.ppo import PPO
from algorithms.reinforce import Reinforce
from algorithms.vpg import VPG
from env_wrappers import NoOpCartPoleEnv
from gym.envs.registration import register

register(
    id='NoOpCartPole-v0',
    entry_point='env_wrappers:NoOpCartPoleEnv',
    reward_threshold=500,
)
register(
    id='GridWorld-v0',
    entry_point='env_wrappers:GridWorldEnv',
)
register(
    id='Eat-v0',
    entry_point='env_wrappers:EatEnv',
)

def init_arg_parser():
    """Initialize the argument parser."""
    parser = argparse.ArgumentParser(description='Reinforce, Asymmetric Self-Play, Hiearchical Self-Play')
    # Training: total number of steps = num_epochs x num_batches x num_steps x num_threads
    parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--num_batches', type=int, default=10, help='number of batches per epoch')
    parser.add_argument('--num_steps', type=int, default=500, help='number of steps per batch (per thread)')
    parser.add_argument('--num_threads', type=int, default=1, help='How many threads to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed (might not work when nthreads > 0)')
    parser.add_argument('--pi_lrate', type=float, default=0.001, help='policy learning rate')
    parser.add_argument('--v_lrate', type=float, default=0.001, help='value function learning rate')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='scale reward before backprop')
    parser.add_argument('--gamma', type=float, default=1.0, help='discount factor between steps')
    parser.add_argument('--gamma_r', type=float, default=0.99, help='discount factor between steps for rewards-to-go')
    parser.add_argument('--gamma_a', type=float, default=0.95, help='additional discount factor between steps for advantages')
    parser.add_argument('--n_v_updates', type=int, default=80, help='how many gradient steps to take per value function update')
    parser.add_argument('--n_pi_updates', type=int, default=80, help='how many gradient steps to take per PPO update')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon in PPO update')
    parser.add_argument('--target_kl', type=float, default=0.05, help='target KL divergence used in PPO early stopping')
    parser.add_argument('--normalize_rewards', action='store_true', default=False, help='normalize rewards in each batch')
    parser.add_argument('--value_coeff', type=float, default=0.05, help='coeff for value loss term')
    parser.add_argument('--entr', type=float, default=0, help='entropy regularization coeff')
    parser.add_argument('--freeze', default=False, action='store_true', help='freeze the model (no learning)')

    # Model
    parser.add_argument('--hid_size', default=64, type=int, help='hidden layer size')
    parser.add_argument('--l', default=2, type=int, help='number of hidden layers')
    parser.add_argument('--mode', default='', type=str, help='model mode: play | self-play | reinforce | vpg | ppo')

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
    parser.add_argument('--sp_reward_coef', default=0, type=float, help='coefficient by which base environment reward is multiplied')
    parser.add_argument('--sp_state_thresh_0', default=0, type=float, help='initial threshold of success for Bob')
    parser.add_argument('--sp_state_thresh_1', default=1, type=float, help='final threshold of success for Bob')
    parser.add_argument('--sp_state_thresh_factor', default=1, type=float, help='final threshold of success for Bob')
    parser.add_argument('--sp_steps', default=5, type=int, help='maximum self-play length')
    parser.add_argument('--sp_test_rate', default=0, type=float, help='percentage of target task episodes')

    # Ergonomics
    parser.add_argument('--display', action='store_true', default=False, help='Display episode from policy after each epoch.')
    parser.add_argument('--save', default='', type=str, help='save the model after training')
    parser.add_argument('--load', default='', type=str, help='load the model')
    parser.add_argument('--verbose', default=0, type=int, help='verbose output')

    return parser

def configure_torch(args):
    """Set parameters and random seeds for `torch`."""
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # seed = (args.seed + 10000 * os.getpid()) % (2 ** 32 - 1)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
    # If we're just testing, let's not take over the entire machine.
    if args.num_threads == 1:
        torch.set_num_threads(1)
    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.set_default_tensor_type('torch.DoubleTensor')

def init_env(args):
    """Initialize the environment."""
    #base_env = gym.make('NoOpCartPole-v0', render_mode=None, new_step_api=True)
    #args.no_op = 1
    base_env = gym.make('CartPole-v1', render_mode=None, new_step_api=True)
    args.no_op = 0
    # base_env = gym.make('Eat-v0', new_step_api=True)
    # args.no_op = 0
    gym_env = GymWrapper(base_env, new_step_api=True)
    env = ResetableTimeLimit(gym_env, max_episode_steps = args.max_steps, new_step_api = True)
    if args.mode == "self-play":
        return SelfPlayWrapper(args, env, new_step_api = True)
    # elif args.mode == "play":
    #     env = SelfPlayWrapper(args, env, new_step_api = True)
    #     return PlayWrapper(args, env, new_step_api = True)
    else:
        return env

def init_runner(args):
    """Initialize the trainer."""
    if args.mode == 'reinforce':
        f = lambda: Reinforce(args, init_env(args), ac_kwargs=dict(hidden_sizes=[args.hid_size]*args.l))
    elif args.mode == 'vpg':
        f = lambda: VPG(args, init_env(args), ac_kwargs=dict(hidden_sizes=[args.hid_size]*args.l))
    elif args.mode == 'ppo':
        f = lambda: PPO(args, init_env(args))
    elif args.mode == 'self-play':
        f = lambda: SelfPlayPPO(args, init_env(args))
    # elif args.mode == 'play':
    #     #f = lambda: Reinforce(args, PlayRunner(args, policy_net, init_env(args)))
    #     pass

    if args.num_threads > 1:
        return ThreadedTrainer(args, f)
    else:
        return f()

def load(args, trainer):
    if args.load != '':
        d = torch.load(args.path)
        trainer.load_state_dict(d['trainer'])

def save(args, trainer):
    if args.save != '':
        d = dict()
        d['trainer'] = trainer.state_dict()
        d['args'] = args
        torch.save(d, args.save)

def visualize_policy(args, trainer):
    if args.display:
        if args.num_threads > 1:
            trainer.trainer.display = True
        else:
            trainer.display = True
        trainer.run_episode(args.max_steps)
        if args.num_threads > 1:
            trainer.trainer.display = False
        else:
            trainer.display = False

def run(args, trainer):
    run_begin_time = time.time()
    for epoch in range(args.num_epochs):
        stat = trainer.run_epoch()
        reward = stat["reward"]
        alice_reward = np.mean([ep['env'].get('reward_alice', 0) for ep in stat["episodes"]])
        bob_reward = np.mean([ep['env'].get('reward_bob', 0) for ep in stat["episodes"]])
        mean_reward = np.mean([ep["reward"] for ep in stat["episodes"] if ep["steps"][-1]["term"] or (ep["steps"][-1]["trunc"] and len(ep["steps"]) == args.max_steps)])
        episodes = sum(1 for ep in stat["episodes"] if ep["steps"][-1]["term"] or (ep["steps"][-1]["trunc"] and len(ep["steps"]) == args.max_steps))
        remainder = sum(len(ep["steps"]) for ep in stat["episodes"] if not (ep["steps"][-1]["term"] or (ep["steps"][-1]["trunc"] and len(ep["steps"]) == args.max_steps)))
        mean_diff = np.mean([s["diff"] for ep in stat["episodes"] for s in ep['steps'] if s.get('diff', None) != None])
        num_steps = stat["num_steps"]
        epoch_time = stat["time"]

        if args.verbose > 0:
            print(f'E: {epoch} R: {mean_reward:.2f} ({reward}) A: {alice_reward:.3f}  B: {bob_reward:.3f} Ep: {episodes} ({remainder}) T: {epoch_time:.2f}s S: {num_steps} D: {mean_diff:.4}', flush = True)

        save(args, trainer)

        if epoch % 10 == 0:
            visualize_policy(args, trainer)
    run_time = time.time() - run_begin_time
    print(f'Run completed in {run_time:.2f}s.')

    return trainer

if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()
    print(args, end="\n\n")

    configure_torch(args)

    env = init_env(args)
    parse_env_args(args, env)
    print(args, end="\n\n")

    trainer = init_runner(args)

    load(args, trainer)

    trainer = run(args, trainer)

    save(args, trainer)

    if sys.flags.interactive == 0 and args.num_threads > 1:
        trainer.quit()
        os._exit(0)
