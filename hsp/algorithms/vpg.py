import numpy as np
import time
import torch as tr
import torch.nn as nn
from torch import optim
from algorithms.networks import MLPActorCritic
from utils import *


class Buffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float64)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float64)
        self.adv_buf = np.zeros(size, dtype=np.float64)
        self.rew_buf = np.zeros(size, dtype=np.float64)
        self.ret_buf = np.zeros(size, dtype=np.float64)
        self.val_buf = np.zeros(size, dtype=np.float64)
        self.logp_buf = np.zeros(size, dtype=np.float64)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def __len__(self):
        return self.ptr

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # The next three lines implement the advantage normalization trick.
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tr.as_tensor(v, dtype=tr.float64) for k,v in data.items()}


class VPG(object):
    def __init__(self, args, env, ac_kwargs=dict()):
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        self.args = args
        self.env = env
        self.ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
        print(f'Number of parameters: a: {count_vars(self.ac.pi)}, c: {count_vars(self.ac.v)}')
        self.buffer = Buffer(obs_dim, act_dim, args.num_steps, args.gamma_r, args.gamma_a)
        self.optimizer_a = optim.Adam(self.ac.pi.parameters(), lr=args.pi_lrate)
        self.optimizer_c = optim.Adam(self.ac.v.parameters(), lr=args.v_lrate)
        self.display = False

    def serialize_step(self, record):
        return f'            Time: {record["t"]}, Reward: {record["reward"]}, Value: {record["value"]}, Action: {record["action"]}'

    def serialize_episode(self, record):
        total_t = record['steps'][-1]['t']
        total_r = sum(s["reward"] for s in record['steps'])
        return f'        (Time: {total_t}, Reward: {total_r:.4f})'

    def run_episode(self, max_episode_steps):
        record = {'steps': []}

        obs = self.env.reset(max_episode_steps = max_episode_steps)

        if self.display:
            self.env.render()

        while True:
            action, v, logp = self.ac.step(tr.as_tensor(obs, dtype=tr.float64))
            step_obs, step_reward, term, trunc, step = self.env.step(action.item())
            step.update({
                'obs': obs,
                'action': action.item(),
                'obs_new': step_obs,
                't': len(record['steps']),
                'term': term,
                'trunc': trunc,
                'done': term or trunc,
                'reward': step_reward.item(),
            })
            self.buffer.store(obs, action, step_reward, v, logp)
            record['steps'].append(step)

            if self.args.verbose > 3:
                print(self.serialize_step(step))

            if self.display:
                self.env.render()

            obs = step_obs

            if step['done']:
                # if trajectory didn't reach terminal state, bootstrap value target
                v = 0 if term else self.ac.step(tr.as_tensor(obs, dtype=tr.float64))[1]
                self.buffer.finish_path(v)
                record['reward'] = sum(s['reward'] for s in record['steps'])
                record['length'] = len(record['steps'])
                break

        if self.args.verbose > 2:
            try:
                record['env'] = self.env.get_stat()
            except:
                record['env'] = {}
            print(self.serialize_episode(record))

        return record

    # only used when single threading
    def run_epoch(self):
        start = time.time()
        record = dict(
            reward = 0,
            episodes = [],
            num_steps = 0
        )
        remaining_steps = self.args.num_steps
        while remaining_steps > 0:
            max_episode_steps = min(self.args.max_steps,remaining_steps)
            episode = self.run_episode(max_episode_steps)
            remaining_steps -= len(episode['steps'])
            record['num_steps'] += len(episode['steps'])
            record['reward'] += sum(s['reward'] for s in episode['steps'])
            record['episodes'].append(episode)
        record['time'] = time.time() - start

        update_record = self.update()
        record.update(update_record)

        return record

        # print(f"Epoch {epoch} Reward {np.mean(ep_rews):.2f} Episodes: {len(ep_lens)} + {steps_per_epoch - sum(ep_lens)} Time {total_time:.2f}s")

    def update(self):
        record = {}
        data = self.buffer.get()

        # Train policy with a single step of gradient descent
        self.optimizer_a.zero_grad()
        loss_a, record['actor_loss'] = self.compute_loss_a(data)
        loss_a.backward()
        if not self.args.freeze:
            self.optimizer_a.step()

        # Value function learning
        for i in range(self.args.n_v_updates):
            self.optimizer_c.zero_grad()
            loss_c, record['critic_loss'] = self.compute_loss_c(data)
            loss_c.backward()
            if not self.args.freeze:
                self.optimizer_c.step()

        return record

    def compute_loss_a(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Loss
        pi, logp = self.ac.pi(obs, act)
        loss = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()

        record = dict(kl=approx_kl, ent=ent, loss=loss)

        return loss, record

    def compute_loss_c(self, data):
        obs, ret = data['obs'], data['ret']
        v = self.ac.v(obs)
        v.sub_(ret)
        v.square_()
        loss = v.mean()
        return loss, dict(loss=loss)
