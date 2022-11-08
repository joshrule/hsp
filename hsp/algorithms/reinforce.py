import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as fn
from torch import optim
from torch.distributions import Categorical
from utils import *


class Buffer(object):
    def __init__(self):
        self.reset()

    def reset_episode(self):
        self.ep_o, self.ep_a, self.ep_r, self.ep_m = [], [], [], []
        self.ep_l = 0

    def reset_epoch(self):
        self.obs_buf, self.acts_buf,self.rtgs_buf, self.misc_buf = [], [], [], []
        self.total_eps = 0

    def reset(self):
        self.reset_epoch()
        self.reset_episode()

    def store_batch(self, gamma_r):
        """when episode is over, appends episode data to batch."""
        ep_obs, ep_acts, ep_rews, ep_misc, ep_len = self.get_episode()

        self.obs_buf += ep_obs
        self.acts_buf += ep_acts
        self.rtgs_buf += tensor(discount_cumsum(tensor(ep_rews).numpy(), gamma_r))
        self.misc_buf += ep_misc
        self.total_eps += 1

    def get_batch(self):
        # NOTE: for continuous action space reshape acts to [batch_size,1]
        b_a = tensor(self.acts_buf).reshape(-1, 1)
        b_o = tr.cat(self.obs_buf, dim=0)
        b_rtg = tensor(self.rtgs_buf).reshape(-1, 1)
        b_misc = self.misc_buf

        return b_o, b_a, b_rtg, b_misc

    def __len__(self):
        return len(self.obs_buf)

    def store_episode(self,o,a,r,info):
        # lists of tensors
        self.ep_o.append(o)
        self.ep_a.append(a)
        self.ep_r.append(r)
        # list of dicts
        self.ep_m.append(info)
        # int
        self.ep_l+=1

    def get_episode(self):
        return self.ep_o,self.ep_a,self.ep_r,self.ep_m,self.ep_l


class Actor(nn.Module):
    def __init__(self, obs_dim, h_dim, act_dim):
        super(Actor, self).__init__()
        self.act_dim = act_dim
        self.layer1 = nn.Linear(obs_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, act_dim)

    def forward(self, x):
        return fn.log_softmax(self.layer3(tr.tanh(self.layer2(tr.tanh(self.layer1(x.double()))))), dim=1)

    def act(self, obs):
        return self.policy(obs.double())[0].detach()

    def policy(self, x):
        logits = self.forward(x.double())
        act = Categorical(logits = logits).sample().unsqueeze(0)
        return act

    def log_prob(self, obs, acts):
        one_hot_acts = (acts.reshape(-1,1).double() == tr.arange(self.act_dim).double()).double()
        logp = tr.sum(one_hot_acts*self.forward(obs), 1)
        return logp


class Reinforce(object):
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.actor = Actor(args.input_dim, args.hid_size, args.num_actions)
        self.buffer = Buffer()
        self.display = False
        self.optimizer = optim.Adam(self.actor.parameters(), lr = args.lrate)

    def serialize_step(self, record):
        return f'            t={record["t"]}, reward={record["reward"]}, value={record["value"]} a={record["action"]}'

    def serialize_episode(self, record):
        total_t = record['steps'][-1]['t']
        total_r = sum(s["reward"] for s in record['steps'])
        return f'        (Time: {total_t}, Reward: {total_r:.4f})'

    def run_episode(self, max_episode_steps):
        self.buffer.reset_episode()
        record = {'steps': []}

        obs = self.env.reset(max_episode_steps = max_episode_steps)

        if self.display:
            self.env.render()

        while True:
            action = self.actor.act(obs.double())
            step_obs, step_reward, term, trunc, step = self.env.step(action)
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
            self.buffer.store_episode(step_obs, action, step_reward, step)
            record['steps'].append(step)

            if self.args.verbose > 3:
                print(self.serialize_step(step))

            if self.display:
                self.env.render()

            if step['done']:
                break
            obs = step_obs

        self.buffer.store_batch(self.args.gamma_r)

        if self.args.verbose > 2:
            try:
                record['env'] = self.env.get_stat()
            except:
                record['env'] = {}
            print(self.serialize_episode(record))

        return record

    def run_batch(self):
        self.buffer.reset()
        record = dict()
        record['reward'] = 0
        record['episodes'] = []
        record['num_steps'] = 0
        remaining_steps = self.args.num_steps
        while remaining_steps > 0:
            max_episode_steps = min(self.args.max_steps,remaining_steps)
            episode = self.run_episode(max_episode_steps)
            remaining_steps -= len(episode['steps'])
            record['num_steps'] += len(episode['steps'])
            record['reward'] += sum(s['reward'] for s in episode['steps'])
            record['episodes'].append(episode)
        return record

    # only used when single threading
    def train_batch(self):
        record = self.run_batch()

        self.optimizer.zero_grad()
        loss, record['loss'] = self.compute_actor_grad()
        loss.backward()
        if not self.args.freeze:
            self.optimizer.step()

        return record

    def compute_actor_grad(self):
        obs, acts, rtgs, misc = self.buffer.get_batch()
        rtgs -= tr.mean(rtgs)
        rtgs /= tr.std(rtgs)

        # Get log-likelihoods of state-action pairs.
        log_p = self.actor.log_prob(obs, acts).reshape(-1, 1)

        # Loss maximizes likelihood*returns.
        loss = -tr.mean(rtgs * log_p)

        record = {'loss': loss.item()}

        return loss, record
