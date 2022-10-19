import torch.nn as nn
from torch.autograd import Variable
from utils import *
from models import MLP
import random
from argparse import Namespace
import action_utils
from env_wrappers import *
from self_play import SPModel

class PlayModel(nn.Module):
    def __init__(self, args):
        super(PlayModel, self).__init__()
        self.affine1 = nn.Linear(args.input_dim, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            raise Exception("Shouldn't be here")
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
            # self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.action_head = nn.Linear(args.hid_size, args.num_actions)
        self.value_head = nn.Linear(args.hid_size, 1)

    def forward(self, x):
        x = x.double()
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        v = self.value_head(x)
        if self.continuous:
            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return F.log_softmax(self.action_head(x), dim=-1), v

class PlayWrapper(EnvWrapper):
    def __init__(self, args, env, **kwargs):
        super(PlayWrapper, self).__init__(env)
        self.args = args
        self.steps = 0
        self.test_steps = 0
        self.play_steps = 0
        self.playing = 0

    @property
    def observation_dim(self):
        dim = self.env.observation_dim # current observation
        dim += 1 # meta information: playing
        return dim

    @property
    def num_actions(self):
        """Assuming discrete actions"""
        return self.env.num_actions + 1

    @property
    def dim_actions(self):
        """Assuming discrete actions"""
        return 1

    @property
    def is_continuous(self):
        self.env.is_continuous

    def reset(self, **kwargs):
        self.stat = dict()
        self.stat['play_actions'] = 0
        self.env.reset(**kwargs)
        self.total_steps = 0
        self.total_test_steps = 0
        self.total_play_steps = 0
        self.playing = 0
        return self.get_state()

    def get_state(self):
        playing = torch.Tensor([[self.playing]])
        env_obs = self.env.get_state()
        return torch.cat((playing, env_obs), dim=1)

    def step(self, action):
        # Time management
        self.steps += 1
        if self.playing:
            self.play_steps += 1
        else:
            self.test_steps += 1


        if action == self.num_actions - 1:
            self.playing = 1 if not self.playing else 0
            self.stat['play_actions'] += 1
            #self.env.toggle_self_play(self.playing)
            obs = self.get_state()
            reward = 4.0 * self.playing
            return obs, reward, False, False, {}
        else:
            _, reward, term, trunc, info = self.env.step(action)
            obs = self.get_state()
            return obs, reward, term, trunc, info

    def render(self):
        obs = self.env.get_state()
        self.display_obs.append(obs)
        self.env.render()
