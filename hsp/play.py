import torch.nn as nn
from torch.autograd import Variable
from utils import *
from models import MLP
import random
from argparse import Namespace
import action_utils
from env_wrappers import *
from self_play import SPModel

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
        return self.env.num_actions

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
        self.playing = False
        return self.get_state()

    def get_state(self):
        playing = torch.Tensor([[self.playing*1]])
        env_obs = self.env.get_state()
        env_obs_1 = env_obs[:,:self.args.input_dim]
        env_obs_2 = env_obs[:,self.args.input_dim:]
        return torch.cat((env_obs_1, playing, env_obs_2), dim=1)

    def step(self, action):
        # Time management
        self.steps += 1
        if self.playing:
            self.play_steps += 1
        else:
            self.test_steps += 1

        if action == self.args.no_op:
            self.playing = not self.playing
            self.stat['play_actions'] += 1
            #self.env.toggle_self_play(self.playing)
            _, reward, term, trunc, info = self.env.step(action)
            obs = self.get_state()
            reward += 1.0 * self.playing
        else:
            _, reward, term, trunc, info = self.env.step(action)
            obs = self.get_state()
        return obs, reward, term, trunc, info

    def render(self):
        obs = self.env.get_state()
        self.display_obs.append(obs)
        self.env.render()
