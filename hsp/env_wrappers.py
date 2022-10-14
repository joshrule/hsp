import time
import numpy as np
import torch
import gym
from gym.spaces.utils import flatdim
from mazebase.torch_featurizers import GridFeaturizer
from utils import merge_stat

def torchify_obs(obs):
    return torch.from_numpy(obs).view(1, -1)

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(EnvWrapper, self).__init__(env, **kwargs)

    @property
    def observation_dim(self):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError

    @property
    def dim_actions(self):
        raise NotImplementedError

    @property
    def is_continuous(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def get_stat(self):
        stat = self.stat if hasattr(self, 'stat') else dict()
        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        elif hasattr(self.env, 'stat'):
            merge_stat(self.env.stat, stat)
        return stat

    def call_recursive(self, func_name, args, defaul_val=None):
        if hasattr(self, func_name):
            return getattr(self, func_name)
        elif hasattr(self.env, func_name):
            return getattr(self.env, func_name)
        elif hasattr(self.env, 'call_recursive'):
            return self.env.call_recursive(func_name, args, defaul_val)
        else:
            return defaul_val

    def property_recursive(self, property_name, defaul_val=None):
        if hasattr(self, property_name):
            return getattr(self, property_name)
        elif hasattr(self.env, property_name):
            return getattr(self.env, property_name)
        elif hasattr(self.env, 'property_recursive'):
            return self.env.property_recursive(property_name, defaul_val)
        else:
            return defaul_val

class GymWrapper(EnvWrapper):
    '''
    This wrapper assumes discrete actions.
    '''
    def __init__(self, env, **kwargs):
        super(GymWrapper, self).__init__(env, **kwargs)
        self.obs = None

    @property
    def observation_dim(self):
        return flatdim(self.env.observation_space)

    @property
    def num_actions(self):
        """Assuming discrete actions"""
        return self.env.action_space.n

    @property
    def dim_actions(self):
        """Assuming discrete actions"""
        return 1

    @property
    def is_continuous(self):
        """Assuming discrete actions"""
        return False

    def get_state(self):
        return torchify_obs(self.obs)

    def set_state(self, state):
        self.env.unwrapped.state = state.numpy()[0]
        self.env.renderer.reset()
        self.env.renderer.render_step()
        self.obs = state.numpy()
        return self.get_state()

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.get_state()

    def step(self, action):
        self.obs, reward, term, trunc, info = self.env.step(action)
        return self.get_state(), reward, term, trunc, info
