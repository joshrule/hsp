import math
import time
import numpy as np
import torch
import gym
from gym.spaces.utils import flatdim
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import TimeLimit
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

class ResetableTimeLimit(TimeLimit):
    def reset(self, max_episode_steps = None, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.
        Args:
            **kwargs: The kwargs to reset the environment with
        Returns:
            The reset environment
        """
        env = super(ResetableTimeLimit, self).reset(**kwargs)
        if max_episode_steps is not None:
            self._max_episode_steps = max_episode_steps
        return env

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

class NoOpCartPoleEnv(CartPoleEnv):
    def __init__(self, **kwargs):
        super(NoOpCartPoleEnv, self).__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 2 else -self.force_mag if action == 0 else 0
        print(f"action: {action}, force: {force}")
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
