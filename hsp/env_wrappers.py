import math
import time
import numpy as np
import torch
import gym
import pygame
from gym.spaces.utils import flatdim
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import TimeLimit
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

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Box(np.array([0, 0, -1, -1, 0, 0]), np.array([size-1, size-1, size-1, size-1, 100, 1]), shape=(6,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = np.array(list(self._agent_location)+list(self._target_location)+[self._agent_health, 1*self.food_available])
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None, return_info=None, **kwargs):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed, **kwargs)

        # Choose the agent's location uniformly at random.
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while list(self._agent_location) == list(self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Initialize health to middling.
        self._agent_health = 50
        self.current_time = 0
        self.food_available = True

        return self._get_obs()

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction agent walks in.
        direction = self._action_to_direction[action]
        # Use `np.clip` to make sure agent doesn't leave the grid.
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        self.current_time += 1
        if self.current_time % 10 == 0 and not self.food_available:
            self.food_available = True
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            while list(self._agent_location) == list(self._target_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        agent_ate = list(self._agent_location) == list(self._target_location) and self.food_available
        if agent_ate:
            # print("            agent ate")
            self._agent_health += min(20,100-self._agent_health)
            self.food_available = False
            self._target_location = np.array([-1, -1])
        elif self._agent_health > 0:
            self._agent_health -= 1

        # An episode is done iff the agent has no health.
        terminated = self._agent_health == 0
        truncated = False
        reward = -(50 - self._agent_health)/50 if self._agent_health < 50 else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels.

        # First we draw the target.
        if self.food_available:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (self._target_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        # Now we draw the agent.
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines.
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
