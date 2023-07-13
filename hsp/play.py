from utils import *
from env_wrappers import *

class PlayWrapper(EnvWrapper):
    def __init__(self, args, env, **kwargs):
        super().__init__(env, **kwargs)
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
        self.play_steps += 1 * self.playing
        self.test_steps += 1 * (not self.playing)

        # NOTE: Don't reward play naively, or agent learns to turn play on and off for reward.
        if action == self.args.no_op:
            print(f"            toggling play to {not self.playing} because action == {action}")
            self.playing = not self.playing
            self.stat['play_actions'] += 1
            self.env.toggle_self_play(self.playing)
            _, reward, term, trunc, info = self.env.step(action)
            obs = self.get_state()
        else:
            _, reward, term, trunc, info = self.env.step(action)
            obs = self.get_state()
        return obs, reward, term, trunc, info

    def render(self):
        self.env.render()
