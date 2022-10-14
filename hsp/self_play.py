import torch.nn as nn
from torch.autograd import Variable
from utils import *
from models import MLP
import random
from argparse import Namespace
import action_utils
from env_wrappers import *


# encode the target observation into a goal vector
class TargetEncoder(nn.Module):
    def __init__(self, args):
        super(TargetEncoder, self).__init__()
        self.args = args
        self.enc = nn.Sequential(
            nn.Linear(args.input_dim, args.hid_size),
            nn.Tanh(),
            nn.Linear(args.hid_size, args.sp_goal_dim))

    def forward(self, x):
        obs_target, obs_current = x
        h_goal = self.enc(obs_target)
        h_current = self.enc(obs_current)
        self.target_emb_snapshot = h_goal.data # used for plotting only
        if self.args.sp_goal_diff:
            h_goal = h_goal - h_current
        return h_goal, h_current

# takes a goal vector as input
class GoalPolicy(nn.Module):
    def __init__(self, args):
        super(GoalPolicy, self).__init__()
        self.args = args
        # current state encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(args.input_dim, args.hid_size),
            nn.Tanh(),
            nn.Linear(args.hid_size, args.hid_size))

        self.goal_enc = nn.Sequential(
            nn.Linear(args.sp_goal_dim, args.hid_size))

        self.final_hid = nn.Sequential(
            nn.Tanh(),
            nn.Linear(args.hid_size, args.hid_size),
            nn.Tanh())

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.action_head = nn.Linear(args.hid_size, args.num_actions)
        self.value_head = nn.Linear(args.hid_size, 1)


    def forward(self, x):
        obs_current, goal_vector = x

        # encode current state
        h_current = self.obs_enc(obs_current)
        h_goal = self.goal_enc(goal_vector)
        h_final = self.final_hid(h_current + h_goal)
        v = self.value_head(h_final)

        if self.continuous:
            action_mean = self.action_mean(h_final)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return F.log_softmax(self.action_head(h_final), dim=-1), v

class Bob(nn.Module):
    def __init__(self, args):
        super(Bob, self).__init__()
        self.args = args
        self.target_enc = TargetEncoder(args)
        self.goal_policy = GoalPolicy(args)

    def forward(self, x):
        obs_current = x[:, :self.args.input_dim]
        obs_current_meta = x[:, :self.args.input_dim+2]
        N = obs_current_meta.size()[1]
        obs_target = x[:, N:N+self.args.input_dim]
        N += obs_target.size()[1]
        goal_vector, enc_vector = self.target_enc([obs_target, obs_current])
        a, v = self.goal_policy([obs_current, goal_vector])
        return a, v, goal_vector, enc_vector

class Alice(nn.Module):
    def __init__(self, args):
        super(Alice, self).__init__()
        self.args = args
        print(f"input_dim {args.input_dim}, num_actions {args.num_actions}")
        self.enc_obs_curr = nn.Linear(args.input_dim, args.hid_size)
        self.enc_obs_init = nn.Linear(args.input_dim, args.hid_size)
        self.enc_meta = nn.Linear(2, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.action_head = nn.Linear(args.hid_size, args.num_actions)
        self.value_head = nn.Linear(args.hid_size, 1)

    def forward(self, x):
        obs_curr = x[:, :self.args.input_dim]
        obs_meta = x[:, self.args.input_dim:self.args.input_dim+2]
        obs_init = x[:, self.args.input_dim+2:self.args.input_dim*2+2]
        h1 = torch.tanh(self.enc_obs_curr(obs_curr) + self.enc_obs_init(obs_init) + self.enc_meta(obs_meta))
        h2 = torch.tanh(self.affine2(h1))
        v = self.value_head(h2)
        if self.continuous:
            action_mean = self.action_mean(h2)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return F.log_softmax(self.action_head(h2), dim=-1), v


class SPModel(nn.Module):
    def __init__(self, args):
        super(SPModel, self).__init__()
        self.args = args
        self.alice = Alice(args)
        self.bob = Bob(args)

    def forward(self, x):
        # x's last element is the mind index.
        mind = x[:,-1:].contiguous()

        amask = (mind == 1)
        bmask = (mind == 2)
        if amask.data.sum() == 0:
            return self.bob(x)
        if bmask.data.sum() == 0:
            y, v = self.alice(x)
            return y, v, None, None

        ax = x[amask.expand(x.size())].view(int(amask.data.sum()), x.size(1))
        bx = x[bmask.expand(x.size())].view(int(bmask.data.sum()), x.size(1))
        ay, av = self.alice(ax)
        by, bv, gv, ev = self.bob(bx)
        y = Variable(mind.data.new(x.size(0), ay.size(1)))
        y.masked_scatter_(amask.expand(y.size()), ay)
        y.masked_scatter_(bmask.expand(y.size()), by)
        v = mind.clone()
        v.masked_scatter_(amask, av)
        v.masked_scatter_(bmask, bv)

        return y, v, gv, ev


class SelfPlayWrapper(EnvWrapper):
    def __init__(self, args, env, **kwargs):
        super(SelfPlayWrapper, self).__init__(env, **kwargs)
        assert args.mode == 'self-play'
        self.args = args
        self.total_steps = 0
        self.total_test_steps = 0
        self.persist_count = self.args.sp_persist
        self.success = None
        self.sp_state_thresh = 10 * self.args.sp_state_thresh_0
        self.alice_limit = 1

    @property
    def observation_dim(self):
        dim = self.env.observation_dim # current observation
        dim += 2 # meta information: mode, time
        dim += self.env.observation_dim # target observation
        dim += 1 # meta information: current mind
        return dim

    @property
    def is_continuous(self):
        self.env.is_continuous

    @property
    def num_actions(self):
        """Assuming discrete actions"""
        return self.env.num_actions

    @property
    def dim_actions(self):
        """Assuming discrete actions"""
        return self.env.dim_actions

    def get_stat(self):
        if hasattr(self.env, 'get_stat'):
            s = self.env.get_stat()
            if 'success' in s:
                if not self.self_play:
                    self.stat['success_test'] = s['success']
                del s['success']
            merge_stat(s, self.stat)
        return self.stat

    def get_state(self):
        current_obs = self.env.get_state()
        if not self.self_play:
            mode = 1
            time = 0
        else:
            mode = -1
            time = self.current_time / self.max_steps
        obs = current_obs
        obs = torch.cat((obs, torch.Tensor([[mode, time]])), dim=1)
        obs = torch.cat((obs, self.target_obs), dim=1)
        obs = torch.cat((obs, torch.Tensor([[self.current_mind]])), dim=1)
        return obs

    def render(self):
        obs = self.env.get_state()
        self.display_obs.append(obs)
        self.env.render()

    def reset(self, max_steps = None, persist=False, self_play=True, **kwargs):
        if persist or self._should_persist():
            if self.args.verbose > 0:
                print(f"        PERSISTING: sub-episode {self.persist_count}")
            self.env.set_state(self.alice_last_state)
            self.persist_count += 1
        else:
            self.stat = dict()
            self.stat['reward_test'] = 0
            self.stat['reward_alice'] = 0
            self.stat['reward_bob'] = 0
            self.stat['num_steps_test'] = 0
            self.stat['num_steps_alice'] = 0
            self.stat['num_steps_bob'] = 0
            self.stat['num_episodes_test'] = 0
            self.stat['num_episodes_alice'] = 0
            self.stat['num_episodes_bob'] = 0
            if self.args.verbose > 0:
                print(f"        FULL RESET")
            self.env.reset()
            self.persist_count = 0
            if self_play:
                f = self.args.sp_state_thresh_factor
                self.sp_state_thresh *= f if self.success else 1/f
                if self.sp_state_thresh <= self.args.sp_state_thresh_1 and self.alice_limit < self.args.sp_steps:
                    print(f"\t\tincreasing alice_limit to {self.alice_limit} and resetting state_thresh from {self.sp_state_thresh:.4} to {(self.alice_limit + 1) * self.args.sp_state_thresh_0}")
                    self.alice_limit += 1
                    self.sp_state_thresh = self.alice_limit * self.args.sp_state_thresh_0
        self.stat['best_diff_value'] = np.inf
        self.stat['best_diff_step'] = np.inf
        self.alice_last_state = None
        self.initial_state = self.env.get_state() # bypass wrapper
        self.current_time = 0
        self.current_mind_time = 0
        self.success = False
        self.display_obs = []
        self.target_obs_curr = []
        self.target_reached_curr = 0
        self.self_play = self_play
        self.max_steps = max_steps if max_steps is not None else self.args.max_steps
        if self.self_play:
            self.target_obs = self.env.get_state()
            self.current_mind = 1
        else:
            self.target_obs = torch.zeros((1, self.env.observation_dim))
            self.current_mind = 2
        print(f"        starting as mind: {self.current_mind}")
        return self.get_state()

    def toggle_self_play(self, target = None):
        '''
        Call only if self has been reset at least once.
        '''
        if target is not None and target == self.self_play:
            return
        self.self_play = not self.self_play if target is None else target
        if self.self_play:
            print("turning on self-play")
            self.target_obs = self.env.get_state()
            self.current_mind = 1
            # Alice is only on during self-play, so mind_time resets.
            self.current_mind_time = 0
        else:
            print("turning off self-play")
            self.target_obs = torch.zeros((1, self.env.observation_dim))
            if self.current_mind == 1:
                self.current_mind_time = 0
            self.current_mind = 2
        return self.self_play

    def step(self, action):
        self.current_time += 1
        self.current_mind_time += 1

        obs_internal, reward, term, trunc, info = self.env.step(action)
        trunc |= self.current_time >= self.max_steps

        self.total_steps += 1

        # Test Mode
        if not self.self_play:
            self.total_test_steps += 1
            self.stat['num_steps_test'] += 1
            self.stat['reward_test'] += reward
            if term or trunc:
                self.stat['num_episodes_test'] += 1
            return self.get_state(), reward, term, trunc, info

        # Alice
        term = self.current_time >= self.args.max_steps
        if self.current_mind == 1:
            self.stat['num_steps_alice'] += 1
            if self.current_time == self.alice_limit:
                self._switch_mind()
                print(f'        switched to mind {self.current_mind} at t={self.current_time}')
                info['sp_switched'] = True
        # Bob
        else:
            self.stat['num_steps_bob'] += 1
            self.bob_last_state = self.env.get_state()
            diff = self._get_bob_diff(obs_internal, self.target_obs)
            if diff < self.stat['best_diff_value']:
                self.stat['best_diff_value'] = diff
                self.stat['best_diff_step'] = self.current_mind_time
            if bool(diff <= self.sp_state_thresh):
                self.success = True
                term = not self._should_persist()
                print(f"            SUCCESS: best diff: {self.stat['best_diff_value']} vs {self.sp_state_thresh}, step {self.stat['best_diff_step']}")

        if self.success or term:
            self.stat['reward_alice'] += self.reward_terminal_mind(1)
            self.stat['num_episodes_alice'] += 1
            self.stat['reward_bob'] += self.reward_terminal_mind(2)
            self.stat['num_episodes_bob'] += 1

        if self.success and not (term or trunc):
            self.reset(persist=True)
            if self.args.sp_persist > 0:
                self.stat['persist_count'] = self.persist_count

        obs = self.get_state()
        return obs, 0, term, trunc, info

    def _get_bob_diff(self, obs, target):
        sp_state_mask = self.env.property_recursive('sp_state_mask')
        if sp_state_mask is not None:
            diff = torch.dist(target.view(-1) * sp_state_mask,
                        obs.view(-1) * sp_state_mask)
        else:
            diff = torch.dist(target, obs)
        return diff

    def _should_persist(self):
        return (
            # haven't run the requisite number of episodes
            self.args.sp_persist - 1 > self.persist_count and
            # have started Bob at least once
            self.alice_last_state is not None and
            # were successful if required
            not (self.args.sp_persist_success and not self.success) and
            # have finished Bob at least once (for separate persistence)
            not (self.args.sp_persist_separate and self.bob_last_state is None)
        )

    def _switch_mind(self):
        self.current_mind_time = 0
        self.alice_last_state = self.env.get_state()
        self.current_mind = 2
        self.stat['switch_time'] = self.current_time
        if self.args.sp_mode == 'reverse':
            pass
        elif self.args.sp_mode == 'repeat':
            self.target_obs = self.env.get_state()
            if self.persist_count > 0 and self.args.sp_persist_separate:
                # start Bob from his previous state.
                self.env.set_state(self.bob_last_state)
            else:
                # start Bob from the/Alice's initial state.
                self.env.set_state(self.initial_state)
        else:
            raise RuntimeError("Invalid sp_mode")

    def reward_terminal_mind(self, mind):
        if not self.self_play:
            return 0
        if mind == 1:
            if self.current_mind == 2:
                return 1 - self.success
            return 0
        return 1 * self.success
