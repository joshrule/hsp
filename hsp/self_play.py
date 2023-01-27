import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as fn
from algorithms.networks import InitCategoricalActor, GoalCategoricalActor, TargetEncoder, MLPCritic
from env_wrappers import EnvWrapper
from torch import optim
from torch.distributions import Categorical
from utils import *

def find_target(env, steps = 1):
    obs_0 = env.reset()
    print(f"initial: {obs_0}")
    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"\taction {action} -> obs {obs}")
    env.reset()
    env.set_state(obs_0)
    print(f"final: {obs} {env.get_state()}")
    return obs

class Buffer:
    """
    A buffer for storing trajectories experienced by a Self-Play agent
    interacting with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.

    Self-play introduces two other special computations:
    - the rewards are post-processed to reflect the results of self-play,
      awarding an extra unit of reward to Alice if Bob fails or Bob if Bob
      succeeds.
    - Rewards-to-go are computed separately based on the mind generating them.
    """

    def __init__(self, args, obs_dim, act_dim, size):
        self.args = args
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float64)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float64)
        self.adv_buf = np.zeros(size, dtype=np.float64)
        self.rew_buf = np.zeros(size, dtype=np.float64)
        self.ret_buf = np.zeros(size, dtype=np.float64)
        self.val_buf = np.zeros(size, dtype=np.float64)
        self.logp_buf = np.zeros(size, dtype=np.float64)
        self.misc_buf = []
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def __len__(self):
        return self.ptr

    def store(self, obs, act, rew, val, logp, misc):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.misc_buf.append(misc)
        self.ptr += 1

    def finish_path(self, env, last_val=0, last_val_a = 0):
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
        miscs = self.misc_buf[path_slice]

        # Identify t_switch
        t_switch = -1
        idx_switch = -1
        for t, misc in enumerate(miscs):
            if misc.get('sp_switched'):
                t_switch = t
                idx_switch = self.path_start_idx + t
                break

        if t_switch > -1:
            alice_slice = slice(self.path_start_idx, idx_switch+1)
            bob_slice = slice(idx_switch+1, self.ptr)

            # Add the last value for GAE estimation.
            rews_a = np.append(self.rew_buf[alice_slice], last_val_a)
            rews_b = np.append(self.rew_buf[bob_slice], last_val)
            vals_a = np.append(self.val_buf[alice_slice], last_val_a)
            vals_b = np.append(self.val_buf[bob_slice], last_val)

            # Update rewards based on results of self-play.
            term = any(misc.get('term') for misc in miscs)
            if env.self_play and term:
                rews_a[t_switch] += env.reward_terminal_mind(1)
                self.misc_buf[idx_switch]['reward'] += env.reward_terminal_mind(1)
            if t_switch > -1 and term and len(rews_b) > 1:
                rews_b[-2] += env.reward_terminal_mind(2)
                self.misc_buf[self.ptr-1]['reward'] += env.reward_terminal_mind(2)

            # Compute the GAE-Lambda advantage.
            deltas_a = rews_a[:-1] + self.args.gamma_r * vals_a[1:] - vals_a[:-1]
            deltas_b = rews_b[:-1] + self.args.gamma_r * vals_b[1:] - vals_b[:-1]
            self.adv_buf[alice_slice] = discount_cumsum(deltas_a, self.args.gamma_r * self.args.gamma_a)
            self.adv_buf[bob_slice] = discount_cumsum(deltas_b, self.args.gamma_r * self.args.gamma_a)

            # Compute rewards-to-go as targets for the value function.
            self.ret_buf[alice_slice] = discount_cumsum(rews_a, self.args.gamma_r)[:-1]
            self.ret_buf[bob_slice] = discount_cumsum(rews_b, self.args.gamma_r)[:-1]

        else:
            # Add the last value for GAE estimation.
            rews = np.append(self.rew_buf[path_slice], last_val)
            vals = np.append(self.val_buf[path_slice], last_val)

            # Update rewards based on results of self-play.
            term = any(misc.get('term') for misc in miscs)
            if env.self_play and term:
                rews[t_switch] += env.reward_terminal_mind(1)
                self.misc_buf[idx_switch]['reward'] += env.reward_terminal_mind(1)
            if t_switch > -1 and term:
                rews[-2] += env.reward_terminal_mind(2)
                self.misc_buf[self.ptr-1]['reward'] += env.reward_terminal_mind(2)

            # Compute the GAE-Lambda advantage.
            deltas = rews[:-1] + self.args.gamma_r * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = discount_cumsum(deltas, self.args.gamma_r * self.args.gamma_a)

            # Compute rewards-to-go as targets for the value function.
            self.ret_buf[path_slice] = discount_cumsum(rews, self.args.gamma_r)[:-1]

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
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        data_dict = {k: tr.as_tensor(v, dtype=tr.float64) for k,v in data.items()}
        data_dict['misc'] = self.misc_buf[:]
        self.misc_buf = []

        return data_dict


# Model components.


class Bob(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.actor = GoalCategoricalActor(args.input_dim, args.hid_size, args.num_actions, args.input_dim)
        # NOTE: Originally, the critics had ReLU output layers, but now they have Tanh.
        self.critic = MLPCritic(args.num_inputs, [args.hid_size]*args.l, nn.Tanh)

    def step(self, obs):
        obs_current = obs[:, :self.args.input_dim]
        n = obs_current.size()[1]+self.args.meta_dim
        obs_target = obs[:, n:n+self.args.input_dim]
        pi = self.actor._distribution([obs_current, obs_target])
        a = pi.sample()
        logp_a = self.actor._log_prob_from_distribution(pi, a)
        v = self.critic(obs)
        return a.detach().numpy(), v.detach().numpy(), logp_a.detach().numpy()


class Alice(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.actor = InitCategoricalActor(args.input_dim, args.hid_size, args.num_actions, args.meta_dim)
        # NOTE: Originally, the critics had ReLU output layers, but now they have Tanh.
        self.critic = MLPCritic(args.num_inputs, [args.hid_size]*args.l, nn.Tanh)

    def step(self, obs):
        pi = self.actor._distribution(obs)
        a = pi.sample()
        logp_a = self.actor._log_prob_from_distribution(pi, a)
        v = self.critic(obs)
        return a.detach().numpy(), v.detach().numpy(), logp_a.detach().numpy()


class SP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.alice = Alice(args)
        self.bob = Bob(args)

    def step(self, x):
        # x's last element is the mind index.
        mind = x[:,-1:].contiguous()

        amask = (mind == 1)
        bmask = (mind == 2)

        if amask.data.sum() == 0:
            return self.bob.step(x)
        if bmask.data.sum() == 0:
            return self.alice.step(x)

        ax = x[amask.expand(x.size())].view(int(amask.data.sum()), x.size(1))
        bx = x[bmask.expand(x.size())].view(int(bmask.data.sum()), x.size(1))
        aa, av, alogp_a = self.alice.step(ax)
        ba, bv, blogp_a = self.bob.step(bx)

        aa = tensor(aa)
        ba = tensor(ba)
        if aa.dim() == 1:
            aa = aa.unsqueeze(1)
            ba = ba.unsqueeze(1)

        a = tr.zeros(x.size(0), aa.size(1))
        a.masked_scatter_(amask.expand(a.size()), aa)
        a.masked_scatter_(bmask.expand(a.size()), ba)

        v = mind.clone()
        v.masked_scatter_(amask, tensor(av))
        v.masked_scatter_(bmask, tensor(bv))

        logp_a = mind.clone()
        logp_a.masked_scatter_(amask, tensor(alogp_a))
        logp_a.masked_scatter_(bmask, tensor(blogp_a))

        return a, v.squeeze(), logp_a.squeeze()

    def logp(self, x, a):
        # x's last element is the mind index.
        mind = x[:,-1:].contiguous()

        amask = (mind == 1)
        bmask = (mind == 2)

        if amask.data.sum() == 0:
            obs_current = x[:, :self.args.input_dim]
            n = obs_current.size()[1]+self.args.meta_dim
            obs_target = x[:, n:n+self.args.input_dim]
            bpi, logp = self.bob.actor([obs_current, obs_target], a)
            return None, bpi, logp
        if bmask.data.sum() == 0:
            api, logp = self.alice.actor(x, a)
            return api, None, logp

        if a.dim() == 1:
            a = a.unsqueeze(1)

        ax = x[amask.expand(-1, x.size(1))].view(int(amask.data.sum()), x.size(1))
        bx = x[bmask.expand(-1, x.size(1))].view(int(bmask.data.sum()), x.size(1))
        aa = a[amask.expand(-1, a.size(1))].view(int(amask.data.sum()), a.size(1))
        ba = a[bmask.expand(-1, a.size(1))].view(int(bmask.data.sum()), a.size(1))

        api, alogp = self.alice.actor(ax, aa.squeeze())

        obs_current = bx[:, :self.args.input_dim]
        n = obs_current.size()[1]+self.args.meta_dim
        obs_target = bx[:, n:n+self.args.input_dim]
        bpi, blogp = self.bob.actor([obs_current, obs_target], ba.squeeze())

        logp = mind.clone()
        logp.masked_scatter_(amask, alogp)
        logp.masked_scatter_(bmask, blogp)

        return api, bpi, logp.squeeze()

    def v(self, x):
        # x's last element is the mind index.
        mind = x[:,-1:].contiguous()

        amask = (mind == 1)
        bmask = (mind == 2)

        if amask.data.sum() == 0:
            return self.bob.critic(x)
        if bmask.data.sum() == 0:
            return self.alice.critic(x)

        ax = x[amask.expand(x.size())].view(int(amask.data.sum()), x.size(1))
        bx = x[bmask.expand(x.size())].view(int(bmask.data.sum()), x.size(1))
        av = self.alice.critic(ax)
        bv = self.bob.critic(bx)

        v = mind.clone()
        v.masked_scatter_(amask, av)
        v.masked_scatter_(bmask, bv)

        return v.squeeze()


# Algorithm.

class SelfPlayPPO(object):
    def __init__(self, args, env):
        obs_dim = args.num_inputs
        act_dim = env.action_space.shape

        self.args = args
        self.env = env
        self.ac = SP(args)
        self.buffer = Buffer(args, obs_dim, act_dim, args.num_steps)
        self.optimizer = optim.Adam(self.ac.parameters(), lr = args.pi_lrate)
        self.display = False

    def serialize_step(self, record):
        return f'            Time: {record["t"]}, Reward: {record["reward"]}, Value: {record["value"]}, Action: {record["action"]}, Mind: {record["mind"]}'

    def serialize_episode(self, record):
        total_t = record['steps'][-1]['t']
        total_r = sum(s["reward"] for s in record['steps'])
        alice_r = self.env.stat['reward_alice']
        bob_r = self.env.stat['reward_bob']
        test_r = self.env.stat['reward_test']
        if self.env.self_play:
            ser = f"        Time: {total_t}, Alice Reward: {alice_r}, Bob Reward: {bob_r} Total Reward: {total_r}"
        else:
            ser = f"        Time: {total_t}, Test Reward: {test_r} Total Reward: {total_r}"
        return ser

    def run_episode(self, max_episode_steps):
        record = {'steps': []}

        obs = self.env.reset(max_episode_steps = max_episode_steps, self_play = True)

        if self.display:
            self.env.render()

        while True:
            action, v, logp = self.ac.step(obs.double())
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
                'value': v.item(),
            })
            self.buffer.store(obs, action, step_reward, v, logp, step)
            record['steps'].append(step)

            if self.args.verbose > 3:
                print(self.serialize_step(step))

            if self.display:
                self.env.render()

            obs = step_obs

            if step['done']:
                # if trajectory didn't reach terminal state, bootstrap value target
                v = 0 if term else self.ac.step(obs.double())[1]
                self.buffer.finish_path(self.env, v)
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
        record = dict(reward = 0, episodes = [], num_steps = 0)
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

    def update(self):
        record = dict(actor_loss = [], critic_loss = [])
        data = self.buffer.get()

        # Train policy with multiple steps of gradient descent
        for i in range(self.args.n_pi_updates):
            self.optimizer.zero_grad()
            loss_a, record_a = self.compute_loss_a(data)
            record['actor_loss'].append(record_a)
            if record['actor_loss'][-1]['kl'] > 1.5 * self.args.target_kl:
                if self.args.verbose > 2:
                    print(f'# Early stopping at step {i} due to reaching max kl.')
                break
            loss_a.backward()
            if not self.args.freeze:
                self.optimizer.step()

        # Value function learning
        for i in range(self.args.n_v_updates):
            self.optimizer.zero_grad()
            loss_c, record_c = self.compute_loss_c(data)
            record['critic_loss'].append(record_c)
            loss_c.backward()
            if not self.args.freeze:
                self.optimizer.step()

        return record

    def compute_loss_a(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Loss
        api, bpi, logp = self.ac.logp(obs, act)
        ratio = tr.exp(logp - logp_old)
        clip_adv = tr.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * adv
        loss = -(tr.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        aent = api.entropy().mean().item() if api else None
        bent = bpi.entropy().mean().item() if bpi else None
        clipped = ratio.gt(1+self.args.eps_clip) | ratio.lt(1-self.args.eps_clip)
        clipfrac = tr.as_tensor(clipped, dtype=tr.float64).mean().item()

        record = dict(
            kl=approx_kl,
            alice_ent=aent,
            bob_ent=bent,
            cf=clipfrac,
            loss=loss.item()
        )

        return loss, record

    def compute_loss_c(self, data):
        obs, ret = data['obs'], data['ret']
        v = self.ac.v(obs)
        v.sub_(ret)
        v.square_()
        loss = v.mean()
        return loss, dict(loss=loss.item())


# Environment components.


class SelfPlayWrapper(EnvWrapper):
    def __init__(self, args, env, **kwargs):
        super().__init__(env, **kwargs)
        self.args = args
        self.total_steps = 0
        self.total_test_steps = 0
        self.persist_count = self.args.sp_persist
        self.success = None
        self.sp_state_thresh = 5 * self.args.sp_state_thresh_0
        self.alice_limit = 1
        self.done = False

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

    def get_state(self, adder = 0):
        current_obs = self.env.get_state()
        if not self.self_play:
            mode = 1
            time = 0
        else:
            mode = -1
            time = (self.current_time + adder) / self.args.max_steps
        obs = current_obs
        obs = tr.cat((obs, tr.Tensor([[mode, time]])), dim=1)
        obs = tr.cat((obs, self.target_obs), dim=1)
        obs = tr.cat((obs, tr.Tensor([[self.current_mind]])), dim=1)
        return obs

    def render(self):
        obs = self.env.get_state()
        self.display_obs.append(obs)
        self.env.render()

    def reset(self, self_play=True, **kwargs):
        if self._should_persist():
            print(f"{self.current_time} < {self.env._max_episode_steps}")
            if self.args.verbose > 1:
                print(f"        PERSISTING: sub-episode {self.persist_count + 1} {self._should_persist()}")
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
            # if self.args.verbose > 1:
            #     print(f"        FULL RESET")
            self.env.reset(**kwargs)
            self.persist_count = 0
            self.current_time = 0
            self.done = False
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
        self.current_mind_time = 0
        self.success = False
        self.display_obs = []
        self.target_obs_curr = []
        self.target_reached_curr = 0
        self.self_play = self_play
        if self.self_play:
            self.target_obs = self.env.get_state()
            self.current_mind = 1
        else:
            self.target_obs = tr.zeros((1, self.env.observation_dim))
            # self.target_obs = tr.tensor(np.random.rand(1, self.env.observation_dim)*0.1 - 0.05)
            # self.target_obs = find_target(self.env, steps = self.alice_limit)
            self.current_mind = 2
        # if self.args.verbose > 1:
        #     print(f"        starting as mind: {self.current_mind}")
        return self.get_state()

    def toggle_self_play(self, target = None):
        '''
        Call only if self has been reset at least once.
        '''
        if target == self.self_play:
            return
        self.self_play = not self.self_play if target is None else target
        self.initial_state = self.env.get_state()
        self.alice_last_state = None
        self.success = False
        if self.self_play:
            print(f"turning on self-play ({self.current_time})")
            self.target_obs = self.initial_state
            self.current_mind = 1
            # Alice is only on during self-play, so mind_time resets.
            self.current_mind_time = 0
        else:
            print(f"turning off self-play ({self.current_time})")
            self.target_obs = tr.zeros((1, self.env.observation_dim))
            if self.current_mind == 1:
                self.current_mind_time = 0
            self.current_mind = 2
        return self.self_play

    def step(self, action):
        self.current_time += 1
        self.current_mind_time += 1

        obs_internal, reward, term, trunc, info = self.env.step(action)

        self.total_steps += 1

        # Test Mode
        if not self.self_play:
            self.total_test_steps += 1
            self.stat['num_steps_test'] += 1
            self.stat['reward_test'] += reward.item()
            # TODO: remove diff lines after debugging Bob.
            diff = self._get_bob_diff(obs_internal, self.target_obs)
            self.success = diff < self.sp_state_thresh
            # term |= diff < self.success
            info['diff'] = diff
            if term or trunc:
                self.done = True
                self.stat['num_episodes_test'] += 1
            info['mask'] = 0 if (term or trunc) else 1
            info['mind'] = self.current_mind
            if self.args.sp_persist > 0:
                info['sp_persist_count'] = self.persist_count
            return self.get_state(), reward, term, trunc, info
            # NOTE: a hacky reward structure for testing Bob
            # return self.get_state(), 1.0 * self.success, term, trunc, info

        # Alice
        term |= self.current_time >= self.args.max_steps
        if self.current_mind == 1:
            self.stat['num_steps_alice'] += 1
            info['diff'] = None
            if self.current_mind_time >= self.alice_limit:
                info['sp_final_obs'] = self.get_state(adder = 1)
                self._switch_mind()
                if self.args.verbose > 3:
                    print(f'        switched to mind {self.current_mind} at t={self.current_time}')
                info['sp_switched'] = True
        # Bob
        else:
            self.stat['num_steps_bob'] += 1
            self.bob_last_state = self.env.get_state()
            diff = self._get_bob_diff(obs_internal, self.target_obs)
            info['diff'] = diff
            if diff < self.stat['best_diff_value']:
                self.stat['best_diff_value'] = diff
                self.stat['best_diff_step'] = self.current_mind_time
            print(f"                diff {diff}, sp_state_thresh {self.sp_state_thresh}")
            if bool(diff <= self.sp_state_thresh):
                self.success = True
                term = not self._should_persist()
                print(f"            SUCCESS: best diff: {self.stat['best_diff_value']} vs {self.sp_state_thresh}, step {self.stat['best_diff_step']}")

        if self.success or term:
            self.stat['reward_alice'] += self.reward_terminal_mind(1)
            self.stat['num_episodes_alice'] += 1
            self.stat['reward_bob'] += self.reward_terminal_mind(2)
            self.stat['num_episodes_bob'] += 1

        # Success
        if self.success and not (term or trunc):
            self.reset()
            if self.args.sp_persist > 0:
                self.stat['persist_count'] = self.persist_count

        # Failure
        if not self.success and (term or trunc):
            print(f"            FAILED: best diff: {self.stat['best_diff_value']} vs {self.sp_state_thresh}, step {self.stat['best_diff_step']}")

        self.done |= term | trunc
        obs = self.get_state()
        info['mask'] = 0 if info.get('sp_switched') else 0 if (term or trunc) else 1
        info['mind'] = self.current_mind
        if self.args.sp_persist > 0:
            info['sp_persist_count'] = self.persist_count
        return obs, self.args.sp_reward_coef * reward, term, trunc, info

    def _get_bob_diff(self, obs, target):
        sp_state_mask = self.env.property_recursive('sp_state_mask')
        if sp_state_mask is not None:
            diff = tr.dist(target.view(-1) * sp_state_mask,
                        obs.view(-1) * sp_state_mask)
        else:
            diff = tr.dist(target, obs)
        return diff

    def _should_persist(self):
        return (
            not self.done and
            # haven't run the requisite number of episodes
            self.args.sp_persist - 1 > self.persist_count and
            # have started Bob at least once
            self.alice_last_state is not None and
            # haven't run out of time: INVARIANT: assumes env has _max_episode_steps
            self.current_time < self.env._max_episode_steps and
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
