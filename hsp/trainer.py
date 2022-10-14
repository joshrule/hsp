from collections import namedtuple
from collections import deque
import random
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from action_utils import *


Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'mask', 'next_state',
                                       'reward', 'misc'))

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.params = [p for p in policy_net.parameters() if p.requires_grad]
        self.optimizer = optim.RMSprop(self.params, lr = args.lrate, alpha=0.97, eps=1e-6)

    def run_policy(self, state):
        raise NotImplementedError

    def _compute_misc(self, info):
        return info

    def _compute_mask(self, done, info):
        return 0 if done else 1

    def _compute_returns(self, returns, rewards, masks, batch):
        prev_return = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            prev_return = returns[i, 0]
        return returns

    def _postprocess_episode(self, episode):
        return episode

    def serialize_step(self, t, reward, done, stat):
        return f't={t}\treward={reward}'

    def serialize_episode(self, t, reward, done, stat):
        return f'        total time: {t}, total reward={reward:.4f}'

    def run_batch(self):
        batch = []
        stat = dict()
        stat['num_episodes'] = 0
        stat['batch_reward'] = 0
        remaining_steps = self.args.num_steps
        while remaining_steps > 0:
            episode, episode_reward = self.get_episode(remaining_steps)
            stat['num_episodes'] += 1
            stat['batch_reward'] += episode_reward
            batch += episode
            remaining_steps -= len(episode)
        stat['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, stat

    # only used when single threading
    def train_batch(self):
        batch, stat = self.run_batch()
        self.optimizer.zero_grad()
        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        if not self.args.freeze:
            self.optimizer.step()
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def get_episode(self, max_episode_steps):
        episode = []

        state = self.env.reset()
        # INVARIANT: assumes gym.wrappers.TimeLimit as outermost interface
        # print(f"remaining max_episode_steps: {max_episode_steps} in {type(self.env)}")
        self.env._max_episode_steps = max_episode_steps

        if self.display:
            self.env.render()

        done = False
        t = 0
        while not done:
            with torch.no_grad():
                action_out, value = self.run_policy(state)
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)

            step_state, step_reward, term, trunc, info = self.env.step(actual)
            t += 1

            done = term or trunc
            mask = self._compute_mask(done, info)

            if self.args.verbose > 1:
                print(self.serialize_step(t, step_reward, done, info))

            if self.display:
                self.env.render()

            misc = self._compute_misc(info)
            episode.append(Transition(state, np.array([action]), action_out, value, mask, step_state, step_reward, misc))

            state = step_state

        episode = self._postprocess_episode(episode)
        reward = sum(step.reward for step in episode)

        if self.args.verbose > 0:
            stat = self.env.get_stat() if hasattr(self.env, 'get_stat') else {}
            print(self.serialize_episode(t, reward, done, stat))

        return episode, reward

    def compute_grad(self, batch):
        stat = dict()
        rewards = torch.Tensor(batch.reward)
        rewards = self.args.reward_scale * rewards
        masks = torch.Tensor(batch.mask)
        actions = torch.from_numpy(np.concatenate(batch.action, 0))
        returns = torch.Tensor(actions.size(0),1)
        returns = self._compute_returns(returns, rewards, masks, batch)
        advantages = torch.Tensor(actions.size(0),1)

        # forward again in batch for speed-up
        states = Variable(torch.cat(batch.state, dim=0), requires_grad=False)
        action_out, values = self.run_policy(states)

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(Variable(actions, requires_grad=False), action_means, action_log_stds, action_stds)
            stat['action_std'] = action_stds.mean(dim=1, keepdim=False).sum().data[0]
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(Variable(actions, requires_grad=False), log_p_a)
        action_loss = -Variable(advantages, requires_grad=False) * log_prob
        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = Variable(returns, requires_grad=False)
        value_loss = (values - targets).pow(2).sum()
        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        loss.backward()
        return stat


class ReinforceTrainer(Trainer):
    def __init__(self, args, policy_net, env):
        super(ReinforceTrainer, self).__init__(args, policy_net, env)

    def run_policy(self, state):
        action_out, value = self.policy_net(Variable(state))
        return action_out, value

class PlayTrainer(Trainer):
    def __init__(self, args, policy_net, env):
        super(PlayTrainer, self).__init__(args, policy_net, env)

    def run_policy(self, state):
        action_out, value = self.policy_net(Variable(state))
        return action_out, value

    def serialize_episode(self, t, reward, done, stat):
        ser = super(PlayTrainer, self).serialize_episode(t, reward, done, stat)
        return ser + f" total plays: {stat['play_actions']}"


class SelfPlayTrainer(Trainer):
    def __init__(self, args, policy_net, env):
        super(SelfPlayTrainer, self).__init__(args, policy_net, env)

    def run_policy(self, state):
        action_out, value, _, _ = self.policy_net(Variable(state))
        return action_out, value

    def serialize_step(self, t, reward, done, stat):
        return f't={t}\treward={reward}\tmind={self.env.current_mind}'

    def serialize_episode(self, t, reward, done, stat):
        ser = f"        total time: {t}, alice reward={stat['reward_alice']}, bob reward={stat['reward_bob']}"
        if not self.env.success:
            ser += f"\n        FAILED: best diff: {self.env.stat['best_diff_value']} vs {self.env.sp_state_thresh}, step {self.env.stat['best_diff_step']}"
        return ser

    def _compute_misc(self, info):
        misc = super(SelfPlayTrainer, self)._compute_misc(info)
        # computed pre-step
        misc['mind'] = self.env.current_mind
        if self.args.sp_persist > 0:
            misc['sp_persist_count'] = self.env.persist_count
        return misc

    def _compute_mask(self, done, info):
        # disconnect episode if sp_switched
        return 0 if info.get('sp_switched') else super(SelfPlayTrainer, self)._compute_mask(done, info)

    def _compute_returns(self, returns, rewards, masks, batch):
        minds = [d['mind'] for d in batch.misc]

        if self.args.sp_persist > 0:
            persist_count = [d['sp_persist_count'] for d in batch.misc]

        prev_return = 0
        prev_alice_return = 0
        prev_bob_return = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            if self.args.sp_persist > 0:
                if minds[i] == 1:
                    # Add Alice's return from the previous episode, but only inside persist episodes.
                    if masks[i] == 0: # At Alice's last step in an episode.
                        returns[i] += self.args.sp_persist_discount * prev_alice_return
                    if persist_count[i] > 0:
                        prev_alice_return = returns[i, 0]
                    else:
                        prev_alice_return = 0

                if minds[i] == 2 and self.args.sp_persist_separate:
                    # do the same with Bob
                    if masks[i] == 0:
                        returns[i] += self.args.sp_persist_discount * prev_bob_return
                    if persist_count[i] > 0:
                        prev_bob_return = returns[i, 0]
                    else:
                        prev_bob_return = 0
            prev_return = returns[i, 0]
        return returns

    def _postprocess_episode(self, episode):
        switch_t = -1
        for t, step in enumerate(episode):
            if step.misc.get('sp_switched'):
                switch_t = t
                break
        if not self.env.test_mode:
            episode[switch_t] = episode[switch_t]._replace(reward=episode[switch_t].reward + self.env.reward_terminal_mind(1))
        if switch_t > -1:
            episode[-1] = episode[-1]._replace(reward=episode[-1].reward + self.env.reward_terminal_mind(2))
        return episode
