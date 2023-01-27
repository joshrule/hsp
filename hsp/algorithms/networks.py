from gym.spaces import Box, Discrete
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import Normal
from utils import mlp


class TargetEncoder(nn.Module):
    """Embed the target observation into a goal vector."""
    def __init__(self, obs_dim, h_dim, goal_dim, goal_diff):
        super().__init__()
        self.goal_diff = goal_diff
        self.enc = mlp([obs_dim, h_dim, goal_dim], nn.Tanh)

    def forward(self, obs):
        obs_target, obs_current = obs
        h_goal = self.enc(obs_target)
        h_current = self.enc(obs_current)
        if self.goal_diff:
            h_goal = h_goal - h_current
        return h_goal, h_current


class GoalNet(nn.Module):
    def __init__(self, obs_dim, h_dim, act_dim, goal_dim):
        super().__init__()
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.h_dim = h_dim

        self.obs_enc = mlp([obs_dim, h_dim, h_dim], nn.Tanh)
        self.goal_enc = mlp([goal_dim, h_dim, h_dim], nn.Tanh)
        self.action_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, act_dim)
        )

    def forward(self, x):
        obs_current, obs_target = x
        h_current = self.obs_enc(obs_current)
        h_target = self.goal_enc(obs_target)
        return self.action_head(h_current + h_target)


class InitNet(nn.Module):
    """Takes a current observation and initial observation pair as input."""
    def __init__(self, obs_dim, h_dim, act_dim, meta_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.meta_dim = meta_dim
        self.act_dim = act_dim
        self.enc_obs_curr = nn.Linear(obs_dim, h_dim)
        self.enc_obs_init = nn.Linear(obs_dim, h_dim)
        self.enc_meta = nn.Linear(meta_dim, h_dim)
        self.action_head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, act_dim),
        )

    def forward(self, x):
        obs_curr = self.enc_obs_curr(x[:, :self.obs_dim])
        obs_meta = self.enc_meta(x[:, self.obs_dim:self.obs_dim+self.meta_dim])
        obs_init = self.enc_obs_init(x[:, self.obs_dim+self.meta_dim:self.obs_dim*2+self.meta_dim])
        return self.action_head(obs_curr + obs_init + obs_meta)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = tr.nn.Parameter(tr.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = tr.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class CategoricalActor(Actor):

    def __init__(self, logits_net):
        super().__init__()
        self.logits_net = logits_net

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCategoricalActor(CategoricalActor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        super().__init__(logits_net)


class GoalCategoricalActor(CategoricalActor):
    """Takes an observation and goal vector pair as input."""
    def __init__(self, obs_dim, h_dim, act_dim, goal_dim):
        logits_net = GoalNet(obs_dim, h_dim, act_dim, goal_dim)
        super().__init__(logits_net)


class InitCategoricalActor(CategoricalActor):
    """Takes a current observation and initial observation pair as input."""
    def __init__(self, obs_dim, h_dim, act_dim, meta_dim):
        logits_net = InitNet(obs_dim, h_dim, act_dim, meta_dim)
        super().__init__(logits_net)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return tr.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with tr.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
