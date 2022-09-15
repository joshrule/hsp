import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
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


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.affine1 = nn.Linear(args.input_dim, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)

    def forward(self, x):
        x, prev_hid = x
        next_hid = F.tanh(self.affine2(prev_hid) + self.affine1(x))
        v = self.value_head(next_hid)
        if self.continuous:
            action_mean = self.action_mean(next_hid)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v, next_hid
        else:
            return [F.log_softmax(head(next_hid)) for head in self.heads], v, next_hid
