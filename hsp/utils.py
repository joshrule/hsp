import numbers
import math
from collections import namedtuple

import numpy as np
import scipy

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time

LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))

def empty_mean(xs):
    return np.NAN if len(xs) == 0 else np.mean(xs)

def merge_stat(src, dest):
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, dict):
            if v['merge_op'] == 'last':
                dest[k] = v
            if v['merge_op'] == 'add':
                dest[k]['data'] = dest[k]['data'] + v['data']
            elif v['merge_op'] == 'concat':
                dest[k]['data'] = tr.cat([dest[k]['data'], v['data']], dim=0)
            elif v['merge_op'] == 'dict_update':
                dest[k]['data'].update(v['data'])
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * tr.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i, actions[i]]
    return log_prob


def multinomials_acc(actions, log_probs):
    acc = 0
    for i in range(len(log_probs)):
        acc += (tr.max(log_probs[i], 1)[1].data == actions[:, i]).float().sum()
    return acc / len(log_probs)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = tr.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = tr.cat(grads)
    return flat_grad


class Timer:
    def __init__(self, msg, sync=False):
        self.msg = msg
        self.sync = sync

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print("{}: {} s".format(self.msg, self.interval))


def pca(X, k=2):
    X_mean = tr.mean(X, dim=0, keepdim=True)
    X = X - X_mean.expand_as(X)
    U,S,V = tr.svd(tr.t(X))
    return tr.mm(X,U[:,:k])


def kl_criterion(mu, logvar):
    bs = mu.size()[0]
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * tr.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= bs

    return KLD


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def tensor(x):
    try:
        y = x.clone().detach()
    except:
        y = tr.tensor(x)
    return y.double()
