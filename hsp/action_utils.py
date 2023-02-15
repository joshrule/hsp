import numpy as np
import torch
from torch.autograd import Variable

def parse_env_args(args, env):
    # TODO: HACK
    if args.mode == "play":
        args.meta_dim = 3
        args.input_dim = env.env.env.observation_dim
        args.num_inputs = env.observation_dim
    elif args.mode == "self-play":
        args.meta_dim = 2
        args.input_dim = env.env.observation_dim
        args.num_inputs = env.observation_dim
    else:
        args.meta_dim = 0
        args.input_dim = env.observation_dim
        args.num_inputs = args.input_dim
    print(f"computing input dim as {args.input_dim} and num_inputs as {args.num_inputs} for env of {type(env)}")
    if env.num_actions > 0:
        # environment takes discrete action
        args.continuous = env.is_continuous
        assert env.dim_actions == 1
        args.num_actions = env.num_actions
        args.naction_heads = [env.num_actions]
    else:
        raise Exception("Not updated. Don't run me.")
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")

def select_action(args, action_out):
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        # print(f"log_p_a: {log_p_a}")
        p_a = log_p_a.exp()
        # print(f"p_a: {p_a}")
        return torch.multinomial(p_a, 1).detach()

def translate_action(args, env, action):
    if args.num_actions > 0:
        # environment takes discrete action
        # print(f"action: {action}")
        action = action.item()
        actual = action
        # print(f"actual: {actual}")
        return action, actual
    elif args.continuous:
        # environment takes continuous action
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                if args.sp_extra_action and i == len(action) - 1:
                    cp_action[-1] = 1 if cp_action[-1] > 0 else 0
                else:
                    low = env.action_space.low[i]
                    high = env.action_space.high[i]
                    cp_action[i] = cp_action[i] * args.action_scale
                    cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                    cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
    else:
        # ???
        actual = np.zeros(len(action))
        for i in range(len(action)):
            if args.sp and i == len(action) - 1:
                actual[-1] = action[-1].squeeze().data[0]
            else:
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
        action = [x.squeeze().data[0] for x in action]
        return action, actual
