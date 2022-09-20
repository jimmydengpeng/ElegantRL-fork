import os
import os.path as osp
from typing import Any, Callable, Optional, Union, List
from xmlrpc.client import ExpatParser
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint
from utils import Color, LogLevel, colorize, debug_msg, debug_print, get_formatted_time

'''config for agent'''


class Arguments:
    def __init__(self, agent_class: Callable, env=None, env_func=None, env_args: Optional[dict] = None):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.get_env_attr_from_env_or_env_args('env_num')  # env_num = 1. In vector env, env_num > 1.
        self.max_step: int = self.get_env_attr_from_env_or_env_args('max_step')  # the max step of an episode
        self.env_name = self.get_env_attr_from_env_or_env_args('env_name')  # the env name. Be used to set 'cwd'.
        self.state_dim = self.get_env_attr_from_env_or_env_args('state_dim')  # vector dimension (feature number) of state
        self.action_dim = self.get_env_attr_from_env_or_env_args('action_dim')  # vector dimension (feature number) of action
        self.if_discrete = self.get_env_attr_from_env_or_env_args('if_discrete')  # discrete or continuous action space
        self.target_return = self.get_env_attr_from_env_or_env_args('target_return')  # target average episode return

        self.agent_class = agent_class  # the class of DRL algorithm
        self.net_dim = 2 ** 8  # the network width
        self.num_layer = 3  # layer number of MLP (Multi-layer perception, `assert layer_num>=2`)
        self.horizon_len = 1  # number of steps per exploration
        self.if_off_policy: bool = self.if_off_policy() #type: ignore # agent is on-policy or off-policy
        if self.if_off_policy:  # off-policy
            self.max_memo = 2 ** 21  # capacity of replay buffer, 2 ** 21 ~= 2e6
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # epoch num
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
            self.num_seed_steps = 2  # the total samples for warm-up is num_seed_steps * env_num * num_steps_per_episode
            self.num_steps_per_episode = 128
            self.n_step = 1  # multi-step TD learning
        else:  # on-policy
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.if_use_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.lambda_critic = 2 ** 0  # the objective coefficient of critic network
        self.learning_rate = 2 ** -15  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus: Union[int, List[int], List[List]] = 0  # `int` means the ID of single GPU, -1 means CPU,

        '''Arguments for evaluate'''
        self.cwd = ""  # current working directory to save model. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_over_write = False  # overwrite the best policy network (actor.pth)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        '''Arguments for evaluate'''
        self.save_gap = 2  # save the policy network (actor.pth) for learning curve, +np.inf means don't save
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return
        self.eval_env_func = None  # eval_env = eval_env_func(*eval_env_args)
        self.eval_env_args = None  # eval_env = eval_env_func(*eval_env_args)

    def init_before_training(self):
        # debug_msg("<Arguments.init_before_training> setting cwd...")
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set cwd'''
        # (project_root)/elegantrl/train/config.py
        prj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))) # 3rd parent directory
        exp_path = osp.join(prj_root, "experiments")
        if not osp.exists(exp_path):
            os.mkdir(exp_path)
        self.cwd =osp.join(exp_path, f'{self.env_name}_{self.agent_class.__name__[5:]}_{self.learner_gpus}_{get_formatted_time()}')
        debug_print("cwd set in:", args=self.cwd, level=LogLevel.SUCCESS)

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            debug_print("<Arguments.init_before_training> Remove cwd:", args=f"{self.cwd}", level=LogLevel.WARNING)
        else:
            debug_print("<Arguments.init_before_training> Keep cwd:", args=f"{self.cwd}", level=LogLevel.WARNING)
        os.makedirs(self.cwd, exist_ok=True)

        if self.env == None:
            debug_msg("<Arguments.init_before_training> args.env is None, building env...")
            self.env = build_env(self.env, self.env_func, self.env_args)

    def get_env_attr_from_env_or_env_args(self, attr: str) -> Any:
        try:
            attribute_value = getattr(self.env, attr) if self.env_args is None else self.env_args[attr]
        except Exception as error:
            print(f"| Arguments.update_attr() Error: {error}")
            attribute_value = None
        return attribute_value

    def if_off_policy(self) -> bool: #type: ignore
        name = self.agent_class.__name__
        if_off_policy = all((name.find('PPO') == -1, name.find('A2C') == -1)) # 当且仅当算法(agent)名称中既没有PPO，也没有A2C时，才是off_policy，也即一旦名称中包含PPO或A2C，即为on_policy，默认算法为off_plicy
        return if_off_policy

    def print(self):
        # prints out args in a neat, readable format
        pprint(vars(self))


'''config for env(simulator)'''


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """get a dict `env_args` about a standard OpenAI gym env information.

    env_args = {
        'env_num': 1,
        'env_name': env_name,            # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,            # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,          # [int] the dimension of state
        'action_dim': action_dim,        # [int] the dimension of action
        'if_discrete': if_discrete,      # [bool] action space is discrete or continuous
        'target_return': target_return,  # [float] We train agent to reach this target episode return.
    }

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env inforamtion.
    :return: env_args [dict]
    """
    import gym

    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1
    target_return = getattr(env, 'target_return') if hasattr(env, 'target_return') else +np.inf

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        print(env_name)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        target_return = getattr(env, 'target_return', None)
        target_return_default = getattr(env.spec, 'reward_threshold', None)
        if target_return is None:
            target_return = target_return_default
        if target_return is None:
            target_return = 2 ** 16

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.discrete.Discrete) #type: ignore
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.box.Box): #type: ignore # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                # print('WARNING: env.action_space.high', env.action_space.high)
                debug_print('WARNING: env.action_space.high: ', args=env.action_space.high, level=LogLevel.WARNING, inline=True)
            if not any(env.action_space.low - 1):
                # print('WARNING: env.action_space.low', env.action_space.low)
                debug_print('WARNING: env.action_space.low', args=env.action_space.low, level=LogLevel.WARNING)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = getattr(env, 'env_num', env_num)
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = getattr(env, 'target_return', target_return)

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete,
                'target_return': target_return, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")

        debug_msg("env_args:", level=LogLevel.INFO)
        print(f"{env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: dict):  # [ElegantRL.2021.12.12]
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    filtered_kwargs = {key: kwargs[key] for key in common_args}
    return filtered_kwargs


def build_env(env=None, env_func: Optional[Callable] = None, env_args: Optional[dict] = None):  # [ElegantRL.2021.12.12]
    debug_msg("<config.py/builing_env> building env...")
    if env is not None:
        debug_msg(f"{__name__}.py/build_env> env is not None, deepcopy...")
        env = deepcopy(env)

    else:
        assert env_func is not None
        assert env_args is not None
        if env_func.__module__ == 'gym.envs.registration':
                                          # ↳__module__ : 表示当前操作的对象（的类定义在）在那个模块
            debug_print("<config.py/build_env> env is None, using", args=colorize("env_func()", Color.RED)+colorize(" initializing env...", color=LogLevel.DEBUG.value, bold=False), inline=True)
            import gym
            gym.logger.set_level(40)  # Block warning
            env = env_func(id=env_args['env_name'])
        else:
            debug_msg("using kwargs_filter...")
            env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    set_attr_for_env(env, env_args)
    debug_print(f"<{__name__}.py/set_attr_for_envbuild_env> env built:", args=env, level=LogLevel.SUCCESS, inline=True)
    return env

def set_attr_for_env(env, env_args):
    debug_msg(f"<{__name__}.py/set_attr_for_env> setattr for env...")
    assert env_args is not None
    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])
    # env.max_step = env.max_step if hasattr(env, 'max_step') else env_args['max_step']
    # env.if_discrete = env.if_discrete if hasattr(env, 'if_discrete') else env_args['if_discrete']