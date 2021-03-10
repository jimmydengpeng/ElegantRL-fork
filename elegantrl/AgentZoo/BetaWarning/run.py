import os
import time
from copy import deepcopy

import torch
import numpy as np
import numpy.random as rd

from elegantrl.agent import ReplayBuffer, ReplayBufferMP
from elegantrl.env import PreprocessEnv
import gym

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

"""Plan to
Arguments(), let Arguments ask AgentXXX to get if_on_policy
Mega PPO and GaePPO into AgentPPO(..., if_gae)
"""

'''DEMO'''


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 8  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.target_step = 2 ** 10  # collect target_step, then update network
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.show_gap = 2 ** 8  # show the Reward and Loss value per show_gap seconds
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main=True):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| There should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.env.env_name}_{self.gpu_id}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def demo1_discrete_action_space():
    args = Arguments(agent=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters

    '''choose an DRL algorithm'''
    from elegantrl.agent import AgentD3QN  # AgentDQN,AgentDuelDQN, AgentDoubleDQN,
    args.agent = AgentD3QN()

    '''choose environment'''
    # args.env = PreprocessEnv(env=gym.make('CartPole-v0'))
    # args.net_dim = 2 ** 7  # change a default hyper-parameters
    # args.batch_size = 2 ** 7
    "TotalStep: 2e3, TargetReward: , UsedTime: 10s"
    args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 8
    "TotalStep: 6e4, TargetReward: 200, UsedTime: 600s"

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo2_continuous_action_space_off_policy():
    args = Arguments(if_on_policy=False)

    '''choose an DRL algorithm'''
    from elegantrl.agent import AgentModSAC  # AgentSAC, AgentTD3, AgentDDPG
    args.agent = AgentModSAC()  # AgentSAC(), AgentTD3(), AgentDDPG()

    '''choose environment'''
    env = gym.make('Pendulum-v0')
    env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
    args.env = PreprocessEnv(env=env)
    args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
    # args.eval_times2 = 2 ** 4  # set a large eval_times to get a precise learning curve
    "TD3    TotalStep: 3e4, TargetReward: -200, UsedTime: 300s"
    "ModSAC TotalStep: 4e4, TargetReward: -200, UsedTime: 400s"
    # args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    # args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302
    "TD3    TotalStep:  9e4, TargetReward: 100, UsedTime: 3ks"
    "TD3    TotalStep: 20e4, TargetReward: 200, UsedTime: 5ks"
    "SAC    TotalStep:  9e4, TargetReward: 200, UsedTime: 3ks"
    "ModSAC TotalStep:  5e4, TargetReward: 200, UsedTime: 1ks"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
    # args.net_dim = 2 ** 8
    # args.break_step = int(2e5)
    # args.if_allow_break = False
    "TotalStep: 2e5, TargetReward: 300, UsedTime: 5000s"

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo2_continuous_action_space_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy

    '''choose an DRL algorithm'''
    from elegantrl.agent import AgentGaePPO  # AgentPPO
    args.agent = AgentGaePPO()

    '''choose environment'''
    # env = gym.make('Pendulum-v0')
    # env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
    # args.env = PreprocessEnv(env=env)
    # args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
    "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
    args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
    args.reward_scale = 2 ** 0  # RewardRange: -800 < -200 < 200 < 302
    "TotalStep: 8e5, TargetReward: 200, UsedTime: 1500s"
    # args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
    # args.reward_scale = 2 ** 0  # RewardRange: -200 < -150 < 300 < 334
    # args.gamma = 0.96
    "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo3_custom_env_fin_rl():
    from elegantrl.agent import AgentGaePPO

    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentGaePPO()  # PPO+GAE (on-policy)

    from elegantrl.env import FinanceMultiStockEnv  # a standard env for ElegantRL, not need PreprocessEnv()
    args.env = FinanceMultiStockEnv(if_train=True, train_beg=0, train_len=1024)
    args.env_eval = FinanceMultiStockEnv(if_train=False, train_beg=0, train_len=1024)  # eva_len = 1699 - train_len
    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
    args.break_step = int(5e6)
    args.net_dim = 2 ** 8
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times1 = 2 ** 2
    args.eval_times2 = 2 ** 4
    args.if_allow_break = False
    "TotalStep:  2e5, TargetReward: 1.25, UsedTime:  200s"
    "TotalStep:  4e5, TargetReward: 1.50, UsedTime:  400s"
    "TotalStep: 10e5, TargetReward: 1.62, UsedTime: 1000s"

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 8
    train_and_evaluate__multiprocessing(args)


def demo4_bullet_mujoco_off_policy():
    args = Arguments(if_on_policy=False)
    args.random_seed = 10086

    from elegantrl.agent import AgentModSAC  # AgentSAC, AgentTD3, AgentDDPG
    args.agent = AgentModSAC()  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent.if_use_dn = True
    args.net_dim = 2 ** 7  # default is 2 ** 8 is too large for if_use_dn = True

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = PreprocessEnv(env=gym.make('AntBulletEnv-v0'))
    args.env.max_step = 2 ** 10  # important, default env.max_step=1800?

    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -2  # RewardRange: -50 < 0 < 2500 < 3340
    args.max_memo = 2 ** 20
    args.batch_size = 2 ** 9
    args.show_gap = 2 ** 8  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder
    "TotalStep: 3e5, TargetReward: 1500, UsedTime:  8ks"
    "TotalStep: 6e5, TargetReward: 2500, UsedTime: 20ks"

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


def demo4_bullet_mujoco_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy

    from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = PreprocessEnv(env=gym.make('AntBulletEnv-v0'))
    args.env.max_step = 2 ** 10

    args.break_step = int(2e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
    args.max_memo = 2 ** 11
    args.repeat_times = 2 ** 3
    args.batch_size = 2 ** 10
    args.net_dim = 2 ** 9
    args.show_gap = 2 ** 8  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder
    "TotalStep:  2e6, TargetReward: 1500, UsedTime:  3ks"
    "TotalStep: 13e6, TargetReward: 2400, UsedTime: 21ks"

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate__multiprocessing(args)


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    env_eval = args.env_eval

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    show_gap = args.show_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    buffer = ReplayBuffer(max_len=max_memo + max_step, if_on_policy=if_on_policy, if_gpu=True,
                          state_dim=state_dim, action_dim=1 if if_discrete else action_dim)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, show_gap=show_gap)  # build Evaluator

    '''prepare for training'''
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)

        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''start training'''
    if_reach_goal = False
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.store_transition(env, buffer, target_step, reward_scale, gamma)

        total_step += steps

        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        with torch.no_grad():  # speed up running
            if_reach_goal = evaluator.evaluate_save(agent.act, steps, obj_a, obj_c)


'''multiprocessing training'''


def train_and_evaluate__multiprocessing(args):
    act_workers = args.rollout_num
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    '''Python.multiprocessing + PyTorch + CUDA --> semaphore_tracker:UserWarning
    https://discuss.pytorch.org/t/issue-with-multiprocessing-semaphore-tracking/22943/4
    '''
    print("| multiprocessing, act_workers:", act_workers)

    import multiprocessing as mp  # Python built-in multiprocessing library
    # mp.set_start_method('spawn', force=True)  # force=True to solve "RuntimeError: context has already been set"
    # mp.set_start_method('fork', force=True)  # force=True to solve "RuntimeError: context has already been set"
    # mp.set_start_method('forkserver', force=True)  # force=True to solve "RuntimeError: context has already been set"

    pipe1_eva, pipe2_eva = mp.Pipe()  # Pipe() for Process mp_evaluate_agent()
    pipe2_exp_list = list()  # Pipe() for Process mp_explore_in_env()

    process_train = mp.Process(target=mp__update_params, args=(args, pipe2_eva, pipe2_exp_list))
    process_evaluate = mp.Process(target=mp_evaluate_agent, args=(args, pipe1_eva))
    process = [process_train, process_evaluate]

    for worker_id in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        pipe2_exp_list.append(exp_pipe1)
        process.append(mp.Process(target=mp_explore_in_env, args=(args, exp_pipe2, worker_id)))

    print("| multiprocessing, None:")
    [p.start() for p in process]
    process_evaluate.join()
    process_train.join()
    import warnings
    warnings.simplefilter('ignore', UserWarning)
    # semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
    # print("[W CudaIPCTypes.cpp:22]← Don't worry about this warning.")
    [p.terminate() for p in process]


def mp__update_params(args, pipe1_eva, pipe1_exp_list):
    args.init_before_training(if_main=False)

    '''basic arguments'''
    env = args.env
    cwd = args.cwd
    agent = args.agent
    rollout_num = args.rollout_num

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer'''
    agent.init(net_dim, state_dim, action_dim)
    if_on_policy = agent.__class__.__name__ in {'AgentPPO', 'AgentGaePPO', 'AgentInterPPO'}

    '''send'''
    pipe1_eva.send(agent.act)  # send
    # act = pipe2_eva.recv()  # recv

    buffer_mp = ReplayBufferMP(max_len=max_memo + max_step * rollout_num, if_on_policy=if_on_policy,
                               state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                               rollout_num=rollout_num, if_gpu=True)

    '''prepare for training'''
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = 0
            for i in range(rollout_num):
                pipe1_exp = pipe1_exp_list[i]

                # pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
                buf_state, buf_other = pipe1_exp.recv()

                steps += len(buf_state)
                buffer_mp.extend_buffer(buf_state, buf_other, i)

        agent.update_net(buffer_mp, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(env, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(env, 'cri_target', None) in dir(
            agent) else None
    total_step = steps
    '''send'''
    pipe1_eva.send((agent.act, steps, 0, 0.5))  # send
    # act, steps, obj_a, obj_c = pipe2_eva.recv()  # recv

    '''start training'''
    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        '''update ReplayBuffer'''
        steps = 0  # send by pipe1_eva
        for i in range(rollout_num):
            pipe1_exp = pipe1_exp_list[i]
            '''send'''
            pipe1_exp.send(agent.act)
            # agent.act = pipe2_exp.recv()
            '''recv'''
            # pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            buf_state, buf_other = pipe1_exp.recv()

            steps += len(buf_state)
            buffer_mp.extend_buffer(buf_state, buf_other, i)
        total_step += steps

        '''update network parameters'''
        obj_a, obj_c = agent.update_net(buffer_mp, target_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        '''send'''
        pipe1_eva.send((agent.act, steps, obj_a, obj_c))
        # q_i_eva_get = pipe2_eva.recv()

        if_solve = pipe1_eva.recv()

        if pipe1_eva.poll():
            '''recv'''
            # pipe2_eva.send(if_solve)
            if_solve = pipe1_eva.recv()

    buffer_mp.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                               env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    '''send'''
    pipe1_eva.send('stop')
    # q_i_eva_get = pipe2_eva.recv()
    time.sleep(4)


def mp_explore_in_env(args, pipe2_exp, worker_id):
    args.init_before_training(if_main=False)

    '''basic arguments'''
    env = args.env
    agent = args.agent
    rollout_num = args.rollout_num

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    target_step = args.target_step
    gamma = args.gamma
    reward_scale = args.reward_scale

    random_seed = args.random_seed
    torch.manual_seed(random_seed + worker_id)
    np.random.seed(random_seed + worker_id)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer'''
    agent.init(net_dim, state_dim, action_dim)
    agent.state = env.reset()

    if_on_policy = agent.__class__.__name__ in {'AgentPPO', 'AgentGaePPO', 'AgentInterPPO'}
    buffer = ReplayBuffer(max_len=max_memo // rollout_num + max_step, if_on_policy=if_on_policy,
                          state_dim=state_dim, action_dim=1 if if_discrete else action_dim, if_gpu=False)

    '''start exploring'''
    exp_step = target_step // rollout_num
    with torch.no_grad():
        if not if_on_policy:
            explore_before_training(env, buffer, exp_step, reward_scale, gamma)

            buffer.update__now_len__before_sample()

            pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            # buf_state, buf_other = pipe1_exp.recv()

            buffer.empty_memories__before_explore()

        while True:
            agent.store_transition(env, buffer, exp_step, reward_scale, gamma)

            buffer.update__now_len__before_sample()

            pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            # buf_state, buf_other = pipe1_exp.recv()

            buffer.empty_memories__before_explore()

            # pipe1_exp.send(agent.act)
            agent.act = pipe2_exp.recv()


def mp_evaluate_agent(args, pipe2_eva):
    args.init_before_training(if_main=True)

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent_id = args.gpu_id
    env_eval = args.env_eval

    '''evaluating arguments'''
    show_gap = args.show_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Evaluator'''
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=torch.device("cpu"), env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, show_gap=show_gap)  # build Evaluator

    '''act_cpu without gradient for pipe1_eva'''
    # pipe1_eva.send(agent.act)
    act = pipe2_eva.recv()

    act_cpu = deepcopy(act).to(torch.device("cpu"))  # for pipe1_eva
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''start evaluating'''
    with torch.no_grad():  # speed up running
        act, steps, obj_a, obj_c = pipe2_eva.recv()  # pipe2_eva (act, steps, obj_a, obj_c)

        if_loop = True
        while if_loop:
            '''update actor'''
            while not pipe2_eva.poll():  # wait until pipe2_eva not empty
                time.sleep(1)
            steps_sum = 0
            while pipe2_eva.poll():  # receive the latest object from pipe
                '''recv'''
                # pipe1_eva.send((agent.act, steps, obj_a, obj_c))
                # pipe1_eva.send('stop')
                q_i_eva_get = pipe2_eva.recv()

                if q_i_eva_get == 'stop':
                    if_loop = False
                    break
                act, steps, obj_a, obj_c = q_i_eva_get
                steps_sum += steps
            act_cpu.load_state_dict(act.state_dict())
            if_solve = evaluator.evaluate_save(act_cpu, steps_sum, obj_a, obj_c)
            '''send'''
            pipe2_eva.send(if_solve)
            # if_solve = pipe1_eva.recv()

            evaluator.save_npy_draw_plot()

    '''save the model, rename the directory'''
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - evaluator.start_time:.0f}')

    while pipe2_eva.poll():  # empty the pipe
        pipe2_eva.recv()


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eva_times1 = eval_times1
        self.eva_times2 = eval_times2
        self.env = env
        self.target_reward = env.target_reward

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def evaluate_save(self, act, steps, obj_a, obj_c):
        reward_list = [get_episode_return(self.env, act, self.device)
                       for _ in range(self.eva_times1)]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            reward_list += [get_episode_return(self.env, act, self.device)
                            for _ in range(self.eva_times2 - self.eva_times1)]
            r_avg = np.average(reward_list)  # episode return average
            r_std = float(np.std(reward_list))  # episode return std
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            '''save actor.pth'''
            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_reward)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return if_reach_goal

    def save_npy_draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return


def save_learning_curve(recorder, cwd='.', save_title='learning curve'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_a = recorder[:, 3]
    obj_c = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    label = 'objA'
    ax11.set_ylabel(label, color=color11)
    ax11.plot(steps, obj_a, label=label, color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/plot_learning_curve.jpg")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, buffer, target_step, reward_scale, gamma):
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_buffer(state, other)

        state = env.reset() if done else next_state
    return steps


'''DEMO wait for updating'''
# def run__demo():
#     import elegantrl.agent as agent
#     from elegantrl.env import prep_env
#     # from elegantrl.main import Arguments, train_and_evaluate, train_and_evaluate__multiprocessing
#     import gym
#
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#
#     """DEMO 1: Discrete action env: CartPole-v0 of gym"""
#     args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters
#
#     args.agent_rl = agent.AgentD3QN  # choose an DRL algorithm
#     args.env = prep_env(env=gym.make('CartPole-v0'))
#     args.net_dim = 2 ** 7  # change a default hyper-parameters
#     # args.env = decorate_env(env=gym.make('LunarLander-v2'))
#     # args.net_dim = 2 ** 8  # change a default hyper-parameters
#
#     train_and_evaluate(args)
#
#     """DEMO 2: Continuous action env, gym.Box2D"""
#     if_on_policy = False
#     args = Arguments(if_on_policy=if_on_policy)  # on-policy has different hyper-parameters from off-policy
#     if if_on_policy:
#         args.agent_rl = agent.AgentGaePPO  # on-policy: AgentPPO, AgentGaePPO
#     else:
#         args.agent_rl = agent.AgentModSAC  # off-policy: AgentSAC, AgentModPPO, AgentTD3, AgentDDPG
#
#     env = gym.make('Pendulum-v0')
#     env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
#     args.env = prep_env(env=env)
#     args.net_dim = 2 ** 7  # change a default hyper-parameters
#     # args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
#     # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
#
#     train_and_evaluate(args)
#
#     """DEMO 3: Custom Continuous action env: FinanceStock-v1"""
#     args = Arguments(if_on_policy=True)
#     args.agent_rl = agent.AgentGaePPO  # PPO+GAE (on-policy)
#
#     from elegantrl.env import FinanceMultiStockEnv
#     args.env = FinanceMultiStockEnv(if_train=True)  # a standard env for ElegantRL, not need decorate_env()
#     args.env_eval = FinanceMultiStockEnv(if_train=False)
#     args.break_step = int(5e6)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
#     args.net_dim = 2 ** 8
#     args.target_step = args.env.max_step
#     args.max_memo = (args.target_step - 1) * 8
#     args.batch_size = 2 ** 11
#     args.repeat_times = 2 ** 4
#     args.eval_times1 = 2 ** 4
#
#     # train_and_evaluate(args)
#     args.rollout_num = 4
#     train_and_evaluate__multiprocessing(args)
#     args.env_eval.draw_cumulative_return(args, torch)
#
#     '''DEMO 4: PyBullet(MuJoCo) Robot Env'''
#     args = Arguments(if_on_policy=True)
#     args.agent_rl = agent.AgentGaePPO  # agent.AgentPPO
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     # args.env = decorate_env(gym.make('AntBulletEnv-v0'))
#     args.env = prep_env(gym.make('ReacherBulletEnv-v0'))
#
#     args.break_step = int(4e4 * 8)  # (4e4) 8e5, UsedTime: (300s) 700s
#     args.eval_times1 = 2 ** 2
#     args.eval_times1 = 2 ** 4
#
#     args.rollout_num = 4
#     train_and_evaluate__multiprocessing(args)
#
#     # args = Arguments(if_on_policy=True)  # on-policy has different hyper-parameters from off-policy
#     # args.agent_rl = agent.AgentGaePPO  # on-policy: AgentPPO, AgentGaePPO
#     #
#     # env_name = 'AntBulletEnv-v0'
#     # assert env_name in {"AntBulletEnv-v0", "Walker2DBulletEnv-v0", "HalfCheetahBulletEnv-v0",
#     #                     "HumanoidBulletEnv-v0", "HumanoidFlagrunBulletEnv-v0", "HumanoidFlagrunHarderBulletEnv-v0"}
#     # import pybullet_envs  # for python-bullet-gym
#     # dir(pybullet_envs)
#     # args.env = decorate_env(gym.make('AntBulletEnv-v0'))
#     #
#     # args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
#     # args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
#     # args.batch_size = 2 ** 8
#     # args.max_memo = 2 ** 20
#     # args.eva_size = 2 ** 2  # for Recorder
#     # args.show_gap = 2 ** 8  # for Recorder
#     #
#     # # train_and_evaluate(args)
#     # args.rollout_num = 4
#     # train_and_evaluate__multiprocessing(args)

# def run__demo_20200202():
#     import gym  # don't worry about 'WARN: Box bound precision lowered by casting to float32'
#     import AgentZoo
#     from AgentEnv import decorate_env
#     args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters
#
#     '''DEMO 1: Discrete action env: CartPole-v0 of gym'''
#     args.agent_rl = AgentZoo.AgentDoubleDQN  # choose an DRL algorithm
#     # args.env = decorate_env(env=gym.make('CartPole-v0'))
#     # args.net_dim = 2 ** 7  # change a default hyper-parameters
#     args.env = decorate_env(env=gym.make('LunarLander-v2'))
#     args.net_dim = 2 ** 8  # change a default hyper-parameters
#     train_and_evaluate(args)
#     exit()
#
#     '''DEMO 2: Continuous action env: gym.box2D'''
#     args.env = decorate_env(env=gym.make('Pendulum-v0'))
#     args.env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
#     # args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
#     # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
#     args.agent_rl = AgentZoo.AgentSAC  # off-policy
#     train_and_evaluate(args)
#     exit()
#
#     args = Arguments(if_on_policy=True)  # on-policy has different hyper-parameters from off-policy
#     # args.env = decorate_env(env=gym.make('Pendulum-v0'))
#     args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
#     # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
#     args.agent_rl = AgentZoo.AgentPPO  # on-policy
#     train_and_evaluate(args)
#     exit()
#
#     '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
#     from AgentEnv import FinanceMultiStockEnv
#     args = Arguments(if_on_policy=True)
#     args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
#     args.agent_rl = AgentZoo.AgentPPO  # PPO+GAE (on-policy)
#
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = (args.max_step - 1) * 16
#     args.batch_size = 2 ** 11
#     args.repeat_times = 2 ** 4
#     train_and_evaluate(args)
#     exit()


# def train__demo():
#     pass
#
#     '''DEMO 1: Standard gym env CartPole-v0 (discrete action) using D3QN (DuelingDoubleDQN, off-policy)'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('CartPole-v0')
#     env = decorate_env(env, if_print=True)
#
#     from AgentZoo import AgentD3QN
#     args = Arguments(rl_agent=AgentD3QN, env=env, gpu_id=0)
#     args.break_step = int(1e5 * 8)  # UsedTime: 60s (reach target_reward 195)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     train_agent(args)
#     exit()
#
#     '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('LunarLanderContinuous-v2')
#     env = decorate_env(env, if_print=True)
#
#     from AgentZoo import AgentModSAC
#     args = Arguments(rl_agent=AgentModSAC, env=env, gpu_id=0)
#     args.break_step = int(6e4 * 8)  # UsedTime: 900s (reach target_reward 200)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     # train_agent(args)  # Train agent using single process. Recommend run on PC.
#     train_agent_mp(args)  # Train using multi process. Recommend run on Server.
#     exit()
#
#     '''DEMO 3: Custom env FinanceStock (continuous action) using PPO (PPO2+GAE, on-policy)'''
#     env = FinanceMultiStockEnv()  # 2020-12-24
#
#     from AgentZoo import AgentPPO
#     args = Arguments(rl_agent=AgentPPO, env=env)
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#     args.rollout_num = 4
#     args.if_break_early = True
#
#     args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = 1699 * 16
#     args.batch_size = 2 ** 10
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     # from AgentZoo import AgentModSAC
#     # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
#     # args.eval_times1 = 1
#     # args.eval_times2 = 2
#     #
#     # args.break_step = 2 ** 22  # UsedTime:
#     # args.net_dim = 2 ** 7
#     # args.max_memo = 2 ** 18
#     # args.batch_size = 2 ** 8
#     # args.init_for_training()
#     # train_agent_mp(args)  # train_agent(args)
#
#
# def train__discrete_action():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentDQN,  # 2014.
#         Zoo.AgentDoubleDQN,  # 2016. stable
#         Zoo.AgentDuelingDQN,  # 2016. stable and fast
#         Zoo.AgentD3QN,  # 2016+ Dueling + Double DQN (Not a creative work)
#     ][3]  # I suggest to use D3QN
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#
#     args.env = decorate_env(gym.make('CartPole-v0'), if_print=True)
#     args.break_step = int(1e4 * 8)  # (3e5) 1e4, used time 20s
#     args.reward_scale = 2 ** 0  # 0 ~ 200
#     args.net_dim = 2 ** 6
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLander-v2'), if_print=True)
#     args.break_step = int(1e5 * 8)  # (2e4) 1e5 (3e5), used time (200s) 1000s (2000s)
#     args.reward_scale = 2 ** -1  # (-1000) -150 ~ 200 (285)
#     args.net_dim = 2 ** 7
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#
# def train__continuous_action__off_policy():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentDDPG,  # 2016. simple, simple, slow, unstable
#         Zoo.AgentBaseAC,  # 2016+ modify DDPG, faster, more stable
#         Zoo.AgentTD3,  # 2018. twin critics, delay target update
#         Zoo.AgentSAC,  # 2018. twin critics, maximum entropy, auto alpha, fix log_prob
#         Zoo.AgentModSAC,  # 2018+ modify SAC, faster, more stable
#         Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
#         Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
#     ][4]  # I suggest to use ModSAC (Modify SAC)
#     # On-policy PPO is not here 'run__off_policy()'. See PPO in 'run__on_policy()'.
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     args.if_break_early = True  # break training if reach the target reward (total return of an episode)
#     args.if_remove_history = True  # delete the historical directory
#
#     env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
#     args.env = decorate_env(env, if_print=True)
#     args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
#     args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
#     args.init_for_training()
#     train_agent(args)  # Train agent using single process. Recommend run on PC.
#     # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
#     args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
#     args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
#     args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
#     args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
#     args.break_step = int(5e4 * 8)  # (4e4) 5e4
#     args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29)
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
#     args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
#     args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
#     args.batch_size = 2 ** 8
#     args.max_memo = 2 ** 20
#     args.eva_size = 2 ** 3  # for Recorder
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
#     args.break_step = int(4e6 * 4)  # (2e6) 4e6
#     args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
#     args.batch_size = (2 ** 8)
#     args.net_dim = int(2 ** 8)
#     args.max_step = 2 ** 11
#     args.max_memo = 2 ** 20
#     args.eval_times2 = 3  # for Recorder
#     args.eval_times2 = 9  # for Recorder
#     args.show_gap = 2 ** 9  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
#     args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334)
#     args.break_step = int(4e6 * 8)  # (2e6) 4e6
#     args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
#     args.max_memo = int(2 ** 21)
#     args.batch_size = int(2 ** 8)
#     args.eval_times2 = 2 ** 5  # for Recorder
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_offline_policy(args)
#     exit()
#
#
# def train__continuous_action__on_policy():
#     import AgentZoo as Zoo
#     args = Arguments(rl_agent=None, env=None, gpu_id=None)
#     args.rl_agent = [
#         Zoo.AgentPPO,  # 2018. PPO2 + GAE, slow but quite stable, especially in high-dim
#         Zoo.AgentInterPPO,  # 2019. Integrated Network, useful in pixel-level task (state2D)
#     ][0]
#
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     args.if_break_early = True  # break training if reach the target reward (total return of an episode)
#     args.if_remove_history = True  # delete the historical directory
#
#     args.net_dim = 2 ** 8
#     args.max_memo = 2 ** 12
#     args.batch_size = 2 ** 9
#     args.repeat_times = 2 ** 4
#     args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#
#     env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
#     args.env = decorate_env(env, if_print=True)
#     args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
#     args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime:  (100s) 200s
#     args.gamma = 0.9  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
#     args.break_step = int(3e5 * 8)  # (2e5) 3e5 , used time: (400s) 600s
#     args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     # exit()
#
#     args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
#     args.break_step = int(8e5 * 8)  # (4e5) 8e5 (4e6), UsedTimes: (600s) 1500s (8000s)
#     args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (325)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
#     args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTimes: 2000s (6000s)
#     args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
#     args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
#     args.reward_scale = 2 ** -3  #
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#     args.net_dim = 2 ** 9
#     args.init_for_training()
#     train_agent_mp(args)
#     exit()
#
#     import pybullet_envs  # for python-bullet-gym
#     dir(pybullet_envs)
#     args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
#     args.break_step = int(1e6 * 8)  # (4e5) 1e6 (8e6)
#     args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.net_dim = 2 ** 8
#     args.max_memo = 2 ** 11
#     args.batch_size = 2 ** 9
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)
#     exit()
#
#     # args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
#     # on-policy (like PPO) is BAD at a environment with so many random factors (like BipedalWalkerHardcore).
#     # exit()
#
#     args.env = fix_car_racing_env(gym.make('CarRacing-v0'))
#     # on-policy (like PPO) is GOOD at learning on a environment with less random factors (like 'CarRacing-v0').
#     # see 'train__car_racing__pixel_level_state2d()'
#
#
# def run__fin_rl():
#     env = FinanceMultiStockEnv()  # 2020-12-24
#
#     from AgentZoo import AgentPPO
#
#     args = Arguments(rl_agent=AgentPPO, env=env)
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#     args.rollout_num = 4
#     args.if_break_early = False
#
#     args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
#     args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
#     args.net_dim = 2 ** 8
#     args.max_step = 1699
#     args.max_memo = 1699 * 16
#     args.batch_size = 2 ** 10
#     args.repeat_times = 2 ** 4
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()
#
#     # from AgentZoo import AgentModSAC
#     #
#     # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
#     # args.eval_times1 = 1
#     # args.eval_times2 = 2
#     #
#     # args.break_step = 2 ** 22  # UsedTime:
#     # args.net_dim = 2 ** 7
#     # args.max_memo = 2 ** 18
#     # args.batch_size = 2 ** 8
#     # args.init_for_training()
#     # train_agent_mp(args)  # train_agent(args)
#
#
# def train__car_racing__pixel_level_state2d():
#     from AgentZoo import AgentPPO
#
#     '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
#     import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
#     gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
#     env = gym.make('CarRacing-v0')
#     env = fix_car_racing_env(env)
#
#     args = Arguments(rl_agent=AgentPPO, env=env, gpu_id=None)
#     args.if_break_early = True
#     args.eval_times2 = 1
#     args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.
#     args.rollout_num = 4  # (num, step, time) (8, 1e5, 1360) (4, 1e4, 1860)
#     args.random_seed += 1943
#
#     args.break_step = int(5e5 * 4)  # (1e5) 2e5 4e5 (8e5) used time (7,000s) 10ks 30ks (60ks)
#     # Sometimes bad luck (5%), it reach 300 score in 5e5 steps and don't increase.
#     # You just need to change the random seed and retrain.
#     args.reward_scale = 2 ** -2  # (-1) 50 ~ 700 ~ 900 (1001)
#     args.max_memo = 2 ** 11
#     args.batch_size = 2 ** 7
#     args.repeat_times = 2 ** 4
#     args.net_dim = 2 ** 7
#     args.max_step = 2 ** 10
#     args.show_gap = 2 ** 8  # for Recorder
#     args.init_for_training()
#     train_agent_mp(args)  # train_agent(args)
#     exit()


if __name__ == '__main__':
    demo1_discrete_action_space()
    # demo2_continuous_action_space_off_policy()
    # demo2_continuous_action_space_on_policy()
    # demo3_custom_env_fin_rl()
    # demo4_bullet_mujoco_off_policy()
    # demo4_bullet_mujoco_on_policy()
