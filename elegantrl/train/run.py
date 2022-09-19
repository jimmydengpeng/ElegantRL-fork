import multiprocessing as mp
from operator import imod
import os
import time
from typing import List, Tuple, Union

import gym
import numpy as np
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.train.config import Arguments
from elegantrl.train.config import build_env
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferList
from utils import LogLevel, debug_msg, debug_print


def init_agent(args: Arguments, gpu_id: int, env=None) -> AgentBase:
    debug_msg("<run.py/init_agent> initializing agent...", level=LogLevel.INFO)
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            state = env.reset() 
            if isinstance(state, Tuple): # -> Tuple[obs: array, info: dict] for NEWEST VERSION of GYM!!!
                debug_msg("New version Gym, reset() -> tuple", level=LogLevel.WARNING)
                state = state[0]
            assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
            assert state.shape in {(args.state_dim,), args.state_dim}
            states = [state, ]
        else: # vector env ?? 
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_agent_isaacgym(args, gpu_id: int, env=None):
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_buffer(args: Arguments, gpu_id: int) -> Union[ReplayBuffer, ReplayBufferList]:
    debug_msg("initializing buffer...", level=LogLevel.INFO)
    if args.if_off_policy:
        debug_msg("  off-policy buffer...", level=LogLevel.INFO)
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_memo,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        assert isinstance(args.cwd, str)
        buffer.save_or_load_history(args.cwd, if_save=False)

    else: # on-policy
        debug_msg("  on-policy buffer...", level=LogLevel.INFO)
        buffer = ReplayBufferList()
    return buffer


def init_evaluator(args: Arguments, gpu_id: int) -> Evaluator:
    debug_msg("initializing evaluator...", level=LogLevel.INFO)
    eval_func = args.eval_env_func if getattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if getattr(args, "eval_env_args") else args.env_args
    debug_msg("<run.py/init_evaluator> building env...")
    eval_env = build_env(args.env, eval_func, eval_args)
    assert isinstance(args.cwd, str)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator


def train_and_evaluate(args: Arguments):
    """
    The training and evaluating loop.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    """
    torch.set_grad_enabled(False)
    args.init_before_training() # if args.env is None, build a env & assign it to args.env
    gpu_id = args.learner_gpus
    assert isinstance(gpu_id, int)

    '''init'''
    env = args.env
    steps = 0
    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)
    assert env is not None and isinstance(env, gym.Env)
    agent.state = env.reset() # assgin new attr `state` to agent
    debug_msg("intialization done!", level=LogLevel.SUCCESS)

    if args.if_off_policy:
        debug_msg("off-policy", level=LogLevel.INFO)
        trajectory = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode)
        buffer.update_buffer(trajectory)

    '''start training'''
    '''copy args then del args'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_allow_break = args.if_allow_break
    if_off_policy = args.if_off_policy
    del args #
    ''' (delete obj) 仅删除上文定义的args，如果之前实例化的时候传入Agent和Evaluator类的参数有args，引用不会被删除 
        (实例化agen/buffer/evaluator的时候也并没有传入args的引用，仅使用其属性的值)'''

    debug_msg("start training...", LogLevel.INFO)
    if_train = True
    while if_train:
        '''trajectory
        [ 
            states: tensor(Size([78, 24(state_dim)])),
            rewards: tensor(Size([78, 1(state_dim)])),
            dones: tensor(Size([78, 1(state_dim)])), 
            actions: tensor(Size([78, 1(actions_dim)])), 
            noises: tensor(Size([78, 1(noises_dim)])), 
        ]
        '''
        trajectory = agent.explore_env(env, horizon_len) 
        # for i, t in enumerate(trajectory):
        #     debug_print(f"t{i}", args=len(t))
        #     debug_print(f"t{i}", args=type(t))
        #     debug_print(f"t{i}", args=t.size())
        #     debug_print(f"t{i}", args=t)
        #     debug_print(f"t{i}", args=type(t[0]))
        #     debug_print(f"t{i}", args=(t[0]))
        # print(trajectory)

        steps, r_exp = buffer.update_buffer((trajectory,))
        if if_off_policy:
            buffer.update_buffer(trajectory)
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        else:
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
            torch.set_grad_enabled(False)

        #r_exp = agent.reward_tracker.mean()
        #step_exp = agent.step_tracker.mean()
        (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.act, steps, r_exp, logging_tuple)
        dont_break = not if_allow_break
        not_reached_goal = not if_reach_goal
        stop_dir_absent = not os.path.exists(f"{cwd}/stop")
        if_train = (
                (dont_break or not_reached_goal)
                and evaluator.total_step <= break_step
                and stop_dir_absent
        )
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)


'''train multiple process'''


def train_and_evaluate_mp(args: Arguments):
    args.init_before_training()

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = PipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    for p in process:
        p.start()

    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent: AgentBase):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args: Arguments, worker_id: int):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        '''loop'''
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


# import wandb
class PipeLearner:
    def __init__(self):
        # wandb.init(project="DDPG_H")
        pass

    @staticmethod
    def run(args: Arguments, comm_eva: mp.Pipe, comm_exp: mp.Pipe):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus
        cwd = args.cwd
        # wandb.init(project="DDPG_H")

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})
            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(cwd, if_save=True)
        print(f'| Learner: Save in {cwd}')

        env = build_env(env_func=args.env_func, env_args=args.env_args)
        buffer.get_state_norm(
            cwd=cwd,
            state_avg=getattr(env, 'state_avg', 0.0),
            state_std=getattr(env, 'state_std', 1.0),
        )
        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps: int, r_exp: float, logging_tuple: tuple) -> (bool, bool):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save_agent = self.pipe1.recv()
            act_state_dict = act.state_dict().copy()  # deepcopy(act.state_dict())
        else:
            if_train = True
            if_save_agent = False
            act_state_dict = None

        self.pipe1.send((act_state_dict, steps, r_exp, logging_tuple))
        return if_train, if_save_agent

    def run(self, args: Arguments):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        '''loop'''
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        save_gap = args.save_gap
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        save_counter = 0
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_save_and_plot(act, steps, r_exp, logging_tuple)

                save_counter += 1
                if save_counter == save_gap:
                    save_counter = 0
                    torch.save(act.state_dict(), f"{cwd}/actor_{evaluator.total_step:012}.pth")
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            self.pipe0.send((if_train, if_save))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()
            self.pipe0.send((False, False))


def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
