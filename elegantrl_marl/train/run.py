import multiprocessing as mp
import os
import time
from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from elegantrl_marl.agents.AgentBase import AgentBase
from elegantrl_marl.agents.AgentPPO import AgentPPO
from elegantrl_marl.train.config import Arguments
from elegantrl_marl.train.config import build_env
from elegantrl_marl.train.evaluator import Evaluator
from elegantrl_marl.train.replay_buffer import ReplayBuffer, ReplayBufferList
from utils import LogLevel, debug_msg, debug_print, pretty_time


def init_agent(args: Arguments, gpu_id: int, env=None) -> AgentPPO: #FIXME
    debug_msg(f"<{__name__}.py/init_agent> initializing agent...", level=LogLevel.INFO)
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    # if args.cwd_exists_pth():
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        agent.last_states_dict = env.reset()
    return agent

def init_buffer(args: Arguments, gpu_id: int) -> ReplayBufferList:
    debug_msg("initializing buffer...", level=LogLevel.INFO)
    # on-policy
    return ReplayBufferList()

def init_evaluator(args: Arguments, gpu_id: int) -> Evaluator:
    debug_msg(f"<{__name__}.py/init_evaluator> calling building_env...")
    eval_env = build_env(args.env)
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
    # agent.states= env.reset() #type:ignore # assgin new attr `state` to agent
    # debug_msg("intialization done!", level=LogLevel.SUCCESS)

    if args.if_off_policy:
        debug_msg("off-policy", level=LogLevel.INFO)
        trajectory = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode)
        buffer.update_buffer(trajectory)

    '''copy args then del args'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_allow_break = args.if_allow_break
    if_off_policy = args.if_off_policy
    del args
    ''' ||> (delete obj) 仅删除上文定义的args，如果之前实例化的时候传入Agent和Evaluator类的参数有args，引用不会被删除 
        (实例化agen/buffer/evaluator的时候也并没有传入args的引用，仅使用其属性的值)'''

    '''start training'''
    if_train = True
    while if_train:
        '''trajectory:
        [ 
            states: tensor(Size([78, 24(state_dim)])),
            rewards: tensor(Size([78, 1])),
            dones: tensor(Size([78, 1])), 
            actions: tensor(Size([78, 4(actions_dim)])), 
            noises: tensor(Size([78, 5(noises_dim)])), 
        ]
        '''
        # debug_print("horizon_len", horizon_len)
        trajectory = agent.explore_env(env, horizon_len) 
        # debug_print("trajectory", len(trajectory))
        # debug_print("state", trajectory[0].size())
        # debug_msg(f"<{__name__}.py/.train_and_evaluate> trajectory:")
        # for i, t in enumerate(trajectory):
        #     debug_print(f"t{i} type:", args=type(t), inline=True)
        #     debug_print(f"t{i} size:", args=t.size(), inline=True)
        #     debug_print(f"t{i} dim:", args=t.dim(), inline=True)
        #     debug_print(f"t{i}", args=t)
        #     debug_print(f"t{i}[0] type", args=type(t[0]), inline=True)
        #     debug_print(f"t{i}[0] size", args=t[0].size(), inline=True)
        #     debug_print(f"t{i}[0]", args=(t[0]))
        steps = horizon_len
        # steps, r_exp = buffer.update_buffer((trajectory,))
        if if_off_policy:
            buffer.update_buffer(trajectory)
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        else:
            r_exp = trajectory[3].mean().item()
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
            torch.set_grad_enabled(False)

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
    mp.set_start_method(method='spawn', force=True)  # force to use 'spawn' methods

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


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps: int, r_exp: float, logging_tuple: tuple) -> Tuple[bool, bool]:
        if self.pipe1.poll():  # 轮询是否有数据可读 # if_evaluator_idle
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

        debug_print("UsedTime:", pretty_time(time.time() - evaluator.start_time), level=LogLevel.SUCCESS, inline=True)
        debug_print("SavedDir:", cwd, level=LogLevel.SUCCESS)
        # print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()
            self.pipe0.send((False, False))


class PipeWorker:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent: AgentBase) -> List[Tuple]: 
        """
        Pass agent.act.state_dict() to Worker Proccesses to explore env then recv traj and return.
        (提供一个接口让 Learner 调用并得到 trajectoy_list)

        return:
            list of WORKER_NUM of traj: 
            [
                (
                    states: Tensor [horizon_len, state_dim], 
                    actions: Tensor [horizon_len, action_dim], 
                    logprobs: Tensor [horizon_len], 
                    rewards: Tensor [horizon_len, 1], 
                    undones: Tensor [horizon_len, 1] 
                ),
                ... * WORKER_NUM 
            ]
        """
        act_dict = agent.act.state_dict() # dict containing a whole state of the module

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
        horizon_len = args.horizon_len
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.horizon_len)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict) # Copy parameters and buffers from state_dict into this module and its descendants
            trajectory = agent.explore_env(env, horizon_len)
            self.pipes[worker_id][0].send(trajectory)



class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args: Arguments, comm_eva: PipeEvaluator, comm_exp: PipeWorker):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus
        cwd = args.cwd

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)
        steps = 0

        '''loop'''
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent) # list of tuple * worker_num: [(t, t, t, t, t),...]
            trajectory = [torch.cat(t_list) for t_list in list(zip(*traj_list))]
            steps = trajectory[0].size()[0]
            # steps, r_exp = buffer.update_buffer(traj_list)
            r_exp = trajectory[3].mean().item()
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
            torch.set_grad_enabled(False)
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})
            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(cwd, if_save=True)
        debug_print("Learner: Save in:", cwd, level=LogLevel.SUCCESS)
        # print(f'| Learner: Save in {cwd}')

        # env = build_env(env_func=args.env_func, env_args=args.env_args)
        # buffer.get_state_norm(
            # cwd=cwd,
            # state_avg=getattr(env, 'state_avg', 0.0),
            # state_std=getattr(env, 'state_std', 1.0),
        # ) # FIXME
        # if hasattr(buffer, 'save_or_load_history'):
        #     print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
        #     buffer.save_or_load_history(cwd, if_save=True)



def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
