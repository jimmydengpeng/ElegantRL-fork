import random, time
import sys
import gym
import argparse
from elegantrl_marl.agents import AgentPPO
from elegantrl_marl.train.config import Arguments, get_gym_env_args, set_attr_for_env
from elegantrl_marl.train.run import *
from utils import debug_msg, debug_print, get_space_dim, LogLevel

from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from metadrive.manager.traffic_manager import TrafficMode

envs = dict(
    Roundabout=MultiAgentRoundaboutEnv,
    Intersection=MultiAgentIntersectionEnv,
    Tollgate=MultiAgentTollgateEnv, # can't run
    Bottleneck=MultiAgentBottleneckEnv,
    Parkinglot=MultiAgentParkingLotEnv, # can't run
    PGMA=MultiAgentMetaDrive
)
''' add args parser '''
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', '-g', type=int, default=0)
parser.add_argument('--worker', type=int, default=4)
parser.add_argument('--thread', type=int, default=4)
parser.add_argument('--desc', type=str, default="")
parser.add_argument('--render', action='store_true')
ter_args = parser.parse_args()

''' platform '''
can_render = not sys.platform == 'darwin'
render_in_linux = False
RENDER_IN_PYGAME = True

''' set args '''
METADRIVE_ENV_CONFIG = dict(
    use_render=False,
    start_seed=random.randint(0, 1000),
    environment_num=100,
    num_agents=15,
    manual_control=False,
    crash_done=False,
    horizon=1000, # default in env
)
ENV_NAME = "Roundabout"
env_func = envs[ENV_NAME]
env = env_func(METADRIVE_ENV_CONFIG)

ENV_CONFIG = dict(
    env_name=ENV_NAME,
    if_discrete=False,
    max_step=1000, # for eval, FIXME
    target_return=400, 
    #FIXME add succ rate
)
ENV_CONFIG = dict(ENV_CONFIG, **METADRIVE_ENV_CONFIG)
ENV_CONFIG["state_dim"] = get_space_dim(env.observation_space)
ENV_CONFIG["action_dim"] = get_space_dim(env.action_space)

args = Arguments(AgentPPO, env=env, env_func=env_func, env_args=ENV_CONFIG)

# debug_print("args.action_dim", args=args.action_dim, inline=True)
# debug_print("args.state_dim", args=args.state_dim, inline=True)

args.target_step = args.max_step * 4 # horizon_len for exploration
args.gamma = 0.98
args.learning_rate = 3e-4
args.ratio_clip = 0.2
args.eval_times = 2 ** 5
args.horizon_len = 8192 # args.batch_size * 4
args.learner_gpus = ter_args.gpuid
args.desc = ter_args.desc
args.if_use_gae = True
# args.repeat_times = 1

render_args = dict(mode="top_down", film_size=(1000, 1000)) if RENDER_IN_PYGAME else {} #for render in eval

# set processes & threads
if sys.platform == "darwin": # only for my mac M1 Pro chip
    debug_msg("On Mac ‍💻")
    args.learner_gpus = 0
    args.worker_num = 1
    args.thread_num = 1
    args.desc = f"{args.worker_num}w{args.thread_num}t_test"
else:
    args.worker_num = ter_args.worker
    args.thread_num = ter_args.thread

# args.print()
debug_print("Agent name:", level=LogLevel.INFO, args=args.agent_class.__name__, inline=True)
debug_print("Env name:", args=args.env_name, level=LogLevel.INFO, inline=True)

''' EXPERIMENT CONSTANTS '''
CONTINUOUS_TRAINING = False

if __name__ == '__main__':
    times = []
    debug_msg(">>> Multi-Process <<<", level=LogLevel.INFO)

    if CONTINUOUS_TRAINING:
        for i in range(1, 11):
            debug_msg(f"=== train_and_evaluate_mp: {i} of 10 ===", level=LogLevel.SUCCESS)
            start = time.time()
            train_and_evaluate_mp(args)
            end = time.time()
            debug_print(f"=== train_and_evaluate_mp: {i} of 10 ===", args=(end-start), level=LogLevel.SUCCESS)
            times.append(end-start)
        debug_print(">>>> total times:", args=times)
    else:
        train_and_evaluate_mp(args)


    # elif FLAG == "MultiGPU":
    #     args.learner_gpus = [0, 1, 2, 3]
    #     train_and_evaluate_mp(args)
    # elif FLAG == "Tournament-based":
    #     args.learner_gpus = [
    #         [i, ] for i in range(4)
    #     ]  # [[0,], [1, ], [2, ]] or [[0, 1], [2, 3]]
    #     python_path = "../bin/python3"
    #     train_and_evaluate_mp(args, python_path) #type: ignore #TODO # multiple processing 
