import random, time
import sys
import gym
# gym.logger.set_level(40)  # Block warning
import argparse
from elegantrl.agents import AgentPPO
from elegantrl.train.config import Arguments, get_gym_env_args, set_attr_for_env
from elegantrl.train.run import *
from utils import debug_msg, debug_print, get_space_dim, LogLevel

'''[
    'MetaDrive-validation-v0', 'MetaDrive-10env-v0', 'MetaDrive-100envs-v0', 'MetaDrive-1000envs-v0', 'SafeMetaDrive-validation-v0', 'SafeMetaDrive-10env-v0', 'SafeMetaDrive-100envs-v0', 'SafeMetaDrive-1000envs-v0', 'MARLTollgate-v0', 'MARLBottleneck-v0', 'MARLRoundabout-v0', 'MARLIntersection-v0', 'MARLParkingLot-v0', 'MARLMetaDrive-v0'
]'''
from metadrive import MetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', '-g', type=int, default=0)
parser.add_argument('--worker', type=int, default=4)
parser.add_argument('--thread', type=int, default=4)
parser.add_argument('--desc', type=str, default="")
ter_args = parser.parse_args()


can_render = not sys.platform == 'darwin'
render_in_linux = False

metadrive_env_config = dict(
        use_render=all([can_render, render_in_linux]),
        traffic_density=0.1,
        traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
        environment_num=100,
        # max_step_per_agent=1000,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        # lane_num=2,
        map=4,  # seven block
        start_seed=random.randint(0, 1000)
)
# env_config = dict(
#     manual_control=False,
#     traffic_density=0.15,
#     # traffic_mode=TrafficMode.Respawn,  # "Respawn", "Trigger"
#     environment_num=100,
#     # random_agent_model=True,
#     random_lane_width=False,
#     random_lane_num=True,
#     map=4,  # seven block
#     # start_seed=random.randint(0, 1000)
#     start_seed=random.randint(0, 1000)
# )
env = MetaDriveEnv(metadrive_env_config)

# 7 major env args, will be set to arg's attributes
env_args = {
    "env_num": 1,
    "env_name": "MetaDrive-Single-Agent",
    "max_step": 1000,
    "state_dim": get_space_dim(env.observation_space),
    "action_dim": get_space_dim(env.action_space),
    "if_discrete": False,
    "target_return": 300,
    # "id": "BipedalWalker-v3",
}
# set_attr_for_env(env, env_args)
# debug_print("env.env_num:", env.env_num, inline=True)
# debug_print("env.env_name:", env.env_name, inline=True)
# get_gym_env_args(env, if_print=True)
env_func = gym.make


args = Arguments(AgentPPO, env=env, env_func=env_func, env_args=env_args)

debug_print("args.state_dim", args=args.state_dim, inline=True)
debug_print("args.action_dim", args=args.action_dim, inline=True)

args.target_step = args.max_step * 4 # horizon_len for exploration
args.gamma = 0.98
args.learning_rate = 3e-4
args.eval_times = 2 ** 4
args.horizon_len = 8192 # args.batch_size * 4
args.learner_gpus = ter_args.gpuid
args.desc = ter_args.desc

# set processes & threads
if sys.platform == "darwin": # only for my mac M1 Pro chip
    debug_msg("On Mac")
    args.learner_gpus = 0
    args.worker_num = 2
    args.thread_num = 2
    args.desc = "2w2t"
else:
    args.worker_num = ter_args.worker
    args.thread_num = ter_args.thread

# args.print()
debug_print("Agent name:", level=LogLevel.INFO, args=args.agent_class.__name__, inline=True)
debug_print("Env   name:", args=args.env_name, level=LogLevel.INFO, inline=True)

''' EXPERIMENT CONSTANTS '''
CONTINUOUS_TRAINING = False
FLAG = "MultiProcess"

if __name__ == '__main__':
    times = []
    if FLAG == "SingleProcess":
        debug_msg(">>> Single Process <<<", level=LogLevel.INFO)
        train_and_evaluate(args)

    elif FLAG == "MultiProcess":
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
    else:
        raise ValueError(f"Unknown flag: {FLAG}")
