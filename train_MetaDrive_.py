import random, time
import gym
import argparse
from elegantrl.agents import AgentPPO
from elegantrl.train.config import Arguments, get_gym_env_args, set_attr_for_env
from elegantrl.train.run import *
from utils import debug_msg, debug_print, get_space_dim, LogLevel

'''[
    'MetaDrive-validation-v0', 'MetaDrive-10env-v0', 'MetaDrive-100envs-v0', 'MetaDrive-1000envs-v0', 'SafeMetaDrive-validation-v0', 'SafeMetaDrive-10env-v0', 'SafeMetaDrive-100envs-v0', 'SafeMetaDrive-1000envs-v0', 'MARLTollgate-v0', 'MARLBottleneck-v0', 'MARLRoundabout-v0', 'MARLIntersection-v0', 'MARLParkingLot-v0', 'MARLMetaDrive-v0'
]'''
from metadrive import MetaDriveEnv

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', '-g', type=int, default=0)
parser.add_argument('--worker', type=int, default=2)
parser.add_argument('--thread', type=int, default=8)
ter_args = parser.parse_args()
# debug_print("ter_args.gpuid: ", ter_args.gpuid)

gym.logger.set_level(40)  # Block warning

env_config = dict(
    use_render=False,
    manual_control=False,
    traffic_density=0.1,
    environment_num=100,
    random_agent_model=True,
    random_lane_width=True,
    random_lane_num=True,
    map=4,  # seven block
    start_seed=random.randint(0, 1000)
)
env = MetaDriveEnv(env_config)

# 7 major env args, will be set to arg's attributes
env_args = {
    "env_num": 1,
    "env_name": "MetaDrive-single",
    "max_step": 1000,
    "state_dim": get_space_dim(env.observation_space),
    "action_dim": get_space_dim(env.action_space),
    "if_discrete": False,
    "target_return": 500,
    # "id": "BipedalWalker-v3",
}
# set_attr_for_env(env, env_args)
# debug_print("env.env_num:", env.env_num, inline=True)
# debug_print("env.env_name:", env.env_name, inline=True)
# get_gym_env_args(env, if_print=True)
env_func = gym.make
# env_args = {
#     "env_num": 1,
#     "env_name": "BipedalWalker-v3",
#     "max_step": 1600,
#     "state_dim": 24,
#     "action_dim": 4,
#     "if_discrete": False,
#     "target_return": 300,
#     "id": "BipedalWalker-v3", 
# }

args = Arguments(AgentPPO, env=env, env_func=env_func, env_args=env_args)

debug_print("args.state_dim", args=args.state_dim)
debug_print("args.action_dim", args=args.action_dim)

args.target_step = args.max_step * 4
args.gamma = 0.98
args.eval_times = 2 ** 4
args.horizon_len = args.batch_size
args.learner_gpus = ter_args.gpuid
args.worker_num = ter_args.worker
args.thread_num = ter_args.thread

# args.print()
debug_print("Agent name:", level=LogLevel.INFO, args=args.agent_class.__name__, inline=True)
debug_print("Env   name:", args=args.env_name, level=LogLevel.INFO, inline=True)

if __name__ == '__main__':
    # flag = "SingleProcess"
    flag = "MultiProcess"
    times = []
    if flag == "SingleProcess":
        debug_msg(">>> Single Process <<<", level=LogLevel.INFO)
        train_and_evaluate(args)

    elif flag == "MultiProcess":
        debug_msg(">>> Multi-Process <<<", level=LogLevel.INFO)
        for i in range(1, 11):
            debug_msg(f"=== train_and_evaluate_mp: {i} of 10 ===", level=LogLevel.SUCCESS)
            start = time.time()
            train_and_evaluate_mp(args)
            end = time.time()
            debug_print(f"=== train_and_evaluate_mp: {i} of 10 ===", args=(end-start), level=LogLevel.SUCCESS)
            times.append(end-start)

    elif flag == "MultiGPU":
        args.learner_gpus = [0, 1, 2, 3]
        train_and_evaluate_mp(args)
    elif flag == "Tournament-based":
        args.learner_gpus = [
            [i, ] for i in range(4)
        ]  # [[0,], [1, ], [2, ]] or [[0, 1], [2, 3]]
        python_path = "../bin/python3"
        train_and_evaluate_mp(args, python_path) #type: ignore #TODO # multiple processing
    else:
        raise ValueError(f"Unknown flag: {flag}")

    debug_print(">>>> total times:", args=times)