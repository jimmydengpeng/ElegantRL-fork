import gym
from elegantrl.agents import AgentPPO
from elegantrl.train.config import get_gym_env_args, Arguments
from elegantrl.train.run import *
from utils import debug_msg, debug_print

gym.logger.set_level(40)  # Block warning

get_gym_env_args(gym.make("BipedalWalker-v3"), if_print=False)

env_func = gym.make
env_args = {
    "env_num": 1,
    "env_name": "BipedalWalker-v3",
    "max_step": 1600,
    "state_dim": 24,
    "action_dim": 4,
    "if_discrete": False,
    "target_return": 300,
    "id": "BipedalWalker-v3",
}
args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

debug_print("Agent name:", level=LogLevel.INFO, args=args.agent_class.__name__, inline=True)

args.target_step = args.max_step * 4
args.gamma = 0.98
args.eval_times = 2 ** 4
args.horizon_len = args.batch_size
# debug("args:", level=LogLevel.INFO)
# args.print()


if __name__ == '__main__':
    flag = "SingleProcess"
    # flag = "MultiProcess"
    if flag == "SingleProcess":
        debug_msg(">>> Single Process <<<", level=LogLevel.INFO)
        args.learner_gpus = 0
        train_and_evaluate(args)
    elif flag == "MultiProcess":
        debug_msg(">>> Multi-Process...", level=LogLevel.INFO)
        args.learner_gpus = 0
        train_and_evaluate_mp(args)
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
