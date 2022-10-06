from ast import parse
from collections import defaultdict
import argparse
import random, time
import sys
import os.path as osp
from tqdm import tqdm
import gym
from metadrive import MetaDriveEnv
import numpy as np
import torch
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from utils import debug_msg, debug_print, get_space_dim, LogLevel, pretty_time

from train_MetaDrive_Single import metadrive_env_config, parser


'''NEVER render in MacOS'''
'''render if terminal arg'''
# parser = argparse.ArgumentParser()
# parser.add_argument('--render', action='store_true')
ter_args = parser.parse_args()

CAN_RENDER = not sys.platform == 'darwin'
RENDER_TER = ter_args.render 
DEFAULT_RENDER = False

use_render = all((CAN_RENDER, any((RENDER_TER, DEFAULT_RENDER)))) 


# build env
# env_config = dict(
#     use_render=all([can_render, linux_render]),
#     traffic_density=0.1,
#     environment_num=100,
#     random_agent_model=False,
#     random_lane_width=False,
#     random_lane_num=True,
#     map=4,  # seven block
#     # map="SCrRX",
#     start_seed=random.randint(0, 1000)
# )
metadrive_env_config['use_render'] = use_render
env = MetaDriveEnv(metadrive_env_config)

env_args = {
    "env_num": 1,
    "env_name": "MetaDrive-Single",
    "max_step": 1000,
    "state_dim": get_space_dim(env.observation_space),
    "action_dim": get_space_dim(env.action_space),
    "if_discrete": False,
}

# cwd
# "MetaDrive-single_PPO_0_20220922_191206"
# agent_dir_name = "MetaDrive-single_PPO_0_20220922_002256"
# agent_dir_name = "MetaDrive-Single_PPO_0_20220922_225932"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20221002_201613_6w6t_test"
# agent_dir_name = "MetaDrive-Single_PPO_0_20220927_202657"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20220927_231734_w6t6"
# agent_dir_name = "MetaDrive-Single_PPO_0_20220923_082311"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20220928_110248_7w7t"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20221003_020700_6w6t_test"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20221004_004143_6w6t_test"
# agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20221005_015719_6w6t_test"
agent_dir_name = "MetaDrive-Single-Agent_PPO_0_20221006_030356_6w6t_test"
exp_dir_name = osp.join("experiments", agent_dir_name)
cwd = osp.join(osp.dirname(osp.abspath(__file__)), exp_dir_name)

# create args
args = Arguments(AgentPPO, env_args=env_args)
args.cwd = cwd

# create agent
agent = AgentPPO(args.net_dim, args.state_dim, args.action_dim, gpu_id=args.learner_gpus, args=args)
agent.save_or_load_agent(cwd=args.cwd, if_save=False)

# first state to decide first action
state = env.reset() 

debug_print("agent.device:", agent.device, inline=True)
# debug_print("state", state, inline=True)

get_action = agent.act.get_deterministic_action # for eval
get_a_to_e = agent.act.get_a_to_e 
episode_reward = 0.0
returns = []
done = False
totle_step_i = 0
epi_step_i = 0
t_0 = time.time()

if_test_success_rate = True
TEST_TIMES = 1000
cur_test_time = 1
test_done_dict = defaultdict(int)

test_start_time = time.time()

with tqdm(total=TEST_TIMES, desc="Test Progress") as pbar:
    while True:
        ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0) # 插入维度 1, 变成 torch.Size([1, state_dim])
        ten_a = get_action(ten_s.to(agent.device)).cpu() # action: torch.Size([1, action_dim])
        action = get_a_to_e(ten_a)[0].detach().cpu().numpy()
        # debug_print("action:", action, inline=True)
        next_s, reward, done, info = env.step(action)
        state = env.reset() if done else next_s
        assert isinstance(reward, float)
        episode_reward += reward

        if done:
            cur_test_time += 1
            pbar.update(1)
            if cur_test_time > TEST_TIMES:
                test_end_time = time.time()
                break

            # test success rate
            for k, v in info.items():
                if isinstance(v, bool) and v == True:
                    # debug_print(k, v, inline=True)
                    test_done_dict[k] += 1
            
            returns.append(episode_reward)
            # print(f'Step {epi_step_i:>6}, Episode return {episode_reward:8.3f}')
            episode_reward = 0
            done = False
            epi_step_i = 0
            done = False
        totle_step_i += 1
        epi_step_i += 1

        if epi_step_i % 1000 == 0:
            # debug_msg(f"step: {epi_step_i}, episode_reward_by_far: {episode_reward}, time_used: {int(time.time() - t_0)}", level=LogLevel.INFO)
            t_0 = time.time()
        # env.render()

if if_test_success_rate:
    test_time_spent = test_end_time - test_start_time
    debug_print(f"Total {TEST_TIMES} test times in", f"{pretty_time(test_time_spent)}", level=LogLevel.INFO, inline=True)

    succ_rate = test_done_dict["arrive_dest"] / TEST_TIMES
    debug_msg(f"{'Success Rate:':<18} {succ_rate * 100:.2f}%", level=LogLevel.SUCCESS)

    crash_rate = test_done_dict["crash"] / TEST_TIMES
    debug_msg(f"{'  Crash Rate:':<20} {crash_rate * 100:.2f}%", level=LogLevel.ERROR)

    out_of_road_rate = test_done_dict["out_of_road"] / TEST_TIMES
    debug_msg(f"{'  Out_of_road Rate:':<20} {out_of_road_rate * 100:.2f}%", level=LogLevel.ERROR)

debug_print("Avg Return:", f"{np.array(returns).mean():.2f}", inline=True)

import matplotlib.pyplot as plt
x = np.linspace(1, len(returns), len(returns))
plt.plot(x, returns)
plt.show()