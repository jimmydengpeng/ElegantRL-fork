import gym
import torch
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from utils import debug_print

# create args
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
# args.cwd = "./experiments/BipedalWalker-v3_PPO_0_20220919_195116"
# test
exp_dir = "./experiments/" 
# _name = "BipedalWalker-v3_PPO_1_20220926_114642_"
_name = "BipedalWalker-v3_PPO_0_20220928_195753"
args.cwd = exp_dir + _name
agent = AgentPPO(args.net_dim, args.state_dim, args.action_dim, gpu_id=args.learner_gpus, args=args)
agent.save_or_load_agent(cwd=args.cwd, if_save=False)

env = build_env(env_func=args.env_func, env_args=args.env_args)
state = env.reset() # first state to decide first action

debug_print("agent.device:", agent.device, inline=True)
debug_print("state", state, inline=True)

get_action = agent.act.get_action
get_a_to_e = agent.act.get_a_to_e 
episode_reward = 0
done = False
totle_step_i = 0
epi_step_i = 0
while totle_step_i < 2 ** 11 or not done:
    # action = agent.select_action(state)

    ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0) # 在指定位置插入维度 1, 变成 torch.Size([1, 24])
    ten_a, ten_n = [
        ten.cpu() for ten in get_action(ten_s.to(agent.device))
    ] # action: torch.Size([1, 4]),  noise: torch.Size([1, 4])
    next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].detach().numpy())

    state = env.reset() if done else next_s

    episode_reward += reward
    if done:
        print(f'Step {epi_step_i:>6}, Episode return {episode_reward:8.3f}')
        episode_reward = 0
        done = False
        epi_step_i = 0
        done = False
    totle_step_i += 1
    epi_step_i += 1
    
    env.render()