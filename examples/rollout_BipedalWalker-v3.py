import torch
from elegantrl.run import train_and_evaluate
from elegantrl.config import Arguments
from elegantrl.train.config import build_env
from elegantrl.agents.AgentPPO import AgentPPO

# train and save
args = Arguments(env=build_env('BipedalWalker-v3'), agent=AgentPPO())

args.cwd = "BipedalWalker-v3_PPO_0_20220919_195116"

# test
agent = AgentPPO()
agent.init(args.net_dim, args.state_dim, args.action_dim)
agent.save_or_load_agent(cwd=args.cwd, if_save=False)

env = build_env('BipedalWalker-v3')
state = env.reset()
episode_reward = 0
for i in range(2 ** 10):
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)

    episode_reward += reward
    if done:
        print(f'Step {i:>6}, Episode return {episode_reward:8.3f}')
        break
    else:
        state = next_state
    env.render()