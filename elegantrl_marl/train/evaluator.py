import os
import time
from typing import Any, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from elegantrl_marl.train.config import Arguments
from utils import debug_msg, sec2hms


class Evaluator:
    def __init__(self, cwd: str, agent_id: int, eval_env, args: Arguments):
        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

        self.cwd = cwd
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = args.eval_gap
        self.eval_times1 = max(1, int(args.eval_times / np.e))
        self.eval_times2 = max(0, int(args.eval_times - self.eval_times1))
        self.target_return = args.target_return

        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")
        '''
        ################################################################################
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        '''

    '''called after every update_net, passing in args of last update_net'''
    def evaluate_save_and_plot(self, act, steps: int, r_exp: float, log_tuple: tuple) -> Tuple[bool, bool]:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            '''evaluate first time'''
            rewards_steps_list = [get_cumulative_returns_and_step(self.eval_env, act)
                                  for _ in range(self.eval_times1)]
            rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)

            r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step

            '''evaluate second time'''
            if r_avg > self.r_max:
                rewards_steps_list.extend([get_cumulative_returns_and_step(self.eval_env, act)
                                           for _ in range(self.eval_times2)])
                rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
                r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step

            r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step

            '''save the policy network'''
            if_save = r_avg > self.r_max
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                act_path = f"{self.cwd}/actor__{self.total_step:012}_{self.r_max:09.3f}.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

            '''record the training information'''
            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                      f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                      f"{'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                      f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                      f"{self.used_time:>8}  ########")

            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")

            if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
                self.eval_env.curriculum_learning_for_evaluator(r_avg)

            '''plot learning curve figure'''
            if len(self.recorder) == 0:
                print("| save_npy_draw_plot() WARNING: len(self.recorder)==0")
                return None

            np.save(self.recorder_path, self.recorder)

            '''draw plot and save as figure'''
            train_time = int(time.time() - self.start_time)
            total_step = int(self.recorder[-1][0])
            # save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
            save_title =f"step_time_(avgStep)_maxR_{int(total_step)}_{int(train_time)}_({sec2hms(train_time)}*{int(total_step/train_time)})_{self.r_max:.3f}"


            save_learning_curve(self.recorder, self.cwd, save_title)

        return if_reach_goal, if_save

    def save_or_load_recoder(self, if_save: bool):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

"""util"""

def get_cumulative_returns_and_step(env, act: nn.Module) -> Tuple[float, int]:  
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pth'

    env = build_env(env_func=env_func, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = 0
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if if_discrete:
            tensor_action = act(tensor_state).argmax(dim=1) 
        else:
            tensor_action = act(tensor_state) # action_avg
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward
        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1
    return returns, steps


def save_learning_curve(
        recorder: List[Tuple[Any]], cwd: str = '.',
        save_title: str = 'learning curve', fig_name: str = 'plot_learning_curve.jpg'
):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder) # type: ignore
    assert isinstance(recorder, np.ndarray)
    steps = recorder[:, 0] # type: ignore # x-axis is training steps
    r_avg = recorder[:, 1] # type: ignore
    r_std = recorder[:, 2] # type: ignore
    r_exp = recorder[:, 3] # type: ignore
    obj_c = recorder[:, 4] # type: ignore
    obj_a = recorder[:, 5] # type: ignore

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]): # action std error 随机策略分布的标准差
        other = recorder[:, plot_i] # type: ignore
        ax10.plot(steps, other, label='stdE', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


if __name__ == '__main__':
    pass
