from tkinter.tix import ExFileSelectBox
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorPPO, ActorDiscretePPO, CriticPPO, SharePPO
from elegantrl.train.config import \
    Arguments  # bug fix:NameError: name 'Arguments' is not defined def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
from elegantrl.train.replay_buffer import ReplayBufferList
from utils import LogLevel, debug_msg, debug_print

"""[ElegantRL.2021.12.12](github.com/AI4Fiance-Foundation/ElegantRL)"""


class AgentPPO(AgentBase):
    """
    Bases: ``AgentBase``

    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(
            self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None
    ):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(
            args, "ratio_clip", 0.25
        )  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(
            args, "lambda_entropy", 0.02
        )  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(
            args, "lambda_entropy", 0.98
        )  # could be 0.95~0.99, GAE (ICLR.2016.)

        if getattr(
                args, "if_use_gae", False
        ):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, horizon_len) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param horizon_len: the total step for the interaction.
        :return:(
                    states: Tensor [horizon_len, state_dim], 
                    actions: Tensor [horizon_len, action_dim], 
                    logprobs: Tensor [horizon_len], 
                    rewards: Tensor [horizon_len, 1], 
                    undones: Tensor [horizon_len, 1] 
                )
        """
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        '''需要第一个state去从模型得到action, 再让env去step, 得到下一个state...'''
        ary_state = self.states[0] # <numpy.ndarray> [state_dim: e.g. 24]

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len): # ~> only collect *horizon_len* length of tajectories
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device) # TenosrSize: [state_dim]
            action, logprob = [t.squeeze() for t in get_action(state.unsqueeze(0))] # 在指定位置插入维度1, 变成 Size([1, state_dim])
            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action) # => only put action of <numpy.ndarray> to env
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state # 总是把最后一次与环境交互得到的（还没有返回的）state 作为 self.states[0]

        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def explore_one_env_old(self, env, target_step, random_exploration=None) -> list:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        traj_list = [] # containing tuples of (ten_s: <Tensor>, r: <float>, d: <Boolean>, a <Tensor>, n <Tensor>)
        last_done = [
            0,
        ]
        assert self.states is not None
        '''需要第一个state去从模型得到action，再让env去step，得到下一个state...'''
        state = self.states[0] # <numpy.ndarray> [24]
        assert isinstance(state, np.ndarray)

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e # action of network to action of environment
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0) # 在指定位置插入维度 1, 变成 torch.Size([1, 24])
            ten_a, ten_n = [
                ten.cpu() for ten in get_action(ten_s.to(self.device))
            ] # action: torch.Size([1, 4]),  noise: torch.Size([1, 4])
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different
            step_i += 1
            state = env.reset() if done else next_s
        self.states[0] = state
        last_done[0] = step_i
        # debug_print(f"<{__name__}.py/explore_one_env> traj_list:", args=len(traj_list), inline=True)
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step, random_exploration=None) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(get_a_to_e(ten_a))

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer: ReplayBufferList):
        """
        Update the neural networks by sampling batch data from `ReplayBuffer`.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )
            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)


        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()

        debug_print("obj_actors:", obj_actors / update_times, level=LogLevel.ERROR, inline=True)
        debug_print("advantages:", advantages, level=LogLevel.ERROR, inline=True)
        debug_print("reward_sums:", reward_sums, level=LogLevel.ERROR, inline=True)
        debug_print("actions:", actions, level=LogLevel.ERROR, inline=True)

        # debug_print("advantages size", advantages.size(), level=LogLevel.ERROR)

        # if (obj_actors / update_times) > 30: exit()

        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def update_net_old(self, buffer):
        """
        Update the neural networks by sampling batch data from `ReplayBuffer`.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]
            buf_len = buf_state.shape[0]

            """get buf_r_sum, buf_logprob"""
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i: i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        """update network"""
        obj_critic = None
        obj_actor = None
        # debug_print(f"<{__name__}.py/AgentPPO.update_net> buf_len:", args=buf_len, inline=True)
        # debug_print(f"<{__name__}.py/AgentPPO.update_net> self.batch_size:", args=self.batch_size, inline=True)
        assert buf_len >= self.batch_size
        for _ in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(
                buf_len,
                size=(self.batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
            self, buf_len, buf_reward, buf_mask, buf_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(
            self, buf_len, ten_reward, ten_mask, ten_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        """
        :param rewards: tensor of size [horizon_len/buffer_len, 1]
        :param undones: tensor of size [horizon_len/buffer_len, 1]
        :param values: tensor of size  [horizon_len/buffer_len]
        :return: tensor of size        [horizon_len/buffer_len]
        """
        advantages = torch.empty_like(values)  # advantage value
        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        if self.states:
            next_state = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
            next_value = self.cri(next_state).detach()[0, 0]
        else:
            next_value = torch.zeros(1)[0] #FIXME only for multi-process

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages


class AgentPPO_isaacgym(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.get_reward_sum = self.get_reward_sum_gae
        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_gae_adv', 0.95)  # could be 0.50~0.99, GAE (ICLR.2016.)
        self.act_update_gap = getattr(args, 'act_update_gap', 1)

    def explore_one_env(self, env, horizon_len: int) -> list:
        traj_list = []
        last_dones = [0, ]
        state = self.state[0]

        i = 0
        done = False
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < horizon_len or not done:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            actions, noises = [item.cpu() for item in get_action(state.to(self.device))]  # different
            next_state, reward, done, _ = env.step(convert(actions)[0].numpy())

            traj_list.append((state, reward, done, actions, noises))  # different

            i += 1
            state = env.reset() if done else next_state
        self.state[0] = state
        last_dones[0] = i
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def explore_vec_env(self, env, horizon_len: int, random_exploration: bool) -> list:
        obs = torch.zeros((horizon_len, self.env_num) + (self.state_dim,)).to(self.device)
        actions = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        noises = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        rewards = torch.zeros((horizon_len, self.env_num)).to(self.device)
        dones = torch.zeros((horizon_len, self.env_num)).to(self.device)

        state = self.state if self.if_use_old_traj else env.reset()
        done = torch.zeros(self.env_num).to(self.device)

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            obs[i] = state
            dones[i] = done

            action, noise = get_action(state)
            next_state, reward, done, _ = env.step(convert(action))
            state = next_state

            actions[i] = action
            noises[i] = noise
            rewards[i] = reward

            self.current_rewards += reward
            self.current_lengths += 1
            env_done_indices = torch.where(done == 1)
            self.reward_tracker.update(self.current_rewards[env_done_indices])
            self.step_tracker.update(self.current_lengths[env_done_indices])
            not_dones = 1.0 - done.float()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        self.state = state

        return (obs, actions, noises, self.reward_scale * rewards, dones, state, done), horizon_len * self.env_num

    def update_net(self, buffer):
        buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum = self.get_reward_sum(buffer)
        buffer_size = buf_state.size()[0]
        assert buffer_size >= self.batch_size

        '''update network'''
        obj_critic_list = []
        obj_actor_list = []
        indices = np.arange(buffer_size)
        for epoch in range(self.repeat_times):
            np.random.shuffle(indices)

            for i in range(0, buffer_size, self.batch_size):
                minibatch_indices = indices[i:i + self.batch_size]
                state = buf_state[minibatch_indices]
                r_sum = buf_r_sum[minibatch_indices]
                adv_v = buf_adv[minibatch_indices]
                adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
                action = buf_action[minibatch_indices]
                logprob = buf_logprob[minibatch_indices]

                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                obj_critic = self.criterion(value, r_sum) * self.lambda_critic
                self.optimizer_update(self.cri_optimizer, obj_critic)
                if self.if_cri_target:
                    self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

                '''PPO: Surrogate objective of Trust Region'''
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.optimizer_update(self.act_optimizer, obj_actor)

                obj_critic_list.append(obj_critic.item())
                obj_actor_list.append(-obj_actor.item())

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return np.array(obj_critic_list).mean(), np.array(obj_actor_list).mean(), action_std_log.item()  # logging_tuple

    def get_reward_sum_gae(self, buffer) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.
        """
        with torch.no_grad():
            buf_state, buf_action, buf_noise, buf_reward, buf_done, next_state, next_done = buffer
            next_state_value = self.cri(next_state)

            buf_adv = torch.zeros_like(buf_reward).to(self.device)
            values = torch.zeros_like(buf_reward).to(self.device)

            lastgaelam = 0
            horizon_len = buf_state.size()[0]
            for t in reversed(range(horizon_len)):
                values[t] = self.cri(buf_state[t]).reshape(-1, )
                if t == horizon_len - 1:
                    nextnonterminal = 1.0 - next_done
                    next_values = next_state_value
                else:
                    nextnonterminal = 1.0 - buf_done[t + 1]
                    next_values = values[t + 1]
                    delta = buf_reward[t] + self.gamma * next_values * nextnonterminal - values[t]
                    buf_adv[t] = lastgaelam = delta + self.gamma * self.lambda_gae_adv * nextnonterminal * lastgaelam
            buf_r_sum = buf_adv + values

            buf_state = buf_state.reshape((-1,) + (self.state_dim,))
            buf_action = buf_action.reshape((-1,) + (self.action_dim,))
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise.reshape((-1,) + (self.action_dim,)))
            buf_logprob = buf_logprob.reshape(-1, )
            buf_adv = buf_adv.reshape(-1, )
            buf_r_sum = buf_r_sum.reshape(-1, )

        return buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum


class AgentDiscretePPO(AgentPPO):
    """
    Bases: ``AgentPPO``

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(
            self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


# FIXME: this class is incomplete
class AgentSharePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
            self,
            net_dim=256,
            state_dim=8,
            action_dim=2,
            reward_scale=1.0,
            gamma=0.99,
            learning_rate=1e-4,
            if_per_or_gae=False,
            env_num=1,
            gpu_id=0,
    ):
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if if_per_or_gae:
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.act = self.cri = SharePPO(state_dim, action_dim, net_dim).to(self.device)

        self.cri_optim = torch.optim.Adam(
            [
                {"params": self.act.enc_s.parameters(), "lr": learning_rate * 0.9},
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.a_std_log,
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
            ],
            lr=learning_rate,
        )
        self.criterion = torch.nn.SmoothL1Loss()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [
                ten.to(self.device) for ten in buffer
            ]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            """get buf_r_sum, buf_logprob"""
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i: i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                    self.lambda_a_value / torch.std(buf_adv_v) + 1e-5
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(
                buf_len, size=(batch_size,), requires_grad=False, device=self.device
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            # it is obj_actor  # todo net.py sharePPO
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple


class AgentPPOHterm(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args=None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self,
                   buffer: ReplayBufferList):  # bug fix:ImportError: cannot import name 'ReplayBufferList' from 'elegantrl.train.replay_buffer'
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            batch_size = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + batch_size]) for i in range(0, buf_len, batch_size)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term_k(buf_state, buf_action, buf_mask, buf_reward)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if i % self.h_term_update_gap == 0:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + self.get_obj_h_term_k()  # todo H-term
                self.optimizer_update(self.act_optimizer, obj_actor)

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), action_std_log.item()  # logging_tuple
