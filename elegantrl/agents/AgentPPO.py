from distutils.log import debug
from tkinter.tix import ExFileSelectBox
from typing import Tuple, List
import gym
from gym.spaces.box import Box
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
            args, "ratio_clip", 0.2
        )  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(
            args, "lambda_entropy", 0.01
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
        ary_state = self.last_states[0] # <numpy.ndarray> [state_dim: e.g. 24]

        get_action = self.act.get_action_logprob
        convert = self.act.convert_action_for_env
        for i in range(horizon_len): # ~> only collect *horizon_len* length of tajectories
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device) # TenosrSize: [state_dim]
            action, logprob = [t.squeeze() for t in get_action(state.unsqueeze(0))] # 在指定位置插入维度1, 变成 Size([1, state_dim])

            
            # for _ in range(10000):
            #     action, logprob = [t.squeeze() for t in get_action(state.unsqueeze(0))] # 在指定位置插入维度1, 变成 Size([1, state_dim])
            #     actions1.append(action[0].item())
            #     logps1.append(logprob.item())

            # for _ in range(10000):
            #     action2, noise = [t.squeeze() for t in self.act.get_action_noise(state.unsqueeze(0))] # 在指定位置插入维度1, 变成 Size([1, state_dim])
            #     actions2.append(action2[0].item())
            #     noise_ = noise.unsqueeze(0)
            #     old_logprob = self.act.get_old_logprob(action2, noise_).squeeze(0)
            #     logps2.append(old_logprob.item())
            #     # debug_print("action1", self.act.get_logprob_entropy(state.unsqueeze(0), action))
            #     # debug_print("action1", self.act.get_logprob_entropy(state.unsqueeze(0), action2))

     
            """ action size[action_dim]"""
            """ logprob size[]"""

            # assert isinstance(env.action_space, Box)
            # ary_action = np.clip(action.cpu().numpy(), env.action_space.low, env.action_space.high)
            # ary_action = action.detach().tanh().cpu().numpy()

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action) # => only put action of <numpy.ndarray> to env
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.last_states[0] = ary_state # 总是把最后一次与环境交互得到的（还没有返回的）state 作为 self.states[0]

        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones


    def explore_vec_env(self, env, target_step, random_exploration=None):
        pass


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
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action) #FIXME obj_entropy should be >=0
            '''   /       |->Tensor size[]''' # [-0.53, 0]
            '''Tensor size[batch_size]''' # (-oo, 0)
            # debug_print("new_logprob", new_logprob[:10])
            # debug_print("logprob", logprob[:10])
            # debug_print("obj_entropy", obj_entropy)
            # debug_print("obj_entropy", obj_entropy.size())
            ratio = (new_logprob - logprob.detach()).exp() # 1 tensor size[batchz_size]
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean() # 求期望Expectation

            # obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()

        action_std = getattr(self.act, 'action_std_log', torch.zeros(1)).exp().mean()

        return obj_critics / update_times, obj_actors / update_times, action_std.item()


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

        if self.last_states:
            next_state = torch.tensor(np.array(self.last_states), dtype=torch.float32).to(self.device)
            next_value = self.cri(next_state).detach()[0, 0]
        else:
            next_value = torch.zeros(1)[0] #FIXME only for multi-process

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages




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
