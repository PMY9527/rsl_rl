from .actor_critic_recurrent import ActorCriticRecurrent
import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

class ActorCriticRecurrentResidual(ActorCriticRecurrent):
    '''
        # Actor
        obs [3+3+3+29+29+29=96]
        ↓
        LSTM Layer 1 [256]
        ↓
        LSTM Layer 2 [256]  ← 2层LSTM
        ↓
        LSTM输出 [256]
        ↓
        MLP Layer 1 [256] + ELU
        ↓
        MLP Layer 2 [256] + ELU  ← 3层MLP
        ↓
        MLP Layer 3 [256] + ELU
        ↓
        输出层 [29] ← Residual


        # Critic
        privileged_obs [96+3+58=157]
        ↓
        LSTM Layer 1 [256]
        ↓
        LSTM Layer 2 [256]  ← 2层LSTM
        ↓
        LSTM输出 [256]
        ↓
        MLP Layer 1 [256] + ELU
        ↓
        MLP Layer 2 [256] + ELU  ← 3层MLP
        ↓
        MLP Layer 3 [256] + ELU
        ↓
        输出层 [1] ← V(s)

    '''

    def act(self, obs, masks=None, hidden_state=None, qref=None): # for train
        
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        out_mem = self.memory_a(obs_tensor, masks, hidden_state).squeeze(0)
        self._update_distribution(out_mem)
        # self.distribution = Normal(mean, std)
        a_res = self.distribution.sample()

        # Residual
        if qref is not None:
            return qref + a_res
        return a_res
    
    def act_inference(self, obs, qref=None): #TODO to return mean instead of sampling. (for play)
        obs_tensor = self.get_actor_obs(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)

    def evaluate(self, obs, masks = None, hidden_state = None, gt_linear_vel=None, motion=None):
        obs_tensor = self.get_critic_obs(obs)
        
        if gt_linear_vel is not None and motion is not None:
            obs_tensor = torch.cat([obs_tensor, gt_linear_vel, motion], dim=-1)

        obs_tensor = self.critic_obs_normalizer(obs_tensor)    
        out_mem = self.memory_c(obs_tensor, masks, hidden_state).squeeze(0)
        value = self.critic(out_mem)

        return value