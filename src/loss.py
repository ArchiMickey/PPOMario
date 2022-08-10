from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

def calc_advantage(gamma: float, lam: float, rewards: Tensor, dones: Tensor, values: Tensor, last_value: Tensor, device, use_ppg=False) -> Tuple[Tensor, Tensor]:
        """Calculate the advantage given rewards, state values, and the last value of episode.
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        # discounted cumulative reward
        gae = 0
        rewards = rewards.to(device)
        dones = dones.to(device)
        if use_ppg:
            values = values + last_value.mean()
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * gamma * lam
            # ic(last_value.shape, done.shape, value.shape)
            gae = gae + reward + gamma * last_value.detach() * (1 - done) - value.detach()
            last_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.stack(R)
        
        # return advantages
        advantages = R - values
        return R, advantages

def cal_actor_loss(action_probs, action, logp_old, adv, clip_ratio, alpha) -> Tensor:
        pi = Categorical(action_probs)
        logp = pi.log_prob(action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio * alpha, 1 + clip_ratio * alpha) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

def cal_critic_loss(qval, value) -> Tensor:
        loss_critic = F.smooth_l1_loss(qval, value.squeeze())
        return loss_critic

def clipped_value_loss(values, rewards, old_values, clip):
        value_clipped = old_values + (values - old_values).clamp(-clip, clip)
        value_loss_1 = (value_clipped.flatten() - rewards) ** 2
        value_loss_2 = (values.flatten() - rewards) ** 2
        return torch.mean(torch.max(value_loss_1, value_loss_2))
    
def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)
