import torch
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical


def cal_actor_loss(logits, action, logp_old, adv, clip_ratio, alpha) -> Tensor:
        pi = Categorical(logits=logits)
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
