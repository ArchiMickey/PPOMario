from typing import Tuple
import numpy as np
from torch import Tensor, nn
import torch.nn.functional as F
import torch


def init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class PPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(state_dim)
        
        self.action_head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_)
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.action_head(conv_out), self.critic_head(conv_out)

class PPG(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.actor = PPGActor(state_dim, hidden_dim, num_actions)
        self.critic = PPOCritic(state_dim, hidden_dim)
        self.apply(init_)
    

class PPOActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(state_dim)
        
        self.action_head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1),
        )
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.action_head(conv_out)

class PPGActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(state_dim)
        
        self.action_head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return self.action_head(conv_out), self.value_head(conv_out)

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(state_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _get_conv_out(self, shape):
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out)
        