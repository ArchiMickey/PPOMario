from typing import Tuple
import numpy as np
from torch import Tensor, nn
import torch

from icecream import ic


class ActorCriticCNN(nn.Module):
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
        
        conv_out = self._get_conv_out(state_dim)
        
        self.actor_head = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(conv_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x) -> Tuple[Tensor, Tensor]:
        """Forward pass through network
        
        Args:
            x: input to network

        Returns:
            action log probs (logits), value
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.actor_head(conv_out), self.critic_head(conv_out)