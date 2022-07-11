import argparse
from statistics import mode
from typing import Any, List, Tuple
import numpy as np

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
# TODO: implement CNN for actor and critic model
from network import ActorCriticCNN
from torch.distributions import Categorical
# from pl_bolts.models.rl.common.networks import MLP, ActorCategorical, ActorContinous
import torch.nn.functional as F
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from env import make_mario
from icecream import ic
from log import log_video

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg("gym")


class PPO(LightningModule):
    """PyTorch Lightning implementation of `Proximal Policy Optimization.
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
    """

    def __init__(
        self,
        # env: str,
        world: int,
        stage: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        beta: float = 0.01,
        lr: float = 1e-4,
        max_episode_len: float = 200,
        hidden_size: int = 128,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        nb_optim_iters: int = 4,
        clip_ratio: float = 0.2,
        render: bool = True,
        render_freq: int = 5000,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This Module requires gym environment which is not installed yet.")

        # Hyperparameters
        self.world = world
        self.stage = stage
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.nb_optim_iters = nb_optim_iters
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.render = render
        self.render_freq = render_freq
        self.save_hyperparameters()

        # self.env = gym.make(env)
        self.env = make_mario(world, stage)
        
        self.net = ActorCriticCNN(self.env.observation_space.shape, self.hidden_size, self.env.action_space.n)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.total_episodes: int = 0
        self.total_steps: int = 0
        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = torch.FloatTensor(self.env.reset())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Passes in a state x through the network and returns the policy and a sampled action.
        Args:
            x: environment state
        Returns:
            Tuple of policy and action
        """
        logits, value = self.net(x)
        pi = Categorical(logits=logits)
        action = pi.sample()

        return pi, action, value

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.
        Args:
            rewards: list of rewards/advantages
            discount: discount factor
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode.
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def eval_1_episode(self):
        print("Evaluating...", end='\r')
        self.state = torch.FloatTensor(self.env.reset())
        self.state = self.state.unsqueeze(0)
        done = False
        episode_reward: float = 0
        frames = []
        durations = []
        while not done:
            with torch.no_grad():
                _, action, _ = self(self.state)
                action = action.squeeze(0).cpu().numpy().item()
            if self.render:
                frame = self.env.render(mode="rgb_array")
                frame = np.array(frame)
                frames.append(frame)
                durations.append(1/24)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.state = torch.FloatTensor(next_state).unsqueeze(0)
        if self.render:
            log_video(env_name='{}-{}'.format(self.world, self.stage), frames=frames, durations=durations, curr_steps=self.global_step,
                      episode_reward=episode_reward, fps=24)
        self.state = torch.FloatTensor(self.env.reset())
    
    def generate_trajectory_samples(self) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating trajectory data to train policy and value network.
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            self.state = self.state.unsqueeze(0).to(device=self.device)
            # self.state.shape: torch.Size([1, 4, 84, 84])

            with torch.no_grad():
                pi, action, value = self(self.state)
                action = action.squeeze(0)
                log_prob = pi.log_prob(action).sum(axis=-1)

            # pi: Categorical(probs: torch.Size([1, 7]), logits: torch.Size([1, 7]))
            # action: tensor([4])
            # value: tensor([[0.0516]])
            # log_prob: tensor(-1.9334)
            next_state, reward, done, _ = self.env.step(action.cpu().numpy().item())

            self.episode_step += 1
            
            if done:
                self.total_episodes += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    self.state = self.state.unsqueeze(0).to(device=self.device)
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes

                self.epoch_rewards.clear()

    def actor_loss(self, logits, action, logp_old, adv) -> Tensor:
        pi = Categorical(logits=logits)
        logp = pi.log_prob(action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, value, qval) -> Tensor:
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.
        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch
        # state.shape: torch.Size([B, 1, 4, 84, 84])
        # action.shape: torch.Size([B])
        # old_logp.shape: torch.Size([B])
        # qval.shape: torch.Size([B])
        # adv.shape: torch.Size([B])
        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("episodes", self.total_episodes, on_step=False, on_epoch=True, prog_bar=True)
        self.log("avg_ep_len", self.avg_ep_len, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, on_step=False, on_epoch=True)

        logits, value = self.net(state.squeeze(1))
        # logits.shape: torch.Size([B, 7])
        # value.shape: torch.Size([B, 1])
        new_policy = F.softmax(logits, dim=1)
        new_m = Categorical(probs=new_policy)
        entropy_loss = torch.mean(new_m.entropy())
        
        actor_loss = self.actor_loss(logits, action, old_logp, adv)
        self.log("actor_loss", actor_loss, prog_bar=True, logger=True)

        critic_loss = self.critic_loss(value, qval)
        self.log("critic_loss", critic_loss, logger=True)
        
        loss = actor_loss + critic_loss - self.beta * entropy_loss
        self.log("total_loss", loss)
        
        return loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:
        if self.global_step % self.render_freq == 0:
            test_score = self.eval_1_episode()
            self.log("test_score", test_score, on_epoch=True, prog_bar=True)

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        return optimizer

    def optimizer_step(self, *args, **kwargs):
        """Run ``nb_optim_iters`` number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="CartPole-v0")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
        parser.add_argument("--lr_actor", type=float, default=3e-4, help="learning rate of actor network")
        parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic network")
        parser.add_argument("--max_episode_len", type=int, default=1000, help="capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=512, help="batch_size when training network")
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=2048,
            help="how many action-state pairs to rollout for trajectory collection per epoch",
        )
        parser.add_argument(
            "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
        )
        parser.add_argument(
            "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
        )

        return parser


def cli_main() -> None:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = Trainer.add_argparse_args(parent_parser)

    parser = PPO.add_model_specific_args(parent_parser)
    args = parser.parse_args()

    model = PPO(1, 1, **vars(args))

    seed_everything(0)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()