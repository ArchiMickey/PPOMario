import argparse
from collections import deque
import random
from statistics import mode
from typing import Any, List, Tuple
import numpy as np

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor, device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
# TODO: implement CNN for actor and critic model
from torch.distributions import Categorical
# from pl_bolts.models.rl.common.networks import MLP, ActorCategorical, ActorContinous
import torch.nn.functional as F
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from .network import ActorCriticCNN
from .env import make_mario
from .multienv import MultiEnv
from icecream import ic
from .log import log_video
from tqdm import tqdm

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
        world: int = 1,
        stage: int = 1,
        gamma: float = 0.9,
        lam: float = 0.99,
        beta: float = 0.01,
        lr: float = 1e-4,
        max_episode_len: float = 200,
        hidden_size: int = 128,
        batch_epoch: int = 10,
        batch_size: int = 32,
        num_workers: int = 1,
        steps_per_epoch: int = 32,
        nb_optim_iters: int = 1,
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
        self.batch_epoch = batch_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.render = render
        self.render_freq = render_freq
        self.save_hyperparameters()

        # self.env = gym.make(env)
        self.demo_env = make_mario(world, stage)
        self.env = MultiEnv(world, stage, num_workers)
        
        self.net = ActorCriticCNN(self.demo_env.observation_space.shape, self.hidden_size, self.demo_env.action_space.n)

        self.epoch_rewards = deque(maxlen=100)

        self.total_episodes: int = 0
        self.total_steps: int = 0
        self.episode_step = 0
        
        self.step_rewards = deque(maxlen=100)
        self.end_rewards = deque(maxlen=100)
        self.end_steps = deque(maxlen=100)
        for _ in range(100):
            self.step_rewards.append(0)
            self.end_rewards.append(0)
            self.end_steps.append(0)
            
        self.ep_rewards = [0 for _ in range(self.num_workers)]
        self.ep_steps = [0 for _ in range(self.num_workers)]
        self.ep_best_reward = 0

        self.state = torch.from_numpy(self.env.reset_all())

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
    
    def calc_advantage(self, rewards: Tensor, dones: Tensor, values: Tensor, last_value: Tensor) -> List[float]:
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
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * self.gamma * self.lam
            gae = gae + reward + self.gamma * last_value.detach() * (1 - done) - value.detach()
            last_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        
        # return advantages
        advantages = R - values
        return R, advantages

    def eval_1_episode(self):
        pbar = tqdm(desc="Evaluating", leave=False)
        state = torch.FloatTensor(self.demo_env.reset()).unsqueeze(0)
        done = False
        episode_reward: float = 0
        frames = []
        durations = []
        while not done:
            with torch.no_grad():
                _, action, _ = self(state.cuda())
                action = action.squeeze(0).cpu().numpy().item()
                pbar.update(1)
            if self.render:
                frame = self.demo_env.render(mode="rgb_array")
                frame = np.array(frame)
                frames.append(frame)
                durations.append(1/24)
            next_state, reward, done, _ = self.demo_env.step(action)
            episode_reward += reward
            state = torch.FloatTensor(next_state).unsqueeze(0)
        if self.render:
            log_video(env_name='{}-{}'.format(self.world, self.stage), frames=frames, durations=durations, curr_steps=self.global_step,
                      episode_reward=episode_reward, fps=24)
        pbar.close()
        return episode_reward
    
    def generate_trajectory_samples(self) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating trajectory data to train policy and value network.
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """
        states = []
        actions = []
        logp = []
        dones = []
        rewards = []
        values = []
        batch_qval = []
        batch_adv = []
        
        self.ep_best_reward = 0
        
        
        for _ in tqdm(range(self.steps_per_epoch), leave=False, desc="Sampling Batch",
                      bar_format="{desc}: {percentage:3.0f}%|{bar:10}{r_bar}"):
            self.state = self.state.to(device=self.device)
            # self.state.shape: torch.Size([1, 4, 84, 84])

            with torch.no_grad():
                pi, action, value = self(self.state)
                action = action.squeeze(0)
                log_prob = pi.log_prob(action)
            
            # pi: Categorical(probs: torch.Size([4, 7]), logits: torch.Size([P, A]))
            # action: tensor([5, 6, 4, 6])
            # value: tensor([[-0.0681],
            #             [ 0.0205],
            #             [ 0.0011],
            #             [ 0.0863]])
            # log_prob: tensor([-2.1821, -1.8158, -1.8556, -1.8856])
            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            for i in range(self.num_workers):
                self.ep_steps[i] += 1
                self.step_rewards.append(reward[i])
                self.ep_rewards[i] += reward[i]
            
            self.episode_step += 1
            self.total_steps += self.num_workers
            for idx, worker_done in enumerate(done):
                if worker_done:
                    self.total_episodes += 1
                    self.end_steps.append(self.ep_steps[idx])
                    self.end_rewards.append(self.ep_rewards[idx])
                    if self.ep_rewards[idx] > self.ep_best_reward:
                        self.ep_best_reward = self.ep_rewards[idx]
                    self.ep_rewards[idx] = 0
                    self.ep_steps[idx] = 0

            states.append(self.state)
            actions.append(action)
            logp.append(log_prob)
            dones.append(done)
            rewards.append(reward)
            values.append(value)

            self.state = torch.FloatTensor(next_state)

        self.state = self.state.to(device=self.device) # self.state.shape: torch.Size([P, 4, 84, 84])
        with torch.no_grad():
            _, _, last_value = self(self.state)
            last_value = last_value.squeeze() # last_value.shape: torch.Size([P])
        
        batch_states = torch.cat(states)
        batch_actions = torch.cat(actions)
        batch_logp = torch.cat(logp)
        ep_values = torch.cat(values).squeeze().detach()
        ep_rewards = torch.FloatTensor(rewards)
        ep_dones = torch.FloatTensor(dones)
        
        # self.batch_states.shape: torch.Size([B, 4, 84, 84])
        # self.batch_actions.shape: torch.Size([B])
        # self.batch_logp.shape: torch.Size([B])
        # self.ep_values.shape: torch.Size([B])
        # self.ep_dones.shape: torch.Size([B])
        # rewards.shape: torch.Size([B, P])
        # dones.shape: torch.Size([B, P])
                        
        # advantage
        qval, adv = self.calc_advantage(ep_rewards, ep_dones, ep_values, last_value)
        batch_qval += qval
        batch_adv += adv
        # logs
        self.epoch_rewards.append(sum(ep_rewards) / self.num_workers)
        # reset params 
        self.episode_step = 0
        self.state = torch.FloatTensor(self.env.reset_all())

        train_data = list(zip(
            batch_states, batch_actions, batch_logp, batch_qval, batch_adv
        ))
        for _ in range(self.batch_epoch):
            random.shuffle(train_data)
            for data in train_data:
                state, action, old_logp, qval, adv = data
                yield state, action, old_logp, qval, adv

    def actor_loss(self, logits, action, logp_old, adv) -> Tensor:
        pi = Categorical(logits=logits)
        logp = pi.log_prob(action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, qval, value) -> Tensor:
        loss_critic = F.smooth_l1_loss(qval, value.squeeze())
        return loss_critic

    def on_train_epoch_start(self):
        self.log("episodes", self.total_episodes, on_epoch=True, prog_bar=True)
        self.log("avg_ep_len", sum(self.end_steps) / len(self.end_steps), on_epoch=True)
        self.log("avg_ep_reward", sum(self.end_rewards) / len(self.end_rewards), prog_bar=True, on_epoch=True)
        self.log("avg_reward", sum(self.step_rewards) / len(self.step_rewards), on_epoch=True)
        self.log("best_ep_reward", self.ep_best_reward, on_epoch=True, prog_bar=True)
        return
    
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
        # state.shape: torch.Size([B, 4, 84, 84])
        # action.shape: torch.Size([B])
        # old_logp.shape: torch.Size([B])
        # qval.shape: torch.Size([B])
        # adv.shape: torch.Size([B])
        
        logits, value = self.net(state)
        actor_loss = self.actor_loss(logits, action, old_logp, adv)
        critic_loss = self.critic_loss(qval, value)
        entropy_loss = torch.mean(Categorical(logits=logits).entropy())
        total_loss = actor_loss + critic_loss - self.beta * entropy_loss
        self.log("actor_loss", actor_loss, on_step=True, logger=True)
        self.log("critic_loss", critic_loss, on_step=True, logger=True)
        self.log("total_loss", total_loss, on_step=True, logger=True)
        
        return total_loss
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, unused: int = 0) -> None:
        if self.global_step % self.render_freq == 0:
            test_score = self.eval_1_episode()
            self.log("test_score", test_score, on_step=True, prog_bar=True)        

    def test_step(self, *args, **kwargs):
        return self.eval_1_episode()
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, capturable=True)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=350)
        
        return [optimizer], [lr_scheduler]

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
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader([0])


def cli_main() -> None:
    model = PPO(world=1, stage=1, num_workers=6, )

    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()